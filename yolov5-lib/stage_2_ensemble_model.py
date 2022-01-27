from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path


def voc2yolo(bboxes, image_height=720, image_width=1280):
    """
    voc  => [x1, y1, x2, y2]
    yolo => [xmid, ymid, w, h] (normalized)
    """

    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / image_height

    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]

    bboxes[..., 0] = bboxes[..., 0] + w / 2
    bboxes[..., 1] = bboxes[..., 1] + h / 2
    bboxes[..., 2] = w
    bboxes[..., 3] = h

    return bboxes


def yolo2voc(bboxes, image_height=720, image_width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y2]

    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

    return bboxes


def coco2yolo(bboxes, image_height=720, image_width=1280):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """

    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    # normolizinig
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / image_height

    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]] / 2

    return bboxes


def yolo2coco(bboxes, image_height=720, image_width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]

    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    # denormalizing
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    # converstion (xmid, ymid) => (xmin, ymin)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2

    return bboxes


def voc2coco(bboxes, image_height=720, image_width=1280):
    bboxes = voc2yolo(bboxes, image_height, image_width)
    bboxes = yolo2coco(bboxes, image_height, image_width)
    return bboxes


def coco2voc(bboxes, image_height=720, image_width=1280):
    bboxes = coco2yolo(bboxes, image_height, image_width)
    bboxes = yolo2voc(bboxes, image_height, image_width)
    return bboxes


def format_prediction(bboxes, confs):
    # bboxes voc
    annot = ""
    if len(confs) == 0:
        return annot

    else:
        bboxes = voc2coco(bboxes)
        for idx in range(len(bboxes)):
            xmin, ymin, w, h = bboxes[idx]
            conf = confs[idx]
            annot += f'{conf:.4f} {xmin:.3f} {ymin:.3f} {w:.3f} {h:.3f}'
            annot += ' '
        annot = annot.strip(' ')
    return annot


def decode_pred_annotation(anno):
    if anno == "":
        return [], []
    anno = np.array(anno.split(" ")).reshape(-1, 5).astype(float)
    conf = anno[:, 0]
    bbox = anno[:, 1:]
    return conf, bbox


def normalize_bbox(bboxes, image_height=720, image_width=1280):
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / image_height
    return bboxes


def denormalize_bbox(bboxes, image_height=720, image_width=1280):
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height
    return bboxes


def wbf_ensemble(pred_files, save_file, wbf_iou=0.5, wbf_min_conf=0.0, weights=None):
    dfs = [pd.read_csv(f).sort_values(['video_id', 'video_frame']).reset_index(drop=True).fillna("")
           for f in pred_files]
    print(f"Preds csv shape {[df.shape for df in dfs]}")
    ensemble_df = dfs[0].copy()
    ensemble_df['preds'] = ''

    nrows = dfs[0].shape[0]
    for r in tqdm(range(nrows)):
        confs = []
        bboxes = []
        labels = []
        for df in dfs:
            conf, bbox = decode_pred_annotation(df.preds[r])
            if len(conf) == 0:
                confs.append([])
                bboxes.append([])
                labels.append([])
                continue

            label = [0] * len(conf)
            bbox = np.clip(normalize_bbox(coco2voc(bbox)), 0, 0.9999)

            confs.append(conf)
            bboxes.append(bbox)
            labels.append(label)
        # print(bboxes)
        esb_bboxes, esb_confs, _ = weighted_boxes_fusion(bboxes, confs, labels, weights=weights,
                                                         iou_thr=wbf_iou, skip_box_thr=wbf_min_conf)
        # print(esb_bboxes)
        esb_bboxes = denormalize_bbox(np.array(esb_bboxes))
        # print(esb_bboxes)
        # print("-" * 30)
        esb_confs = np.array(esb_confs)
        ensemble_df.preds[r] = format_prediction(esb_bboxes, esb_confs)

    ensemble_df.to_csv(save_file, index=None)
    print(f"Save in {save_file}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pred_files", type=str, help="preds csv")
#     parser.add_argument("--save", type=str, help="result save path/ file name")
#     parser.add_argument("--nms_conf", default=None, type=float)
#     parser.add_argument("--wbf_iou", default=None, type=float)
#     args = parser.parse_args()
#     Path(args.save).parent.mkdir(parents=True, exist_ok=True)
#
#     # pred_files = eval(args.pred_files)
#     pred_files = args.pred_files.split(",")
