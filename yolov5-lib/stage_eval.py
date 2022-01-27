import os

from pathlib import Path
import glob
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams['figure.figsize'] = 30, 30
np.set_printoptions(precision=3, suppress=True)
from ast import literal_eval
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm

tqdm.pandas()
from norfair import Detection, Tracker
import os
import cv2
import glob
import shutil
import sys
import torch
from PIL import Image


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


def load_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_bboxes(img, bboxes, classes, class_ids, colors=None, show_classes=None,
                bbox_format='yolo', class_name=False, line_thickness=2):
    image = img.copy()
    show_classes = classes if show_classes is None else show_classes
    colors = (0, 255, 0) if colors is None else colors

    if bbox_format == 'yolo':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = round(float(bbox[0]) * image.shape[1])
                y1 = round(float(bbox[1]) * image.shape[0])
                w = round(float(bbox[2]) * image.shape[1] / 2)  # w/2
                h = round(float(bbox[3]) * image.shape[0] / 2)

                voc_bbox = (x1 - w, y1 - h, x1 + w, y1 + h)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls,
                             line_thickness=line_thickness)

    elif bbox_format == 'coco':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                w = int(round(bbox[2]))
                h = int(round(bbox[3]))

                voc_bbox = (x1, y1, x1 + w, y1 + h)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls if class_name else str(cls_id),
                             line_thickness=line_thickness)

    elif bbox_format == 'voc':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                x2 = int(round(bbox[2]))
                y2 = int(round(bbox[3]))
                voc_bbox = (x1, y1, x2, y2)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls if class_name else str(cls_id),
                             line_thickness=line_thickness)
    else:
        raise ValueError('wrong bbox format')

    return image


def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes


np.random.seed(32)
colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) \
          for idx in range(1)]


def decode_annotations(annotaitons_str):
    """decode annotations in string to list of dict"""
    return literal_eval(annotaitons_str)


def load_image_with_annotations(video_id, video_frame, image_dir, annotaitons_str):
    img = load_image(video_id, video_frame, image_dir)
    annotations = decode_annotations(annotaitons_str)
    if len(annotations) > 0:
        for ann in annotations:
            cv2.rectangle(img, (ann['x'], ann['y']),
                          (ann['x'] + ann['width'], ann['y'] + ann['height']),
                          (255, 0, 0), thickness=2, )
    return img


def draw_predictions(img, pred_bboxes):
    img = img.copy()
    if len(pred_bboxes) > 0:
        for bbox in pred_bboxes:
            conf = bbox[0]
            x, y, w, h = bbox[1:].round().astype(int)
            cv2.rectangle(img, (x, y),
                          (x + w, y + h),
                          (0, 255, 255), thickness=2, )
            cv2.putText(
                img,
                f"{conf:.2}",
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                thickness=1,
            )
    return img


def show_img(img, bboxes, name, color, bbox_format='voc'):
    names = [name] * len(bboxes)
    labels = [0] * len(bboxes)
    img = draw_bboxes(img=img,
                      bboxes=bboxes,
                      classes=names,
                      class_ids=labels,
                      class_name=False,
                      colors=color,
                      bbox_format=bbox_format,
                      line_thickness=2)
    return img


def calc_iou(bboxes1, bboxes2, bbox_mode='xywh'):
    # xmin ymin weight height
    assert len(bboxes1.shape) == 2 and bboxes1.shape[1] == 4
    assert len(bboxes2.shape) == 2 and bboxes2.shape[1] == 4

    bboxes1 = bboxes1.copy()
    bboxes2 = bboxes2.copy()

    if bbox_mode == 'xywh':
        bboxes1[:, 2:] += bboxes1[:, :2]
        bboxes2[:, 2:] += bboxes2[:, :2]

    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def f_beta(tp, fp, fn, beta=2):
    return (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + 1e-6)


def calc_is_correct_at_iou_th(gt_bboxes, pred_bboxes, iou_th, verbose=False):
    gt_bboxes = gt_bboxes.copy()
    pred_bboxes = pred_bboxes.copy()

    tp = 0
    fp = 0
    for k, pred_bbox in enumerate(pred_bboxes):  # fixed in ver.7
        ious = calc_iou(gt_bboxes, pred_bbox[None, 1:])
        max_iou = ious.max()
        if max_iou > iou_th:
            tp += 1
            gt_bboxes = np.delete(gt_bboxes, ious.argmax(), axis=0)
        else:
            fp += 1
        if len(gt_bboxes) == 0:
            fp += len(pred_bboxes) - (k + 1)  # fix in ver.7
            break

    fn = len(gt_bboxes)
    return tp, fp, fn


def calc_is_correct(gt_bboxes, pred_bboxes):
    """
    gt_bboxes: (N, 4) np.array in xywh format
    pred_bboxes: (N, 5) np.array in conf+xywh format
    """
    if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, 0
        return tps, fps, fns

    elif len(gt_bboxes) == 0:
        tps, fps, fns = 0, len(pred_bboxes), 0
        return tps, fps, fns

    elif len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, len(gt_bboxes)
        return tps, fps, fns

    pred_bboxes = pred_bboxes[pred_bboxes[:, 0].argsort()[::-1]]  # sort by conf

    tps, fps, fns = 0, 0, 0
    for iou_th in np.arange(0.3, 0.85, 0.05):
        tp, fp, fn = calc_is_correct_at_iou_th(gt_bboxes, pred_bboxes, iou_th)
        tps += tp
        fps += fp
        fns += fn
    return tps, fps, fns


def calc_f2_score(gt_bboxes_list, pred_bboxes_list, verbose=False):
    """
    gt_bboxes_list: list of (N, 4) np.array in xywh format
    pred_bboxes_list: list of (N, 5) np.array in conf+xywh format
    """
    tps, fps, fns = 0, 0, 0
    for gt_bboxes, pred_bboxes in zip(gt_bboxes_list, pred_bboxes_list):

        tp, fp, fn = calc_is_correct(gt_bboxes, pred_bboxes)
        tps += tp
        fps += fp
        fns += fn
        if verbose:
            num_gt = len(gt_bboxes)
            num_pred = len(pred_bboxes)
            print(f'num_gt:{num_gt:<3} num_pred:{num_pred:<3} tp:{tp:<3} fp:{fp:<3} fn:{fn:<3}')

    precision = tps / (tps + fps + 0.1)
    recall = tps / (tps + fns + 0.1)

    f2 = f_beta(tps, fps, fns, beta=2)

    print(f'tps:{tps}  fps:{fps}  fns:{fns}, p: {precision:.3f}, r: {recall:.3f}, f2: {f2:.3f}')
    return f2


def preds2array(x):
    if len(x) == 0:
        return np.array([])
    return np.array(x.split(" ")).reshape(-1, 5).astype(float)


c1 = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
c2 = (np.random.randint(255), np.random.randint(255), np.random.randint(255))


def evaluate(df_path, min_conf=0, save_video=False, **kw):
    # output video config
    frame_id = 0
    fps = 24  # don't know exact value
    width = 1280
    height = 720

    if save_video:
        if not save_video.endswith(".mp4"):
            save_video += ".mp4"
        print("Video save in :", save_video)
        output_video = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height))

    df = pd.read_csv(df_path).fillna("")
    print(f"load predict file shape {df.shape}")
    print(f"filter bbox conf lower than {min_conf:.3f} ...")

    gts = [np.array(b) for b in df.bboxes.apply(eval).tolist()]
    preds = df['preds'].apply(preds2array).tolist()
    keep_preds = []
    for pred in preds:

        if (pred.size == 0):
            keep_preds.append(np.array([]))
        else:
            keep = pred[:, 0] > min_conf
            # if (np.sum(keep) == 0): keep_preds.append(np.array([]))
            keep_preds.append(pred[keep])

    for i, row in tqdm(df.iterrows()):

        if save_video:
            img = load_image(row.image_path)[:, :, ::-1]
            plt.figure(figsize=(16, 12))
            img = show_img(img, keep_preds[i], 'PRED', c1, 'coco')
            img = show_img(img, gts[i], 'GT', c2, 'coco')
            output_video.write(img)
            plt.close()
        frame_id += 1

    score = calc_f2_score(gts, keep_preds, verbose=False)
    print(f"F2 score {score: .5f}")
    return score


