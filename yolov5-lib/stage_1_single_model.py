import os
import argparse
import time
import glob
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 30, 30
np.set_printoptions(precision=3, suppress=True)
from ast import literal_eval
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
import pandas as pd
from norfair import Detection, Tracker
import os
import cv2
import matplotlib.pyplot as plt
import glob
import shutil
import sys
import torch
from PIL import Image
import ast
import timm
import torch
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import *


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


def load_yolo_model(ckpt_path, conf=None, iou=None, max_det=None):
    model = torch.hub.load('./',
                           'custom',
                           path=ckpt_path,
                           source='local',
                           force_reload=True)  # local repo
    if conf is not None:
        model.conf = conf  # NMS confidence threshold
    if iou is not None:
        model.iou = iou  # NMS IoU threshold
    model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    if model.max_det is not None:
        model.max_det = max_det  # maximum number of detections per image
    print("--------------  load yolov5 model ...-----------------")
    print(f"NMS IOU {model.iou},  NMS CONF {model.conf},  MAX_DET {model.max_det}")
    return model


def yolo_predict(model, img, conf, size=3600, augment=False, return_str=False):
    results = model(img, size=size, augment=augment)  # custom inference size
    preds = results.pandas().xyxy[0]
    bboxes = preds[['xmin', 'ymin', 'xmax', 'ymax']].values
    confs = preds.confidence.values
    if conf is not None:
        keep = confs > conf
        bboxes, confs = bboxes[keep], confs[keep]
    # voc format : 'xmin','ymin','xmax','ymax'
    if return_str:
        return format_prediction(bboxes, confs)
    return bboxes, confs


def format_prediction(bboxes, confs):
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


def load_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def run_single_model(ckpts, data_files, save_path, size, nms_conf, nms_iou, aug):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    results = []
    start = time.time()
    for ckpt, file in zip(ckpts, data_files):
        df = pd.read_csv(file)
        val = df[df.fold == 'val'].reset_index(drop=True)
        val = val.sort_values(['video_id', 'video_frame'])
        # print(ckpt)
        yolo_model = load_yolo_model(ckpt, nms_conf, nms_iou, max_det=1000)
        r = _run_fold_predict(yolo_model, val, size, nms_conf, aug)
        results.append(r)
    use_minute = (time.time() - start) / 60
    pd.concat(results, axis=0).reset_index(drop=True).to_csv(save_path, index=None)
    print(f"finished use {use_minute:.1f} minutes, save in {save_path}")


def _run_fold_predict(model, val, size, infer_conf, aug):
    preds = []
    for i, row in tqdm(val.iterrows()):
        image = load_image(row.image_path)
        pred_bbox, pred_conf = yolo_predict(model, image, infer_conf, size, aug)
        pred_annotation = format_prediction(pred_bbox, pred_conf)
        preds.append(pred_annotation)

    val['preds'] = preds

    return val.fillna("")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="model path")
    parser.add_argument("--data", type=str, help="data csv path")
    parser.add_argument("--size", type=int, help="infer image size")
    parser.add_argument("--save", type=str, help="result save path/ file name")
    parser.add_argument("--nms_conf", default=None, type=float)
    parser.add_argument("--nms_iou", default=None, type=float)
    parser.add_argument("--infer_conf", default=None, type=float)
    parser.add_argument("--aug", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # env setup
    print("setup env ...")
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    # load model
    print("load model ...")
    yolo_model = load_yolo_model(args.ckpt, args.nms_conf, args.nms_iou, max_det=1000)
    print(f"Aug {args.aug}")

    # load data
    print("load data ...")
    df = pd.read_csv(args.data)
    val = df[df.fold == 'val'].reset_index(drop=True)
    val = val.sort_values(['video_id', 'video_frame'])

    # print(f"debug {args.debug}")
    if args.debug:
        val = val[val["bboxes"].apply(eval).apply(len) > 1].head(50)
        print(" ----------------- DEBUG MODE ---------------------- ")
    print(f"data shape : {val.shape}")

    # predict & save
    start = time.time()
    preds = []
    for i, row in tqdm(val.iterrows()):
        image = load_image(row.image_path)
        pred_bbox, pred_conf = yolo_predict(yolo_model, image, args.infer_conf, args.size, args.aug)
        pred_annotation = format_prediction(pred_bbox, pred_conf)
        preds.append(pred_annotation)
    use_minute = (time.time() - start) / 60
    val['preds'] = preds
    val.fillna("").to_csv(args.save, index=None)
    print(f"finished use {use_minute:.1f} minutes, save in {args.save}")
