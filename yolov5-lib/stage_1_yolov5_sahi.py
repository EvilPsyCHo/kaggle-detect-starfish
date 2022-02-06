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
from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
from sahi.utils.yolov5 import (
    download_yolov5s6_model,
)
from sahi.prediction import ObjectPrediction
from sahi.model import DetectionModel
from typing import Dict, List, Optional, Union
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
#CUSTOM_YOLO5_CLASS const (we can execute using standard SAHI predict or custom one implemented in this notebook).
CUSTOM_YOLO5_CLASS = True

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import cv2
import torch
from PIL import Image as Img
from IPython.display import display


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


def load_yolo_model_with_sahi(ckpt_path, conf=None, iou=None, max_det=None):
    detection_model = COTSYolov5DetectionModel(
        model_path=ckpt_path,
        confidence_threshold=conf,
        device="cuda",
    )

    if conf is not None:
        detection_model.model.conf = conf  # NMS confidence threshold
    if iou is not None:
        detection_model.model.iou = iou  # NMS IoU threshold
    detection_model.model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    detection_model.model.multi_label = False  # NMS multiple labels per box
    if max_det is not None:
        detection_model.model.max_det = max_det  # maximum number of detections per image
    print("--------------  load yolov5 model with SAHI...-----------------")
    print(f"NMS IOU {detection_model.model.iou},  NMS CONF {detection_model.model.conf},  MAX_DET {detection_model.model.max_det}")
    return detection_model


def yolo_predict(model, img, conf=None, size=3600, augment=False, return_str=False):
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


def show_prediction(img, bboxes, scores, show=True):
    colors = [(0, 0, 255)]

    obj_names = ["s"]

    for box, score in zip(bboxes, scores):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (255, 0, 0), 2)
        cv2.putText(img, f'{score}', (int(box[0]), int(box[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    cv2.LINE_AA)

    if show:
        img = Img.fromarray(img).resize((1280, 720))
    return img


class COTSYolov5DetectionModel(DetectionModel):

    def load_model(self):
        model = torch.hub.load('./',
                               'custom',
                               path=self.model_path,
                               source='local',
                               force_reload=True)

        model.conf = self.confidence_threshold
        self.model = model

        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        if image_size is not None:
            warnings.warn("Set 'image_size' at DetectionModel init.", DeprecationWarning)
            prediction_result = self.model(image, size=image_size, augment=False)
        elif self.image_size is not None:
            prediction_result = self.model(image, size=self.image_size, augment=False)
        else:
            prediction_result = self.model(image)
        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        has_mask = self.model.with_mask
        return has_mask

    @property
    def category_names(self):
        return self.model.names

    def _create_object_prediction_list_from_original_predictions(
            self,
            shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
            full_shape_list: Optional[List[List[int]]] = None, ):

        original_predictions = self._original_predictions
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions.xyxy):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1 = int(prediction[0])
                y1 = int(prediction[1])
                x2 = int(prediction[2])
                y2 = int(prediction[3])
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # ignore invalid predictions
                if bbox[0] > bbox[2] or bbox[1] > bbox[3] or bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 0 or bbox[3] < 0:
                    print(f"ignoring invalid prediction with bbox: {bbox}")
                    continue
                if full_shape is not None and (
                        bbox[1] > full_shape[0]
                        or bbox[3] > full_shape[0]
                        or bbox[0] > full_shape[1]
                        or bbox[2] > full_shape[1]
                ):
                    print(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


def yolov5_sahi_predict(img, model, sw, sh, ohr, owr, pmt, img_size, verb):
    result = get_sliced_prediction(img,
                                   model,
                                   slice_width=sw,
                                   slice_height=sh,
                                   overlap_height_ratio=ohr,
                                   overlap_width_ratio=owr,
                                   postprocess_match_threshold=pmt,
                                   image_size=img_size,
                                   verbose=verb,
                                   perform_standard_pred=True)

    bboxes = []
    scores = []
    result_len = result.to_coco_annotations()
    for pred in result_len:
        bboxes.append(coco2voc(np.array(pred['bbox'])))
        scores.append(pred['score'])
    # return voc
    return np.array(bboxes), np.array(scores)


def run_yolov5_sahi(ckpts, data_files, save_path, nms_conf, nms_iou,
                     sw, sh, ohr, owr, pmt, size, max_det=1000, verb=0):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    results = []
    start = time.time()
    for ckpt, file in zip(ckpts, data_files):
        df = pd.read_csv(file)
        val = df[df.fold == 'val'].reset_index(drop=True)
        val = val.sort_values(['video_id', 'video_frame'])
        # print(ckpt)
        yolo_model = load_yolo_model_with_sahi(ckpt, nms_conf, nms_iou, max_det=max_det)
        r = _run_fold_predict(yolo_model, val, sw, sh, ohr, owr, pmt, size, verb=verb)
        results.append(r)
    use_minute = (time.time() - start) / 60
    pd.concat(results, axis=0).reset_index(drop=True).to_csv(save_path, index=None)
    print(f"finished use {use_minute:.1f} minutes, save in {save_path}")


def _run_fold_predict(model, val, sw, sh, ohr, owr, pmt, size, verb):
    preds = []
    for i, row in tqdm(val.iterrows()):
        image = load_image(row.image_path)
        pred_bbox, pred_conf = yolov5_sahi_predict(image, model, sw, sh, ohr, owr, pmt, size, verb)
        pred_annotation = format_prediction(pred_bbox, pred_conf)
        preds.append(pred_annotation)

    val['preds'] = preds

    return val.fillna("")
