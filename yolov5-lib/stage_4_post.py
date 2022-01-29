import os

from pathlib import Path
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


def decode_pred_annotation(anno):
    if anno == "":
        return [], []
    anno = np.array(anno.split(" ")).reshape(-1, 5).astype(float)
    conf = anno[:, 0]
    bbox = anno[:, 1:]
    return conf, bbox


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


tfm = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(mean=[0], std=[1], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)


class RegHead(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(RegHead, self).__init__()
        # self.norm = nn.LayerNorm(in_feat)
        self.fc1 = nn.Linear(in_feat, out_feat)
        # self.fc2 = nn.Linear(in_feat//2, out_feat)

    def forward(self, x):
        # x = self.norm(x)

        x = self.fc1(x)
        # x = F.relu_(x)
        # x = self.fc2(x)
        # x = x.sigmoid() * 100.
        return x


class PawModel(nn.Module):
    def __init__(self, arch='tf_efficientnet_b0_ns', pretrained=True, n_class=1):
        super(PawModel, self).__init__()

        self.model = timm.create_model(arch, pretrained=pretrained)

        # print(self.model); assert False

        try:
            if 'efficientnet' in arch:
                self.fc = RegHead(self.model.classifier.in_features, 1)
                '''
                for k, v in self.model.named_parameters():
                    if 'blocks.4' in k or 'blocks.5' in k or 'blocks.6' in k:
                        print(f'{k} unfrozen')
                        v.requires_grad = True
                    else:
                        v.requires_grad = False
                '''
                self.model.classifier = nn.Identity()
            elif 'swin' in arch or 'vit' in arch:
                self.fc = RegHead(self.model.head.in_features, n_class)  # RegHead(self.model.head.in_features, 1)

                '''
                # 11-17

                unfreeze_kws = ['layers.2.blocks.16', 'layers.2.blocks.17', 'layers.2.downsample', 'layers.3']

                for k, v in self.model.named_parameters():
                    if any([kw in k for kw in unfreeze_kws]):
                        #print(f'{k} unfrozen')
                        v.requires_grad = True
                    else:
                        v.requires_grad = False
                '''
                self.model.head = nn.Identity()
            else:
                self.fc = RegHead(self.model.fc.in_features, 1)
                self.model.fc = nn.Identity()

        except:
            print(self.model)
            # print(self.model.layers[3].blocks[1].mlp.fc2)
            assert False

    def forward(self, imgs):

        # with torch.no_grad():
        x = self.model(imgs)  # (b*d, img_f)

        # x = F.normalize(x)

        x = self.fc(x)

        return x


def cls_predict(cls_model, img, crops):
    conf = []

    with torch.no_grad():
        for crop in crops:
            x, y, x2, y2 = crop.astype(int)

            img_ = img[y:y2, x:x2, :]
            # print("crop", crop, img.shape, img_.shape)
            img_pt = tfm(image=img_)['image'].unsqueeze(0).to('cuda:0')
            p = cls_model(img_pt)[0].sigmoid().detach().cpu().numpy()[0]
            conf += [p]

    return np.array(conf)


def _run_single_fold(df, post_model, CONF):
    cnt = 0
    cnt_all = 0
    df = df.copy().fillna("")
    preds = []

    for _, row in tqdm(df.iterrows()):
        anno = row.preds
        if anno == "":
            preds.append(anno)
            continue
        score, bbox = decode_pred_annotation(anno)
        voc_bbox = coco2voc(bbox)
        image = cv2.imread(row.image_path)[:,:,::-1]
        cls_conf = cls_predict(post_model, image, voc_bbox)

        keep = cls_conf > CONF
        cnt += np.sum(cls_conf < CONF)
        cnt_all += len(cls_conf)
        bbox = bbox[keep]
        score = score[keep]

        preds.append(format_prediction(bbox, score))

    df['preds'] = preds
    return df, cnt, cnt_all


def run_post(df_path, post_models, save, CLS_CONF):
    Path(save).parent.mkdir(exist_ok=True, parents=True)

    device = "cuda:0"
    df = pd.read_csv(df_path).sort_values(['video_id', 'video_frame']).reset_index(drop=True).fillna("")

    result = []
    cnts = 0
    cnts_all = 0

    for video_id in range(3):
        print(f"process video {video_id} ...")
        model = PawModel('swin_large_patch4_window7_224', pretrained=False, n_class=1).to(device)
        model.load_state_dict(torch.load(post_models[video_id], map_location=device))
        model.eval()
        df_video, cnt, cnt_all = _run_single_fold(df.loc[df.video_id == video_id], model, CLS_CONF)
        result.append(df_video)
        cnts += cnt
        cnts_all += cnt_all

    results = pd.concat(result, axis=0).sort_values(['video_id', 'video_frame']).reset_index(drop=True)
    results.to_csv(save, index=None)

    print(f"Finished, post model drop {cnts/cnts_all*100:.1f}% bbox")


if __name__ == "__main__":
    cls0 = "./checkpoints/KH/cls_v2/fold_0_ep_1"
    cls1 = "./checkpoints/KH/cls_v2/fold_1_ep_1"
    cls2 = "./checkpoints/KH/cls_v2/fold_2_ep_1"
    models = [cls0, cls1, cls2]
    run_post("./stage_cache/track/test.csv", models, "./stage_cache/post/test.csv", 0.9)
