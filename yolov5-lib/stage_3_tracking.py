import numpy as np
import pandas as pd
from norfair import Detection, Tracker
from tqdm import tqdm
from pathlib import Path


# Helper to convert bbox in format [x_min, y_min, x_max, y_max, score] to norfair.Detection class
def coco2norfair(detects, confs, frame_id):
    result = []
    for (x_min, y_min, w, h), score in zip(detects, confs):
        xc, yc = x_min + w / 2, y_min + h /  2
        result.append(Detection(points=np.array([xc, yc]), scores=np.array([score]), data=np.array([w, h, frame_id])))

    return result


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def decode_pred_annotation(anno):
    if anno == "":
        return [], []
    anno = np.array(anno.split(" ")).reshape(-1, 5).astype(float)
    conf = anno[:, 0]
    bbox = anno[:, 1:]
    return conf, bbox


def run_track(df_path, save, distance_threshold=30, hit_inertia_min=3, hit_inertia_max=6, initialization_delay=1):

    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=distance_threshold,
        hit_inertia_min=hit_inertia_min,
        hit_inertia_max=hit_inertia_max,
        initialization_delay=initialization_delay,
    )
    Path(save).parent.mkdir(exist_ok=True, parents=True)
    frame_id = 0
    cnt = 0
    df = pd.read_csv(df_path).sort_values(['video_id', 'video_frame']).reset_index(drop=True).fillna("")

    result = []
    for idx, row in tqdm(df.iterrows()):
        conf, bbox = decode_pred_annotation(row.preds)
        anno = row.preds + " "
        tracked_objects = tracker.update(detections=coco2norfair(bbox, conf, frame_id))
        for tobj in tracked_objects:
            cnt += 1
            bbox_width, bbox_height, last_detected_frame_id = tobj.last_detection.data
            if last_detected_frame_id == frame_id:  # Skip objects that were detected on current frame
                continue

            # Add objects that have no detections on current frame to predictions
            xc, yc = tobj.estimate[0]
            x_min, y_min = int(round(xc - bbox_width / 2)), int(round(yc - bbox_height / 2))
            score = tobj.last_detection.scores[0]
            anno += '{} {} {} {} {} '.format(score, x_min, y_min, bbox_width, bbox_height)

        result.append(anno.strip(' '))
        frame_id += 1

    df['preds'] = result
    df.to_csv(save, index=None)
    print(f"Add {cnt} bbox to predict, result save in {save}")


if __name__ == "__main__":
    run_track("./stage_cache/ensemble/test.csv", "./stage_cache/track/test.csv")