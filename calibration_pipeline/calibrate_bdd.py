import os
import json
import argparse
import numpy as np
import pickle
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB -xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def load_bdd_gt(label_path):
    with open(label_path) as f:
        data = json.load(f)
    objects = data["frames"][0]["objects"]
    gt_boxes = []
    gt_labels = []

    for obj in objects:
        if "box2d" not in obj:
            continue
        box = obj["box2d"]
        gt_boxes.append([
            box["x1"],
            box["y1"],
            box["x2"],
            box["y2"]
        ])
        gt_labels.append(obj["category"])
    return gt_boxes, gt_labels

def build_dataset(image_dir, label_dir, detections_json, num_images=1000):
    with open(detections_json) as f:
        detections = json.load(f)
    image_files = os.listdir(image_dir)

    if num_images > 0 and num_images < len(image_files):
        image_files = np.random.choice(
            image_files,
            num_images,
            replace=False
        )
    all_rows = []
    for img_name in tqdm(image_files):
        image_id = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_dir, image_id + ".json")
        if not os.path.exists(label_path):
            continue
        gt_boxes, gt_labels = load_bdd_gt(label_path)
        img_dets = [d for d in detections if d["image_id"]==image_id]
        if len(img_dets) == 0:
            continue
        for det in img_dets:
            pred_box = det["bbox"]
            score = det["score"]
            pred_cls = det["category"]
            best_iou = 0
            for gt_box, gt_cls in zip(gt_boxes, gt_labels):
                if gt_cls != pred_cls:
                    continue
                iou = compute_iou(pred_box, gt_box)
                best_iou = max(best_iou, iou)
            all_rows.append([score, best_iou, pred_cls])
    return np.array(all_rows)

def train_calibrator(dets, calibration_file, calibration_type="IR", class_agnostic=True):
    calibrator = {}
    if class_agnostic:
        scores = dets[:, 0].astype(float)
        ious = dets[:, 1].astype(float)
        scores = scores.reshape(-1, 1)
        if calibration_type == "IR":
            model = IsotonicRegression(
                y_min=0.,
                y_max=1.,
                out_of_bounds="clip"
            ).fit(scores, ious)
        else:
            model = LinearRegression().fit(scores, ious)
        classes = np.unique(dets[:, 2])
        for cls in classes:
            calibrator[cls] = model

    else:
        classes = np.unique(dets[:, 2])
        for cls in classes:
            idx = np.where(dets[:, 2] == cls)[0]
            scores = dets[idx, 0].astype(float).reshape(-1, 1)
            ious = dets[idx, 1].astype(float)

            if len(scores) == 0:
                continue

            if calibration_type == "IR":
                model = IsotonicRegression(
                    y_min = 0.,
                    y_max= 1.,
                    out_of_bounds = "clip"
                ).fit(scores, ious)
            else:
                model = LinearRegression().fit(scores, ious)

            calibrator[cls] = model

    os.makedirs(os.path.dirname(calibration_file), exist_ok=True)
    with open(calibration_file, "wb") as f:
        pickle.dump(calibrator, f)

    print("calibrator saved:", calibration_file)
    return calibrator

def predict(calibrator, dets):
    calibrated = np.zeros(len(dets))
    for i, (score, _, cls) in enumerate(dets):
        if cls not in calibrator:
            calibrated[i] = score
        else:
            calibrated[i] = calibrator[cls].predict([[float(score)]])[0]
    return calibrated   
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--type", default="IR")
    parser.add_argument("--class_agnostic", action="store_true")

    args = parser.parse_args()

    print("Building calibration dataset...")
    dets = build_dataset(
        args.image_dir,
        args.label_dir,
        args.detections,
        args.num_images
    )

    print("Training calibrator...")
    calibrator = train_calibrator(
        dets,
        args.output,
        calibration_type=args.type,
        class_agnostic=args.class_agnostic
    )

    print("Done.")