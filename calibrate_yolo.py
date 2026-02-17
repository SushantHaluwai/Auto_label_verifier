from ultralytics import YOLO
import joblib
from sklearn.isotonic import IsotonicRegression
from collections import defaultdict
from dataset import BDDDataset
from utils import compute_iou
from tqdm import tqdm
import os

model = YOLO("yolov8n.pt")

dataset = BDDDataset(
    "bdd100k_images_100k/100k/val",
    "bdd100k_labels/100k/val"
)

scores = defaultdict(list)
ious = defaultdict(list)

for image, gt_boxes, gt_labels in tqdm(dataset):

    results = model(image)[0]

    for box, score, cls in zip(
        results.boxes.xyxy,
        results.boxes.conf,
        results.boxes.cls
    ):
        if score < 0.05:
            continue

        best_iou = 0
        for gt_box in gt_boxes:
            iou = compute_iou(box.tolist(), gt_box.tolist())
            best_iou = max(best_iou, iou)

        cls_id = int(cls.item())
        scores[cls_id].append(score.item())
        ious[cls_id].append(best_iou)

os.makedirs("models/calibrations", exist_ok=True)

for cls in scores:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores[cls], ious[cls])
    joblib.dump(ir, f"models/calibrations/yolo_{cls}.pkl")

print("YOLO calibration complete.")
