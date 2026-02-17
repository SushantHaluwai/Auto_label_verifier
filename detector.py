# detector.py

import torch
import torchvision
from ultralytics import YOLO
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

# RCNN
rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
rcnn.to(device)
rcnn.eval()

# YOLO
yolo = YOLO("yolov8n.pt")


def detect(image):
    """
    Returns RAW detections only.
    No calibration.
    No NMS.
    """

    detections = []

    # RCNN
    image_tensor = ToTensor()(image).to(device)

    with torch.no_grad():
        outputs = rcnn([image_tensor])[0]

    for box, score, label in zip(
        outputs["boxes"],
        outputs["scores"],
        outputs["labels"]
    ):
        if score < 0.05:
            continue

        detections.append({
            "bbox": box.cpu().tolist(),
            "score": float(score),
            "class": int(label),
            "model": "rcnn"
        })

    # YOLO
    results = yolo(image)[0]

    for box, score, cls in zip(
        results.boxes.xyxy,
        results.boxes.conf,
        results.boxes.cls
    ):
        if score < 0.05:
            continue

        detections.append({
            "bbox": box.tolist(),
            "score": float(score),
            "class": int(cls),
            "model": "yolo"
        })

    return detections
