# detector.py
import torch
from ultralytics import YOLO
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, 
    FasterRCNN_ResNet50_FPN_Weights)

coco_to_bdd = {
    "bicycle":"bike",
    "motorcycle": "motor"
}
device = "cuda" if torch.cuda.is_available() else "cpu"

#RCNN
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
rcnn_model = fasterrcnn_resnet50_fpn(weights=weights)
rcnn_model.to(device)
rcnn_model.eval()
rcnn_categories = weights.meta["categories"]

#YOLO
yolo_model = YOLO("yolov8n.pt")
yolo_categories = yolo_model.names

def detect(image: Image.Image):
    detections = []
    image_tensor = ToTensor()(image).to(device)
    with torch.no_grad():
        outputs = rcnn_model([image_tensor])[0]
    for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        if score < 0.05:
            continue
        cls_name = rcnn_categories[int(label)]
        if cls_name in coco_to_bdd:
            cls_name = coco_to_bdd[cls_name]

        detections.append({
            "bbox" : box.cpu().tolist(),
            "score" : float(score),
            "category": cls_name,
            "model": "rcnn"
        })
    #YOLO
    results = yolo_model(image)[0]
    for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if score < 0.05:
            continue
        cls_name = yolo_categories[int(cls)]
        if cls_name in coco_to_bdd:
            cls_name = coco_to_bdd[cls_name]
        detections.append({
            "bbox": box.tolist(),
            "score": float(score),
            "category": cls_name,
            "model": "yolo"
        })
    return detections
