import pandas as pd
import json
import os
import torch
import torchvision
import argparse
from ultralytics import YOLO
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)

coco_to_bdd = {"bicycle": "bike",
              "motorcycle": "motor"}
BDD_CLASSES = {'area/alternative', 'area/drivable', 'area/unknown', 'bike', 'bus', 'car', 
                'lane/crosswalk', 'lane/double other', 'lane/double white', 'lane/double yellow',
                'lane/road curb', 'lane/single other', 'lane/single white', 'lane/single yellow',
                'motor', 'person', 'rider', 'traffic light', 'traffic sign', 'train', 'truck'}

def load_model(model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == "rcnn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model.to(device)
        model.eval()
        return model, device
    elif model_type == "yolo":
        model = YOLO("yolov8n.pt")
        return model, device
    else:
        raise ValueError("Unknown model type")

#categories of fasterRCNN
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
rcnn_categories = weights.meta["categories"]

#categories of YOLO
model = YOLO("yolov8n.pt")
yolo_categories = model.names

def detect_image(model, device, model_type, image):
    detections = []
    if model_type == "rcnn":
        image_tensor = ToTensor()(image).to(device)
        with torch.no_grad():
            outputs = model([image_tensor])[0]
        for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
            if score < 0.05:
                continue
            cls_name = rcnn_categories[int(label)]
            if cls_name in coco_to_bdd:
                cls_name = coco_to_bdd[cls_name]
            if cls_name not in BDD_CLASSES:
                continue

            detections.append({
                "bbox": box.cpu().tolist(),
                "score": float(score),
                "category": cls_name
            })
    elif model_type == "yolo":
        results = model(image)[0]
        for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if score < 0.05:
                continue
            cls_name = yolo_categories[int(cls)]
            if cls_name in coco_to_bdd:
                cls_name = coco_to_bdd[cls_name]
            if cls_name not in BDD_CLASSES:
                continue
            detections.append({
                "bbox": box.tolist(),
                "score": float(score),
                "category": cls_name
            })
    return detections   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, choices=["rcnn", "yolo"])
    args = parser.parse_args()
    model, device = load_model(args.model)
    image_files = [
        f for f in os.listdir(args.image_dir)
        if f.endswith(".jpg")
    ]
    all_detections = []
    for img_name in tqdm(image_files):
        image_path = os.path.join(args.image_dir, img_name)
        image_id = os.path.splitext(img_name)[0]
        image = Image.open(image_path).convert("RGB")
        detections = detect_image(
            model,
            device, 
            args.model,
            image
        )
        for det in detections:
            det["image_id"] = image_id
            all_detections.append(det)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_detections, f)
    print("Detections saved to:", args.output)   