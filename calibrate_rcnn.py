import torch
import torchvision
import joblib
from sklearn.isotonic import IsotonicRegression
from collections import defaultdict
from dataset import BDDDataset
from utils import compute_iou
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = BDDDataset(
    "bdd100k_images_100k/100k/val",
    "bdd100k_labels/100k/val"
)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
#model.load_state_dict(torch.load("models/rcnn.pth"))
model.to(device)
model.eval()

scores = defaultdict(list)
ious = defaultdict(list)

for image, gt_boxes, gt_labels in tqdm(dataset):

    image_tensor = ToTensor()(image).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    for box, score, label in zip(
        outputs["boxes"],
        outputs["scores"],
        outputs["labels"]
    ):
        if score < 0.05:
            continue

        best_iou = 0
        for gt_box in gt_boxes:
            iou = compute_iou(box.cpu().tolist(), gt_box.tolist())
            best_iou = max(best_iou, iou)

        cls = int(label.item())
        scores[cls].append(score.item())
        ious[cls].append(best_iou)

os.makedirs("models/calibrations", exist_ok=True)

for cls in scores:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores[cls], ious[cls])
    joblib.dump(ir, f"models/calibrations/rcnn_{cls}.pkl")

print("RCNN calibration complete.")
