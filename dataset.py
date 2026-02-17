import os
import json
from PIL import Image
import torch

CLASSES = [
    "person","rider","car","truck",
    "bus","train","motorcycle","bicycle",
    "traffic light","traffic sign"
]

class BDDDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.files = [f for f in os.listdir(label_dir) if f.endswith(".json")]

    def __getitem__(self, idx):
        file = self.files[idx]

        with open(os.path.join(self.label_dir, file)) as f:
            data = json.load(f)

        image_path = os.path.join(
            self.image_dir,
            file.replace(".json",".jpg")
        )

        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []

        for obj in data["frames"][0]["objects"]:
            if "box2d" not in obj:
                continue
            if obj["category"] not in CLASSES:
                continue

            b = obj["box2d"]
            boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
            labels.append(CLASSES.index(obj["category"]))

        return image, torch.tensor(boxes), torch.tensor(labels)

    def __len__(self):
        return len(self.files)
