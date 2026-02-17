# critique.py

import torch
import json
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/qwen2.5"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

def critique(image: Image.Image, detections):

    # ------------------------------
    # 1️⃣ Convert to clean readable format
    # ------------------------------

    label_map = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat"
    }

    clean_detections = []

    for d in detections:
        clean_detections.append({
            "label": label_map.get(d["class"], f"class_{d['class']}"),
            "confidence": round(float(d["calibrated_score"]), 2),
            "bbox": [round(v, 1) for v in d["bbox"]]
        })

    # ------------------------------
    # 2️⃣ Strict Auto Label Checker Prompt
    # ------------------------------

    prompt = f"""
You are an Auto Label Checker AI for object detection quality control.

You must strictly evaluate the detections.

Detected Objects:
{json.dumps(clean_detections, indent=2)}

Tasks:

1. Check if any visible objects are missing.
2. Check if any bounding boxes are badly aligned.
3. Check if any labels are incorrect.
4. Decide if human review is required.

Rules:
- Be objective.
- Do NOT hallucinate objects.
- Only use what is visible.
- If detections look correct, say "No issues detected."
- Final line MUST be either:
  APPROVED
  or
  NEEDS_REVIEW

Respond in this format:

Missing Objects: <yes/no + explanation>
Bounding Box Issues: <yes/no + explanation>
Label Issues: <yes/no + explanation>
Final Decision: <APPROVED or NEEDS_REVIEW>
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.2  # lower randomness = more consistent QC
    )

    response = processor.batch_decode(
        output,
        skip_special_tokens=True
    )[0]

    return response


