from fastapi import FastAPI, UploadFile
from PIL import Image
import io

from detector import detect
from critique import critique
from calibration_loader import calibrate_score
from nms import fuse_detections

app = FastAPI()


CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5


@app.post("/process")
async def process(file: UploadFile):

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # -----------------------------
    # 1️⃣ Raw detections
    # -----------------------------
    detections = detect(image)

    # -----------------------------
    # 2️⃣ Calibrate scores
    # -----------------------------
    calibrated_detections = []

    for det in detections:
        raw_score = det["score"]
        cls = det["class"]
        model_name = det["model"]

        calibrated = calibrate_score(model_name, cls, raw_score)

        if calibrated >= CONF_THRESHOLD:
            calibrated_detections.append({
                **det,
                "calibrated_score": float(calibrated)
            })

    # -----------------------------
    # 3️⃣ Cross-model NMS
    # -----------------------------
    final_detections = fuse_detections(
    calibrated_detections,
    iou_threshold=0.5
)

    # -----------------------------
    # 4️⃣ Critique
    # -----------------------------
    critique_output = critique(image, final_detections)

    # -----------------------------
    # 5️⃣ Human decision logic
    # -----------------------------
    status = "approved"

    critique_lower = critique_output.lower()

    if (
        "missing" in critique_lower
        or "wrong" in critique_lower
        or "incorrect" in critique_lower
        or any(det["calibrated_score"] < 0.6 for det in final_detections)
    ):
        status = "needs_review"

    return {
        "detections": final_detections,
        "critique": critique_output,
        "status": status
    }
