from fastapi import FastAPI, UploadFile
from PIL import Image
import io
from detector import detect
from critique import critique
from calibration_loader import calibrate_score
from nms import refining_nms

app = FastAPI()


CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5


@app.post("/process")
async def process(file: UploadFile):

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    detections = detect(image)

    raw_by_model = {}
    calibrated_by_model = {}

    # ----------------------------------------
    # Separate raw & calibrated
    # ----------------------------------------
    for det in detections:

        model_name = det["model"]

        raw_by_model.setdefault(model_name, [])
        calibrated_by_model.setdefault(model_name, [])

        raw_by_model[model_name].append(det)

        calibrated_score = calibrate_score(
            model_name,
            det["category"],
            det["score"]
        )

        calibrated_by_model[model_name].append({
            **det,
            "calibrated_score": float(calibrated_score)
        })

    # ----------------------------------------
    # Single-model refining NMS
    # ----------------------------------------
    raw_refined_single = {}
    calibrated_refined_single = {}

    for model in raw_by_model:

        raw_refined_single[model] = refining_nms(
            raw_by_model[model],
            score_key="score",
            iou_threshold=0.5,
            output_model_name=model
        )

        calibrated_refined_single[model] = refining_nms(
            calibrated_by_model[model],
            score_key="calibrated_score",
            iou_threshold=0.5,
            output_model_name=model
        )

    # ----------------------------------------
    # Cross-model refining NMS
    # ----------------------------------------
    all_raw = []
    all_calibrated = []

    for model in raw_by_model:
        all_raw.extend(raw_by_model[model])
        all_calibrated.extend(calibrated_by_model[model])

    raw_refined_cross = refining_nms(
        all_raw,
        score_key="score",
        iou_threshold=0.5,
        output_model_name="ensemble"
    )

    calibrated_refined_cross = refining_nms(
        all_calibrated,
        score_key="calibrated_score",
        iou_threshold=0.5,
        output_model_name="ensemble"
    )

    # ----------------------------------------
    # Return Everything (UI-compatible)
    # ----------------------------------------
    return {
        "raw": raw_by_model,
        "raw_refined_single": raw_refined_single,
        "raw_refined_cross": raw_refined_cross,
        "calibrated": calibrated_by_model,
        "calibrated_refined_single": calibrated_refined_single,
        "calibrated_refined_cross": calibrated_refined_cross
    }






#         cls = det["class"]
#         model_name = det["model"]

#         calibrated = calibrate_score(model_name, cls, raw_score)

#         if calibrated >= CONF_THRESHOLD:
#             calibrated_detections.append({
#                 **det,
#                 "calibrated_score": float(calibrated)
#             })

#     # -----------------------------
#     # 3️⃣ Cross-model NMS
#     # -----------------------------
#     final_detections = fuse_detections(
#     calibrated_detections,
#     iou_threshold=0.5
# )

#     # # -----------------------------
#     # # 4️⃣ Critique
#     # # -----------------------------
#     # critique_output = critique(image, final_detections)

#     # # -----------------------------
#     # # 5️⃣ Human decision logic
#     # # -----------------------------
#     # status = "approved"

#     # critique_lower = critique_output.lower()

#     # if (
#     #     "missing" in critique_lower
#     #     or "wrong" in critique_lower
#     #     or "incorrect" in critique_lower
#     #     or any(det["calibrated_score"] < 0.6 for det in final_detections)
#     # ):
#     #     status = "needs_review"

#     # return {
#     #     "detections": final_detections,
#     #     "critique": critique_output,
#     #     "status": status
#     # }
