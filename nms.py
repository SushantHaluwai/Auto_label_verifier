import numpy as np
from utils import compute_iou

def refining_nms(
    detections,
    score_key="calibrated_score",
    iou_threshold=0.5,
    score_threshold=0.001,
    output_model_name=None
):

    final_detections = []

    classes = set([d["category"] for d in detections])

    for cls in classes:

        cls_dets = [d.copy() for d in detections if d["category"] == cls]

        while len(cls_dets) > 0:

            cls_dets.sort(key=lambda x: x[score_key], reverse=True)

            best = cls_dets[0]
            best_box = best["bbox"]
            best_score = best[score_key]

            overlaps = []

            # Find overlaps for Score Voting
            for det in cls_dets:
                iou = compute_iou(best_box, det["bbox"])
                if iou > iou_threshold:
                    overlaps.append((det, iou))

            # -----------------------
            # Score Voting
            # -----------------------
            num = np.zeros(4)
            deno = 0.0

            for det, iou in overlaps:
                weight = det[score_key] * iou
                num += np.array(det["bbox"]) * weight
                deno += weight

            refined_box = (num / deno).tolist() if deno > 0 else best_box

            contributors = list(set([d["model"] for d, _ in overlaps]))

            final_detections.append({
                "bbox": refined_box,
                "category": cls,
                score_key: float(best_score),
                "model": output_model_name if output_model_name else best["model"],
                "contributors": contributors
            })

            # -----------------------
            # Soft-NMS decay (correct)
            # -----------------------
            new_dets = []

            for det in cls_dets[1:]:

                iou = compute_iou(best_box, det["bbox"])

                det[score_key] *= (1 - iou)

                if det[score_key] > score_threshold:
                    new_dets.append(det)

            cls_dets = new_dets

    return final_detections