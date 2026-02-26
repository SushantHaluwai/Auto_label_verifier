import numpy as np
from utils import compute_iou

def refining_nms(detections, iou_threshold=0.5, score_threshold=0.04):
    final_detections = []
    classes = set([d["category"] for d in detections])
    for cls in classes:
        cls_dets = [d.copy() for d in detections
                    if d["category"] == cls
                    ]
        while len(cls_dets) > 0:
            cls_dets.sort(
                key = lambda x: x["calibrated_score"],
                reverse=True
            )
            best = cls_dets[0]
            best_box = best["bbox"]
            best_score = best["calibrated_score"]
            overlaps = []
            remaining = []
            for det in cls_dets:
                iou = compute_iou(best_box, det["bbox"])
                if iou > iou_threshold:
                    overlaps.append((det, iou))
                else:
                    remaining.append(det)
            num = np.zeroes(4)
            deno = 0.0
            for det, iou in overlaps:
                weight = det["calibrated_score"] * iou
                num += np.array(det["bbox"]) * weight
                deno += weight

            if deno > 0:
                refined_box = (num/deno).tolist()
            else:
                refined_box = best_box
            
            final_detections.append({
                "bbox": refined_box,
                "category": cls,
                "calibrated_score": float(best_score),
                "model": "ensemble"
            })

            new_dets = []
            for det in remaining:
                iou = compute_iou(best_box, det["bbox"])
                if iou > iou_threshold:
                    det["calibrated_score"] *= (1 - iou)
                if det["calibrated_score"] > score_threshold:
                    new_dets.append(det)
            cls_dets = new_dets
    return  final_detections

