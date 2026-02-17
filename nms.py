import math

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)


def fuse_detections(detections, iou_threshold=0.5):
    """
    True MoCAE-style fusion.
    Returns ONE box per object.
    """

    detections = sorted(
        detections,
        key=lambda x: x["calibrated_score"],
        reverse=True
    )

    fused = []
    used = set()

    for i, det in enumerate(detections):

        if i in used:
            continue

        cluster = [det]
        used.add(i)

        for j in range(i + 1, len(detections)):

            if j in used:
                continue

            if det["class"] != detections[j]["class"]:
                continue

            iou = compute_iou(det["bbox"], detections[j]["bbox"])

            if iou > iou_threshold:
                cluster.append(detections[j])
                used.add(j)

        # 🔥 Weighted box fusion
        total_weight = sum(d["calibrated_score"] for d in cluster)

        fused_box = [0, 0, 0, 0]

        for d in cluster:
            weight = d["calibrated_score"]
            for k in range(4):
                fused_box[k] += d["bbox"][k] * weight

        fused_box = [x / total_weight for x in fused_box]

        # 🔥 Pick best expert
        best_model = max(cluster, key=lambda x: x["calibrated_score"])["model"]
        best_score = max(cluster, key=lambda x: x["calibrated_score"])["calibrated_score"]

        fused.append({
            "bbox": fused_box,
            "class": det["class"],
            "calibrated_score": best_score,
            "model": best_model
        })

    return fused
