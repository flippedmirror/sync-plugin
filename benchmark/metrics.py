import math


def compute_iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Standard IoU between two [x1, y1, x2, y2] bboxes."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def compute_center_distance_normalized(
    pred_bbox: list[float], gt_bbox: list[float], image_size: tuple[int, int]
) -> float:
    """Euclidean distance between bbox centers, normalized by image diagonal."""
    w, h = image_size
    diagonal = math.sqrt(w * w + h * h)
    if diagonal == 0:
        return float("inf")

    pc = bbox_center(pred_bbox)
    gc = bbox_center(gt_bbox)
    dist = math.sqrt((pc[0] - gc[0]) ** 2 + (pc[1] - gc[1]) ** 2)
    return dist / diagonal


def compute_aggregate_metrics(
    per_pair_results: list[dict], iou_thresholds: list[float] = None
) -> dict:
    """Compute aggregate metrics from per-pair results.

    Each entry in per_pair_results must have keys:
        iou, center_distance_normalized, step_latencies (dict)
    """
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.5, 0.7]

    n = len(per_pair_results)
    if n == 0:
        return {}

    ious = [r["iou"] for r in per_pair_results]
    center_dists = [r["center_distance_normalized"] for r in per_pair_results]

    # Collect all latency step keys
    all_step_keys = set()
    for r in per_pair_results:
        all_step_keys.update(r.get("step_latencies", {}).keys())

    mean_step_latencies = {}
    for key in sorted(all_step_keys):
        vals = [r["step_latencies"][key] for r in per_pair_results if key in r.get("step_latencies", {})]
        mean_step_latencies[key] = sum(vals) / len(vals) if vals else 0.0

    total_latencies = [sum(r.get("step_latencies", {}).values()) for r in per_pair_results]

    metrics = {
        "mean_iou": sum(ious) / n,
        "mean_center_distance_normalized": sum(center_dists) / n,
        "mean_latency_seconds": sum(total_latencies) / n,
        "mean_step_latencies": mean_step_latencies,
    }

    for t in iou_thresholds:
        hits = sum(1 for iou in ious if iou >= t)
        metrics[f"accuracy_at_{t}"] = hits / n

    return metrics
