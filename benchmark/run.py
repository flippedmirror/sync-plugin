from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from benchmark.dataset import TestPair, load_dataset
from benchmark.metrics import (
    compute_aggregate_metrics,
    compute_center_distance_normalized,
    compute_iou,
)
from benchmark.model import FlorenceModel
from benchmark.pipeline import STRATEGIES, PipelineResult
from benchmark.visualize import draw_comparison, save_visualization


def _evaluate_pair(
    result: PipelineResult, gt_bbox: list[float], target_img_size: tuple[int, int]
) -> dict:
    """Compute per-pair metrics from a pipeline result."""
    if result.predicted_bbox is None:
        iou = 0.0
        center_dist = 1.0  # max normalized distance
    else:
        iou = compute_iou(result.predicted_bbox, gt_bbox)
        center_dist = compute_center_distance_normalized(
            result.predicted_bbox, gt_bbox, target_img_size
        )

    src_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    img_area = target_img_size[0] * target_img_size[1]
    element_area_fraction = src_area / img_area if img_area > 0 else 0.0

    return {
        "iou": iou,
        "center_distance_normalized": center_dist,
        "predicted_bbox": result.predicted_bbox,
        "ground_truth_bbox": gt_bbox,
        "description": result.description,
        "ocr_text": result.ocr_text,
        "num_candidates": result.num_candidates,
        "step_latencies": result.step_latencies,
        "element_area_fraction": element_area_fraction,
    }


def main(
    data_dir: str,
    output_dir: str,
    strategies: list[str] | None = None,
    device: str | None = None,
    visualize: bool = False,
    model_name: str = "microsoft/Florence-2-base-ft",
):
    if strategies is None:
        strategies = list(STRATEGIES.keys())

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from {data_dir}...")
    pairs = load_dataset(data_dir)
    print(f"Loaded {len(pairs)} test pairs.")

    model = FlorenceModel(model_name=model_name, device=device)

    results = {
        "metadata": {
            "model": model_name,
            "device": model.device,
            "dtype": str(model.dtype),
            "num_pairs": len(pairs),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "strategies": {},
    }

    for strategy_name in strategies:
        strategy_fn = STRATEGIES[strategy_name]
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*60}")

        per_pair_results = []

        for i, pair in enumerate(pairs):
            print(f"  [{i+1}/{len(pairs)}] {pair.id}...", end=" ", flush=True)

            pipeline_result = strategy_fn(
                model, pair.source_image, pair.source_bbox, pair.target_image
            )

            pair_metrics = _evaluate_pair(
                pipeline_result, pair.target_bbox, pair.target_image.size
            )
            pair_metrics["id"] = pair.id
            pair_metrics["tags"] = pair.tags

            per_pair_results.append(pair_metrics)

            iou = pair_metrics["iou"]
            print(f"IoU={iou:.3f} | candidates={pipeline_result.num_candidates}")

            if visualize and pipeline_result.predicted_bbox is not None:
                viz_img = draw_comparison(
                    pair.target_image,
                    pipeline_result.predicted_bbox,
                    pair.target_bbox,
                    pair.id,
                    strategy_name,
                    iou,
                )
                save_visualization(viz_img, output_dir, pair.id, strategy_name)

        aggregate = compute_aggregate_metrics(per_pair_results)

        results["strategies"][strategy_name] = {
            "aggregate": aggregate,
            "per_pair": per_pair_results,
        }

        # Print summary
        print(f"\n  --- {strategy_name} Summary ---")
        print(f"  Mean IoU:        {aggregate.get('mean_iou', 0):.3f}")
        print(f"  Acc@0.3:         {aggregate.get('accuracy_at_0.3', 0):.1%}")
        print(f"  Acc@0.5:         {aggregate.get('accuracy_at_0.5', 0):.1%}")
        print(f"  Acc@0.7:         {aggregate.get('accuracy_at_0.7', 0):.1%}")
        print(f"  Mean center dist:{aggregate.get('mean_center_distance_normalized', 0):.4f}")
        print(f"  Mean latency:    {aggregate.get('mean_latency_seconds', 0):.2f}s")

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
