"""Click-to-click benchmark runner.

Usage:
    python -m benchmark.click_benchmark --data-dir data/click_sample --output-dir results/click/
    python -m benchmark.click_benchmark --data-dir data/click_sample --output-dir results/click/ --onnx
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone

from PIL import Image

from benchmark.click_pipeline import ClickResult, run_click_pipeline


def _load_click_dataset(data_dir: str) -> list[dict]:
    with open(os.path.join(data_dir, "annotations.json")) as f:
        data = json.load(f)

    pairs = []
    for p in data["pairs"]:
        pairs.append({
            "id": p["id"],
            "source_image": Image.open(os.path.join(data_dir, p["source"]["image"])).convert("RGB"),
            "source_click": p["source"]["click"],
            "target_image": Image.open(os.path.join(data_dir, p["target"]["image"])).convert("RGB"),
            "target_click": p["target"]["click"],
            "element_type": p.get("element_type", ""),
            "element_text": p.get("element_text", ""),
        })
    return pairs


def _pixel_distance(a: list[int], b: list[int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _normalized_distance(a: list[int], b: list[int], img_size: tuple[int, int]) -> float:
    diag = math.sqrt(img_size[0] ** 2 + img_size[1] ** 2)
    return _pixel_distance(a, b) / diag if diag > 0 else float("inf")


def run_click_benchmark(
    data_dir: str,
    output_dir: str,
    device: str | None = None,
    model_name: str = "microsoft/Florence-2-base-ft",
    crop_radius: int = 150,
    visualize: bool = False,
    fast: bool = False,
    onnx_dir: str | None = None,
    onnx_fp32: bool = False,
    compile_model: bool = False,
    coreml_dir: str | None = None,
    coreml_int8: bool = False,
    image_size: int | None = None,
    parallel: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading click dataset from {data_dir}...")
    pairs = _load_click_dataset(data_dir)
    print(f"Loaded {len(pairs)} click pairs.")

    use_parallel = parallel
    if use_parallel:
        from benchmark.model_parallel import FlorenceParallelModel, run_parallel_click_pipeline
        model = FlorenceParallelModel(model_name=model_name, device=device, compile_model=compile_model)
    elif coreml_dir:
        from benchmark.model_coreml_hybrid import FlorenceCoreMLHybridModel
        model = FlorenceCoreMLHybridModel(coreml_dir, use_int8=coreml_int8)
    elif onnx_dir and not onnx_fp32:
        from benchmark.model_hybrid import FlorenceHybridModel
        model = FlorenceHybridModel(onnx_dir, model_name=model_name, use_int8=True)
    elif onnx_dir:
        from benchmark.model_onnx import FlorenceONNXModel
        model = FlorenceONNXModel(onnx_dir, use_int8=False)
    else:
        from benchmark.model import FlorenceModel
        model = FlorenceModel(model_name=model_name, device=device, fast=fast, compile_model=compile_model, image_size=image_size)

    results_list = []
    hit_thresholds = [20, 50, 100, 150]  # pixel distance thresholds

    print(f"\n{'='*70}")
    print(f"Click-to-Click Benchmark (crop_radius={crop_radius}){' [PARALLEL ENCODER]' if use_parallel else ''}")
    print(f"{'='*70}")

    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {pair['id']} ({pair['element_text']})...", end=" ", flush=True)

        if use_parallel:
            result = run_parallel_click_pipeline(
                model,
                pair["source_image"],
                pair["source_click"],
                pair["target_image"],
                crop_radius=crop_radius,
            )
        else:
            result = run_click_pipeline(
                model,
                pair["source_image"],
                pair["source_click"],
                pair["target_image"],
                crop_radius=crop_radius,
            )

        gt_click = pair["target_click"]
        target_size = pair["target_image"].size

        if result.predicted_click is None:
            px_dist = float("inf")
            norm_dist = 1.0
        else:
            px_dist = _pixel_distance(result.predicted_click, gt_click)
            norm_dist = _normalized_distance(result.predicted_click, gt_click, target_size)

        total_latency = sum(result.step_latencies.values())

        entry = {
            "id": pair["id"],
            "element_type": pair["element_type"],
            "element_text": pair["element_text"],
            "source_click": pair["source_click"],
            "target_click_gt": gt_click,
            "target_click_pred": result.predicted_click,
            "pixel_distance": px_dist,
            "normalized_distance": norm_dist,
            "method": result.method,
            "query_text": result.query_text,
            "num_candidates": result.num_candidates,
            "step_latencies": result.step_latencies,
            "total_latency": total_latency,
        }
        results_list.append(entry)

        status = f"dist={px_dist:.0f}px" if px_dist != float("inf") else "MISS"
        print(f"{status} | method={result.method} | query='{result.query_text}' | {total_latency:.2f}s")

        if visualize and result.predicted_click is not None:
            _save_click_viz(pair["target_image"], result.predicted_click, gt_click,
                           pair["id"], px_dist, output_dir)

    # Aggregate metrics
    valid = [r for r in results_list if r["pixel_distance"] != float("inf")]
    n = len(results_list)

    agg = {
        "total_pairs": n,
        "successful": len(valid),
        "success_rate": len(valid) / n if n > 0 else 0,
        "mean_pixel_distance": sum(r["pixel_distance"] for r in valid) / len(valid) if valid else float("inf"),
        "median_pixel_distance": sorted([r["pixel_distance"] for r in valid])[len(valid) // 2] if valid else float("inf"),
        "mean_normalized_distance": sum(r["normalized_distance"] for r in valid) / len(valid) if valid else 1.0,
        "mean_latency": sum(r["total_latency"] for r in results_list) / n if n > 0 else 0,
    }

    for thresh in hit_thresholds:
        hits = sum(1 for r in results_list if r["pixel_distance"] <= thresh)
        agg[f"hit_rate_{thresh}px"] = hits / n if n > 0 else 0

    # Method breakdown
    method_counts = {}
    for r in results_list:
        method_counts[r["method"]] = method_counts.get(r["method"], 0) + 1
    agg["method_breakdown"] = method_counts

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Success rate:       {agg['success_rate']:.1%} ({agg['successful']}/{agg['total_pairs']})")
    print(f"  Mean pixel dist:    {agg['mean_pixel_distance']:.1f}px")
    print(f"  Median pixel dist:  {agg['median_pixel_distance']:.1f}px")
    for thresh in hit_thresholds:
        print(f"  Hit rate @{thresh}px:   {agg[f'hit_rate_{thresh}px']:.1%}")
    print(f"  Mean latency:       {agg['mean_latency']:.2f}s")
    print(f"  Methods used:       {agg['method_breakdown']}")

    output = {
        "metadata": {
            "model": f"coreml:{coreml_dir}" if coreml_dir else (f"onnx:{onnx_dir}" if onnx_dir else model_name),
            "device": f"coreml-{'int8' if coreml_int8 else 'fp16'}" if coreml_dir else ("onnxruntime-cpu" if onnx_dir else model.device),
            "dtype": "int8" if coreml_int8 else ("int8" if (onnx_dir and not onnx_fp32) else str(getattr(model, 'dtype', 'fp32'))),
            "crop_radius": crop_radius,
            "num_pairs": n,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "aggregate": agg,
        "per_pair": results_list,
    }

    results_path = os.path.join(output_dir, "click_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


def _save_click_viz(target_img, pred_click, gt_click, pair_id, px_dist, output_dir):
    from PIL import ImageDraw
    img = target_img.copy()
    draw = ImageDraw.Draw(img)
    r = 20

    # GT — green circle
    draw.ellipse([gt_click[0]-r, gt_click[1]-r, gt_click[0]+r, gt_click[1]+r],
                 outline=(0, 200, 0), width=4)
    draw.line([gt_click[0]-r, gt_click[1], gt_click[0]+r, gt_click[1]], fill=(0, 200, 0), width=2)
    draw.line([gt_click[0], gt_click[1]-r, gt_click[0], gt_click[1]+r], fill=(0, 200, 0), width=2)

    # Pred — red circle
    draw.ellipse([pred_click[0]-r, pred_click[1]-r, pred_click[0]+r, pred_click[1]+r],
                 outline=(220, 0, 0), width=4)
    draw.line([pred_click[0]-r, pred_click[1], pred_click[0]+r, pred_click[1]], fill=(220, 0, 0), width=2)
    draw.line([pred_click[0], pred_click[1]-r, pred_click[0], pred_click[1]+r], fill=(220, 0, 0), width=2)

    # Line between them
    draw.line([gt_click[0], gt_click[1], pred_click[0], pred_click[1]], fill=(255, 165, 0), width=2)

    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    img.save(os.path.join(viz_dir, f"{pair_id}_dist{px_dist:.0f}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Click-to-click cross-platform benchmark")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results/click/")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default=None)
    parser.add_argument("--model", default="microsoft/Florence-2-base-ft")
    parser.add_argument("--crop-radius", type=int, default=150)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Use greedy decoding for lower latency")
    parser.add_argument("--onnx-dir", default=None, help="Path to ONNX model directory")
    parser.add_argument("--onnx-fp32", action="store_true", help="Use FP32 ONNX models instead of INT8")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for graph optimization")
    parser.add_argument("--coreml-dir", default=None, help="Path to CoreML model directory")
    parser.add_argument("--coreml-int8", action="store_true", help="Use INT8 CoreML models")
    parser.add_argument("--image-size", type=int, default=None, help="Override image resolution (default: 768)")
    parser.add_argument("--parallel", action="store_true", help="Use parallel encoder execution")
    args = parser.parse_args()

    run_click_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        model_name=args.model,
        crop_radius=args.crop_radius,
        visualize=args.visualize,
        fast=args.fast,
        onnx_dir=args.onnx_dir,
        onnx_fp32=args.onnx_fp32,
        compile_model=args.compile,
        coreml_dir=args.coreml_dir,
        coreml_int8=args.coreml_int8,
        image_size=args.image_size,
        parallel=args.parallel,
    )
