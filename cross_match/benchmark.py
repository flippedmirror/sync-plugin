"""Benchmark CrossMatch model on a test set.

Reports accuracy (hit rates, pixel distances) and latency metrics.

Usage:
    python -m cross_match.benchmark --checkpoint checkpoints/cross_match_5k_best.pt --data-dir data/cross_match_test --device mps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from PIL import Image

from cross_match.predict import CrossMatchPredictor

REF_DIAG = math.sqrt(1170**2 + 2532**2)


def load_test_actions(data_dir: str) -> list[dict]:
    with open(os.path.join(data_dir, "annotations.json")) as f:
        data = json.load(f)

    actions = []
    for pair in data["pairs"]:
        src_path = os.path.join(data_dir, pair["source"]["image"])
        tgt_path = os.path.join(data_dir, pair["target"]["image"])
        tgt_size = pair["target"].get("size", [1170, 2532])

        for act in pair["actions"]:
            actions.append({
                "pair_id": pair["id"],
                "source_image_path": src_path,
                "target_image_path": tgt_path,
                "target_size": tgt_size,
                "action": act,
            })
    return actions


def compute_distance(action_type, gt_coords, pred_coords):
    if action_type == "click":
        gt = gt_coords["at"]
        pred = pred_coords["at"]
        return math.sqrt((gt[0] - pred[0])**2 + (gt[1] - pred[1])**2)
    else:
        gt_f = gt_coords["from_arg"]
        pred_f = pred_coords["from_arg"]
        gt_t = gt_coords["to_arg"]
        pred_t = pred_coords["to_arg"]
        d_from = math.sqrt((gt_f[0] - pred_f[0])**2 + (gt_f[1] - pred_f[1])**2)
        d_to = math.sqrt((gt_t[0] - pred_t[0])**2 + (gt_t[1] - pred_t[1])**2)
        return (d_from + d_to) / 2


def main(args):
    print("Loading model from {}...".format(args.checkpoint))
    predictor = CrossMatchPredictor(args.checkpoint, device=args.device)
    print("Device: {}".format(predictor.device))

    actions = load_test_actions(args.data_dir)
    print("Test actions: {} ({} pairs)".format(len(actions), len(set(a["pair_id"] for a in actions))))

    # Warmup
    src = Image.open(actions[0]["source_image_path"]).convert("RGB")
    tgt = Image.open(actions[0]["target_image_path"]).convert("RGB")
    act = actions[0]["action"]
    _ = predictor.predict(src, tgt, act["type"], act["source_coords"])

    print("\nRunning benchmark...")
    print("=" * 90)

    results = []
    click_dists = []
    scroll_dists = []
    latencies = []
    action_correct = 0

    for i, item in enumerate(actions):
        src = Image.open(item["source_image_path"]).convert("RGB")
        tgt = Image.open(item["target_image_path"]).convert("RGB")
        act = item["action"]

        result = predictor.predict(src, tgt, act["type"], act["source_coords"])
        latencies.append(result["latency"])

        dist = compute_distance(act["type"], act["target_coords"], result["target_coords"])

        if result["type"] == act["type"]:
            action_correct += 1

        entry = {"pair_id": item["pair_id"], "type": act["type"], "dist": dist, "latency": result["latency"]}
        results.append(entry)

        if act["type"] == "click":
            click_dists.append(dist)
        else:
            scroll_dists.append(dist)

        if (i + 1) % 100 == 0:
            print("  Processed {}/{}...".format(i + 1, len(actions)))

    all_dists = [r["dist"] for r in results]
    sorted_dists = sorted(all_dists)
    n = len(all_dists)

    print("=" * 90)
    print()
    print("ACCURACY METRICS")
    print("-" * 50)
    print("  Total actions:        {}".format(n))
    print("  Click actions:        {}".format(len(click_dists)))
    print("  Scroll actions:       {}".format(len(scroll_dists)))
    print("  Action type accuracy: {:.1%}".format(action_correct / n))
    print()
    print("  Overall pixel distance:")
    print("    Mean:               {:.1f}px".format(sum(all_dists) / n))
    print("    Median:             {:.1f}px".format(sorted_dists[n // 2]))
    print("    P90:                {:.1f}px".format(sorted_dists[int(n * 0.9)]))
    print("    P95:                {:.1f}px".format(sorted_dists[int(n * 0.95)]))
    print("    Max:                {:.1f}px".format(max(all_dists)))

    for label, dists in [("Click", click_dists), ("Scroll", scroll_dists)]:
        if dists:
            sd = sorted(dists)
            print()
            print("  {} pixel distance (n={}):".format(label, len(dists)))
            print("    Mean:               {:.1f}px".format(sum(dists) / len(dists)))
            print("    Median:             {:.1f}px".format(sd[len(sd) // 2]))

    print()
    print("  Hit rates:")
    for thresh in [10, 20, 30, 50, 75, 100, 150]:
        hits = sum(1 for d in all_dists if d <= thresh)
        print("    @{:3d}px:             {:.1%} ({}/{})".format(thresh, hits / n, hits, n))

    print()
    print("LATENCY METRICS")
    print("-" * 50)
    # Exclude first call (warmup already done, but first test item might still be slower)
    post_warmup = latencies[1:]
    print("  Mean:                 {:.0f}ms".format(sum(post_warmup) / len(post_warmup) * 1000))
    print("  Median:              {:.0f}ms".format(sorted(post_warmup)[len(post_warmup) // 2] * 1000))
    print("  P95:                  {:.0f}ms".format(sorted(post_warmup)[int(len(post_warmup) * 0.95)] * 1000))
    print("  Min:                  {:.0f}ms".format(min(post_warmup) * 1000))
    print("  Max:                  {:.0f}ms".format(max(post_warmup) * 1000))
    print("  Throughput:           {:.0f} actions/sec".format(1 / (sum(post_warmup) / len(post_warmup))))

    # Save results
    output = {
        "checkpoint": args.checkpoint,
        "device": str(predictor.device),
        "test_data": args.data_dir,
        "num_actions": n,
        "accuracy": {
            "action_type_accuracy": action_correct / n,
            "mean_px": sum(all_dists) / n,
            "median_px": sorted_dists[n // 2],
            "p90_px": sorted_dists[int(n * 0.9)],
            "p95_px": sorted_dists[int(n * 0.95)],
            "click_mean_px": sum(click_dists) / len(click_dists) if click_dists else 0,
            "scroll_mean_px": sum(scroll_dists) / len(scroll_dists) if scroll_dists else 0,
            "hit_10px": sum(1 for d in all_dists if d <= 10) / n,
            "hit_20px": sum(1 for d in all_dists if d <= 20) / n,
            "hit_30px": sum(1 for d in all_dists if d <= 30) / n,
            "hit_50px": sum(1 for d in all_dists if d <= 50) / n,
            "hit_100px": sum(1 for d in all_dists if d <= 100) / n,
        },
        "latency": {
            "mean_ms": sum(post_warmup) / len(post_warmup) * 1000,
            "median_ms": sorted(post_warmup)[len(post_warmup) // 2] * 1000,
            "p95_ms": sorted(post_warmup)[int(len(post_warmup) * 0.95)] * 1000,
            "min_ms": min(post_warmup) * 1000,
            "max_ms": max(post_warmup) * 1000,
        },
        "per_action": results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print()
    print("Results saved to {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="results/cross_match")
    args = parser.parse_args()
    main(args)
