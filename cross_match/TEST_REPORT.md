# CrossMatch Test Report — 17th April 2026

Benchmark of the 5K-trained CrossMatch model on a held-out test set of 200 unseen synthetic pairs (576 actions). Model was never exposed to this data during training.

## Test Configuration

| Parameter | Value |
|---|---|
| Checkpoint | `cross_match_5k_best.pt` (epoch 10, trained on 5K pairs) |
| Test data | 200 pairs, seed=777 (synthetic_v2 generator) |
| Test actions | 576 (385 clicks, 191 scrolls) |
| Device | Apple Silicon MPS (fp16) |
| DINOv2 encoder | dinov2_vits14 (22M, frozen pretrained) |

## Accuracy Results

### Overall

| Metric | Value |
|---|---|
| **Mean pixel distance** | **20.7px** |
| **Median pixel distance** | **19.4px** |
| P90 pixel distance | 35.5px |
| P95 pixel distance | 43.0px |
| Max pixel distance | 76.8px |
| Action type accuracy | 100.0% |

### Hit Rates

| Threshold | Hit Rate | Count |
|---|---|---|
| @10px | 15.1% | 87/576 |
| **@20px** | **53.3%** | **307/576** |
| **@30px** | **82.5%** | **475/576** |
| **@50px** | **97.9%** | **564/576** |
| @75px | 99.8% | 575/576 |
| **@100px** | **100.0%** | **576/576** |

### By Action Type

| Action | Count | Mean px | Median px |
|---|---|---|---|
| Click | 385 | 17.5px | 17.0px |
| Scroll | 191 | 27.0px | 24.8px |

Click actions are more precise than scrolls — expected since clicks target a single point while scrolls require matching both start and end coordinates.

## Latency Results

| Metric | Value |
|---|---|
| **Mean** | **18ms** |
| **Median** | **16ms** |
| P95 | 25ms |
| Min | 13ms |
| Max | 199ms |
| **Throughput** | **57 actions/sec** |

Single forward pass, no autoregressive decoding. Latency is dominated by DINOv2 encoding two 518x518 images.

## Comparison to Florence-2

| | Florence-2 (MPS+compile) | **CrossMatch (this test)** |
|---|---|---|
| Mean distance | 251px (all) / 3px (hits) | **20.7px** |
| Hit @20px | 53% | **53%** |
| Hit @50px | 53% | **98%** |
| Hit @100px | 53% | **100%** |
| Mean latency | 1,840ms | **18ms** |
| Throughput | 0.5 actions/sec | **57 actions/sec** |
| Forward passes | 2 (autoregressive) | 1 (single pass) |
| Model size | 232M | 33M (11M trainable) |

CrossMatch achieves 100x faster inference with dramatically better accuracy at all thresholds above 20px. Florence-2 is bimodal (perfect or catastrophically wrong), while CrossMatch has a tight distribution centered at ~20px.

## Distribution Analysis

The error distribution is well-behaved:
- No catastrophic failures (max 76.8px vs Florence-2's 1,742px)
- Tight clustering: 82.5% within 30px, 97.9% within 50px
- The remaining 2.1% between 50-77px are likely edge cases with ambiguous element positioning or dense UI layouts

## Notes

- Tested on **synthetic data only** — real app screenshots will have more visual complexity. Accuracy on real data is expected to be lower until trained on real pairs.
- The DINOv2 encoder was loaded fresh (HuggingFace backend on Python 3.9) while training used torch.hub (Python 3.12). The 175 encoder keys were skipped during checkpoint loading — only the trained cross-attention head weights were restored. Since the encoder is frozen pretrained, this has no impact on accuracy (same underlying DINOv2-small weights).
- Scroll actions have ~10px higher error than clicks due to compounding error across two coordinate pairs (from + to).

## Artifacts

| File | Path |
|---|---|
| Full results JSON | `results/cross_match/benchmark_results.json` |
| Test data | `data/cross_match_test/` (200 pairs, seed=777) |
| Checkpoint | `checkpoints/cross_match_5k_best.pt` |
