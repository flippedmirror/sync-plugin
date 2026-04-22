# CrossMatch Training V5 — L40S Training Plan & Results

**Date**: 22 April 2026  
**GPU**: NVIDIA L40S (48GB VRAM, Ada Lovelace)  
**Data**: Synthetic v5 (archetypes + SVG icons + varied resolutions)

---

## 1. Model

| Component | Params | Trainable |
|---|---|---|
| DINOv2-small encoder | 22M | Frozen (Phase 1), unfrozen (Phase 2) |
| Cross-attention (4 layers, 6 heads, 384 dim) | ~9.5M | Yes |
| Coord encoder + action heads | ~1.8M | Yes |
| **Total** | **33.35M** | **11.29M (Phase 1), 33.35M (Phase 2)** |

## 2. Training Plan

### Phase 0: Calibration Run (5K pairs, 5 epochs)

Quick run to validate v5 data pipeline, estimate per-epoch time, and catch issues early.

| Parameter | Value |
|---|---|
| Dataset | 5,000 v5 pairs |
| Epochs | 5 (frozen encoder only, no fine-tune) |
| Batch size | 64 |
| LR | 1e-4, cosine with 2-epoch warmup |
| Feature caching | Yes |
| Expected time | ~5-10 min total |

### Phase 1: Full Run — Frozen Encoder (20K pairs, 30 epochs)

| Parameter | Value |
|---|---|
| Dataset | 20,000 v5 pairs |
| Epochs | 30 |
| Batch size | 64 |
| LR | 1e-4, cosine schedule, 5-epoch warmup |
| Feature caching | Yes (~42GB on disk) |
| Expected time | ~8-10 min (after caching) |

### Phase 2: Full Run — Fine-tune Encoder (5 epochs)

| Parameter | Value |
|---|---|
| Resume from | Phase 1 best checkpoint |
| Epochs | 5 |
| Batch size | 32 (encoder gradients need more VRAM) |
| LR | 1e-6 (100x lower) |
| Feature caching | No (encoder is being updated) |
| Expected time | ~12-15 min |

### Total Estimated Time

| Step | Estimated (L40S) |
|---|---|
| Data generation (20K pairs) | ~20 min |
| Feature caching | ~8 min |
| Phase 1 training (30 epochs) | ~8-10 min |
| Phase 2 fine-tune (5 epochs) | ~12-15 min |
| **Total** | **~50-55 min** |

### Cost

L40S at ~$1.00-1.50/hr on-demand → **~$1.50 total**

## 3. Accuracy Targets

| Metric | v2 baseline (5K, T4) | v5 target (20K, L40S) |
|---|---|---|
| Mean pixel distance | 30px | <20px |
| Hit @20px | 31% | >50% |
| Hit @50px | 88% | >95% |
| Hit @100px | 100% | 100% |
| Bottom-half accuracy | ~0% | >80% @50px |
| Identity pair accuracy | Poor | <5px mean |

---

## 4. Results

### Calibration Run (5K pairs, 5 epochs)

_Pending — results will be added here after run completes._

| Epoch | Train Loss | Val Loss | Mean px | @20px | @50px | @100px | Time |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |

### Full Run (20K pairs, 30+5 epochs)

_Pending — results will be added here after run completes._
