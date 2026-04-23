# CrossMatch Training V5/V5.1 — L40S Training Plan & Results

**Date**: 22-23 April 2026  
**GPU**: NVIDIA L40S (48GB VRAM, Ada Lovelace)  
**Data**: V5 (archetypes + SVG icons + varied resolutions), V5.1 (+ pairing strategies for semantic matching)

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

### V5 Calibration Run (5K v5 pairs, 5+25 epochs, L40S)

**Data**: 5K v5 pairs (proportional coordinate mapping only, no pairing strategies)  
**Total training time**: 3.8 min (5 epochs) + 10.6 min (25 more epochs) = ~15 min  
**Feature caching**: 10K images at 20 img/s = ~8 min

| Epoch | Train Loss | Val Loss | Mean px | Median px | p95 | @10px | @20px | @50px | @100px | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0711 | 0.0512 | 884 | 903 | 1470 | 0% | 0% | 0% | 0% | 95s |
| 2 | 0.0125 | 0.0022 | 127 | 118 | 255 | 0% | 2% | 12% | 39% | 35s |
| 4 | 0.0004 | 0.0003 | 63 | 56 | 137 | 2% | 8% | 43% | 85% | 32s |
| **5** | **0.0002** | **0.0002** | **54** | **48** | **117** | **3%** | **12%** | **53%** | **91%** | **29s** |
| *Resumed from epoch 5 checkpoint (lr=5e-5, warmup=3):* |
| +7 | 0.0001 | 0.0001 | 39 | 34 | 81 | 7% | 23% | 73% | 98% | 28s |
| +9 | 0.0001 | 0.0001 | 33 | 29 | 71 | 9% | 30% | 81% | 99% | 30s |
| +15 | 0.0000 | 0.0001 | 32 | 30 | 63 | 9% | 29% | 87% | 100% | 28s |
| **+17** | **0.0000** | **0.0000** | **28** | **26** | **59** | **12%** | **36%** | **91%** | **100%** | **29s** |
| +21 | 0.0000 | 0.0000 | 29 | 26 | 59 | 10% | 33% | 91% | 100% | 31s |
| +25 | 0.0000 | 0.0000 | 29 | 26 | 59 | 11% | 33% | 91% | 100% | 29s |

**Converged at epoch ~17**: 28px mean, 91% @50px, 100% @100px.

**Key observation**: The model achieved good synthetic accuracy but learned to match by **position scaling** rather than visual content. On real device screenshots, accuracy was poor because the model couldn't find elements at positions different from the training distribution.

### V5.1 Training (5K-15K v5.1 pairs, pairing strategies)

_Pending — training with new pairing strategy data (proportional 45% + independent 40% + shuffled 15% + identity 12%)._

### Full Run (15-20K pairs, 30+5 epochs)

_Pending._
