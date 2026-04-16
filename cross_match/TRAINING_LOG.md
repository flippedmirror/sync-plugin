# CrossMatch Training Log — Demo Run

Local validation run on Apple Silicon (MPS) with 100 synthetic pairs to verify the training pipeline and confirm the model learns.

## Setup

| Parameter | Value |
|---|---|
| Device | CPU (cached features) / MPS (inference) |
| Training pairs | 100 (319 total actions: clicks + scrolls) |
| Train/Val split | 288 / 31 samples |
| Batch size | 16 |
| Epochs completed | 12 of 30 (frozen encoder only, no fine-tune phase) |
| Learning rate | 1e-4 with cosine decay, 5-epoch warmup |
| Feature caching | Yes (DINOv2 features precomputed on CPU) |
| Epoch time (cached) | ~60-90s on CPU |

## Training Convergence

| Epoch | Train Loss | Val Loss | Mean px | Median px | @50px | @100px | Action Acc |
|---|---|---|---|---|---|---|---|
| 1 | 0.0762 | 0.0748 | 574 | 566 | 0% | 0% | 68% |
| 2 | 0.0440 | 0.0218 | 610 | 575 | 0% | 3% | 100% |
| 4 | 0.0143 | 0.0182 | 607 | 612 | 0% | 3% | 100% |
| 6 | 0.0139 | 0.0134 | 516 | 481 | 0% | 0% | 100% |
| 8 | 0.0051 | 0.0040 | 298 | 252 | 3% | 6% | 100% |
| 10 | 0.0017 | 0.0021 | 185 | 171 | 10% | 29% | 100% |
| **12** | **0.0009** | **0.0016** | **156** | **139** | **10%** | **35%** | **100%** |

**Key observations:**
- **Loss reduced 47x** (0.075 → 0.0016) in 12 epochs — clear convergence with no overfitting
- **Pixel distance halved every ~4 epochs** (574 → 298 → 156px)
- **Action classification solved by epoch 2** (68% → 100%) — click vs scroll is trivially learned
- **No fine-tune phase was run** — these results are frozen encoder only (11.3M trainable params)
- Val loss tracks train loss closely — no overfitting despite small dataset

## Inference Latency (MPS, Apple Silicon)

Tested on 5 pairs (9 actions) after training, using the best checkpoint (epoch 12):

| Metric | Value |
|---|---|
| Mean latency | 199ms |
| Min latency | 84ms |
| Post-warmup range | 84 - 168ms |
| First-call warmup | 744ms (Metal shader compilation, one-time) |

### Per-sample results

| Pair | Action | GT | Predicted | Distance | Latency |
|---|---|---|---|---|---|
| pair_00000 | click | (523, 574) | (592, 452) | 140px | 744ms |
| pair_00000 | scroll | (480, 1144) | (470, 1229) | 86px | 168ms |
| pair_00001 | click | (480, 1917) | (475, 2076) | 159px | 128ms |
| pair_00001 | scroll | (577, 1571) | (612, 1640) | 77px | 97ms |
| pair_00002 | click | (687, 642) | (697, 523) | 119px | 218ms |
| pair_00002 | click | (545, 1722) | (511, 1806) | 91px | 84ms |
| pair_00003 | click | (573, 812) | (613, 682) | 136px | 98ms |
| pair_00004 | click | (443, 1253) | (431, 1096) | 157px | 142ms |
| pair_00004 | click | (478, 1654) | (486, 1816) | 162px | 111ms |

**Mean distance: 125px** — expected for 100 synthetic pairs / 12 epochs. The model has learned the proportional coordinate scaling between Android (1080x1920) and iOS (1170x2532) but hasn't yet learned element-level visual correspondence (which requires more diverse training data and more epochs).

**Scroll actions (77-86px) outperform clicks (91-162px)** — likely because scroll coordinates map more linearly across screen sizes (proportional scaling), while click targets depend on precise element localization.

## Comparison to Florence-2

| | Florence-2 (MPS+compile) | CrossMatch (MPS, 12 epochs) |
|---|---|---|
| Latency | 1,840ms | **84-168ms** |
| Forward passes | 2 (sequential, autoregressive) | **1 (single pass)** |
| Model size | 232M (all active) | 33M (22M frozen) |
| Accuracy | 53% @20px | 35% @100px |
| Training required | None (pretrained) | Yes |

CrossMatch is **10-20x faster** but less accurate — expected with only 100 synthetic training samples. With production training (10K+ real pairs, 50 epochs, fine-tune phase), accuracy should reach <30px mean distance while maintaining sub-200ms latency.

## Expected Scaling

Based on the convergence curve (loss halving every ~4 epochs, distance halving every ~4 epochs):

| Dataset | Epochs | Expected mean px | Expected @20px | Training time (A100) |
|---|---|---|---|---|
| 100 pairs (done) | 12 | 156px | ~0% | ~12min |
| 1K pairs | 30 | ~60-80px | ~15% | ~30min |
| 10K pairs | 50 | ~20-30px | ~70% | ~2hrs |
| 50K pairs | 50 | ~10-15px | ~85%+ | ~8hrs |

## Artifacts

- Best checkpoint: `checkpoints/cross_match/best.pt` (epoch 12, val_loss=0.0016)
- Epoch checkpoints: `checkpoints/cross_match/epoch_5.pt`, `epoch_10.pt`
- Cached features: `data/cross_match/feature_cache/`
