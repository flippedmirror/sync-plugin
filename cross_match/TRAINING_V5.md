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

### V5.1 Calibration Run (5K v5.1 pairs, 25 epochs, L40S)

**Data**: 5K v5.1 batch 1 (with pairing strategies: proportional 45% + independent 40% + shuffled 15% + identity 12%)  
**Instance**: L40S, 48GB VRAM, 30GB RAM, EBS gp3 storage  
**Total training time**: ~13 min  
**Feature caching**: 10K images at 20 img/s = ~8 min  

| Epoch | Train Loss | Val Loss | Mean px | Median px | p95 | @50px | @100px | Time |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0673 | 0.0506 | 885 | 908 | 1460 | 0% | 0% | 96s |
| 3 | 0.0132 | 0.0109 | 455 | 330 | 1454 | 3% | 8% | 29s |
| 5 | 0.0114 | 0.0103 | 426 | 273 | 1455 | 6% | 18% | 30s |
| 8 | 0.0109 | 0.0096 | 400 | 232 | 1381 | 7% | 21% | 29s |
| 13 | 0.0099 | 0.0100 | 388 | 186 | 1384 | 9% | 30% | 28s |
| **17** | **0.0090** | **0.0103** | **435** | **274** | **1307** | **7%** | **19%** | **29s** |
| 25 | 0.0061 | 0.0113 | 406 | 185 | 1491 | 12% | 31% | 27s |

**Observations**:
- v5.1 data is much harder than v5 — at epoch 25, mean distance is 406px (vs 29px for v5)
- **Bimodal distribution**: Median (185px) much better than mean (406px). The proportional pairs are learned well, the independent-layout pairs produce outliers (p95 ~1400px)
- **Val loss plateaued at epoch ~8** (0.0096) and started rising — overfitting on 5K training samples
- Train loss keeps dropping while val loss rises — classic overfitting signal
- 5K pairs with 40% independent strategy = only ~2K independent pairs — insufficient for the model to learn visual matching
- **Conclusion**: Need more data (10K-20K) for the independent strategy to work. The model can't learn semantic matching from just ~2K independent pairs with 11M trainable params.

---

## 5. Infrastructure Issues & Learnings

### 5.1 cuBLAS Compatibility (cublasLtCreate)

**Error**: `Invalid handle. Cannot load symbol cublasLtCreate` on L40S with PyTorch 2.11 + CUDA 13.0 driver.

**Impact**: DINOv2 encoder forward pass crashes at batch sizes > 16 when run through the training script's full model.forward() path. Does NOT affect:
- Standalone `model._encode_image()` calls (caching works at batch_size=16)
- Basic CUDA operations (matmul, linear layers)

**Root cause**: Version mismatch between PyTorch-bundled cuBLAS and system CUDA 13.0 cuBLAS. The encoder uses operations (`torch.nn.functional.linear` in attention layers) that trigger `cublasLtCreate` which fails with the bundled library.

**Workaround**: 
- Feature caching at `--cache-batch-size 16` works (uses `torch.no_grad()` path)
- No-cache training requires batch_size ≤ 16 (too slow for 20K data: ~600s/epoch)
- `LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64` did NOT fix it

### 5.2 Feature Cache I/O Bottleneck

**The core scaling blocker for 10K+ pair training.**

**Architecture**: Feature caching precomputes DINOv2 encoder outputs once, saves as individual `.pt` files (one per image, ~2MB each), then training loads them via DataLoader workers each epoch.

**What works**:
- 5K pairs = 10K `.pt` files, ~20GB total cache → **30s/epoch** ✅
- OS page cache (20GB available on 30GB RAM instance) holds the entire 5K cache after epoch 1
- All subsequent epochs read from RAM, not disk

**What breaks**:
- 10K pairs = 20K files, ~40GB cache → **epochs never complete** ❌
- 20K pairs = 40K files, ~80GB cache → **same** ❌
- Cache exceeds OS page cache → every epoch re-reads from EBS disk
- EBS gp3: 3,000 IOPS baseline, 125 MB/s throughput
- 20K random `torch.load()` calls per epoch: each involves file open + read + pickle deserialize + close
- 4 DataLoader workers compete for I/O bandwidth + Python GIL during deserialization

**Per-batch theoretical timing** (batch_size=64):

| Step | Data size | Speed | Time |
|---|---|---|---|
| EBS → RAM (64 random .pt reads) | 128MB | 3,000 IOPS | ~21ms |
| RAM → GPU (PCIe Gen4) | 128MB | 25 GB/s | ~5ms |
| GPU compute (cross-attention fwd+bwd) | — | — | ~2-5ms |
| **Total per batch** | | | **~28-31ms** |

**Per-epoch theoretical** (10K pairs, 660 batches): 660 × 30ms = **~20s** — should work.

**Why theory doesn't match reality**: `torch.load()` is much more expensive than raw file I/O:
- Python pickle deserialization: ~0.2ms per file (adds ~4s/epoch for 20K files)
- GIL contention: 4 workers all deserializing simultaneously serialize on the GIL
- Memory allocation: each `torch.load()` allocates a new tensor (~0.1ms)
- File handle overhead: open/close 20K files per epoch
- OS page cache thrashing: 40GB cache, 20GB page cache → continuous eviction

**Attempted fixes that didn't work**:
- `num_workers=8` with `persistent_workers=True`: Caused DataLoader deadlock
- `prefetch_factor=4`: No improvement on EBS random reads
- No-cache mode: Hits cublasLtCreate bug at batch_size > 16

### 5.3 Instance Constraints

| Resource | Available | 5K needs | 10K needs | 20K needs |
|---|---|---|---|---|
| GPU VRAM | 48GB | ~3.5GB | ~3.5GB | ~3.5GB |
| RAM | 30GB | 20GB cache fits in page cache | 40GB — doesn't fit | 80GB — way over |
| EBS IOPS | 3,000 | Sufficient | Bottleneck | Severe bottleneck |
| EBS throughput | 125 MB/s | OK | Marginal | Insufficient |

### 5.4 Proposed Solutions

| # | Approach | How it works | Effort | Instance requirement |
|---|---|---|---|---|
| **1** | **Memory-mapped numpy** | Save all features as 2 flat `.npy` files. Dataset reads via `numpy.memmap` — OS handles paging, zero pickle overhead, zero file open/close | Medium | Any (OS manages paging) |
| 2 | Consolidated chunk files | Save features in ~100 chunk files of ~200 samples each. Reduces IOPS from 20K to ~100 per epoch | Medium | Any |
| 3 | Preload into RAM | Load all .pt into a dict at startup. Zero disk I/O during training | Low | 96GB+ RAM |
| 4 | Local NVMe instance | Use instance with local SSD (e.g., g5.xlarge). 100K+ IOPS, 1-7 GB/s | None | Different instance type |
| 5 | Higher-throughput EBS | Provision io2 volume with 10K+ IOPS | None | ~$100/month storage cost |

**Resolution**: Use an instance with local NVMe SSD (Option 4). The EBS I/O bottleneck is a storage problem, not a code problem. Local NVMe provides 100K+ IOPS and 1-7 GB/s — eliminates the bottleneck without any code changes.

---

## 6. Revised Plan — g5.2xlarge with NVMe

### 6.1 Why g5.2xlarge

Previous runs used L40S on EBS storage. Two problems emerged:
1. cuBLAS incompatibility on the L40S AMI (CUDA 13.0 + PyTorch 2.11)
2. EBS I/O bottleneck for 10K+ cached feature files

The g5.2xlarge solves both:
- **A10G GPU**: Different CUDA stack, no cuBLAS issue expected (needs verification)
- **450GB local NVMe SSD**: 100K+ IOPS, eliminates I/O bottleneck
- **32GB RAM**: ~26GB page cache, partial cache fits. NVMe handles the rest.
- **$1.21/hr**: Cheapest viable option

| Spec | g5.2xlarge |
|---|---|
| GPU | 1x NVIDIA A10G (24GB VRAM) |
| RAM | 32GB |
| Storage | 450GB local NVMe SSD |
| vCPUs | 8 |
| Cost | $1.21/hr |
| A10G vs T4 | ~3x faster encoder, ~2x faster training |

### 6.2 Training Plan (Phase 1 Only)

Phase 2 (encoder fine-tuning) is deferred — adds ~10 hrs of training with uncertain benefit. Will reconsider after Phase 1 results.

**Data**: 20K v5.1 pairs (4 batches × 5K, all generated locally)  
**Strategy mix**: Proportional 45% + Independent 40% + Shuffled 15% + Identity 12%

| Step | What | Expected time |
|---|---|---|
| 1. Test run (100 pairs, 2 epochs) | Validate: deps install, CUDA works, caching works, training completes, no cuBLAS crash | ~5 min |
| 2. Upload 20K data | SCP 4 × 1.2GB tarballs to instance | ~15 min |
| 3. Cache features | DINOv2 encoder on 40K images, batch_size=16, save .pt files to NVMe | ~56 min |
| 4. Train Phase 1 | 50 epochs, batch_size=64, frozen encoder, cosine LR 1e-4, warmup 5 epochs | ~6 min |
| 5. Download checkpoint | SCP best.pt (~127MB) locally | ~1 min |
| **Total** | | **~83 min** |
| **Cost** | ~1.4 hrs × $1.21/hr | **~$1.70** |

### 6.3 Test Run Protocol (Step 1)

Before committing to the full run, validate the entire pipeline with minimal data:

```bash
# 1. Install deps
pip3 install torch torchvision pillow

# 2. Verify CUDA
python3 -c 'import torch; print(torch.cuda.get_device_name(0))'

# 3. Clone repo
git clone https://github.com/flippedmirror/sync-plugin.git

# 4. Generate 100 test pairs
python3 -u -m cross_match.synthetic_v5 --output-dir data/test_100 --num-pairs 100 --seed 99

# 5. Run 2-epoch training with caching
python3 -u -m cross_match.train \
  --data-dir data/test_100 \
  --output-dir checkpoints/test \
  --epochs 2 --finetune-epochs 0 \
  --batch-size 16 --lr 1e-4 --warmup-epochs 1 \
  --cache-batch-size 16 \
  --device cuda
```

**Pass criteria**: Both epochs complete, progress.json shows "complete", GPU utilization > 0% during training, no cublasLtCreate errors.

**If test fails**: Debug on the test instance before uploading 20K data. Do not proceed until test passes.

### 6.4 Accuracy Targets

| Metric | v5 (5K, proportional only) | v5.1 target (20K, mixed strategies) |
|---|---|---|
| Mean pixel distance | 29px | <100px |
| @50px | 91% | >40% |
| @100px | 100% | >70% |
| Independent pair @100px | N/A | >50% |
| Identity pair mean | N/A | <10px |

Note: Targets are conservative. v5.1's independent layout pairs make this a fundamentally harder task than v5. The model must learn visual matching, not position scaling. Lower absolute numbers are expected.

### 6.5 Decision Points

After Phase 1 completes:

1. **If @100px > 70%**: Export to ONNX, test on real device screenshots. Consider Phase 2 only if real-world accuracy needs improvement.
2. **If @100px 40-70%**: Run 50 more epochs from checkpoint (same data). If still plateaued, consider more data or architecture changes.
3. **If @100px < 40%**: The frozen DINOv2 features may be insufficient for semantic matching. Consider Phase 2 on a faster GPU (L40S with compatible CUDA), or switch to a different encoder.

---

## 7. Checkpoints & Data Inventory (Local)

| Item | Path | Size |
|---|---|---|
| v5 checkpoint (5K, 25ep) | `checkpoints/v5_5k_25ep/best_25ep.pt` | 127MB |
| v5 ONNX model | `plugin/models/cross_match_v5.onnx` | 128MB |
| v5.1 checkpoint (5K, 25ep) | `checkpoints/v5.1_5k/best.pt` | 127MB |
| v5.1 batch 1 data (seed=500) | `data/cross_match_v5.1_batch1/` | 1.3GB |
| v5.1 batch 2 data (seed=600) | `data/cross_match_v5.1_batch2/` | 1.3GB |
| v5.1 batch 3 data (seed=700) | `data/cross_match_v5.1_batch3/` | 1.3GB |
| v5.1 batch 4 data (seed=800) | `data/cross_match_v5.1_batch4/` | 1.3GB |
| v5.1 batch tarballs | `data/cross_match_v5.1_batch{1-4}.tar.gz` | 4 × 1.2GB |
| Original v5 data (5K) | `data/cross_match_v5_5k/` | 1.3GB |
| Original v2 ONNX model | `plugin/models/cross_match.onnx` | 128MB |
