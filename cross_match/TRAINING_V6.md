# CrossMatch V6 — SigLIP2-small Encoder

**Date**: 24 April 2026  
**Change**: Replace DINOv2-small encoder with SigLIP2-small  
**Data**: Same 20K v5.1 pairs (pairing strategies)

---

## 1. Why SigLIP2

V5/V5.1 training with DINOv2-small showed the frozen encoder is the bottleneck:
- DINOv2 was trained on natural images via self-supervised learning
- It captures shapes and textures but can't semantically distinguish UI elements
- Independent-layout pairs (40% of data) require recognizing "this is the Chrome icon" regardless of position — DINOv2 features can't do this
- Result: val loss plateaued at epoch 15 and rose thereafter (overfitting)

SigLIP2-small is trained on **image-text pairs** (400M web pairs). It learns that visual content has semantic meaning — "this icon with a compass shape relates to the concept 'Safari'". This should produce more discriminative features for matching identical UI elements across different screen layouts.

## 2. Model Architecture

| Component | V5 (DINOv2) | V6 (SigLIP2) |
|---|---|---|
| Encoder | DINOv2-small (22M) | SigLIP2-small (22M) |
| Encoder dim | 384 | 384 |
| Patch size | 14 | 16 |
| Input size | 518 | 256 |
| Patch tokens | 1,369 | 256 |
| Cross-attention head | 4 layers, 6 heads, 384 dim | Same (drop-in) |
| Head params | ~9.5M | ~9.5M |
| Coord encoder + action | ~1.8M | ~1.8M |
| **Total** | **~33M** | **~33M** |
| **Trainable (Phase 1)** | **11.3M** | **11.3M** |

### Key differences:
- **5x fewer patch tokens** (256 vs 1,369): cross-attention is O(n²), so this is ~25x less compute in the attention layers
- **Smaller input** (256 vs 518): faster image preprocessing, less GPU memory
- **Text-aware features**: SigLIP2 understands that text and icons have semantic meaning
- **Same head architecture**: no changes to cross-attention, coord encoder, or action heads

### Expected impact on inference:
- Fewer tokens = faster cross-attention = potentially **faster** browser inference despite same param count
- Smaller input = faster image resize in browser preprocessing
- ONNX model size similar (~128MB)

## 3. Training Plan

**Instance**: g5.xlarge (A10G, 16GB RAM, 250GB NVMe, $1.00/hr)  
**Data**: Same 20K v5.1 pairs already on instance

| Step | What | Expected time |
|---|---|---|
| 1. Test run (2 epochs) | Validate SigLIP2 loads, caching works, training completes | ~15 min |
| 2. Cache features | SigLIP2 encoder on 40K images, batch_size=16 | ~40 min (est.) |
| 3. Train Phase 1 | 50 epochs, batch_size=64, frozen encoder | TBD after test |
| 4. Evaluate | Compare vs V5.1 DINOv2 results | — |

### Test run protocol:
1. Modify `model.py` to support SigLIP2 encoder
2. Run 2 epochs on 20K data with caching
3. **Pass criteria**: caching completes, both epochs finish, no errors, GPU utilization > 0%
4. If test passes: launch full 50-epoch run

## 4. Code Changes Required

### model.py
- Add SigLIP2 encoder option alongside DINOv2
- Load via HuggingFace `transformers` (`SiglipVisionModel.from_pretrained`)
- Adjust `_encode_image()` to use SigLIP2's `last_hidden_state` output
- Handle different input preprocessing (SigLIP uses mean=0.5, std=0.5 vs ImageNet normalization)

### config.py
- Add `encoder_name` option: `"siglip2_small"` vs `"dinov2_vits14"`
- Update `image_size` default to 256 for SigLIP2

### dataset.py
- Make image transforms configurable based on encoder choice

### train.py
- No changes needed (encoder-agnostic)

### export_onnx.py
- Handle SigLIP2 input size for dummy tensors

## 5. Results

_Pending — will be filled after training._

## 6. Checkpoints

_Pending._
