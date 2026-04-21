# CrossMatch: Cross-Platform UI Action Matching Model

A lightweight model that translates user actions (clicks, scrolls) from one device screenshot to another in a **single forward pass** — no autoregressive generation, no multi-call pipeline.

## Problem

Given a user action on device A (source), predict the equivalent action coordinates on device B (target), where A and B may be different platforms (Android/iOS) with different screen sizes, resolutions, and UI frameworks.

### Actions Supported

```
click:
  input:  { at: (x, y) }              # tap location on source
  output: { at: (x, y) }              # predicted tap location on target

scroll:
  input:  { from_arg: (x, y), to_arg: (x, y) }   # swipe start/end on source
  output: { from_arg: (x, y), to_arg: (x, y) }   # predicted swipe start/end on target
```

## Architecture

```
                    ┌─────────────────────┐
                    │  DINOv2-small (22M)  │  ← frozen pretrained encoder
                    │  shared weights      │     (weight-shared for both images)
                    └──────┬──────┬───────┘
                           │      │
                source_img │      │ target_img
                           ▼      ▼
                     src_features  tgt_features
                     (1369, 384)   (1369, 384)
                           │
            ┌──────────────┼──────────────┐
            │              │              │
     ┌──────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
     │  Coordinate  │ │  Action  │ │   Global    │
     │   Encoder    │ │ Encoder  │ │   Pool      │
     │ (sinusoidal  │ │ (embed)  │ │ (mean of    │
     │  PE + MLP)   │ │          │ │  src_feat)  │
     └──────┬───────┘ └────┬─────┘ └──────┬──────┘
            │              │              │
            ▼              ▼              ▼
         coord_tok      action_tok     global_tok
         (1, 384)       (1, 384)       (1, 384)
            │              │              │
            └──────────┬───┘──────────────┘
                       ▼
              queries (3, 384)   ← 3 query tokens
                       │
                       ▼
            ┌─────────────────────┐
            │  Self-Attention (2L) │  ← queries interact with each other
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ Cross-Attention (4L) │  ← queries attend to target_features
            │  Q: queries          │
            │  K,V: tgt_features   │
            └──────────┬──────────┘
                       │
                       ▼
              mean pool queries → (384,)
                       │
              ┌────────┴────────┐
              ▼                 ▼
       ┌─────────────┐  ┌─────────────┐
       │  Coord Head  │  │ Action Head  │
       │ MLP → (4,)   │  │ Linear → (2,)│
       │ + Sigmoid     │  │              │
       └──────┬───────┘  └──────┬───────┘
              ▼                 ▼
        target_coords      action_logits
     (at_x, at_y, to_x,   (click, scroll)
      to_y) in [0, 1]
```

## Components

### 1. Vision Encoder — DINOv2-small (frozen)

- **Model:** `facebook/dinov2-small` (ViT-S/14)
- **Params:** 22.06M (all frozen)
- **Input:** 518x518 image (37x37 = 1369 patches, patch size 14)
- **Output:** (1369, 384) patch feature tokens
- **Why DINOv2:** Self-supervised training on visual correspondence — the model learns patch-level feature matching, which is exactly our task. DINOv2 features have been shown to provide strong spatial correspondence across viewpoints, which translates well to cross-platform UI element matching.
- **Weight-shared:** Same encoder instance processes both source and target screenshots.

### 2. Coordinate Encoder — Sinusoidal PE + MLP

Encodes the 4 input coordinates (click: `x, y, 0, 0`; scroll: `from_x, from_y, to_x, to_y`) into a single embedding vector.

- **Encoding:** Log-spaced sinusoidal frequencies (64 bands), applied independently to each coordinate, producing sin and cos components.
  ```
  input:  (4,) normalized coords in [0, 1]
  → sinusoidal: (4 × 2 × 64) = (512,)
  → MLP: Linear(512, 384) → GELU → Linear(384, 384)
  output: (384,)
  ```
- **Params:** ~50K
- **Why sinusoidal:** Smooth, continuous representation of spatial position. The log-spaced frequencies give multi-scale resolution — fine for nearby elements, coarse for global position.

### 3. Action Encoder — Learnable Embedding

- **Embedding table:** 2 × 384 (click=0, scroll=1)
- **Params:** 768
- **Purpose:** Conditions the cross-attention on action semantics. A click needs precise point correspondence; a scroll needs start/end trajectory mapping.

### 4. Query Construction

Three query tokens are assembled:
1. **coord_token** — where the action happened on the source
2. **action_token** — what type of action it was
3. **global_token** — mean-pooled source features (overall screen context)

These pass through a linear projection, then 2 layers of self-attention (so the tokens can exchange information before attending to the target).

### 5. Cross-Attention Transformer — 4 layers

Each layer:
```
query = LayerNorm(query + MultiheadAttention(Q=query, K=target_features, V=target_features))
query = LayerNorm(query + FFN(query))
```
- **Layers:** 4
- **Heads:** 6 (head_dim = 64)
- **FFN:** 384 → 1536 → 384 (GELU activation)
- **Dropout:** 0.1
- **Params:** ~6M

The cross-attention mechanism learns to localize: given the source context (what/where/how), attend to the target patches that correspond to the same UI element.

### 6. Output Heads

After cross-attention, the 3 query tokens are mean-pooled to a single (384,) vector.

**Coordinate Head:**
```
Linear(384, 384) → GELU → Linear(384, 4) → Sigmoid
```
Outputs 4 normalized coordinates in [0, 1]. For clicks, only the first 2 are used (the loss masks out coords 3-4).

**Action Head:**
```
Linear(384, 2)
```
Classifies click vs scroll. Serves as a consistency check — the action type shouldn't change during translation, so this head validates the model is understanding the task.

## Parameter Budget

| Component | Params | Trainable |
|---|---|---|
| DINOv2-small encoder | 22.06M | No |
| Coordinate encoder (sinusoidal PE + MLP) | ~0.25M | Yes |
| Action encoder (embedding) | ~0.001M | Yes |
| Query projection | ~0.15M | Yes |
| Self-attention (2 layers) | ~2.4M | Yes |
| Cross-attention (4 layers) | ~6M | Yes |
| Coordinate head | ~0.15M | Yes |
| Action head | ~0.001M | Yes |
| **Total** | **33.35M** | **11.29M** |

**On disk:** ~127MB (fp32), ~64MB (fp16), ~32MB (INT8)

## Inference Latency

| Hardware | Latency (single sample) | Notes |
|---|---|---|
| CPU (Apple M1) | ~568ms | Dominated by DINOv2 encoder (2 images) |
| MPS (Apple Metal) | ~50-80ms (est.) | GPU acceleration for both encoder + head |
| CUDA T4 | ~10-15ms (est.) | |
| CUDA A100 | ~5-8ms (est.) | |

This is a **single forward pass** — no autoregressive decoding, no multi-call pipeline. Compare to Florence-2's best result of 1.84s (2 calls × ~0.9s).

## Training Plan

### Phase 1: Frozen Encoder (epochs 1–45)

- **Encoder:** Frozen. All DINOv2 weights are fixed.
- **What trains:** Coordinate encoder, action encoder, self-attention, cross-attention, output heads (~11.3M params)
- **Feature caching:** DINOv2 features are precomputed for all images and saved to disk. Training epochs only run the lightweight head, making each epoch very fast.
- **Optimizer:** AdamW, lr=1e-4, weight_decay=1e-2
- **Schedule:** Linear warmup (5 epochs) → cosine decay

### Phase 2: Fine-tune Encoder (epochs 46–50)

- **Encoder:** Unfrozen with very low lr (1e-6)
- **Feature caching:** Disabled (must recompute features since encoder weights change)
- **Purpose:** Adapt DINOv2's patch features slightly toward UI-specific correspondence. Limited to 5 epochs to avoid catastrophic forgetting of general visual features.
- **Encoder lr:** 1e-6 (100x lower than head lr at this point)

### Loss Function

```
L = coord_weight × L_coord + action_weight × L_action
```

**Coordinate loss (smooth L1):**
- Click actions: loss on coords [0, 1] (at_x, at_y); coords [2, 3] masked to zero
- Scroll actions: loss on all coords [0, 1, 2, 3] (from_x, from_y, to_x, to_y)
- Per-sample masking ensures the model isn't penalized for unused coordinate slots

**Action loss (cross-entropy):**
- Weight: 0.1 (secondary objective)
- Purpose: regularization signal ensuring the model understands action type

**Gradient clipping:** max_norm=1.0

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Batch size | 32 | Fits in 16GB GPU with frozen encoder |
| Learning rate | 1e-4 | Standard for small transformers |
| Weight decay | 1e-2 | Regularize cross-attention weights |
| Warmup epochs | 5 | Stabilize early training |
| Total epochs | 50 | 45 frozen + 5 fine-tune |
| Coord loss weight | 1.0 | Primary objective |
| Action loss weight | 0.1 | Auxiliary signal |
| Fine-tune lr | 1e-6 | Conservative encoder adaptation |

### Estimated Training Time

Training only the 11.3M trainable params with cached features:

| Dataset size | Samples (actions) | A100 | T4 | M-series MPS |
|---|---|---|---|---|
| 1K pairs | ~3K | ~5 min | ~15 min | ~30 min |
| 10K pairs | ~30K | ~30 min | ~2 hrs | ~4 hrs |
| 50K pairs | ~150K | ~2 hrs | ~8 hrs | ~16 hrs |
| 100K pairs | ~300K | ~4 hrs | ~16 hrs | ~32 hrs |

Fine-tuning phase (5 epochs, no caching) adds ~30% to total time since the encoder must run during training.

## Data Requirements

### Annotation Format

```json
{
  "pairs": [
    {
      "id": "pair_001",
      "source": {
        "image": "source/screen_001.png",
        "platform": "android",
        "size": [1080, 1920]
      },
      "target": {
        "image": "target/screen_001.png",
        "platform": "ios",
        "size": [1170, 2532]
      },
      "actions": [
        {
          "type": "click",
          "source_coords": { "at": [540, 960] },
          "target_coords": { "at": [585, 1266] }
        },
        {
          "type": "scroll",
          "source_coords": { "from_arg": [540, 1200], "to_arg": [540, 600] },
          "target_coords": { "from_arg": [585, 1583], "to_arg": [585, 791] }
        }
      ]
    }
  ]
}
```

Coordinates are in **pixel space** relative to the original screenshot dimensions. The dataset loader normalizes them to [0, 1] using the `size` field.

### Data Sources

1. **Synthetic generation** — progressively improved through 3 generator versions:

   - **v1** (`cross_match/synthetic.py`) — flat colored rectangles on solid backgrounds. Good for initial validation but limited visual diversity.

   - **v2** (`cross_match/synthetic_v2.py`) — gradient backgrounds, shadows, icons, cards, toggles, nav bars, platform-specific palettes. Significant visual diversity improvement, but has a **Y-distribution bias**: top-down `y_cursor` layout causes 77% of click actions to land in the top 25% of the screen, with 0% below the midpoint. Root cause: elements are placed sequentially and exhaust vertical space before reaching the lower half.

   - **v4** (`cross_match/synthetic_v4.py`) — current generator, fixes all known v2 issues:
     - **Slot-based layout:** screen divided into 6–12 equal vertical slots (6%–90% Y range), one element per slot, guaranteeing uniform spatial coverage
     - **Multi-resolution:** 4 Android devices (720×1600 to 1440×2560) + 4 iOS devices (750×1334 to 1290×2796), any-to-any pairing including same-OS different-device
     - **Resolution-scaled fonts:** text sizes scale proportionally to `screen_height / 1920`, keeping visuals consistent across resolutions
     - **Icon-only elements:** FAB (floating action button with + icon), icon buttons (hamburger, back arrow, close X, bell), icon-only nav bars — ~39% of click actions target icon-based elements
     - **Distribution stats in annotations.json:** source + target Y quartile distributions, resolution counts, and element type breakdowns are computed and embedded automatically for validation
     - Validated Y-distribution: ~21/28/29/22% across quartiles (source and target)

2. **BrowserStack same-OS sync recordings** — record action mappings between two same-OS devices (e.g., Android to Android), then use those mappings as ground truth for cross-platform training by pairing with a different platform's screenshot of the same app state.

3. **Web/app screenshot pairs** — capture the same app on both platforms, manually or semi-automatically annotate corresponding elements.

4. **Augmentation strategies:**
   - Random crop/pad (simulate different screen regions)
   - Color jitter on backgrounds (vary app themes)
   - Resolution variation (different device densities) — now built into v4 generator
   - Element reordering (same elements, different layout)

### What the model needs to learn

1. **Visual correspondence** — "this blue button on source is the same as that blue button on target" even though positions, sizes, and surrounding context differ
2. **Spatial awareness** — understanding that a click at (50%, 25%) on Android likely maps to roughly (50%, 25%) on iOS, adjusted for UI differences
3. **Action semantics** — a scroll's start/end points define a trajectory that maps proportionally across screen sizes
4. **Platform conventions** — Android navigation bar is at bottom, iOS has a notch at top; status bar heights differ

## Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| **Mean pixel distance** | L2 distance between predicted and GT click/scroll points | <30px |
| **Hit rate @20px** | % of predictions within 20px of GT | >85% |
| **Hit rate @50px** | % of predictions within 50px of GT | >95% |
| **Action accuracy** | % of correct action type predictions | >99% |
| **Scroll trajectory error** | Mean angular + magnitude error for scroll vectors | <10% |

## Usage

```bash
# Generate synthetic training data (v4 — multi-resolution, uniform Y distribution)
python -m cross_match.synthetic_v4 --output-dir data/cross_match_v4 --num-pairs 5000

# Train (CUDA)
python -m cross_match.train --data-dir data/cross_match --output-dir checkpoints/cross_match --device cuda

# Train without feature caching (slower, required if modifying encoder)
python -m cross_match.train --data-dir data/cross_match --output-dir checkpoints/cross_match --no-cache

# Inference
python -c "
from cross_match.predict import CrossMatchPredictor
from PIL import Image

p = CrossMatchPredictor('checkpoints/cross_match/best.pt')
result = p.predict(
    source_image=Image.open('source.png'),
    target_image=Image.open('target.png'),
    action_type='click',
    source_coords={'at': (540, 960)},
)
print(result)
# {'type': 'click', 'target_coords': {'at': (585, 1266)}, 'latency': 0.07}
"
```
