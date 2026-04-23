# Synthetic Data V5 — Strategy & Design

**Date**: 22 April 2026  
**Status**: Implemented, pending large-scale generation + training

---

## 1. Problem Statement

The CrossMatch model (DINOv2-small + cross-attention, 33M params) was trained on v2 synthetic data — 5,000 pairs of simplistic screens with buttons, inputs, labels, and basic geometric icons. When tested on real BrowserStack App Live device screens, several accuracy issues emerged:

### Observed Failures

1. **Bottom-of-screen elements**: Clicks on elements in the lower 50% of the screen produced wildly inaccurate predictions. Root cause: 77% of training actions were in the top 25% of the screen, 0% below the midpoint.

2. **Icon recognition**: Real app icons (home screen grids, toolbar icons) are detailed rounded-square images with internal gradients and symbols. The model had only seen flat geometric shapes (circles, triangles, stars) and couldn't match real icons.

3. **Identity mapping failure**: Even with the same image as source and target, the model produced inaccurate coordinate mappings — indicating it hadn't learned the trivial identity case.

4. **Text label confusion**: Real screens have blurry small text at device resolution. Synthetic data used clean, large text exclusively.

5. **Layout mismatch**: Real screens use grid layouts (home screen icon grids, product card grids), scrollable lists, and complex navigation patterns. Synthetic data only generated top-down sequential layouts.

### Adversarial Test Images

Three real-world screenshots were collected to test the model:
- **iOS Home Screen (light)**: 4-column app icon grid, gradient wallpaper, bottom dock
- **iOS Home Screen (blue)**: Same structure, different wallpaper and icon arrangement
- **E-commerce App**: Product listing with photos, prices, "Add to cart" buttons, heart icons, filter bar

The model failed on all three — particularly on the icon grids and bottom-of-screen elements.

## 2. Analysis: Synthetic v2 vs Real Screens

| Aspect | Synthetic v2 | Real Screens | Gap Severity |
|---|---|---|---|
| Icons | Flat geometric shapes (circle, square, triangle) | Rounded-square app icons with gradients, internal symbols, brand logos | Critical |
| Layout | Top-down sequential sections only | Grid layouts (icon grids, product grids), complex navigation | Critical |
| Y-distribution | 77% in top 25%, 0% below midpoint | Elements across full screen including bottom dock/nav | Critical |
| Images/photos | None | Product photos, wallpapers, avatar images | High |
| Screen sizes | Fixed 1080x1920 / 1170x2532 | Many device resolutions (iPhone SE to Pro Max, Pixel to Galaxy) | High |
| Backgrounds | Solid colors, simple gradients | Complex wallpapers, mesh gradients, photo backgrounds | Medium |
| Text sizes | Uniform medium/large | Tiny icon labels, small prices, large headers, blurry at low res | Medium |
| Status bar | Not present | Always present with time, battery, signal | Low-Medium |
| Screen types | One generic layout | Home screens, product lists, settings, forms, chat | High |

## 3. Solution: V5 Synthetic Data Generator

### 3.1 Screen Archetypes

Instead of one generic layout, v5 generates five distinct screen archetypes matching common real-world screen types:

| Archetype | Weight | What It Generates |
|---|---|---|
| `home_screen` | 30% | Icon grid (3-5 cols, 3-6 rows) with SVG icons on colored rounded squares, mesh gradient wallpaper, bottom dock, tiny icon labels |
| `product_list` | 20% | Product cards with image placeholders, prices, "Add to cart" buttons, heart/favorite icons, filter bar, bottom text nav |
| `settings` | 20% | Toggle rows with colored circle icons, chevrons, separators, evenly spaced across full height |
| `form` | 15% | Login/registration with logo, input fields, primary button, social login row (Google/Apple/GitHub), terms text at bottom |
| `chat` | 15% | Message bubbles (left/right aligned), contact header with avatar, message input bar + send button at bottom |

Weights are configurable via `--archetype-weights 'home_screen:40,settings:30,form:30'`.

### 3.2 Real SVG Icon Packs

Downloaded and integrated two icon packs (6,450 SVGs total):

- **Phosphor Icons** (~3,024 icons: 1,512 fill + 1,512 regular): MIT license, covers every common UI action — settings gear, search, home, bell, camera, etc.
- **Simple Icons** (~3,426 icons): CC0 license, brand/logo icons — Chrome, WhatsApp, Instagram, GitHub, Spotify, etc.

Icon rendering pipeline:
1. Pick random SVG from a random pack
2. Recolor to white (for contrast on colored backgrounds)
3. Render to PNG at target size via `rsvg-convert` CLI (or `cairosvg` if available)
4. Composite onto a vibrant colored rounded-square background (20 colors)
5. Optional gradient overlay on background for depth

Fallback when no SVG renderer is available: random uppercase letter on colored rounded square.

Icons stored at `data/icons/phosphor/` and `data/icons/simple-icons/`.

### 3.3 Uniform Y-Distribution

Every archetype is designed to place elements across the full screen height (6-94%):

- **Home screen**: Icon grid starts at ~7% and fills to ~80%, bottom dock at ~92%
- **Product list**: Cards stack from ~10% downward, bottom nav at ~94%
- **Settings**: Row height is calculated as `available_height / num_rows` to spread evenly from ~8% to ~92%
- **Form**: Input fields at 20-50%, primary button at ~55%, social login buttons at 72-82%, terms text at ~92%
- **Chat**: Message spacing calculated as `available_height / num_messages` to distribute from ~10% to ~88%, input bar at ~94%

### 3.4 Screen Size Variety

Instead of fixed resolutions, each pair randomly selects from real device sizes:

**Android (8 sizes)**:
- 1080x1920, 1080x2340, 1080x2400, 1440x3040, 1440x3200, 720x1280, 720x1600, 1080x2160

**iOS (8 sizes)**:
- 1170x2532, 1179x2556, 1290x2796, 1125x2436, 828x1792, 750x1334, 1242x2688, 1284x2778

This teaches the model to handle different aspect ratios and resolutions, not just memorize one fixed mapping.

### 3.5 Pairing Strategies (Position Variability)

A critical discovery during initial testing: when target coordinates are always a proportional scaling of source coordinates (`target = source * resolution_ratio + jitter`), the model learns a simple linear mapping without looking at the images. It matches by position, not by visual content.

**Problem**: The model should learn "find the Chrome icon on the target screen" — not "scale coordinates by 1.08x, 1.32y".

**Solution**: Four pairing strategies with different levels of position variability:

| Strategy | Weight | How Target Coords Are Determined | What It Teaches |
|---|---|---|---|
| **Proportional** | 45% | `source_bbox * (target_size / source_size) + jitter(±12px)` | Basic resolution-aware coordinate scaling between devices |
| **Independent** | 40% | Target screen regenerated with same elements but different layout RNG. Coords matched by semantic identity (type + text). | Visual element matching — model MUST look at images |
| **Shuffled** | 15% | Proportional scaling but element bboxes randomly reordered across positions | Breaks pure position memorization |
| **Identity** | ~12% | Same image, same coordinates | Exact coordinate preservation |

**How independent pairing works**:
1. Source screen is generated with a set of elements (e.g., 12 app icons: Chrome, Safari, Maps...)
2. Target screen is generated with the **same element names** but a **different layout RNG** — different grid columns, different spacing, different margins
3. Elements are matched by `(type, text)` identity — "icon/Chrome" on source maps to "icon/Chrome" on target, regardless of position
4. Unmatched elements (present on source but not fitting on target layout) are excluded from training actions

**Example**: Home screen pair with independent strategy:
- Source: 4-column grid, Chrome at row 2 col 3 → position (810, 450)
- Target: 3-column grid, Chrome at row 3 col 1 → position (195, 600)
- The model must visually find Chrome, not just scale coordinates

**Impact on training data**: Independent pairs have slightly fewer actions per pair (~2.1 vs ~2.6 for proportional) because some elements don't match between layouts. This is acceptable — quality > quantity for these pairs.

### 3.6 Identity Pairs

12% of generated pairs use the **same image** for both source and target (same size too). This forces the model to learn the trivial identity mapping — if source and target are identical, the output coordinates should equal the input coordinates. This directly addresses the observed "same image gives wrong coords" failure.

### 3.7 Complex Backgrounds

- **Home screens**: Full mesh gradient wallpapers (3-4 color blobs blended with Gaussian blur)
- **Other archetypes**: 30% chance of subtle vertical gradient instead of flat solid
- 7 distinct color palettes (light, dark, warm, indigo, purple, green, orange) for visual diversity

### 3.7 Status Bar on All Screens

Every archetype renders a status bar at the top with:
- Time (random HH:MM)
- Battery indicator (rectangle with fill level)
- Signal strength bars (4 ascending rectangles)

### 3.9 Realistic Image Placeholders

Product cards contain photo-like rectangular areas rendered with one of 4 styles:
- Vertical gradient (two random colors)
- Diagonal split (two colors, triangular)
- Color blocks (random grid of colored rectangles)
- Radial gradient (center-out color blend)

These simulate the visual weight and variety of real product photos without requiring actual images.

### 3.10 Slight Blur

15% of generated images receive a subtle Gaussian blur (radius=0.8) to simulate the blurriness of real device streams captured at lower-than-native resolution via WebRTC.

### 3.11 Y-Distribution Metrics

The generator computes and saves Y-distribution metrics at two levels:

**Per-pair** (in each annotation entry):
- Source/target action Y positions (normalized 0-1)
- Source element Y positions
- Quartile percentages (Q1: 0-25%, Q2: 25-50%, Q3: 50-75%, Q4: 75-100%)
- Min/max Y

**Aggregate** (in `dataset_stats`):
- Overall quartile distribution across all action coordinates
- Per-archetype quartile breakdown
- Min/max/mean Y

This enables monitoring for distribution bias without external analysis.

## 4. File Structure

```
cross_match/
  synthetic_v5.py          <-- Generator script

data/
  icons/
    phosphor/              <-- Phosphor icon pack (SVGs)
      core-main/assets/fill/       (~1,512 filled icons)
      core-main/assets/regular/    (~1,512 line icons)
    simple-icons/          <-- Simple Icons pack (SVGs)
      simple-icons-develop/icons/  (~3,426 brand icons)
  cross_match_v5/          <-- Generated dataset (output)
    source/                <-- Source (Android) screen images
    target/                <-- Target (iOS) screen images
    annotations.json       <-- Pair annotations + Y-distribution + stats
```

## 5. Usage

```bash
# Generate 10,000 pairs (default)
python -m cross_match.synthetic_v5 --output-dir data/cross_match_v5 --num-pairs 10000

# Preview mode (20 pairs, verbose output)
python -m cross_match.synthetic_v5 --output-dir data/cross_match_v5_preview --num-pairs 20 --preview

# Custom archetype weights
python -m cross_match.synthetic_v5 --num-pairs 5000 \
  --archetype-weights 'home_screen:40,product_list:20,settings:20,form:10,chat:10'

# Different seed for variety
python -m cross_match.synthetic_v5 --num-pairs 10000 --seed 123
```

### Prerequisites

- Python 3.9+
- Pillow (`pip install Pillow`)
- SVG rendering: either `librsvg` (`brew install librsvg` for `rsvg-convert` CLI) or `cairosvg` (`pip install cairosvg` + `brew install cairo`)
- Icon packs downloaded to `data/icons/` (see file structure above)

## 6. Sample Output (20 pairs, seed=789)

```
Pairs: 20  Actions: 54  Identity: 2
Archetypes: {'product_list': 4, 'form': 5, 'chat': 3, 'settings': 5, 'home_screen': 3}
Screen sizes used: 14

Y-Distribution (action coords):
  Overall:      Q1(0-25%)=14.3%  Q2(25-50%)=42.9%  Q3(50-75%)=17.5%  Q4(75-100%)=25.4%
  home_screen:  Q1=14.3%  Q2=42.9%  Q3=14.3%  Q4=28.6%
  product_list: Q1=10.0%  Q2=30.0%  Q3=40.0%  Q4=20.0%
  settings:     Q1=18.2%  Q2=63.6%  Q3=9.1%   Q4=9.1%
  form:         Q1=0.0%   Q2=50.0%  Q3=0.0%   Q4=50.0%
  chat:         Q1=40.0%  Q2=40.0%  Q3=0.0%   Q4=20.0%
```

Mean Y = 0.5 (centered), min = 0.113, max = 0.963.

## 7. Expected Impact on Model Accuracy

| Issue | V2 Behavior | V5 Fix | Expected Improvement |
|---|---|---|---|
| Bottom-screen clicks | 0% training data below midpoint | Uniform Y-distribution across 6-94% | Significant accuracy gain for bottom elements |
| Icon matching | Geometric shapes only | 6,450 real SVG icons on colored backgrounds | Model learns icon matching by shape/position |
| Identity mapping | Not trained | 12% identity pairs | Exact coord reproduction for same-image pairs |
| Grid layouts | Never seen | Home screen icon grids, product card layouts | Model learns grid-based spatial reasoning |
| Resolution variety | Fixed 2 sizes | 16 real device resolutions | Better generalization across devices |
| Screen type diversity | 1 generic layout | 5 distinct archetypes | Model learns screen-type-aware matching |

## 8. Next Steps

1. **Generate large dataset**: 15,000-25,000 pairs for full training run
2. **Train on GPU**: Resume from current checkpoint or train fresh on v5 data
3. **Benchmark**: Compare v5-trained model accuracy against v2-trained on the adversarial test images
4. **Iterate**: Analyze remaining failure modes and feed back into v6 if needed
5. **Real data augmentation**: Once synthetic accuracy is acceptable, fine-tune with a small set of real device screenshot pairs
