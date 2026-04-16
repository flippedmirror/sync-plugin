# Florence-2 Cross-Platform UI Grounding Benchmark

Benchmarking Florence-2-base-ft (232M params) for cross-platform action sync in BrowserStack App Live. The task: given a user click on device A (Android), find the corresponding click point on device B (iOS).

## Task Definition

**Input:** Source screenshot + click (x, y) coordinate + target screenshot
**Output:** Predicted click (x, y) on target screenshot
**Pipeline:** Crop around click on source -> caption/OCR the crop (call 1) -> locate element on target via grounding (call 2)

This requires **2 sequential inference calls** per action — the fundamental architectural constraint.

## Model

- **Model:** `microsoft/Florence-2-base-ft` (232M parameters, fine-tuned)
- **Architecture:** DaViT vision encoder + BART-like text encoder-decoder
- **Internal resolution:** All inputs resized to 768x768
- **Task tokens used:** `<CAPTION>`, `<OCR>`, `<OPEN_VOCABULARY_DETECTION>`, `<CAPTION_TO_PHRASE_GROUNDING>`

## Benchmark Dataset

Synthetic cross-platform pairs: Android (1080x1920) and iOS (1170x2532) screenshots with colored UI elements (buttons, inputs, labels) at proportionally shifted positions with jitter. 15 click-coordinate test pairs covering buttons, inputs, labels, and toggles.

Ground truth: manually defined corresponding click coordinates on both platforms.

## Grounding Strategy Results (bbox-to-bbox benchmark)

First benchmark tested 3 strategies using annotated bounding boxes (not realistic for production, but validates model capability):

| Strategy | Mean IoU | Acc@0.5 | Acc@0.7 | Mean Latency (CPU) |
|---|---|---|---|---|
| desc_only (REGION_TO_DESCRIPTION -> OVD) | 0.000 | 0.0% | 0.0% | 5.97s |
| **desc_ocr** (REGION_TO_DESCRIPTION + OCR -> grounding) | **0.494** | **60.0%** | **60.0%** | **10.35s** |
| ocr_only (OCR -> OVD) | 0.000 | 0.0% | 0.0% | 7.42s |

**Finding:** Combined description + OCR is the only viable strategy. Pure description is too generic, pure OCR text can't be matched through OVD.

## Click-to-Click Benchmark (production-realistic)

Realistic pipeline: only (x, y) click coordinates available, no bounding boxes. Pipeline crops around click point, captions/OCRs the crop, then locates on target.

### Accuracy Metrics

Measured across all runtime configurations (accuracy is consistent where noted):

| Metric | Value |
|---|---|
| Success rate | 100% (element found on target) |
| Hit rate @20px | 46.7% - 53.3% |
| Hit rate @50px | 46.7% - 53.3% |
| Median pixel distance (hits) | 1 - 4px |
| Mean pixel distance (all) | 225 - 303px |

**Interpretation:** When the model correctly identifies the element (~50%), it's extremely precise (1-4px). Failures are catastrophic misses (>200px), not near-misses. Failure modes:
- Plain text labels with no visual container (e.g., "Notifications" as bare text)
- Elements with OCR misreads (e.g., "Messages" -> "Mossaas")
- `unanswerable` captions for ambiguous crops (input fields, small toggles)
- Duplicate similar elements where spatial prior picks wrong one

### Latency Comparison (all configurations tested)

| # | Configuration | Mean Latency | Per-call | Hit @20px | Notes |
|---|---|---|---|---|---|
| 1 | PyTorch CPU fp32, beam search (num_beams=3) | ~10s | ~5s | 53.3% | Original baseline |
| 2 | PyTorch CPU fp32, greedy (num_beams=1) | 6.33s | ~3.2s | 53.3% | Greedy cuts ~40% |
| 3 | ONNX FP32 (no KV cache) | 17.94s | ~9s | 53.3% | O(n^2) decoding, worse than PT |
| 4 | ONNX INT8 MatMul-only quant (no KV cache) | 8.89s | ~4.5s | 46.7% | INT8 helps but no KV cache kills it |
| 5 | Hybrid ONNX INT8 encoder + PyTorch decoder | 5.53s | ~2.8s | 46.7% | Best CPU config |
| 6 | **PyTorch MPS fp16, greedy** | **2.45s** | **~0.95s** | **46.7%** | Apple Metal GPU |
| 7 | **PyTorch MPS fp16 + torch.compile** | **1.84s** | **~0.87s** | **53.3%** | **Best local result** |
| 8 | CoreML fp16 encoder+decoder (no KV cache) | ~25s | ~12s | — | Garbage output, fp16 drift |
| 9 | CoreML fp16 encoder + PyTorch decoder (hybrid) | 21.1s | ~10s | 0% | Precision mismatch |
| 10 | MPS fp16 + parallel target pre-encoding | ~4.4s | — | 53.3% | MPS can't parallelize (see below) |
| 11 | MPS fp16, image_size=384 | 1.23s | ~0.47s | 0% | Fast but model breaks |
| 12 | MPS fp16, image_size=512 | 1.83s | ~0.75s | 0% | Model outputs "unanswerable" |
| 13 | MPS fp16, image_size=672 | 2.39s | ~1.0s | 0% | OCR works but grounding fails |

### Latency Breakdown (MPS + compile, best config)

| Component | Time |
|---|---|
| First call warmup (shader compilation) | ~2.5s (one-time) |
| Call 1: Caption source crop (300x300 -> 768x768) | ~0.85s |
| Call 2: OVD/grounding on target (1170x2532 -> 768x768) | ~0.90s |
| Post-processing (bbox selection, coordinate mapping) | <1ms |
| **Total per-click (post-warmup)** | **~1.75s** |

## Optimization Attempts — Detailed Findings

### Resolution patching (rows 11-13)

Florence-2's processor resizes all inputs to 768x768. We patched the image processor to use smaller resolutions (384, 512, 672) to reduce the DaViT encoder's patch count and speed up inference.

**Result: Complete failure at all reduced resolutions.**

- **384x384:** 2x faster encoder but model generates degraded captions. Grounding fails because descriptions become too vague or miss key details.
- **512x512:** Model outputs "unanswerable" for nearly all crops. The 768->512 shift breaks the learned token-to-text correspondence entirely.
- **672x672:** OCR starts working (extracts correct text like "Submit", "Continue"), but OVD/grounding can't match bare single-word queries. The caption path also breaks — descriptions are truncated to 1-2 words.

**Why:** Florence-2 was trained exclusively at 768x768. The DaViT encoder's positional embeddings, attention patterns, and the decoder's learned correspondence between visual patch tokens and text tokens are all calibrated for that exact resolution. Even a 12% reduction (672) disrupts the visual-language alignment.

**Conclusion:** Resolution is not a tunable knob for Florence-2. Any input size other than 768x768 produces unreliable output.

### Parallel encoder pre-encoding (row 10)

The target image's DaViT encoding is independent of the grounding text prompt. We pre-encode the target vision features before call 1, then reuse them in call 2 (decoder-only, skipping the encoder).

**Per-step breakdown (post-warmup):**

| Step | Parallel | Baseline |
|---|---|---|
| Target vision encode | 0.16s | (included in call 2) |
| Call 1: caption (full) | 1.0s | 0.85s |
| Call 2: grounding | **0.35s** (decoder only) | 0.90s (full) |
| **Total** | **~1.5s internal** | **~1.75s** |

The grounding call dropped from 0.9s to 0.35s — the pre-encoding works. But wall-clock time was **worse** (4.4s vs 2.45s) because:

1. **MPS single command queue:** Apple's Metal GPU serializes all operations on one queue. The pre-encode step can't overlap with call 1 — it just adds sequential overhead.
2. **Split forward path overhead:** Calling `encode_vision()` separately from `language_model.generate()` introduces extra MPS synchronization points that don't exist in the unified `model.generate()` path.
3. **Caption call regression:** The caption step took ~1.0s instead of ~0.85s due to MPS kernel scheduling overhead from the mixed forward paths.

**This optimization would work on CUDA** where separate CUDA streams allow true parallel kernel execution. On MPS, the synchronization overhead negates the encoder savings.

**Conclusion:** Pre-encoding is architecturally sound but requires multi-stream GPU execution. Not viable on MPS/Apple Silicon.

### ONNX Runtime (rows 3-5)

Exported encoder and decoder to ONNX separately, with INT8 dynamic quantization (MatMul-only, skipping Conv ops for ARM compatibility).

**Key finding: ONNX autoregressive decoding without KV caching is O(n^2) in sequence length.** Each generated token requires a full decoder forward pass over all previous tokens. PyTorch's `generate()` uses KV caching natively, making it O(n). This fundamental difference means ONNX is slower than PyTorch for generation tasks despite faster individual op execution.

The hybrid approach (ONNX encoder + PyTorch decoder) captured the best of both: ONNX-optimized encoder + PyTorch's KV-cached decoder. This was the best CPU config at 5.53s.

**ARM INT8 limitation:** ONNX Runtime on Apple Silicon (ARM) doesn't implement `ConvInteger` ops required for INT8 convolution layers. We worked around this by only quantizing MatMul/Gemm ops, but this limits the INT8 speedup to ~50% size reduction with modest speed gains.

### CoreML (rows 8-9)

Exported encoder and decoder to CoreML (.mlpackage) with fp16 precision and INT8 weight quantization via coremltools.

**Key finding: Florence-2's architecture doesn't convert cleanly to CoreML.**

- **Pure CoreML (row 8):** The DaViT vision encoder has data-dependent assertions and dynamic shapes that get baked as constants during tracing. The autoregressive decoder without KV cache produces garbage after ~10 tokens due to accumulated fp16 precision drift. Mean latency was ~25s with nonsensical output.
- **Hybrid CoreML encoder + PyTorch decoder (row 9):** The CoreML encoder's fp16 output differs numerically from PyTorch's output enough that the PyTorch decoder generates "unanswerable" for every input. The encoder-decoder precision coupling in Florence-2 is too tight for mixed-runtime execution.

**Conclusion:** CoreML is not viable for Florence-2 without significant architecture-specific conversion work (custom op implementations, precision calibration). The DaViT encoder in particular has control flow patterns that don't map to CoreML's static graph representation.

### INT8/quantized inference on MPS

PyTorch's MPS backend does not support INT8 inference. The `bitsandbytes` library (used for INT8/INT4 quantization in PyTorch) is CUDA-only. There is no path to quantized inference on Apple Silicon through PyTorch.

CoreML supports INT8 natively on the ANE, but the CoreML conversion issues (above) block this path for Florence-2.

## Key Technical Findings

### What works
- **MPS (Metal) + fp16 + torch.compile** is the best local inference path on Apple Silicon
- **Greedy decoding (num_beams=1)** has negligible accuracy impact vs beam search but ~40% faster
- **Caption-based grounding** outperforms description-based and OCR-only strategies for the click-to-click task
- **Spatial prior heuristic** (pick candidate closest in normalized position to source) effectively disambiguates multiple detection results
- **Pre-encoding target vision features** saves ~0.55s on the grounding call (confirmed by step-level timing), but only benefits CUDA with multi-stream execution

### What doesn't work
- **ONNX without KV cache** — O(n^2) autoregressive decoding negates per-op speedups
- **CoreML** — precision drift between CoreML and PyTorch breaks encoder-decoder coupling
- **Resolution patching** — model trained at 768x768 produces unreliable output at any other resolution
- **Parallel execution on MPS** — single command queue means no true concurrency
- **Pure OCR pipeline** — OCR text alone can't be matched through OVD; the model needs rich descriptive captions
- **INT8 on Apple Silicon** — no viable path through PyTorch MPS or CoreML for this model

### Architectural constraints
The 2-call pipeline is fundamental to Florence-2's design:
1. Florence-2 accepts **1 image per call** (no multi-image input)
2. It has **fixed task tokens**, not free-form instruction following
3. There is no "match this element" task — must describe then locate
4. Input resolution is locked to 768x768 (trained, non-configurable)

This means the minimum latency is 2x per-call latency, which on MPS is **~1.75s total — the hard floor for Florence-2 on local Apple Silicon hardware**.

## Paths to Sub-Second Total Latency

| Approach | Expected Total | Complexity | Trade-offs |
|---|---|---|---|
| **Server-side GPU (T4/A10)** | **~0.3-0.4s** | **Low** | Adds network round-trip (~50ms); natural fit for BrowserStack since devices are already server-hosted. Parallel encoder + TensorRT feasible. |
| **CrossMatch custom model** (see `cross_match/ARCHITECTURE.md`) | **~70ms CPU, ~10ms GPU** | **High** | Single forward pass, no autoregression. Requires training data + training time. DINOv2-small encoder (22M frozen) + 11M trainable cross-attention head. |
| Lightweight OCR + embedding match + VLM fallback | ~100ms (text) / ~1.75s (non-text) | Medium | Fast for ~60-70% of clicks (text elements); VLM fallback for icons/images. Doesn't solve the non-text case. |
| Instruction-following VLM (Qwen2-VL, InternVL) | ~0.3-0.5s (GPU) | Medium | Single-call matching with multi-image support. Models are 2B+ params — larger but capable. Server-side only. |

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic test data
python scripts/create_click_data.py --output-dir data/click_sample --num-pairs 15

# Run click-to-click benchmark (auto-detects MPS on Apple Silicon)
python -m benchmark.click_benchmark --data-dir data/click_sample --output-dir results/ --fast --visualize

# With torch.compile (best local performance)
python -m benchmark.click_benchmark --data-dir data/click_sample --output-dir results/ --fast --compile --visualize

# Force specific device
python -m benchmark.click_benchmark --data-dir data/click_sample --output-dir results/ --device cpu --fast

# With parallel encoder pre-encoding (only beneficial on CUDA)
python -m benchmark.click_benchmark --data-dir data/click_sample --output-dir results/ --parallel --compile

# With patched image resolution (for testing — accuracy degrades below 768)
python -m benchmark.click_benchmark --data-dir data/click_sample --output-dir results/ --fast --compile --image-size 384

# Export ONNX models
python scripts/export_onnx.py --output-dir models/florence2-onnx

# Run with ONNX hybrid (ONNX encoder + PyTorch decoder)
python -m benchmark.click_benchmark --data-dir data/click_sample --output-dir results/ --onnx-dir models/florence2-onnx

# Export CoreML models (macOS only)
python scripts/export_coreml.py --output-dir models/florence2-coreml
```

## File Structure

```
sync-plugin/
├── benchmark/
│   ├── __init__.py
│   ├── __main__.py              # Original bbox-based benchmark CLI
│   ├── model.py                 # PyTorch Florence-2 wrapper (CPU/MPS/CUDA, +compile, +image_size)
│   ├── model_onnx.py            # Pure ONNX Runtime inference (encoder + decoder, no KV cache)
│   ├── model_hybrid.py          # ONNX INT8 encoder + PyTorch decoder (KV cache)
│   ├── model_coreml.py          # Pure CoreML inference (broken — precision issues)
│   ├── model_coreml_hybrid.py   # CoreML encoder + PyTorch decoder (broken — precision mismatch)
│   ├── model_parallel.py        # Split encoder/decoder with target pre-encoding
│   ├── click_pipeline.py        # Click-to-click pipeline (caption -> grounding)
│   ├── click_benchmark.py       # Click-to-click benchmark runner (all configs)
│   ├── pipeline.py              # Original bbox-based grounding strategies
│   ├── run.py                   # Original bbox benchmark orchestration
│   ├── dataset.py               # Dataset loading/validation
│   ├── metrics.py               # IoU, center distance, accuracy metrics
│   └── visualize.py             # Bbox/click visualization
├── cross_match/                 # Custom single-pass model (see ARCHITECTURE.md)
│   ├── model.py                 # CrossMatch: DINOv2 + cross-attention + coord/action heads
│   ├── dataset.py               # Dataset with click/scroll action annotations
│   ├── train.py                 # Training loop (frozen encoder + fine-tune phases)
│   ├── predict.py               # Inference wrapper
│   ├── synthetic.py             # Synthetic training data generator
│   ├── config.py                # Model and training configs
│   └── ARCHITECTURE.md          # Detailed architecture and training plan
├── scripts/
│   ├── create_synthetic_data.py # Generate bbox-annotated test pairs
│   ├── create_click_data.py     # Generate click-coordinate test pairs
│   ├── export_onnx.py           # Export to ONNX + INT8 quantization
│   ├── export_coreml.py         # Export to CoreML + INT8 quantization
│   └── requantize_onnx.py       # Re-quantize ONNX (skip Conv ops for ARM)
├── models/                      # Exported model artifacts (ONNX, CoreML)
├── data/                        # Test datasets
├── results/                     # Benchmark output (JSON + visualizations)
├── requirements.txt
└── BENCHMARK.md                 # This file
```
