# CrossMatch Browser Plugin — Implementation Notes

In-browser inference service for cross-platform UI action matching, powered by ONNX Runtime Web with WebGPU acceleration.

## Architecture

```
Chrome Extension / Web Page
├── cross_match_service.js    ← inference module (ES module)
│   ├── ONNX Runtime Web      ← WebGPU/WASM runtime
│   ├── cross_match.onnx      ← 128MB model (loaded from CDN, cached in IndexedDB)
│   └── Image preprocessing   ← resize 518x518, ImageNet normalize, HWC→CHW
└── demo.html                 ← interactive demo page
```

### Inference Flow

```
User clicks on source image
  → Source indicator drawn immediately (flush paint via rAF + setTimeout)
  → Cached source tensor reused (or preprocessed if first click)
  → Cached target tensor reused (or preprocessed if first upload)
  → session.run() on WebGPU (single forward pass through full model)
  → Decode output: normalized coords → pixel coords on target
  → Draw predicted point on target image
  → Log metrics (latency, coordinates)
```

## ONNX Model Export

The CrossMatch PyTorch model (DINOv2-small encoder + cross-attention head) is exported as a single ONNX graph.

**Export command:**
```python
torch.onnx.export(
    model,
    (source_image, source_coords, source_action, target_image),
    "cross_match.onnx",
    input_names=["source_image", "source_coords", "source_action", "target_image"],
    output_names=["target_coords", "action_logits"],
    opset_version=14,
)
```

**Inputs:**

| Name | Type | Shape | Description |
|---|---|---|---|
| source_image | float32 | [1, 3, 518, 518] | Source screenshot (ImageNet normalized) |
| source_coords | float32 | [1, 4] | Normalized action coords [0,1] |
| source_action | **int32** | [1] | Action type: 0=click, 1=scroll |
| target_image | float32 | [1, 3, 518, 518] | Target screenshot (ImageNet normalized) |

**Outputs:**

| Name | Type | Shape | Description |
|---|---|---|---|
| target_coords | float32 | [1, 4] | Predicted normalized coords [0,1] |
| action_logits | float32 | [1, 2] | Action type logits [click_score, scroll_score] |

**Critical: `source_action` must be int32, not int64.** WebGPU and WebGL backends in ONNX Runtime Web do not support int64 tensors. The model must be exported with `torch.tensor([0], dtype=torch.int32)` as the action input.

## Execution Backends

| Backend | Latency (per inference) | Notes |
|---|---|---|
| **WebGPU** | **~500-800ms** | Best browser performance. Uses GPU compute shaders via Chrome's Dawn backend. Requires Chrome 113+. |
| WebGL | Not working | Fails on int64 even with int32 export (ORT bug). Falls back to WASM. |
| WASM | ~5-10s | Single-threaded CPU. Functional but too slow for interactive use. |
| WASM (multi-thread) | ~2-3s | Requires `SharedArrayBuffer` which needs COOP/COEP headers. |

**Recommendation: Always use WebGPU with WASM fallback.**

## Performance Optimizations

### 1. GPU Pipeline Warmup (saves ~1.5s on first click)

WebGPU compiles GPU shaders on first inference. We run a dummy inference during `init()` so this cost is paid at load time, not when the user first clicks.

```
Without warmup: init 0ms → first click ~2.5s (1.5s compile + 1s inference)
With warmup:    init ~1.8s (includes compile) → first click ~0.8s
```

### 2. Image Tensor Caching (saves ~50-100ms per click)

Both source and target images are preprocessed (resize to 518x518, normalize, convert to CHW float32) once on upload. The resulting tensors are cached and reused for all subsequent clicks on the same images.

```
Without caching: each click preprocesses 2 images × ~50ms = ~100ms
With caching:    first click 100ms, subsequent clicks 0ms preprocessing
```

### 3. Target Preloading

`service.preloadTarget(image)` can be called when the target image is uploaded, before any clicks happen. This ensures the target tensor is ready in GPU memory when the first prediction is requested.

### 4. Immediate Visual Feedback

Source action indicators (crosshairs, scroll arrows) are drawn to canvas before inference starts. A `requestAnimationFrame` + `setTimeout` flush ensures the browser paints the indicators before the GPU-blocking inference call.

## Model Delivery

The 128MB ONNX model is too large to bundle in a Chrome extension directly (~10MB soft limit). Strategy:

1. **First load:** Download from CDN, show progress indicator
2. **Cache in IndexedDB:** Model bytes stored in browser's IndexedDB
3. **Subsequent loads:** Read from IndexedDB (instant, no network)
4. **Version management:** Cache key includes model URL — new URL = new download

```js
const service = new CrossMatchService();
await service.init("https://cdn.example.com/models/cross_match_v1.onnx");
// First load: downloads 128MB, caches in IndexedDB
// Subsequent loads: reads from IndexedDB (~instant)
```

## API Reference

```js
import { CrossMatchService } from "./cross_match_service.js";

const service = new CrossMatchService();

// Initialize (downloads model on first use, cached after)
await service.init(modelUrl, {
  executionProvider: "webgpu",  // "webgpu", "wasm", or omit for auto
  useCache: true,               // cache in IndexedDB (default: true)
});

// Pre-cache target image tensor (optional, saves ~50ms on first predict)
service.preloadTarget(targetImageElement);

// Predict
const result = await service.predict({
  sourceScreenshot: sourceImageElement,   // HTMLImageElement, HTMLCanvasElement, or ImageData
  targetScreenshot: targetImageElement,
  action: {
    type: "click",                        // "click" or "scroll"
    sourceCoords: { at: [540, 960] },     // click: {at: [x,y]}
    // sourceCoords: { from_arg: [540, 1200], to_arg: [540, 600] },  // scroll
  },
  sourceSize: [1080, 1920],               // source screen [width, height]
  targetSize: [1170, 2532],               // target screen [width, height]
});

// result:
// {
//   type: "click",
//   targetCoords: { at: [585, 1266] },
//   latencyMs: 650,
// }

// Cleanup
await service.dispose();
```

## Browser Compatibility

| Feature | Minimum Version |
|---|---|
| WebGPU | Chrome 113+, Edge 113+ |
| WASM fallback | All modern browsers |
| IndexedDB caching | All modern browsers |
| ES Modules | Chrome 61+ |

Safari WebGPU support is available from Safari 18+ (macOS Sequoia / iOS 18).

## Known Limitations

1. **Model size (128MB):** Large initial download. Mitigated by IndexedDB caching. Could be reduced to ~64MB with fp16 ONNX export (not yet implemented).
2. **WebGPU shader compilation (~1.8s):** One-time cost on init. Mitigated by warmup inference.
3. **Single ONNX graph:** Both DINOv2 encoder passes (source + target) run inside one `session.run()`. Splitting into encoder + head would allow caching encoder outputs across clicks, potentially halving inference time.
4. **No Web Worker:** Inference runs on main thread. Could be moved to a Web Worker for non-blocking UI, but would require transferring image data.

## Files

| File | Purpose |
|---|---|
| `src/cross_match_service.js` | Inference service module (ES export) |
| `src/test_service.mjs` | Node.js test script (onnxruntime-node) |
| `demo.html` | Interactive demo with click/scroll visualization |
| `models/cross_match.onnx` | ONNX model (128MB, not committed — symlink to export) |
| `package.json` | Dependencies (onnxruntime-web) |
