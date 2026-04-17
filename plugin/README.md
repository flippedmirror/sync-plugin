# CrossMatch Plugin

In-browser inference service that translates user actions (clicks, scrolls) between device screenshots across platforms. Powered by a custom 33M-parameter model running on WebGPU via ONNX Runtime Web.

**Single forward pass. No server. ~500-700ms per action in Chrome.**

## Quick Start

```js
import { CrossMatchService } from "./src/cross_match_service.js";

const service = new CrossMatchService();
await service.init("https://cdn.example.com/models/cross_match.onnx");

const result = await service.predict({
  sourceScreenshot: sourceImg,
  targetScreenshot: targetImg,
  action: { type: "click", sourceCoords: { at: [540, 960] } },
  sourceSize: [1080, 1920],
  targetSize: [1170, 2532],
});

console.log(result.targetCoords.at); // [585, 1266]
console.log(result.latencyMs);       // ~650
```

## Installation

```bash
cd plugin
npm install
```

The ONNX model (`cross_match.onnx`, 128MB) is not bundled — it's loaded from a URL at runtime and cached in IndexedDB.

To generate the model from a trained checkpoint:

```bash
# From project root
python3 -c "
import torch
from cross_match.model import CrossMatchModel
from cross_match.config import ModelConfig

model = CrossMatchModel(ModelConfig())
# ... load checkpoint weights ...
model.eval()

torch.onnx.export(
    model,
    (torch.randn(1,3,518,518), torch.tensor([[.5,.5,0,0]]), torch.tensor([0], dtype=torch.int32), torch.randn(1,3,518,518)),
    'plugin/models/cross_match.onnx',
    input_names=['source_image','source_coords','source_action','target_image'],
    output_names=['target_coords','action_logits'],
    opset_version=14,
)
"
```

## API

### `new CrossMatchService()`

Creates a new service instance.

### `service.init(modelUrl, options?)`

Loads the ONNX model and initializes the inference session.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `modelUrl` | `string` | required | URL to the ONNX model file |
| `options.executionProvider` | `string` | auto | `"webgpu"`, `"wasm"`, or omit for auto-detection |
| `options.useCache` | `boolean` | `true` | Cache model in IndexedDB after first download |

```js
await service.init("https://cdn.example.com/cross_match.onnx", {
  executionProvider: "webgpu",
});
```

On first call, downloads the model (~128MB) and caches it. Subsequent page loads read from IndexedDB (instant). A warmup inference runs automatically to pre-compile GPU shaders.

### `service.predict(input)`

Translates an action from the source device to the target device.

**Input:**

```js
{
  sourceScreenshot: HTMLImageElement | HTMLCanvasElement | ImageData,
  targetScreenshot: HTMLImageElement | HTMLCanvasElement | ImageData,
  action: {
    type: "click" | "scroll",
    sourceCoords: object,
  },
  sourceSize: [number, number],  // source screen [width, height] in pixels
  targetSize: [number, number],  // target screen [width, height] in pixels
}
```

**Action coordinate formats:**

```js
// Click — single tap point
{ type: "click", sourceCoords: { at: [x, y] } }

// Scroll — swipe from start to end
{ type: "scroll", sourceCoords: { from_arg: [x1, y1], to_arg: [x2, y2] } }
```

All coordinates are in **pixel space** relative to the device's actual screen resolution.

**Returns:**

```js
{
  type: "click" | "scroll",
  targetCoords: {
    at: [x, y],                   // click
    // or
    from_arg: [x1, y1],           // scroll start
    to_arg: [x2, y2],             // scroll end
  },
  latencyMs: number,              // inference time in milliseconds
}
```

### `service.preloadTarget(image)`

Pre-caches the target image tensor. Call when the target device's screen changes (navigation, new screen state). Saves ~50ms on the next `predict()` call.

```js
service.preloadTarget(targetImageElement);
```

### `service.dispose()`

Releases the ONNX session and frees GPU resources.

```js
await service.dispose();
```

### Properties

| Property | Type | Description |
|---|---|---|
| `service.ready` | `boolean` | `true` after `init()` completes |
| `service.backend` | `string` | Active backend: `"webgpu"`, `"wasm"`, etc. |

## Integration Example

```js
// 1. Initialize on app startup
const service = new CrossMatchService();
await service.init(MODEL_URL);

// 2. When target device session starts
const targetImg = await captureTargetScreenshot();
service.preloadTarget(targetImg);

// 3. On each user action on source device
function onSourceAction(actionType, coords) {
  const sourceImg = captureSourceScreenshot();

  service.predict({
    sourceScreenshot: sourceImg,
    targetScreenshot: targetImg,
    action: { type: actionType, sourceCoords: coords },
    sourceSize: [1080, 1920],
    targetSize: [1170, 2532],
  }).then(result => {
    // Execute on target device
    if (result.type === "click") {
      targetDevice.tap(result.targetCoords.at[0], result.targetCoords.at[1]);
    } else {
      targetDevice.swipe(
        result.targetCoords.from_arg[0], result.targetCoords.from_arg[1],
        result.targetCoords.to_arg[0], result.targetCoords.to_arg[1],
      );
    }
  });
}

// 4. When target screen changes
function onTargetScreenChange() {
  targetImg = captureTargetScreenshot();
  service.preloadTarget(targetImg);
}
```

## Demo

Run the interactive demo locally:

```bash
cd plugin
python3 -m http.server 8080
# Open http://localhost:8080/demo.html
```

Upload source and target screenshots, then click on the source image to see predicted target points. The demo shows action indicators (green = source, red = predicted target) and logs latency metrics.

## Performance

| Metric | Value |
|---|---|
| Model size | 128MB (fp32 ONNX) |
| WebGPU inference | ~500-700ms |
| WASM inference | ~5-10s |
| Init time (first load) | ~5s (download + warmup) |
| Init time (cached) | ~3s (IndexedDB read + warmup) |
| Image preprocessing | ~50ms (cached after first call) |

### Optimization Tips

- **Always use WebGPU** — 10x faster than WASM.
- **Call `preloadTarget()`** when the target screen changes, not during action handling.
- **Reuse the service instance** — don't create a new one per action. The session holds compiled GPU pipelines.
- **Image tensors are cached** — same source/target images across multiple clicks don't re-preprocess.

## Browser Compatibility

| Browser | WebGPU | WASM Fallback |
|---|---|---|
| Chrome 113+ | Yes | Yes |
| Edge 113+ | Yes | Yes |
| Firefox | Nightly only | Yes |
| Safari 18+ | Yes | Yes |

## Model Details

The ONNX model is exported from **CrossMatch** — a custom architecture:

- **Encoder:** DINOv2-small (22M params, frozen) — shared between source and target images
- **Head:** Cross-attention transformer (11M params, trained) — attends from source context to target features
- **Output:** 4 coordinate values (normalized [0,1]) + 2 action logits
- **Training:** 5K synthetic pairs, 10 epochs on NVIDIA T4
- **Accuracy:** 20.7px mean distance, 88% within 50px, 100% within 100px (on synthetic test set)

See `cross_match/ARCHITECTURE.md` for full architecture details and `cross_match/TEST_REPORT.md` for benchmark results.
