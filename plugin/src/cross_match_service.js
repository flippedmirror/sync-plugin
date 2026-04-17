/**
 * CrossMatch Inference Service
 *
 * Runs the CrossMatch ONNX model in-browser via onnxruntime-web.
 * Single forward pass: 2 screenshots + action → translated action.
 *
 * Usage:
 *   const service = new CrossMatchService();
 *   await service.init("https://cdn.example.com/models/cross_match.onnx");
 *
 *   const result = await service.predict({
 *     sourceScreenshot: imageDataOrCanvas,  // source device screenshot
 *     targetScreenshot: imageDataOrCanvas,  // target device screenshot
 *     action: {
 *       type: "click",                      // "click" or "scroll"
 *       sourceCoords: { at: [540, 960] },   // click: {at: [x,y]}, scroll: {from_arg: [x,y], to_arg: [x,y]}
 *     },
 *     sourceSize: [1080, 1920],             // source screen dimensions [w, h]
 *     targetSize: [1170, 2532],             // target screen dimensions [w, h]
 *   });
 *
 *   // result:
 *   // {
 *   //   type: "click",
 *   //   targetCoords: { at: [585, 1266] },
 *   //   latencyMs: 150,
 *   // }
 */

const IMAGE_SIZE = 518;
const IMAGE_MEAN = [0.485, 0.456, 0.406];
const IMAGE_STD = [0.229, 0.224, 0.225];
const ACTION_TYPES = ["click", "scroll"];
const DB_NAME = "cross_match_cache";
const DB_STORE = "models";
const DB_VERSION = 1;

export class CrossMatchService {
  constructor() {
    this.session = null;
    this.ready = false;
  }

  /**
   * Initialize the service by loading the ONNX model.
   * @param {string} modelUrl - URL to the ONNX model file
   * @param {object} [options] - Options
   * @param {string} [options.executionProvider] - "webgl", "wasm", or "webgpu" (default: auto)
   * @param {boolean} [options.useCache] - Cache model in IndexedDB (default: true)
   */
  async init(modelUrl, options = {}) {
    const { executionProvider, useCache = true } = options;
    const ort = globalThis.ort || await import("onnxruntime-web");

    let modelBuffer;

    // Try loading from IndexedDB cache
    if (useCache) {
      modelBuffer = await this._loadFromCache(modelUrl);
    }

    // Fetch from URL if not cached
    if (!modelBuffer) {
      console.log(`[CrossMatch] Downloading model from ${modelUrl}...`);
      const response = await fetch(modelUrl);
      if (!response.ok) throw new Error(`Failed to fetch model: ${response.status}`);
      modelBuffer = await response.arrayBuffer();

      // Cache for next time
      if (useCache) {
        await this._saveToCache(modelUrl, modelBuffer);
      }
      console.log(`[CrossMatch] Model downloaded (${(modelBuffer.byteLength / 1024 / 1024).toFixed(1)} MB)`);
    } else {
      console.log(`[CrossMatch] Model loaded from cache (${(modelBuffer.byteLength / 1024 / 1024).toFixed(1)} MB)`);
    }

    // Create inference session with best available backend
    const sessionOptions = {};
    if (executionProvider) {
      sessionOptions.executionProviders = [executionProvider, "wasm"];
    } else {
      // Auto: prefer WebGPU > WebGL > WASM
      const providers = ["wasm"];
      if (typeof navigator !== "undefined" && navigator.gpu) providers.unshift("webgpu");
      sessionOptions.executionProviders = providers;
    }

    console.log("[CrossMatch] Creating session with providers:", sessionOptions.executionProviders);
    this.session = await ort.InferenceSession.create(modelBuffer, sessionOptions);
    this.ort = ort;
    this.backend = sessionOptions.executionProviders[0];

    // Warmup: run a dummy inference to compile GPU shaders/pipelines
    console.log("[CrossMatch] Warming up GPU pipelines...");
    const warmupT0 = performance.now();
    const n = IMAGE_SIZE * IMAGE_SIZE;
    const dummyImg = new ort.Tensor("float32", new Float32Array(3 * n), [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
    const dummyCoords = new ort.Tensor("float32", new Float32Array([0.5, 0.5, 0, 0]), [1, 4]);
    const dummyAction = new ort.Tensor("int32", Int32Array.from([0]), [1]);
    await this.session.run({ source_image: dummyImg, source_coords: dummyCoords, source_action: dummyAction, target_image: dummyImg });
    console.log(`[CrossMatch] Warmup done in ${Math.round(performance.now() - warmupT0)}ms`);

    this._cachedTargetTensor = null;
    this._cachedTargetKey = null;
    this.ready = true;
    console.log("[CrossMatch] Model loaded and ready. Backend:", this.backend);
  }

  /**
   * Predict translated action on target device.
   * @param {object} input
   * @param {ImageData|HTMLCanvasElement|HTMLImageElement} input.sourceScreenshot
   * @param {ImageData|HTMLCanvasElement|HTMLImageElement} input.targetScreenshot
   * @param {object} input.action - {type: "click"|"scroll", sourceCoords: {at:[x,y]} or {from_arg:[x,y], to_arg:[x,y]}}
   * @param {number[]} input.sourceSize - [width, height] of source screen
   * @param {number[]} input.targetSize - [width, height] of target screen
   * @returns {Promise<{type: string, targetCoords: object, latencyMs: number}>}
   */
  async predict(input) {
    if (!this.ready) throw new Error("Service not initialized. Call init() first.");

    const { sourceScreenshot, targetScreenshot, action, sourceSize, targetSize } = input;
    const ort = this.ort;

    // Cache image tensors — both images stay the same across clicks, only coords change
    const srcKey = sourceScreenshot.src || "source";
    let srcTensor;
    if (this._cachedSourceKey === srcKey && this._cachedSourceTensor) {
      srcTensor = this._cachedSourceTensor;
    } else {
      srcTensor = this._preprocessImage(sourceScreenshot, ort);
      this._cachedSourceTensor = srcTensor;
      this._cachedSourceKey = srcKey;
    }

    const targetKey = targetScreenshot.src || "target";
    let tgtTensor;
    if (this._cachedTargetKey === targetKey && this._cachedTargetTensor) {
      tgtTensor = this._cachedTargetTensor;
    } else {
      tgtTensor = this._preprocessImage(targetScreenshot, ort);
      this._cachedTargetTensor = tgtTensor;
      this._cachedTargetKey = targetKey;
    }

    // Encode action coordinates (normalized to [0, 1])
    const coordsNorm = this._normalizeCoords(action, sourceSize);
    const coordsTensor = new ort.Tensor("float32", new Float32Array(coordsNorm), [1, 4]);

    // Action type index
    const actionIdx = ACTION_TYPES.indexOf(action.type);
    if (actionIdx === -1) throw new Error(`Unknown action type: ${action.type}`);
    const actionTensor = new ort.Tensor("int32", Int32Array.from([actionIdx]), [1]);

    // Run inference
    const t0 = performance.now();
    const results = await this.session.run({
      source_image: srcTensor,
      source_coords: coordsTensor,
      source_action: actionTensor,
      target_image: tgtTensor,
    });
    const latencyMs = performance.now() - t0;

    // Decode output
    const predCoords = results.target_coords.data; // Float32Array [4]
    const actionLogits = results.action_logits.data; // Float32Array [2]
    const predActionIdx = actionLogits[0] > actionLogits[1] ? 0 : 1;
    const predType = ACTION_TYPES[predActionIdx];

    // Denormalize coords to target pixel space
    const [tw, th] = targetSize;
    const targetCoords = this._denormalizeCoords(predCoords, predType, tw, th);

    return {
      type: predType,
      targetCoords,
      latencyMs: Math.round(latencyMs),
    };
  }

  /**
   * Pre-cache the target image tensor. Call when target image is loaded/changed.
   * Saves ~40-60ms per predict() by skipping target preprocessing.
   */
  preloadTarget(targetImage) {
    if (!this.ready) return;
    const tgtTensor = this._preprocessImage(targetImage, this.ort);
    this._cachedTargetTensor = tgtTensor;
    this._cachedTargetKey = targetImage.src || "preloaded";
    console.log("[CrossMatch] Target image pre-cached.");
  }

  /**
   * Preprocess an image to a normalized tensor [1, 3, 518, 518].
   */
  _preprocessImage(imageSource, ort) {
    // Get pixel data via canvas
    const canvas = document.createElement("canvas");
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    const ctx = canvas.getContext("2d");

    if (imageSource instanceof ImageData) {
      // Create temp canvas for resize
      const tmp = document.createElement("canvas");
      tmp.width = imageSource.width;
      tmp.height = imageSource.height;
      tmp.getContext("2d").putImageData(imageSource, 0, 0);
      ctx.drawImage(tmp, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    } else {
      // HTMLCanvasElement or HTMLImageElement
      ctx.drawImage(imageSource, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    }

    const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const { data } = imageData; // RGBA Uint8ClampedArray

    // Convert to CHW float32 with ImageNet normalization
    const n = IMAGE_SIZE * IMAGE_SIZE;
    const floats = new Float32Array(3 * n);

    for (let i = 0; i < n; i++) {
      const r = data[i * 4] / 255.0;
      const g = data[i * 4 + 1] / 255.0;
      const b = data[i * 4 + 2] / 255.0;

      floats[i] = (r - IMAGE_MEAN[0]) / IMAGE_STD[0];           // R channel
      floats[n + i] = (g - IMAGE_MEAN[1]) / IMAGE_STD[1];       // G channel
      floats[2 * n + i] = (b - IMAGE_MEAN[2]) / IMAGE_STD[2];   // B channel
    }

    return new ort.Tensor("float32", floats, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  }

  /**
   * Normalize action coordinates to [0, 1].
   */
  _normalizeCoords(action, sourceSize) {
    const [sw, sh] = sourceSize;
    if (action.type === "click") {
      const [x, y] = action.sourceCoords.at;
      return [x / sw, y / sh, 0, 0];
    } else {
      const [fx, fy] = action.sourceCoords.from_arg;
      const [tx, ty] = action.sourceCoords.to_arg;
      return [fx / sw, fy / sh, tx / sw, ty / sh];
    }
  }

  /**
   * Denormalize predicted coords to target pixel space.
   */
  _denormalizeCoords(predCoords, actionType, tw, th) {
    if (actionType === "click") {
      return {
        at: [Math.round(predCoords[0] * tw), Math.round(predCoords[1] * th)],
      };
    } else {
      return {
        from_arg: [Math.round(predCoords[0] * tw), Math.round(predCoords[1] * th)],
        to_arg: [Math.round(predCoords[2] * tw), Math.round(predCoords[3] * th)],
      };
    }
  }

  // IndexedDB caching

  async _openDB() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => req.result.createObjectStore(DB_STORE);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async _loadFromCache(key) {
    try {
      const db = await this._openDB();
      return new Promise((resolve) => {
        const tx = db.transaction(DB_STORE, "readonly");
        const req = tx.objectStore(DB_STORE).get(key);
        req.onsuccess = () => resolve(req.result || null);
        req.onerror = () => resolve(null);
      });
    } catch {
      return null;
    }
  }

  async _saveToCache(key, data) {
    try {
      const db = await this._openDB();
      const tx = db.transaction(DB_STORE, "readwrite");
      tx.objectStore(DB_STORE).put(data, key);
    } catch {
      // Caching is best-effort
    }
  }

  /**
   * Clean up resources.
   */
  async dispose() {
    if (this.session) {
      await this.session.release();
      this.session = null;
      this.ready = false;
    }
  }
}

export default CrossMatchService;
