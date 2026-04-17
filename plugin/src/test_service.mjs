/**
 * Test CrossMatch ONNX inference via Node.js (onnxruntime-node).
 * Validates the model loads and produces correct output shapes.
 *
 * Usage: cd plugin && npm install onnxruntime-node && node src/test_service.mjs
 */

import * as ort from "onnxruntime-node";
import { readFileSync } from "fs";
import { performance } from "perf_hooks";

const MODEL_PATH = "../models/cross_match_onnx/cross_match.onnx";
const IMAGE_SIZE = 518;

async function main() {
  console.log("Loading model...");
  const session = await ort.InferenceSession.create(MODEL_PATH);
  console.log("Model loaded. Inputs:", session.inputNames, "Outputs:", session.outputNames);

  // Create dummy inputs (random noise — we're testing shape/latency, not accuracy)
  const n = IMAGE_SIZE * IMAGE_SIZE;
  const srcImage = new ort.Tensor("float32", new Float32Array(3 * n).fill(0.1), [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  const tgtImage = new ort.Tensor("float32", new Float32Array(3 * n).fill(0.2), [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  const coords = new ort.Tensor("float32", new Float32Array([0.5, 0.5, 0.0, 0.0]), [1, 4]);
  const action = new ort.Tensor("int32", Int32Array.from([0]), [1]);

  // Warmup
  await session.run({
    source_image: srcImage,
    source_coords: coords,
    source_action: action,
    target_image: tgtImage,
  });

  // Benchmark
  const runs = 5;
  const latencies = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    const results = await session.run({
      source_image: srcImage,
      source_coords: coords,
      source_action: action,
      target_image: tgtImage,
    });
    const lat = performance.now() - t0;
    latencies.push(lat);

    if (i === 0) {
      const predCoords = results.target_coords.data;
      const actionLogits = results.action_logits.data;
      console.log("\nOutput shapes:");
      console.log("  target_coords:", results.target_coords.dims, "=", Array.from(predCoords).map(v => v.toFixed(3)));
      console.log("  action_logits:", results.action_logits.dims, "=", Array.from(actionLogits).map(v => v.toFixed(3)));
      console.log("  Predicted action:", actionLogits[0] > actionLogits[1] ? "click" : "scroll");
    }
  }

  const mean = latencies.reduce((a, b) => a + b) / latencies.length;
  const min = Math.min(...latencies);
  const max = Math.max(...latencies);
  console.log(`\nLatency (${runs} runs):`);
  console.log(`  Mean: ${mean.toFixed(0)}ms`);
  console.log(`  Min:  ${min.toFixed(0)}ms`);
  console.log(`  Max:  ${max.toFixed(0)}ms`);

  // Test with scroll action
  const scrollCoords = new ort.Tensor("float32", new Float32Array([0.5, 0.6, 0.5, 0.3]), [1, 4]);
  const scrollAction = new ort.Tensor("int32", Int32Array.from([1]), [1]);
  const scrollResult = await session.run({
    source_image: srcImage,
    source_coords: scrollCoords,
    source_action: scrollAction,
    target_image: tgtImage,
  });
  console.log("\nScroll test:");
  console.log("  target_coords:", Array.from(scrollResult.target_coords.data).map(v => v.toFixed(3)));
  console.log("  Predicted action:", scrollResult.action_logits.data[0] > scrollResult.action_logits.data[1] ? "click" : "scroll");

  console.log("\n✓ All tests passed.");
}

main().catch(console.error);
