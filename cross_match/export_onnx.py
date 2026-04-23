"""Export CrossMatch model to ONNX format.

Usage:
    python -m cross_match.export_onnx --checkpoint checkpoints/v5_5k_25ep/best_25ep.pt --output plugin/models/cross_match_v5.onnx
    python -m cross_match.export_onnx --checkpoint checkpoints/v5_5k_25ep/best_25ep.pt --output plugin/models/cross_match_v5.onnx --int32-action
"""

from __future__ import annotations

import argparse
import os

import torch

from cross_match.config import ModelConfig
from cross_match.model import CrossMatchModel


def export_onnx(checkpoint_path, output_path, device="cpu", int32_action=True):
    print("Loading checkpoint:", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", ModelConfig())

    model = CrossMatchModel(model_config).to(device)

    # Load state_dict with key compatibility handling
    saved_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    compatible = {}
    for k, v in saved_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            compatible[k] = v
    model.load_state_dict(compatible, strict=False)
    model.eval()

    loaded = len(compatible)
    total = len(model_state)
    print("  Loaded {}/{} keys ({} skipped)".format(loaded, total, total - loaded))

    # Dummy inputs
    img_size = model_config.image_size
    source_image = torch.randn(1, 3, img_size, img_size, device=device)
    target_image = torch.randn(1, 3, img_size, img_size, device=device)
    source_coords = torch.tensor([[0.5, 0.5, 0.0, 0.0]], device=device)

    # WebGPU/WebGL doesn't support int64 — use int32
    if int32_action:
        source_action = torch.tensor([0], dtype=torch.int32, device=device)
    else:
        source_action = torch.tensor([0], dtype=torch.int64, device=device)

    print("  Exporting to ONNX...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        (source_image, source_coords, source_action, target_image),
        output_path,
        input_names=["source_image", "source_coords", "source_action", "target_image"],
        output_names=["target_coords", "action_logits"],
        dynamic_axes={
            "source_image": {0: "batch"},
            "target_image": {0: "batch"},
            "source_coords": {0: "batch"},
            "source_action": {0: "batch"},
            "target_coords": {0: "batch"},
            "action_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print("  Exported: {} ({:.1f} MB)".format(output_path, file_size))

    # Verify
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        result = sess.run(None, {
            "source_image": source_image.numpy(),
            "source_coords": source_coords.numpy(),
            "source_action": source_action.numpy(),
            "target_image": target_image.numpy(),
        })
        print("  Verification OK: target_coords={}, action_logits={}".format(
            result[0].shape, result[1].shape))
    except ImportError:
        print("  Skipping verification (onnxruntime not installed)")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CrossMatch to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--int32-action", action="store_true", default=True,
                        help="Use int32 for action tensor (WebGPU compat, default)")
    parser.add_argument("--int64-action", action="store_true",
                        help="Use int64 for action tensor")
    args = parser.parse_args()
    int32 = not args.int64_action
    export_onnx(args.checkpoint, args.output, args.device, int32)
