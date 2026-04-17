"""Inference for CrossMatch model.

Usage:
    from cross_match.predict import CrossMatchPredictor
    predictor = CrossMatchPredictor("checkpoints/cross_match/best.pt", device="mps")

    # Click
    result = predictor.predict(
        source_image=source_pil,
        target_image=target_pil,
        action_type="click",
        source_coords={"at": (540, 960)},
    )
    # result: {"type": "click", "target_coords": {"at": (585, 1266)}, "latency": 0.07}

    # Scroll
    result = predictor.predict(
        source_image=source_pil,
        target_image=target_pil,
        action_type="scroll",
        source_coords={"from_arg": (540, 1200), "to_arg": (540, 600)},
    )
    # result: {"type": "scroll", "target_coords": {"from_arg": (585, 1583), "to_arg": (585, 791)}, "latency": 0.07}
"""

from __future__ import annotations

import time

import torch
from PIL import Image
from torchvision import transforms

from cross_match.config import ModelConfig
from cross_match.dataset import ACTION_TO_IDX, IDX_TO_ACTION
from cross_match.model import CrossMatchModel


class CrossMatchPredictor:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = checkpoint.get("model_config", ModelConfig())
        self.model = CrossMatchModel(model_config).to(device)

        # Load state_dict, handling encoder key mismatches between torch.hub and HF DINOv2.
        # The encoder is frozen/pretrained — only the head weights matter from the checkpoint.
        saved_state = checkpoint["model_state_dict"]
        model_state = self.model.state_dict()

        # Filter: load keys that exist in both and have matching shapes
        compatible = {}
        skipped_encoder = 0
        for k, v in saved_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                compatible[k] = v
            elif k.startswith("encoder."):
                skipped_encoder += 1

        model_state.update(compatible)
        self.model.load_state_dict(model_state)

        if skipped_encoder > 0:
            print("  Note: {} encoder keys skipped (different DINOv2 backend, using fresh pretrained weights)".format(skipped_encoder))
        self.model.eval()

        self.image_size = model_config.image_size
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        action_type: str,
        source_coords: dict,
    ) -> dict:
        """Predict translated action on target device.

        Args:
            source_image: PIL Image of source device
            target_image: PIL Image of target device
            action_type: "click" or "scroll"
            source_coords: click: {"at": (x, y)}, scroll: {"from_arg": (x,y), "to_arg": (x,y)}

        Returns:
            {"type": str, "target_coords": dict, "latency": float}
        """
        src_w, src_h = source_image.size
        tgt_w, tgt_h = target_image.size

        # Normalize source coords to [0, 1]
        if action_type == "click":
            at = source_coords["at"]
            coords_norm = [at[0] / src_w, at[1] / src_h, 0.0, 0.0]
        elif action_type == "scroll":
            f = source_coords["from_arg"]
            t = source_coords["to_arg"]
            coords_norm = [f[0] / src_w, f[1] / src_h, t[0] / src_w, t[1] / src_h]
        else:
            raise ValueError(f"Unknown action: {action_type}")

        # Prepare tensors
        src_tensor = self.transform(source_image).unsqueeze(0).to(self.device)
        tgt_tensor = self.transform(target_image).unsqueeze(0).to(self.device)
        coords_tensor = torch.tensor([coords_norm], dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor([ACTION_TO_IDX[action_type]], dtype=torch.long, device=self.device)

        # Inference
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(src_tensor, coords_tensor, action_tensor, tgt_tensor)
        latency = time.perf_counter() - start

        # Denormalize predicted coords to target pixel space
        pred_coords = outputs["target_coords"][0].cpu().tolist()
        pred_action_idx = outputs["action_logits"][0].argmax().item()
        pred_action = IDX_TO_ACTION[pred_action_idx]

        if action_type == "click":
            target_coords = {
                "at": (int(pred_coords[0] * tgt_w), int(pred_coords[1] * tgt_h)),
            }
        else:
            target_coords = {
                "from_arg": (int(pred_coords[0] * tgt_w), int(pred_coords[1] * tgt_h)),
                "to_arg": (int(pred_coords[2] * tgt_w), int(pred_coords[3] * tgt_h)),
            }

        return {
            "type": pred_action,
            "target_coords": target_coords,
            "latency": latency,
        }
