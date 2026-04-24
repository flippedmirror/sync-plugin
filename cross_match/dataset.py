"""Dataset for cross-platform action matching.

Annotation format (annotations.json):
{
    "pairs": [
        {
            "id": "pair_001",
            "source": {
                "image": "source/screen_001.png",
                "platform": "android"
            },
            "target": {
                "image": "target/screen_001.png",
                "platform": "ios"
            },
            "actions": [
                {
                    "type": "click",
                    "source_coords": {"at": [540, 960]},
                    "target_coords": {"at": [585, 1266]}
                },
                {
                    "type": "scroll",
                    "source_coords": {"from_arg": [540, 1200], "to_arg": [540, 600]},
                    "target_coords": {"from_arg": [585, 1583], "to_arg": [585, 791]}
                }
            ]
        }
    ]
}
"""

from __future__ import annotations

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ACTION_TO_IDX = {"click": 0, "scroll": 1}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


def _normalize_coords(coords: list[float], image_size: tuple[int, int]) -> list[float]:
    """Normalize pixel coordinates to [0, 1] based on image dimensions."""
    w, h = image_size
    # coords are [x1, y1, x2, y2] or [x, y, 0, 0]
    return [
        coords[0] / w, coords[1] / h,
        coords[2] / w if len(coords) > 2 else 0.0,
        coords[3] / h if len(coords) > 3 else 0.0,
    ]


def _action_to_coords(action: dict, image_size: tuple[int, int]) -> tuple[list[float], int]:
    """Convert action dict to normalized 4-coord vector + action index.

    Returns:
        (coords_4, action_idx) where coords_4 is [c1, c2, c3, c4] normalized to [0,1]
    """
    action_type = action["type"]
    action_idx = ACTION_TO_IDX[action_type]

    if action_type == "click":
        at = action["at"]
        raw = [at[0], at[1], 0.0, 0.0]
    elif action_type == "scroll":
        from_arg = action["from_arg"]
        to_arg = action["to_arg"]
        raw = [from_arg[0], from_arg[1], to_arg[0], to_arg[1]]
    else:
        raise ValueError(f"Unknown action type: {action_type}")

    return _normalize_coords(raw, image_size), action_idx


class CrossMatchDataset(Dataset):
    """Dataset that yields (source_image, source_coords, action_idx, target_image, target_coords)."""

    def __init__(self, data_dir: str, image_size: int = 518, split: str = "train", encoder_name: str = None):
        self.data_dir = data_dir
        self.image_size = image_size

        annotations_path = os.path.join(data_dir, "annotations.json")
        with open(annotations_path) as f:
            data = json.load(f)

        # Flatten: one sample per action (a pair can have multiple actions)
        self.samples = []
        for pair in data["pairs"]:
            src_img_path = os.path.join(data_dir, pair["source"]["image"])
            tgt_img_path = os.path.join(data_dir, pair["target"]["image"])

            # We need image sizes for normalization — read lazily or store in annotations
            src_size = pair["source"].get("size")  # [w, h] if present
            tgt_size = pair["target"].get("size")

            for action_pair in pair["actions"]:
                self.samples.append({
                    "source_image_path": src_img_path,
                    "target_image_path": tgt_img_path,
                    "source_action": {
                        "type": action_pair["type"],
                        **action_pair["source_coords"],
                    },
                    "target_action": {
                        "type": action_pair["type"],
                        **action_pair["target_coords"],
                    },
                    "source_size": src_size,
                    "target_size": tgt_size,
                })

        # SigLIP uses mean=0.5, std=0.5; DINOv2/ImageNet uses standard normalization
        if encoder_name and encoder_name.startswith("siglip"):
            norm_mean, norm_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        src_img = Image.open(sample["source_image_path"]).convert("RGB")
        tgt_img = Image.open(sample["target_image_path"]).convert("RGB")

        src_size = sample["source_size"] or [src_img.width, src_img.height]
        tgt_size = sample["target_size"] or [tgt_img.width, tgt_img.height]

        # Convert actions to normalized coords
        src_coords, action_idx = _action_to_coords(sample["source_action"], tuple(src_size))
        tgt_coords, _ = _action_to_coords(sample["target_action"], tuple(tgt_size))

        # Transform images
        src_tensor = self.transform(src_img)
        tgt_tensor = self.transform(tgt_img)

        return {
            "source_image": src_tensor,
            "source_coords": torch.tensor(src_coords, dtype=torch.float32),
            "source_action": torch.tensor(action_idx, dtype=torch.long),
            "target_image": tgt_tensor,
            "target_coords": torch.tensor(tgt_coords, dtype=torch.float32),
        }


class CachedFeatureDataset(Dataset):
    """Dataset using precomputed DINOv2 features for fast training.

    Expects feature cache files:
        cache_dir/source/{image_stem}.pt  — (N_patches, D) tensor
        cache_dir/target/{image_stem}.pt  — (N_patches, D) tensor
    """

    def __init__(self, data_dir: str, cache_dir: str, image_size: int = 518):
        self.cache_dir = cache_dir

        annotations_path = os.path.join(data_dir, "annotations.json")
        with open(annotations_path) as f:
            data = json.load(f)

        self.samples = []
        for pair in data["pairs"]:
            src_stem = os.path.splitext(os.path.basename(pair["source"]["image"]))[0]
            tgt_stem = os.path.splitext(os.path.basename(pair["target"]["image"]))[0]

            src_img_path = os.path.join(data_dir, pair["source"]["image"])
            tgt_img_path = os.path.join(data_dir, pair["target"]["image"])
            src_size = pair["source"].get("size")
            tgt_size = pair["target"].get("size")

            for action_pair in pair["actions"]:
                self.samples.append({
                    "source_cache": os.path.join(cache_dir, "source", f"{src_stem}.pt"),
                    "target_cache": os.path.join(cache_dir, "target", f"{tgt_stem}.pt"),
                    "source_image_path": src_img_path,
                    "target_image_path": tgt_img_path,
                    "source_action": {
                        "type": action_pair["type"],
                        **action_pair["source_coords"],
                    },
                    "target_action": {
                        "type": action_pair["type"],
                        **action_pair["target_coords"],
                    },
                    "source_size": src_size,
                    "target_size": tgt_size,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        src_features = torch.load(sample["source_cache"], weights_only=True)
        tgt_features = torch.load(sample["target_cache"], weights_only=True)

        # Resolve image sizes for normalization
        if sample["source_size"]:
            src_size = tuple(sample["source_size"])
        else:
            src_img = Image.open(sample["source_image_path"])
            src_size = (src_img.width, src_img.height)

        if sample["target_size"]:
            tgt_size = tuple(sample["target_size"])
        else:
            tgt_img = Image.open(sample["target_image_path"])
            tgt_size = (tgt_img.width, tgt_img.height)

        src_coords, action_idx = _action_to_coords(sample["source_action"], src_size)
        tgt_coords, _ = _action_to_coords(sample["target_action"], tgt_size)

        return {
            "source_features": src_features,
            "source_coords": torch.tensor(src_coords, dtype=torch.float32),
            "source_action": torch.tensor(action_idx, dtype=torch.long),
            "target_features": tgt_features,
            "target_coords": torch.tensor(tgt_coords, dtype=torch.float32),
        }
