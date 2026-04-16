import json
import os
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class TestPair:
    id: str
    source_image: Image.Image
    source_bbox: list[float]
    source_platform: str
    target_image: Image.Image
    target_bbox: list[float]
    target_platform: str
    tags: list[str] = field(default_factory=list)


def validate_annotations(annotations: dict, data_dir: str) -> list[str]:
    """Return list of validation errors (empty if valid)."""
    errors = []

    if "pairs" not in annotations:
        errors.append("Missing 'pairs' key in annotations")
        return errors

    for i, pair in enumerate(annotations["pairs"]):
        prefix = f"pairs[{i}]"

        for key in ("id", "source", "target"):
            if key not in pair:
                errors.append(f"{prefix}: missing '{key}'")

        if "source" in pair:
            src = pair["source"]
            if "image" not in src:
                errors.append(f"{prefix}.source: missing 'image'")
            elif not os.path.exists(os.path.join(data_dir, src["image"])):
                errors.append(f"{prefix}.source: image not found: {src['image']}")
            if "bbox" not in src:
                errors.append(f"{prefix}.source: missing 'bbox'")
            elif len(src["bbox"]) != 4:
                errors.append(f"{prefix}.source: bbox must have 4 values")

        if "target" in pair:
            tgt = pair["target"]
            if "image" not in tgt:
                errors.append(f"{prefix}.target: missing 'image'")
            elif not os.path.exists(os.path.join(data_dir, tgt["image"])):
                errors.append(f"{prefix}.target: image not found: {tgt['image']}")
            if "bbox" not in tgt:
                errors.append(f"{prefix}.target: missing 'bbox'")
            elif len(tgt["bbox"]) != 4:
                errors.append(f"{prefix}.target: bbox must have 4 values")

    return errors


def load_dataset(data_dir: str) -> list[TestPair]:
    """Load annotations.json and return list of TestPair."""
    annotations_path = os.path.join(data_dir, "annotations.json")
    with open(annotations_path) as f:
        annotations = json.load(f)

    errors = validate_annotations(annotations, data_dir)
    if errors:
        raise ValueError(f"Dataset validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    pairs = []
    for pair in annotations["pairs"]:
        src = pair["source"]
        tgt = pair["target"]

        source_img = Image.open(os.path.join(data_dir, src["image"])).convert("RGB")
        target_img = Image.open(os.path.join(data_dir, tgt["image"])).convert("RGB")

        pairs.append(TestPair(
            id=pair["id"],
            source_image=source_img,
            source_bbox=src["bbox"],
            source_platform=src.get("platform", "unknown"),
            target_image=target_img,
            target_bbox=tgt["bbox"],
            target_platform=tgt.get("platform", "unknown"),
            tags=pair.get("tags", []),
        ))

    return pairs
