from __future__ import annotations

import math
from dataclasses import dataclass, field

from PIL import Image

from benchmark.model import FlorenceModel


@dataclass
class PipelineResult:
    predicted_bbox: list[float] | None
    description: str
    ocr_text: str
    num_candidates: int
    strategy: str
    step_latencies: dict[str, float] = field(default_factory=dict)


def _crop_with_padding(image: Image.Image, bbox: list[float], pad_fraction: float = 0.2) -> Image.Image:
    """Crop image around bbox with padding. Returns cropped PIL Image."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_x = w * pad_fraction
    pad_y = h * pad_fraction

    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(image.width, int(x2 + pad_x))
    cy2 = min(image.height, int(y2 + pad_y))

    return image.crop((cx1, cy1, cx2, cy2))


def select_best_bbox(
    candidates: list[list[float]],
    source_bbox: list[float],
    source_img_size: tuple[int, int],
    target_img_size: tuple[int, int],
) -> list[float] | None:
    """Pick best bbox from candidates using spatial-prior + area-ratio heuristic."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Normalize source bbox center to [0, 1]
    sw, sh = source_img_size
    src_cx = (source_bbox[0] + source_bbox[2]) / 2 / sw
    src_cy = (source_bbox[1] + source_bbox[3]) / 2 / sh
    src_area_frac = ((source_bbox[2] - source_bbox[0]) * (source_bbox[3] - source_bbox[1])) / (sw * sh)

    tw, th = target_img_size
    best = None
    best_score = float("inf")

    for bbox in candidates:
        # Normalized center of candidate
        cx = (bbox[0] + bbox[2]) / 2 / tw
        cy = (bbox[1] + bbox[3]) / 2 / th
        area_frac = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (tw * th)

        # Spatial distance (weighted more heavily)
        spatial_dist = math.sqrt((cx - src_cx) ** 2 + (cy - src_cy) ** 2)

        # Area ratio difference
        area_diff = abs(area_frac - src_area_frac) / max(src_area_frac, 1e-6)

        score = spatial_dist * 2.0 + area_diff * 0.5
        if score < best_score:
            best_score = score
            best = bbox

    return best


def run_desc_only(
    model: FlorenceModel,
    source_img: Image.Image,
    source_bbox: list[float],
    target_img: Image.Image,
) -> PipelineResult:
    """Strategy 1: REGION_TO_DESCRIPTION → OPEN_VOCABULARY_DETECTION"""
    desc, desc_lat = model.region_to_description(source_img, source_bbox)

    if not desc:
        return PipelineResult(
            predicted_bbox=None, description="", ocr_text="",
            num_candidates=0, strategy="desc_only",
            step_latencies={"region_to_desc": desc_lat, "ovd": 0.0},
        )

    det_result, det_lat = model.open_vocabulary_detection(target_img, desc)
    candidates = det_result.get("bboxes", [])

    predicted = select_best_bbox(candidates, source_bbox, source_img.size, target_img.size)

    return PipelineResult(
        predicted_bbox=predicted,
        description=desc,
        ocr_text="",
        num_candidates=len(candidates),
        strategy="desc_only",
        step_latencies={"region_to_desc": desc_lat, "ovd": det_lat},
    )


def run_desc_ocr(
    model: FlorenceModel,
    source_img: Image.Image,
    source_bbox: list[float],
    target_img: Image.Image,
) -> PipelineResult:
    """Strategy 2: REGION_TO_DESCRIPTION + OCR (cropped) → CAPTION_TO_PHRASE_GROUNDING"""
    desc, desc_lat = model.region_to_description(source_img, source_bbox)

    cropped = _crop_with_padding(source_img, source_bbox)
    ocr_result, ocr_lat = model.ocr_with_region(cropped)
    ocr_labels = ocr_result.get("labels", [])
    ocr_text = " ".join(ocr_labels).strip()

    # Combine description with OCR text
    if desc and ocr_text:
        query = f"{desc} with text {ocr_text}"
    elif desc:
        query = desc
    elif ocr_text:
        query = ocr_text
    else:
        return PipelineResult(
            predicted_bbox=None, description="", ocr_text="",
            num_candidates=0, strategy="desc_ocr",
            step_latencies={"region_to_desc": desc_lat, "ocr": ocr_lat, "grounding": 0.0},
        )

    ground_result, ground_lat = model.caption_to_phrase_grounding(target_img, query)
    candidates = ground_result.get("bboxes", [])

    predicted = select_best_bbox(candidates, source_bbox, source_img.size, target_img.size)

    return PipelineResult(
        predicted_bbox=predicted,
        description=desc,
        ocr_text=ocr_text,
        num_candidates=len(candidates),
        strategy="desc_ocr",
        step_latencies={"region_to_desc": desc_lat, "ocr": ocr_lat, "grounding": ground_lat},
    )


def run_ocr_only(
    model: FlorenceModel,
    source_img: Image.Image,
    source_bbox: list[float],
    target_img: Image.Image,
) -> PipelineResult:
    """Strategy 3: OCR_WITH_REGION (cropped) → OPEN_VOCABULARY_DETECTION"""
    cropped = _crop_with_padding(source_img, source_bbox)
    ocr_result, ocr_lat = model.ocr_with_region(cropped)
    ocr_labels = ocr_result.get("labels", [])
    ocr_text = " ".join(ocr_labels).strip()

    if not ocr_text:
        return PipelineResult(
            predicted_bbox=None, description="", ocr_text="",
            num_candidates=0, strategy="ocr_only",
            step_latencies={"ocr": ocr_lat, "ovd": 0.0},
        )

    det_result, det_lat = model.open_vocabulary_detection(target_img, ocr_text)
    candidates = det_result.get("bboxes", [])

    predicted = select_best_bbox(candidates, source_bbox, source_img.size, target_img.size)

    return PipelineResult(
        predicted_bbox=predicted,
        description="",
        ocr_text=ocr_text,
        num_candidates=len(candidates),
        strategy="ocr_only",
        step_latencies={"ocr": ocr_lat, "ovd": det_lat},
    )


STRATEGIES = {
    "desc_only": run_desc_only,
    "desc_ocr": run_desc_ocr,
    "ocr_only": run_ocr_only,
}
