"""Click-to-click pipeline: translate (x, y) click from source to target screenshot.

The only input is a screenshot + click coordinate. No bounding boxes, no labels.

Pipeline (OCR-first, fast path):
  1. Crop a region around the click point on source screenshot
  2. Run <OCR> on the crop to get text at the click location
  3. Run <OPEN_VOCABULARY_DETECTION> on target with that text
  4. Return center of best-matching bbox as predicted click

Fallback (if OCR finds no text):
  1. Run <CAPTION> on the crop to describe what's there
  2. Run <CAPTION_TO_PHRASE_GROUNDING> on target with caption
  3. Return center of best-matching bbox
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from PIL import Image

from benchmark.model import FlorenceModel


@dataclass
class ClickResult:
    predicted_click: list[int] | None  # [x, y] on target, or None if failed
    query_text: str                     # what we searched for
    method: str                         # "ocr" or "caption" or "failed"
    num_candidates: int
    step_latencies: dict[str, float] = field(default_factory=dict)


def _strip_loc_tokens(text: str) -> str:
    """Remove Florence-2 <loc_NNN> tokens from text."""
    return re.sub(r"<loc_\d+>", "", text).strip()


def _crop_around_click(image: Image.Image, x: int, y: int, radius: int = 150) -> Image.Image:
    """Crop a square region around the click point."""
    cx1 = max(0, x - radius)
    cy1 = max(0, y - radius)
    cx2 = min(image.width, x + radius)
    cy2 = min(image.height, y + radius)
    return image.crop((cx1, cy1, cx2, cy2))


def _bbox_center(bbox: list[float]) -> list[int]:
    return [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]


def _select_best_bbox_for_click(
    candidates: list[list[float]],
    source_click: list[int],
    source_img_size: tuple[int, int],
    target_img_size: tuple[int, int],
) -> list[float] | None:
    """Pick candidate bbox whose center is closest in normalized position to source click."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    sw, sh = source_img_size
    src_nx = source_click[0] / sw
    src_ny = source_click[1] / sh

    tw, th = target_img_size
    best = None
    best_dist = float("inf")

    for bbox in candidates:
        cx = (bbox[0] + bbox[2]) / 2 / tw
        cy = (bbox[1] + bbox[3]) / 2 / th
        dist = math.sqrt((cx - src_nx) ** 2 + (cy - src_ny) ** 2)
        if dist < best_dist:
            best_dist = dist
            best = bbox

    return best


def run_click_pipeline(
    model: FlorenceModel,
    source_img: Image.Image,
    source_click: list[int],
    target_img: Image.Image,
    crop_radius: int = 150,
) -> ClickResult:
    """Translate a click from source to target screenshot."""

    # Step 1: Crop around click point
    crop = _crop_around_click(source_img, source_click[0], source_click[1], crop_radius)

    # Step 2: Try OCR on the crop (fast path — 2 inference calls total)
    ocr_text, ocr_lat = model.ocr(crop)
    ocr_text = _strip_loc_tokens(ocr_text).strip()

    if ocr_text and len(ocr_text) >= 2:
        # Step 3a: Search target for this text via OVD
        det_result, det_lat = model.open_vocabulary_detection(target_img, ocr_text)
        candidates = det_result.get("bboxes", [])

        best = _select_best_bbox_for_click(candidates, source_click, source_img.size, target_img.size)
        if best:
            return ClickResult(
                predicted_click=_bbox_center(best),
                query_text=ocr_text,
                method="ocr",
                num_candidates=len(candidates),
                step_latencies={"crop_ocr": ocr_lat, "target_ovd": det_lat},
            )

    # Step 3b: Fallback — caption the crop, then ground on target
    caption_text, caption_lat = model.caption(crop)
    caption_text = _strip_loc_tokens(caption_text).strip()

    if caption_text:
        ground_result, ground_lat = model.caption_to_phrase_grounding(target_img, caption_text)
        candidates = ground_result.get("bboxes", [])
        best = _select_best_bbox_for_click(candidates, source_click, source_img.size, target_img.size)
        if best:
            return ClickResult(
                predicted_click=_bbox_center(best),
                query_text=caption_text,
                method="caption",
                num_candidates=len(candidates),
                step_latencies={"crop_ocr": ocr_lat, "crop_caption": caption_lat, "target_ground": ground_lat},
            )

    # Total failure
    return ClickResult(
        predicted_click=None,
        query_text=ocr_text or caption_text or "",
        method="failed",
        num_candidates=0,
        step_latencies={"crop_ocr": ocr_lat, "crop_caption": caption_lat if caption_text else 0.0},
    )
