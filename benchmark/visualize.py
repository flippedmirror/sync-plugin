import os

from PIL import Image, ImageDraw, ImageFont


def _try_load_font(size: int):
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_comparison(
    target_image: Image.Image,
    predicted_bbox: list[float],
    ground_truth_bbox: list[float],
    pair_id: str,
    strategy: str,
    iou: float,
) -> Image.Image:
    """Draw GT (green) and predicted (red) bboxes on target image. Returns new image."""
    img = target_image.copy()
    draw = ImageDraw.Draw(img)
    font = _try_load_font(28)

    # Ground truth — green
    gt = [int(v) for v in ground_truth_bbox]
    draw.rectangle(gt, outline=(0, 200, 0), width=4)
    draw.text((gt[0], gt[1] - 32), "GT", fill=(0, 200, 0), font=font)

    # Predicted — red (ensure valid bbox ordering)
    pred = [int(v) for v in predicted_bbox]
    pred_rect = [min(pred[0], pred[2]), min(pred[1], pred[3]), max(pred[0], pred[2]), max(pred[1], pred[3])]
    draw.rectangle(pred_rect, outline=(220, 0, 0), width=4)
    draw.text((pred_rect[0], pred_rect[3] + 4), "Pred", fill=(220, 0, 0), font=font)

    # IoU annotation at top
    header_font = _try_load_font(36)
    draw.text(
        (10, 10),
        f"{pair_id} | {strategy} | IoU={iou:.3f}",
        fill=(0, 0, 0),
        font=header_font,
    )

    return img


def save_visualization(image: Image.Image, output_dir: str, pair_id: str, strategy: str):
    viz_dir = os.path.join(output_dir, "viz", strategy)
    os.makedirs(viz_dir, exist_ok=True)
    path = os.path.join(viz_dir, f"{pair_id}.png")
    image.save(path)
