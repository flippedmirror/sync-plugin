"""Generate synthetic cross-platform test pairs with CLICK coordinates.

Instead of bounding boxes, each pair has a source click (x, y) and
expected target click (x, y) — matching real-world usage where we only
know where the user tapped.

Usage:
    python scripts/create_click_data.py --output-dir data/click_sample --num-pairs 15
"""

import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

ANDROID_SIZE = (1080, 1920)
IOS_SIZE = (1170, 2532)
ANDROID_BG = (245, 245, 245)
IOS_BG = (242, 242, 247)

ELEMENT_STYLES = [
    {"type": "button", "fill": (33, 150, 243), "text_color": (255, 255, 255),
     "texts": ["Sign In", "Submit", "Continue", "Next", "Save", "Cancel", "OK", "Delete", "Send", "Search"]},
    {"type": "button", "fill": (76, 175, 80), "text_color": (255, 255, 255),
     "texts": ["Accept", "Confirm", "Done", "Apply", "Start", "Connect"]},
    {"type": "button", "fill": (244, 67, 54), "text_color": (255, 255, 255),
     "texts": ["Reject", "Remove", "Logout", "Clear", "Stop"]},
    {"type": "input", "fill": (255, 255, 255), "outline": (189, 189, 189), "text_color": (117, 117, 117),
     "texts": ["Email", "Password", "Username", "Search...", "Enter name", "Phone number"]},
    {"type": "label", "fill": None, "text_color": (33, 33, 33),
     "texts": ["Settings", "Profile", "Home", "Notifications", "Messages", "Account"]},
]


def _try_load_font(size):
    for path in ["/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_element(draw, bbox, style, text, font):
    x1, y1, x2, y2 = bbox
    if style["type"] == "button":
        draw.rounded_rectangle([x1, y1, x2, y2], radius=12, fill=style["fill"])
        tw = draw.textlength(text, font=font)
        tx = x1 + (x2 - x1 - tw) / 2
        ty = y1 + (y2 - y1 - font.size) / 2
        draw.text((tx, ty), text, fill=style["text_color"], font=font)
    elif style["type"] == "input":
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=style["fill"],
                               outline=style.get("outline", (189, 189, 189)), width=2)
        draw.text((x1 + 12, y1 + (y2 - y1 - font.size) / 2), text, fill=style["text_color"], font=font)
    elif style["type"] == "label":
        draw.text((x1, y1), text, fill=style["text_color"], font=font)


def _generate_elements(screen_size, num_elements, seed):
    rng = random.Random(seed)
    w, h = screen_size
    top_margin = int(h * 0.06)
    bottom_margin = int(h * 0.90)
    elements = []

    for _ in range(num_elements):
        style = rng.choice(ELEMENT_STYLES)
        text = rng.choice(style["texts"])

        if style["type"] == "button":
            ew = rng.randint(int(w * 0.3), int(w * 0.6))
            eh = rng.randint(int(h * 0.03), int(h * 0.045))
        elif style["type"] == "input":
            ew = rng.randint(int(w * 0.6), int(w * 0.85))
            eh = rng.randint(int(h * 0.03), int(h * 0.04))
        else:
            ew = rng.randint(int(w * 0.2), int(w * 0.5))
            eh = int(h * 0.025)

        x1 = rng.randint(int(w * 0.05), max(int(w * 0.05) + 1, w - ew - int(w * 0.05)))
        y1 = rng.randint(top_margin, max(top_margin + 1, bottom_margin - eh))
        elements.append({"bbox": [x1, y1, x1 + ew, y1 + eh], "style": style, "text": text})

    return elements


def create_click_data(output_dir, num_pairs, seed=42):
    source_dir = os.path.join(output_dir, "source")
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    font = _try_load_font(36)
    annotations = {"pairs": []}
    rng = random.Random(seed)

    for i in range(num_pairs):
        pair_id = f"click_{i:03d}"
        num_elems = rng.randint(5, 9)
        android_elems = _generate_elements(ANDROID_SIZE, num_elems, seed=seed + i)

        # Build iOS elements with proportional scaling + jitter
        jitter_rng = random.Random(seed + i + 5000)
        ios_elems = []
        for elem in android_elems:
            ax1, ay1, ax2, ay2 = elem["bbox"]
            sx = IOS_SIZE[0] / ANDROID_SIZE[0]
            sy = IOS_SIZE[1] / ANDROID_SIZE[1]
            jx = jitter_rng.randint(-12, 12)
            jy = jitter_rng.randint(-18, 18)
            ios_elems.append({
                "bbox": [
                    max(0, int(ax1 * sx + jx)), max(0, int(ay1 * sy + jy)),
                    min(IOS_SIZE[0], int(ax2 * sx + jx)), min(IOS_SIZE[1], int(ay2 * sy + jy)),
                ],
                "style": elem["style"], "text": elem["text"],
            })

        # Draw Android
        android_img = Image.new("RGB", ANDROID_SIZE, ANDROID_BG)
        android_draw = ImageDraw.Draw(android_img)
        android_draw.rectangle([0, 0, ANDROID_SIZE[0], int(ANDROID_SIZE[1] * 0.04)], fill=(33, 150, 243))
        for elem in android_elems:
            _draw_element(android_draw, elem["bbox"], elem["style"], elem["text"], font)
        android_img.save(os.path.join(output_dir, f"source/{pair_id}.png"))

        # Draw iOS
        ios_img = Image.new("RGB", IOS_SIZE, IOS_BG)
        ios_draw = ImageDraw.Draw(ios_img)
        for elem in ios_elems:
            _draw_element(ios_draw, elem["bbox"], elem["style"], elem["text"], font)
        ios_img.save(os.path.join(output_dir, f"target/{pair_id}.png"))

        # Pick test element — click at its CENTER (not bbox)
        test_idx = jitter_rng.randint(0, len(android_elems) - 1)
        src_bbox = android_elems[test_idx]["bbox"]
        tgt_bbox = ios_elems[test_idx]["bbox"]

        src_click = [int((src_bbox[0] + src_bbox[2]) / 2), int((src_bbox[1] + src_bbox[3]) / 2)]
        tgt_click = [int((tgt_bbox[0] + tgt_bbox[2]) / 2), int((tgt_bbox[1] + tgt_bbox[3]) / 2)]

        annotations["pairs"].append({
            "id": pair_id,
            "source": {
                "image": f"source/{pair_id}.png",
                "click": src_click,
                "platform": "android",
            },
            "target": {
                "image": f"target/{pair_id}.png",
                "click": tgt_click,
                "platform": "ios",
            },
            "element_type": android_elems[test_idx]["style"]["type"],
            "element_text": android_elems[test_idx]["text"],
        })
        print(f"  {pair_id}: click on '{android_elems[test_idx]['text']}' ({android_elems[test_idx]['style']['type']})"
              f" @ ({src_click[0]},{src_click[1]}) → ({tgt_click[0]},{tgt_click[1]})")

    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"\nGenerated {num_pairs} click pairs in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/click_sample")
    parser.add_argument("--num-pairs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_click_data(args.output_dir, args.num_pairs, args.seed)
