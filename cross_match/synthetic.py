"""Synthetic data generation for CrossMatch training.

Generates Android/iOS screenshot pairs with UI elements and corresponding
click/scroll action annotations in the CrossMatch annotation format.

Usage:
    python -m cross_match.synthetic --output-dir data/cross_match --num-pairs 1000

NOT run automatically — call explicitly when you need training data.
"""

from __future__ import annotations

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
     "texts": ["Sign In", "Submit", "Continue", "Next", "Save", "Cancel", "OK", "Delete", "Send", "Search",
               "Back", "Forward", "Refresh", "Share", "Edit", "Done", "Apply", "Reset", "Close", "Open"]},
    {"type": "button", "fill": (76, 175, 80), "text_color": (255, 255, 255),
     "texts": ["Accept", "Confirm", "Done", "Apply", "Start", "Connect", "Enable", "Activate", "Verify", "Proceed"]},
    {"type": "button", "fill": (244, 67, 54), "text_color": (255, 255, 255),
     "texts": ["Reject", "Remove", "Logout", "Clear", "Stop", "Disable", "Deny", "Block", "Unsubscribe"]},
    {"type": "input", "fill": (255, 255, 255), "outline": (189, 189, 189), "text_color": (117, 117, 117),
     "texts": ["Email", "Password", "Username", "Search...", "Enter name", "Phone number",
               "Address", "City", "Zip code", "Company", "Message", "Note"]},
    {"type": "label", "fill": None, "text_color": (33, 33, 33),
     "texts": ["Settings", "Profile", "Home", "Notifications", "Messages", "Account",
               "Privacy", "Security", "About", "Help", "Feedback", "Language"]},
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


def _generate_elements(screen_size, num_elements, rng):
    w, h = screen_size
    top_margin = int(h * 0.06)
    bottom_margin = int(h * 0.90)
    elements = []
    for _ in range(num_elements):
        style = rng.choice(ELEMENT_STYLES)
        text = rng.choice(style["texts"])
        if style["type"] == "button":
            ew = rng.randint(int(w * 0.25), int(w * 0.6))
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


def _scale_bbox(bbox, src_size, tgt_size, rng, jitter=15):
    sx = tgt_size[0] / src_size[0]
    sy = tgt_size[1] / src_size[1]
    jx = rng.randint(-jitter, jitter)
    jy = rng.randint(-jitter, jitter)
    return [
        max(0, int(bbox[0] * sx + jx)),
        max(0, int(bbox[1] * sy + jy)),
        min(tgt_size[0], int(bbox[2] * sx + jx)),
        min(tgt_size[1], int(bbox[3] * sy + jy)),
    ]


def _bbox_center(bbox):
    return [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]


def generate_dataset(output_dir: str, num_pairs: int, seed: int = 42):
    source_dir = os.path.join(output_dir, "source")
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    font = _try_load_font(36)
    rng = random.Random(seed)
    annotations = {"pairs": []}

    for i in range(num_pairs):
        pair_id = f"pair_{i:05d}"
        num_elems = rng.randint(5, 10)

        android_elems = _generate_elements(ANDROID_SIZE, num_elems, rng)
        jitter_rng = random.Random(seed + i + 10000)
        ios_elems = [
            {**e, "bbox": _scale_bbox(e["bbox"], ANDROID_SIZE, IOS_SIZE, jitter_rng)}
            for e in android_elems
        ]

        # Draw Android screenshot
        android_img = Image.new("RGB", ANDROID_SIZE, ANDROID_BG)
        android_draw = ImageDraw.Draw(android_img)
        android_draw.rectangle([0, 0, ANDROID_SIZE[0], int(ANDROID_SIZE[1] * 0.04)], fill=(33, 150, 243))
        for e in android_elems:
            _draw_element(android_draw, e["bbox"], e["style"], e["text"], font)
        android_img.save(os.path.join(source_dir, f"{pair_id}.png"))

        # Draw iOS screenshot
        ios_img = Image.new("RGB", IOS_SIZE, IOS_BG)
        ios_draw = ImageDraw.Draw(ios_img)
        for e in ios_elems:
            _draw_element(ios_draw, e["bbox"], e["style"], e["text"], font)
        ios_img.save(os.path.join(target_dir, f"{pair_id}.png"))

        # Generate actions for this pair
        actions = []
        # 1-3 click actions on random elements
        num_clicks = rng.randint(1, min(3, num_elems))
        click_indices = rng.sample(range(num_elems), num_clicks)
        for idx in click_indices:
            src_center = _bbox_center(android_elems[idx]["bbox"])
            tgt_center = _bbox_center(ios_elems[idx]["bbox"])
            actions.append({
                "type": "click",
                "source_coords": {"at": src_center},
                "target_coords": {"at": tgt_center},
            })

        # 0-2 scroll actions
        num_scrolls = rng.randint(0, 2)
        for _ in range(num_scrolls):
            # Random scroll: start from mid-screen area, swipe up or down
            src_from_x = rng.randint(int(ANDROID_SIZE[0] * 0.3), int(ANDROID_SIZE[0] * 0.7))
            src_from_y = rng.randint(int(ANDROID_SIZE[1] * 0.3), int(ANDROID_SIZE[1] * 0.7))
            scroll_dy = rng.randint(int(ANDROID_SIZE[1] * 0.1), int(ANDROID_SIZE[1] * 0.3))
            scroll_dir = rng.choice([-1, 1])
            src_to_y = max(0, min(ANDROID_SIZE[1], src_from_y + scroll_dir * scroll_dy))

            # Translate to iOS coordinates
            sx = IOS_SIZE[0] / ANDROID_SIZE[0]
            sy = IOS_SIZE[1] / ANDROID_SIZE[1]
            tgt_from_x = int(src_from_x * sx) + jitter_rng.randint(-10, 10)
            tgt_from_y = int(src_from_y * sy) + jitter_rng.randint(-10, 10)
            tgt_to_y = int(src_to_y * sy) + jitter_rng.randint(-10, 10)

            actions.append({
                "type": "scroll",
                "source_coords": {
                    "from_arg": [src_from_x, src_from_y],
                    "to_arg": [src_from_x, src_to_y],
                },
                "target_coords": {
                    "from_arg": [tgt_from_x, tgt_from_y],
                    "to_arg": [tgt_from_x, tgt_to_y],
                },
            })

        annotations["pairs"].append({
            "id": pair_id,
            "source": {
                "image": f"source/{pair_id}.png",
                "platform": "android",
                "size": list(ANDROID_SIZE),
            },
            "target": {
                "image": f"target/{pair_id}.png",
                "platform": "ios",
                "size": list(IOS_SIZE),
            },
            "actions": actions,
        })

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_pairs} pairs")

    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=2)

    total_actions = sum(len(p["actions"]) for p in annotations["pairs"])
    print(f"\nGenerated {num_pairs} pairs with {total_actions} total actions in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/cross_match")
    parser.add_argument("--num-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.num_pairs, args.seed)
