"""Generate synthetic cross-platform test pairs for benchmarking.

Creates simple UI-like screenshots with buttons, text fields, and labels
at known positions, simulating Android and iOS layout differences.

Usage:
    python scripts/create_synthetic_data.py --output-dir data/sample --num-pairs 10
"""

import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont


# Screen dimensions (simulating mobile devices)
ANDROID_SIZE = (1080, 1920)
IOS_SIZE = (1170, 2532)

# Color palettes
ANDROID_BG = (245, 245, 245)
IOS_BG = (242, 242, 247)

ELEMENT_STYLES = [
    {"type": "button", "fill": (33, 150, 243), "text_color": (255, 255, 255), "texts": ["Sign In", "Submit", "Continue", "Next", "Save", "Cancel", "OK", "Delete", "Send", "Search"]},
    {"type": "button", "fill": (76, 175, 80), "text_color": (255, 255, 255), "texts": ["Accept", "Confirm", "Done", "Apply", "Start", "Connect"]},
    {"type": "button", "fill": (244, 67, 54), "text_color": (255, 255, 255), "texts": ["Reject", "Remove", "Logout", "Clear", "Stop"]},
    {"type": "input", "fill": (255, 255, 255), "outline": (189, 189, 189), "text_color": (117, 117, 117), "texts": ["Email", "Password", "Username", "Search...", "Enter name", "Phone number"]},
    {"type": "label", "fill": None, "text_color": (33, 33, 33), "texts": ["Settings", "Profile", "Home", "Notifications", "Messages", "Account"]},
    {"type": "toggle", "fill": (33, 150, 243), "text_color": (255, 255, 255), "texts": ["ON", "OFF"]},
]


def _try_load_font(size: int):
    """Try to load a TTF font, fall back to default."""
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


def _draw_element(draw: ImageDraw.Draw, bbox: list[int], style: dict, text: str, font):
    """Draw a UI element on the image."""
    x1, y1, x2, y2 = bbox
    elem_type = style["type"]

    if elem_type == "button":
        # Rounded rectangle button
        draw.rounded_rectangle([x1, y1, x2, y2], radius=12, fill=style["fill"])
        # Center text
        tw = draw.textlength(text, font=font)
        tx = x1 + (x2 - x1 - tw) / 2
        ty = y1 + (y2 - y1 - font.size) / 2
        draw.text((tx, ty), text, fill=style["text_color"], font=font)

    elif elem_type == "input":
        # Input field with border
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=style["fill"], outline=style["outline"], width=2)
        tx = x1 + 12
        ty = y1 + (y2 - y1 - font.size) / 2
        draw.text((tx, ty), text, fill=style["text_color"], font=font)

    elif elem_type == "label":
        # Just text
        draw.text((x1, y1), text, fill=style["text_color"], font=font)

    elif elem_type == "toggle":
        # Simple toggle switch
        draw.rounded_rectangle([x1, y1, x2, y2], radius=(y2 - y1) // 2, fill=style["fill"])
        # Circle indicator
        r = (y2 - y1) // 2 - 4
        cx = x2 - r - 6 if text == "ON" else x1 + r + 6
        cy = (y1 + y2) // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))


def _generate_layout(screen_size: tuple[int, int], num_elements: int, seed: int) -> list[dict]:
    """Generate random non-overlapping element positions for a screen."""
    rng = random.Random(seed)
    w, h = screen_size
    elements = []

    # Status bar region (top 5%)
    top_margin = int(h * 0.05)
    # Navigation bar (bottom 8%)
    bottom_margin = int(h * 0.92)

    for _ in range(num_elements):
        style = rng.choice(ELEMENT_STYLES)
        text = rng.choice(style["texts"])

        # Element size varies by type
        if style["type"] == "button":
            ew = rng.randint(int(w * 0.25), int(w * 0.6))
            eh = rng.randint(int(h * 0.025), int(h * 0.04))
        elif style["type"] == "input":
            ew = rng.randint(int(w * 0.6), int(w * 0.85))
            eh = rng.randint(int(h * 0.025), int(h * 0.035))
        elif style["type"] == "toggle":
            ew = rng.randint(int(w * 0.08), int(w * 0.12))
            eh = rng.randint(int(h * 0.015), int(h * 0.02))
        else:  # label
            ew = rng.randint(int(w * 0.2), int(w * 0.5))
            eh = int(h * 0.02)

        # Random position (centered horizontally with jitter)
        x1 = rng.randint(int(w * 0.05), max(int(w * 0.05) + 1, w - ew - int(w * 0.05)))
        y1 = rng.randint(top_margin, max(top_margin + 1, bottom_margin - eh))

        bbox = [x1, y1, x1 + ew, y1 + eh]
        elements.append({"bbox": bbox, "style": style, "text": text})

    return elements


def _generate_pair(pair_idx: int, base_seed: int) -> dict:
    """Generate a source (Android) and target (iOS) screenshot pair."""
    # Generate layout on Android dimensions
    num_elements = random.Random(base_seed + pair_idx).randint(4, 8)
    android_elements = _generate_layout(ANDROID_SIZE, num_elements, seed=base_seed + pair_idx)

    # Create corresponding iOS elements with shifted positions
    # Simulate cross-platform layout differences: different screen size, slight position shifts
    rng = random.Random(base_seed + pair_idx + 1000)
    ios_elements = []
    for elem in android_elements:
        ax1, ay1, ax2, ay2 = elem["bbox"]
        aw, ah = ANDROID_SIZE
        iw, ih = IOS_SIZE

        # Scale proportionally + add jitter
        scale_x = iw / aw
        scale_y = ih / ah
        jitter_x = rng.randint(-15, 15)
        jitter_y = rng.randint(-20, 20)

        ix1 = max(0, int(ax1 * scale_x + jitter_x))
        iy1 = max(0, int(ay1 * scale_y + jitter_y))
        ix2 = min(iw, int(ax2 * scale_x + jitter_x))
        iy2 = min(ih, int(ay2 * scale_y + jitter_y))

        ios_elements.append({
            "bbox": [ix1, iy1, ix2, iy2],
            "style": elem["style"],
            "text": elem["text"],
        })

    # Pick a random element as the test target
    test_idx = rng.randint(0, len(android_elements) - 1)

    return {
        "android_elements": android_elements,
        "ios_elements": ios_elements,
        "test_element_idx": test_idx,
    }


def create_synthetic_data(output_dir: str, num_pairs: int, seed: int = 42):
    source_dir = os.path.join(output_dir, "source")
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    font = _try_load_font(36)
    annotations = {"pairs": []}

    for i in range(num_pairs):
        pair_data = _generate_pair(i, seed)
        pair_id = f"pair_{i:03d}"

        # Draw Android screenshot
        android_img = Image.new("RGB", ANDROID_SIZE, ANDROID_BG)
        android_draw = ImageDraw.Draw(android_img)
        # Add a subtle header bar
        android_draw.rectangle([0, 0, ANDROID_SIZE[0], int(ANDROID_SIZE[1] * 0.04)], fill=(33, 150, 243))

        for elem in pair_data["android_elements"]:
            _draw_element(android_draw, elem["bbox"], elem["style"], elem["text"], font)

        source_path = f"source/{pair_id}.png"
        android_img.save(os.path.join(output_dir, source_path))

        # Draw iOS screenshot
        ios_img = Image.new("RGB", IOS_SIZE, IOS_BG)
        ios_draw = ImageDraw.Draw(ios_img)
        # Add iOS-style status bar area
        ios_draw.rectangle([0, 0, IOS_SIZE[0], int(IOS_SIZE[1] * 0.035)], fill=(242, 242, 247))

        for elem in pair_data["ios_elements"]:
            _draw_element(ios_draw, elem["bbox"], elem["style"], elem["text"], font)

        target_path = f"target/{pair_id}.png"
        ios_img.save(os.path.join(output_dir, target_path))

        # Annotation entry
        test_idx = pair_data["test_element_idx"]
        annotations["pairs"].append({
            "id": pair_id,
            "source": {
                "image": source_path,
                "bbox": pair_data["android_elements"][test_idx]["bbox"],
                "platform": "android",
                "element_description": f"{pair_data['android_elements'][test_idx]['style']['type']}: {pair_data['android_elements'][test_idx]['text']}",
            },
            "target": {
                "image": target_path,
                "bbox": pair_data["ios_elements"][test_idx]["bbox"],
                "platform": "ios",
            },
            "tags": [pair_data["android_elements"][test_idx]["style"]["type"]],
        })

        print(f"  Created {pair_id}: {pair_data['android_elements'][test_idx]['text']} ({pair_data['android_elements'][test_idx]['style']['type']})")

    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nGenerated {num_pairs} pairs in {output_dir}")
    print(f"Annotations: {annotations_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cross-platform UI test pairs")
    parser.add_argument("--output-dir", default="data/sample", help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=10, help="Number of test pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    create_synthetic_data(args.output_dir, args.num_pairs, args.seed)
