"""Enhanced synthetic data generation for CrossMatch training (v2).

Improvements over v1:
  - Gradient and solid backgrounds with varied colors
  - Shadows and depth on buttons
  - Icon-like shapes (circles, triangles, squares) for non-text elements
  - Multiple font sizes and weights
  - Navigation bars, tab bars, headers
  - More realistic layout grids (vertically stacked forms, horizontal tab bars)
  - Varied element densities per screen
  - Platform-specific styling (Material Design vs iOS HIG colors)
  - Random noise/texture overlays for visual diversity

Usage:
    python -m cross_match.synthetic_v2 --output-dir data/cross_match_v2 --num-pairs 10000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter

ANDROID_SIZE = (1080, 1920)
IOS_SIZE = (1170, 2532)

# Expanded color palettes
ANDROID_PALETTES = [
    {"bg": (250, 250, 250), "primary": (33, 150, 243), "accent": (76, 175, 80), "danger": (244, 67, 54), "header": (33, 150, 243)},
    {"bg": (18, 18, 18), "primary": (100, 181, 246), "accent": (129, 199, 132), "danger": (239, 154, 154), "header": (30, 30, 30)},  # dark theme
    {"bg": (255, 248, 225), "primary": (255, 152, 0), "accent": (0, 150, 136), "danger": (211, 47, 47), "header": (255, 152, 0)},   # warm
    {"bg": (232, 234, 246), "primary": (63, 81, 181), "accent": (0, 188, 212), "danger": (233, 30, 99), "header": (63, 81, 181)},   # indigo
    {"bg": (243, 229, 245), "primary": (156, 39, 176), "accent": (255, 87, 34), "danger": (244, 67, 54), "header": (156, 39, 176)},  # purple
]

IOS_PALETTES = [
    {"bg": (242, 242, 247), "primary": (0, 122, 255), "accent": (52, 199, 89), "danger": (255, 59, 48), "header": (249, 249, 249)},
    {"bg": (0, 0, 0), "primary": (10, 132, 255), "accent": (48, 209, 88), "danger": (255, 69, 58), "header": (28, 28, 30)},         # dark
    {"bg": (255, 251, 235), "primary": (255, 149, 0), "accent": (0, 199, 190), "danger": (255, 59, 48), "header": (255, 251, 235)},  # warm
    {"bg": (235, 235, 245), "primary": (88, 86, 214), "accent": (0, 199, 190), "danger": (255, 59, 48), "header": (242, 242, 247)},  # indigo
    {"bg": (245, 235, 248), "primary": (175, 82, 222), "accent": (255, 149, 0), "danger": (255, 59, 48), "header": (245, 235, 248)}, # purple
]

BUTTON_TEXTS = [
    "Sign In", "Submit", "Continue", "Next", "Save", "Cancel", "OK", "Delete", "Send", "Search",
    "Back", "Forward", "Refresh", "Share", "Edit", "Done", "Apply", "Reset", "Close", "Open",
    "Accept", "Confirm", "Start", "Connect", "Enable", "Activate", "Verify", "Proceed",
    "Reject", "Remove", "Logout", "Clear", "Stop", "Disable", "Deny", "Block",
    "Add to Cart", "Buy Now", "Subscribe", "Download", "Upload", "Play", "Pause",
]

INPUT_LABELS = [
    "Email", "Password", "Username", "Search...", "Enter name", "Phone number",
    "Address", "City", "Zip code", "Company", "Message", "Note", "URL",
    "First name", "Last name", "Date of birth", "Card number",
]

NAV_LABELS = [
    "Settings", "Profile", "Home", "Notifications", "Messages", "Account",
    "Privacy", "Security", "About", "Help", "Feedback", "Language",
    "Dashboard", "Analytics", "Reports", "Users", "Orders", "Products",
]

ICON_SHAPES = ["circle", "square", "triangle", "star", "heart"]


def _try_load_fonts():
    fonts = {}
    paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
             "/System/Library/Fonts/Helvetica.ttc"]
    base_path = None
    for p in paths:
        if os.path.exists(p):
            base_path = p
            break

    if base_path:
        for size_name, size in [("sm", 24), ("md", 32), ("lg", 40), ("xl", 52)]:
            try:
                fonts[size_name] = ImageFont.truetype(base_path, size)
            except Exception:
                fonts[size_name] = ImageFont.load_default()
    else:
        default = ImageFont.load_default()
        for name in ["sm", "md", "lg", "xl"]:
            fonts[name] = default

    return fonts


def _draw_gradient_bg(img, color1, color2, vertical=True):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(h if vertical else w):
        ratio = i / (h if vertical else w)
        r = int(color1[0] + (color2[0] - color1[0]) * ratio)
        g = int(color1[1] + (color2[1] - color1[1]) * ratio)
        b = int(color1[2] + (color2[2] - color1[2]) * ratio)
        if vertical:
            draw.line([(0, i), (w, i)], fill=(r, g, b))
        else:
            draw.line([(i, 0), (i, h)], fill=(r, g, b))


def _draw_shadow_rect(draw, bbox, fill, radius=12, shadow_offset=4):
    x1, y1, x2, y2 = bbox
    # Shadow
    shadow_color = (0, 0, 0, 40)
    draw.rounded_rectangle([x1 + shadow_offset, y1 + shadow_offset, x2 + shadow_offset, y2 + shadow_offset],
                           radius=radius, fill=(180, 180, 180))
    # Main rect
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill)


def _draw_icon(draw, cx, cy, size, shape, color):
    r = size // 2
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        s = int(r * 0.8)
        draw.rounded_rectangle([cx - s, cy - s, cx + s, cy + s], radius=4, fill=color)
    elif shape == "triangle":
        points = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        draw.polygon(points, fill=color)
    elif shape == "star":
        # Simple 5-point star
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            rad = r if i % 2 == 0 else r * 0.4
            points.append((cx + int(rad * math.cos(angle)), cy - int(rad * math.sin(angle))))
        draw.polygon(points, fill=color)
    elif shape == "heart":
        # Approximate heart
        draw.ellipse([cx - r, cy - r // 2, cx, cy + r // 4], fill=color)
        draw.ellipse([cx, cy - r // 2, cx + r, cy + r // 4], fill=color)
        draw.polygon([(cx - r, cy), (cx, cy + r), (cx + r, cy)], fill=color)


def _generate_screen(screen_size, palette, rng, fonts, is_ios=False):
    w, h = screen_size
    elements = []

    # Background
    img = Image.new("RGB", screen_size, palette["bg"])
    draw = ImageDraw.Draw(img)

    # Optionally use gradient background
    if rng.random() < 0.3:
        bg2 = tuple(max(0, min(255, c + rng.randint(-30, 30))) for c in palette["bg"])
        _draw_gradient_bg(img, palette["bg"], bg2)
        draw = ImageDraw.Draw(img)

    # Header bar
    header_h = int(h * 0.06) if is_ios else int(h * 0.045)
    draw.rectangle([0, 0, w, header_h], fill=palette["header"])
    # Header title
    title = rng.choice(["Home", "Settings", "Profile", "Dashboard", "Messages", "Search"])
    font_lg = fonts["lg"]
    tw = draw.textlength(title, font=font_lg)
    draw.text(((w - tw) / 2, header_h // 2 - font_lg.size // 2), title, fill=(255, 255, 255), font=font_lg)

    # Content area
    y_cursor = header_h + int(h * 0.03)
    max_y = int(h * 0.88)
    num_sections = rng.randint(2, 5)

    for section in range(num_sections):
        if y_cursor >= max_y:
            break

        section_type = rng.choices(
            ["button_row", "input_field", "label", "icon_row", "card", "toggle_row"],
            weights=[3, 2, 2, 1, 2, 1]
        )[0]

        if section_type == "button_row":
            num_btns = rng.randint(1, 3)
            btn_w = int(w * rng.uniform(0.25, 0.55))
            btn_h = int(h * rng.uniform(0.028, 0.042))
            for b in range(num_btns):
                if y_cursor + btn_h >= max_y:
                    break
                x1 = rng.randint(int(w * 0.05), max(int(w * 0.05) + 1, w - btn_w - int(w * 0.05)))
                y1 = y_cursor
                color_key = rng.choice(["primary", "accent", "danger"])
                fill = palette[color_key]

                if rng.random() < 0.4:
                    _draw_shadow_rect(draw, [x1, y1, x1 + btn_w, y1 + btn_h], fill)
                else:
                    draw.rounded_rectangle([x1, y1, x1 + btn_w, y1 + btn_h], radius=rng.randint(6, 16), fill=fill)

                text = rng.choice(BUTTON_TEXTS)
                font = fonts[rng.choice(["md", "lg"])]
                tw = draw.textlength(text, font=font)
                tx = x1 + (btn_w - tw) / 2
                ty = y1 + (btn_h - font.size) / 2
                draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

                elements.append({"bbox": [x1, y1, x1 + btn_w, y1 + btn_h], "type": "button", "text": text})
                y_cursor = y1 + btn_h + int(h * rng.uniform(0.01, 0.03))

        elif section_type == "input_field":
            inp_w = int(w * rng.uniform(0.6, 0.88))
            inp_h = int(h * rng.uniform(0.028, 0.038))
            x1 = rng.randint(int(w * 0.05), max(int(w * 0.05) + 1, w - inp_w - int(w * 0.05)))
            y1 = y_cursor

            bg = (255, 255, 255) if palette["bg"][0] > 128 else (45, 45, 45)
            outline = (189, 189, 189) if palette["bg"][0] > 128 else (100, 100, 100)
            draw.rounded_rectangle([x1, y1, x1 + inp_w, y1 + inp_h], radius=8, fill=bg, outline=outline, width=2)

            text = rng.choice(INPUT_LABELS)
            font = fonts["md"]
            text_color = (150, 150, 150) if palette["bg"][0] > 128 else (120, 120, 120)
            draw.text((x1 + 12, y1 + (inp_h - font.size) / 2), text, fill=text_color, font=font)

            elements.append({"bbox": [x1, y1, x1 + inp_w, y1 + inp_h], "type": "input", "text": text})
            y_cursor = y1 + inp_h + int(h * rng.uniform(0.01, 0.03))

        elif section_type == "label":
            text = rng.choice(NAV_LABELS)
            font = fonts[rng.choice(["md", "lg", "xl"])]
            x1 = rng.randint(int(w * 0.05), int(w * 0.3))
            y1 = y_cursor
            text_color = (33, 33, 33) if palette["bg"][0] > 128 else (220, 220, 220)
            draw.text((x1, y1), text, fill=text_color, font=font)

            tw = draw.textlength(text, font=font)
            elements.append({"bbox": [x1, y1, x1 + int(tw), y1 + font.size], "type": "label", "text": text})
            y_cursor = y1 + font.size + int(h * rng.uniform(0.01, 0.025))

        elif section_type == "icon_row":
            num_icons = rng.randint(3, 6)
            icon_size = rng.randint(int(h * 0.02), int(h * 0.035))
            spacing = (w - int(w * 0.1)) // num_icons
            y1 = y_cursor
            for ic in range(num_icons):
                cx = int(w * 0.05) + ic * spacing + spacing // 2
                cy = y1 + icon_size
                shape = rng.choice(ICON_SHAPES)
                color = palette[rng.choice(["primary", "accent", "danger"])]
                _draw_icon(draw, cx, cy, icon_size, shape, color)

                elements.append({
                    "bbox": [cx - icon_size // 2, cy - icon_size // 2, cx + icon_size // 2, cy + icon_size // 2],
                    "type": "icon", "text": shape,
                })
            y_cursor = y1 + icon_size * 2 + int(h * 0.02)

        elif section_type == "card":
            card_w = int(w * rng.uniform(0.7, 0.9))
            card_h = int(h * rng.uniform(0.06, 0.1))
            x1 = (w - card_w) // 2
            y1 = y_cursor
            card_bg = (255, 255, 255) if palette["bg"][0] > 128 else (40, 40, 40)
            _draw_shadow_rect(draw, [x1, y1, x1 + card_w, y1 + card_h], card_bg, radius=16, shadow_offset=6)

            # Card title + subtitle
            font = fonts["lg"]
            title = rng.choice(NAV_LABELS)
            text_color = (33, 33, 33) if palette["bg"][0] > 128 else (220, 220, 220)
            draw.text((x1 + 20, y1 + 12), title, fill=text_color, font=font)

            subtitle = rng.choice(["Tap to view", "Updated recently", "3 new items", "Active", "Pending"])
            sub_color = (120, 120, 120)
            draw.text((x1 + 20, y1 + 12 + font.size + 4), subtitle, fill=sub_color, font=fonts["sm"])

            elements.append({"bbox": [x1, y1, x1 + card_w, y1 + card_h], "type": "card", "text": title})
            y_cursor = y1 + card_h + int(h * rng.uniform(0.015, 0.03))

        elif section_type == "toggle_row":
            x1 = int(w * 0.05)
            y1 = y_cursor
            label_text = rng.choice(NAV_LABELS)
            font = fonts["md"]
            text_color = (33, 33, 33) if palette["bg"][0] > 128 else (220, 220, 220)
            draw.text((x1, y1), label_text, fill=text_color, font=font)

            # Toggle switch
            tog_w = int(w * 0.1)
            tog_h = int(h * 0.018)
            tog_x = w - int(w * 0.05) - tog_w
            is_on = rng.random() < 0.5
            tog_color = palette["primary"] if is_on else (180, 180, 180)
            draw.rounded_rectangle([tog_x, y1, tog_x + tog_w, y1 + tog_h], radius=tog_h // 2, fill=tog_color)
            circle_r = tog_h // 2 - 3
            circle_x = tog_x + tog_w - circle_r - 5 if is_on else tog_x + circle_r + 5
            draw.ellipse([circle_x - circle_r, y1 + 3, circle_x + circle_r, y1 + tog_h - 3], fill=(255, 255, 255))

            elements.append({"bbox": [tog_x, y1, tog_x + tog_w, y1 + tog_h], "type": "toggle", "text": label_text})
            y_cursor = y1 + tog_h + int(h * rng.uniform(0.015, 0.03))

    # Bottom nav bar (50% of screens)
    if rng.random() < 0.5:
        nav_h = int(h * 0.06)
        nav_y = h - nav_h
        nav_bg = palette["header"] if not is_ios else (249, 249, 249) if palette["bg"][0] > 128 else (28, 28, 30)
        draw.rectangle([0, nav_y, w, h], fill=nav_bg)
        nav_items = rng.sample(NAV_LABELS[:8], min(5, len(NAV_LABELS[:8])))
        spacing = w // len(nav_items)
        for i, label in enumerate(nav_items):
            cx = i * spacing + spacing // 2
            cy = nav_y + nav_h // 2
            font = fonts["sm"]
            tw = draw.textlength(label, font=font)
            color = palette["primary"] if i == 0 else (150, 150, 150)
            draw.text((cx - tw / 2, cy - font.size / 2), label, fill=color, font=font)

    # Optional noise/texture (10% of screens)
    if rng.random() < 0.1:
        noise = Image.effect_noise(screen_size, 15)
        img = Image.blend(img, noise.convert("RGB"), 0.03)

    return img, elements


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

    fonts = _try_load_fonts()
    rng = random.Random(seed)
    annotations = {"pairs": []}

    for i in range(num_pairs):
        pair_id = "pair_{:05d}".format(i)

        # Pick palettes (same index for visual consistency within a pair)
        pal_idx = rng.randint(0, len(ANDROID_PALETTES) - 1)
        android_pal = ANDROID_PALETTES[pal_idx]
        ios_pal = IOS_PALETTES[pal_idx]

        # Generate Android screen
        android_img, android_elems = _generate_screen(ANDROID_SIZE, android_pal, rng, fonts, is_ios=False)

        # Generate iOS screen with same elements at scaled positions
        ios_img = Image.new("RGB", IOS_SIZE, ios_pal["bg"])
        ios_draw = ImageDraw.Draw(ios_img)

        # Apply same gradient if applicable
        if rng.random() < 0.3:
            bg2 = tuple(max(0, min(255, c + rng.randint(-30, 30))) for c in ios_pal["bg"])
            _draw_gradient_bg(ios_img, ios_pal["bg"], bg2)
            ios_draw = ImageDraw.Draw(ios_img)

        jitter_rng = random.Random(seed + i + 50000)
        ios_elems = []
        for elem in android_elems:
            scaled_bbox = _scale_bbox(elem["bbox"], ANDROID_SIZE, IOS_SIZE, jitter_rng)
            ios_elems.append({**elem, "bbox": scaled_bbox})

        # Re-draw elements on iOS image with iOS palette colors
        for elem in ios_elems:
            x1, y1, x2, y2 = elem["bbox"]
            etype = elem["type"]
            if etype == "button":
                color = ios_pal[rng.choice(["primary", "accent", "danger"])]
                ios_draw.rounded_rectangle([x1, y1, x2, y2], radius=12, fill=color)
                font = fonts[rng.choice(["md", "lg"])]
                tw = ios_draw.textlength(elem["text"], font=font)
                ios_draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - font.size) / 2),
                              elem["text"], fill=(255, 255, 255), font=font)
            elif etype == "input":
                bg = (255, 255, 255) if ios_pal["bg"][0] > 128 else (45, 45, 45)
                ios_draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=bg, outline=(189, 189, 189), width=2)
                ios_draw.text((x1 + 12, y1 + (y2 - y1 - fonts["md"].size) / 2),
                              elem["text"], fill=(150, 150, 150), font=fonts["md"])
            elif etype == "label":
                color = (33, 33, 33) if ios_pal["bg"][0] > 128 else (220, 220, 220)
                ios_draw.text((x1, y1), elem["text"], fill=color, font=fonts["md"])
            elif etype == "icon":
                color = ios_pal[rng.choice(["primary", "accent"])]
                sz = max(10, x2 - x1)
                _draw_icon(ios_draw, (x1 + x2) // 2, (y1 + y2) // 2, sz, elem["text"], color)
            elif etype == "card":
                card_bg = (255, 255, 255) if ios_pal["bg"][0] > 128 else (40, 40, 40)
                ios_draw.rounded_rectangle([x1, y1, x2, y2], radius=16, fill=card_bg)
                color = (33, 33, 33) if ios_pal["bg"][0] > 128 else (220, 220, 220)
                ios_draw.text((x1 + 20, y1 + 12), elem["text"], fill=color, font=fonts["lg"])
            elif etype == "toggle":
                is_on = rng.random() < 0.5
                color = ios_pal["primary"] if is_on else (180, 180, 180)
                th = y2 - y1
                ios_draw.rounded_rectangle([x1, y1, x2, y2], radius=max(1, th // 2), fill=color)

        android_img.save(os.path.join(source_dir, "{}.png".format(pair_id)))
        ios_img.save(os.path.join(target_dir, "{}.png".format(pair_id)))

        # Generate actions
        actions = []
        clickable = [e for e in android_elems if e["type"] in ("button", "input", "card", "icon", "toggle")]
        if clickable:
            num_clicks = min(rng.randint(1, 3), len(clickable))
            for elem in rng.sample(clickable, num_clicks):
                idx = android_elems.index(elem)
                src_center = _bbox_center(android_elems[idx]["bbox"])
                tgt_center = _bbox_center(ios_elems[idx]["bbox"])
                actions.append({
                    "type": "click",
                    "source_coords": {"at": src_center},
                    "target_coords": {"at": tgt_center},
                })

        # Scroll actions
        num_scrolls = rng.randint(0, 2)
        for _ in range(num_scrolls):
            src_x = rng.randint(int(ANDROID_SIZE[0] * 0.3), int(ANDROID_SIZE[0] * 0.7))
            src_from_y = rng.randint(int(ANDROID_SIZE[1] * 0.3), int(ANDROID_SIZE[1] * 0.7))
            scroll_dy = rng.randint(int(ANDROID_SIZE[1] * 0.1), int(ANDROID_SIZE[1] * 0.3))
            src_to_y = max(0, min(ANDROID_SIZE[1], src_from_y + rng.choice([-1, 1]) * scroll_dy))

            sx = IOS_SIZE[0] / ANDROID_SIZE[0]
            sy = IOS_SIZE[1] / ANDROID_SIZE[1]
            actions.append({
                "type": "scroll",
                "source_coords": {"from_arg": [src_x, src_from_y], "to_arg": [src_x, src_to_y]},
                "target_coords": {
                    "from_arg": [int(src_x * sx) + jitter_rng.randint(-10, 10),
                                 int(src_from_y * sy) + jitter_rng.randint(-10, 10)],
                    "to_arg": [int(src_x * sx) + jitter_rng.randint(-10, 10),
                               int(src_to_y * sy) + jitter_rng.randint(-10, 10)],
                },
            })

        if not actions:
            # Fallback: at least 1 click somewhere
            src_center = _bbox_center(android_elems[0]["bbox"]) if android_elems else [ANDROID_SIZE[0] // 2, ANDROID_SIZE[1] // 2]
            tgt_center = _bbox_center(ios_elems[0]["bbox"]) if ios_elems else [IOS_SIZE[0] // 2, IOS_SIZE[1] // 2]
            actions.append({"type": "click", "source_coords": {"at": src_center}, "target_coords": {"at": tgt_center}})

        annotations["pairs"].append({
            "id": pair_id,
            "source": {"image": "source/{}.png".format(pair_id), "platform": "android", "size": list(ANDROID_SIZE)},
            "target": {"image": "target/{}.png".format(pair_id), "platform": "ios", "size": list(IOS_SIZE)},
            "actions": actions,
        })

        if (i + 1) % 500 == 0:
            print("  Generated {}/{} pairs".format(i + 1, num_pairs))

    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f)

    total_actions = sum(len(p["actions"]) for p in annotations["pairs"])
    print("\nGenerated {} pairs with {} total actions in {}".format(num_pairs, total_actions, output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/cross_match_v2")
    parser.add_argument("--num-pairs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.num_pairs, args.seed)
