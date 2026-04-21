"""Enhanced synthetic data generation for CrossMatch training (v4).

Changes from v3:
  - Multi-resolution: 4 Android + 4 iOS real-world device sizes, any-to-any pairing
  - Resolution-scaled fonts: text stays proportionally consistent across screen sizes
  - New icon-only element types: FAB, icon_button, icon_nav_bar
  - Bumped icon_row weight from 1 to 2
  - Target Y distribution tracked alongside source in annotations stats
  - Resolution distribution stats in annotations

Usage:
    python -m cross_match.synthetic_v4 --output-dir data/cross_match_v4 --num-pairs 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --- Device resolution pools (4 each) ---
ANDROID_DEVICES = [
    {"name": "Pixel 2 / Galaxy S7",     "size": (1080, 1920)},
    {"name": "Pixel 5 / Galaxy S21",     "size": (1080, 2400)},
    {"name": "Galaxy S8 / Pixel XL",     "size": (1440, 2560)},
    {"name": "Budget modern",            "size": (720, 1600)},
]

IOS_DEVICES = [
    {"name": "iPhone SE / 6 / 7 / 8",   "size": (750, 1334)},
    {"name": "iPhone XR / 11",           "size": (828, 1792)},
    {"name": "iPhone 12 / 13 / 14",      "size": (1170, 2532)},
    {"name": "iPhone 14 Pro Max / 15+",  "size": (1290, 2796)},
]

# Reference height for font scaling
_REF_HEIGHT = 1920

# Color palettes
PALETTES = [
    {"bg": (250, 250, 250), "primary": (33, 150, 243), "accent": (76, 175, 80), "danger": (244, 67, 54), "header": (33, 150, 243)},
    {"bg": (18, 18, 18), "primary": (100, 181, 246), "accent": (129, 199, 132), "danger": (239, 154, 154), "header": (30, 30, 30)},
    {"bg": (255, 248, 225), "primary": (255, 152, 0), "accent": (0, 150, 136), "danger": (211, 47, 47), "header": (255, 152, 0)},
    {"bg": (232, 234, 246), "primary": (63, 81, 181), "accent": (0, 188, 212), "danger": (233, 30, 99), "header": (63, 81, 181)},
    {"bg": (243, 229, 245), "primary": (156, 39, 176), "accent": (255, 87, 34), "danger": (244, 67, 54), "header": (156, 39, 176)},
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

# v4: added fab, icon_button, icon_nav_bar; bumped icon_row weight to 2
SECTION_TYPES = [
    "button_row", "input_field", "label", "icon_row", "card", "toggle_row",
    "fab", "icon_button", "icon_nav_bar",
]
SECTION_WEIGHTS = [3, 2, 2, 2, 2, 1, 1, 2, 1]


def _load_fonts_for_height(screen_height):
    """Load fonts scaled proportionally to screen height."""
    scale = screen_height / _REF_HEIGHT
    fonts = {}
    paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
             "/System/Library/Fonts/Helvetica.ttc"]
    base_path = None
    for p in paths:
        if os.path.exists(p):
            base_path = p
            break

    base_sizes = {"sm": 24, "md": 32, "lg": 40, "xl": 52}
    if base_path:
        for name, base_sz in base_sizes.items():
            sz = max(10, int(base_sz * scale))
            try:
                fonts[name] = ImageFont.truetype(base_path, sz)
            except Exception:
                fonts[name] = ImageFont.load_default()
    else:
        default = ImageFont.load_default()
        for name in base_sizes:
            fonts[name] = default

    return fonts


# --- Font cache to avoid reloading for same height ---
_font_cache = {}


def _get_fonts(screen_height):
    if screen_height not in _font_cache:
        _font_cache[screen_height] = _load_fonts_for_height(screen_height)
    return _font_cache[screen_height]


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
    draw.rounded_rectangle([x1 + shadow_offset, y1 + shadow_offset, x2 + shadow_offset, y2 + shadow_offset],
                           radius=radius, fill=(180, 180, 180))
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill)


def _draw_icon(draw, cx, cy, size, shape, color):
    r = size // 2
    if r < 2:
        r = 2
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        s = int(r * 0.8)
        draw.rounded_rectangle([cx - s, cy - s, cx + s, cy + s], radius=4, fill=color)
    elif shape == "triangle":
        points = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        draw.polygon(points, fill=color)
    elif shape == "star":
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            rad = r if i % 2 == 0 else r * 0.4
            points.append((cx + int(rad * math.cos(angle)), cy - int(rad * math.sin(angle))))
        draw.polygon(points, fill=color)
    elif shape == "heart":
        draw.ellipse([cx - r, cy - r // 2, cx, cy + r // 4], fill=color)
        draw.ellipse([cx, cy - r // 2, cx + r, cy + r // 4], fill=color)
        draw.polygon([(cx - r, cy), (cx, cy + r), (cx + r, cy)], fill=color)


def _draw_hamburger(draw, cx, cy, size, color):
    """Draw a 3-line hamburger menu icon."""
    r = size // 2
    bar_h = max(2, size // 8)
    gap = max(3, size // 5)
    for offset in [-gap, 0, gap]:
        y = cy + offset
        draw.rounded_rectangle([cx - r, y - bar_h // 2, cx + r, y + bar_h // 2],
                               radius=max(1, bar_h // 2), fill=color)


def _draw_plus(draw, cx, cy, size, color):
    """Draw a + icon for FAB."""
    r = size // 2
    bar = max(2, size // 6)
    draw.rounded_rectangle([cx - r, cy - bar, cx + r, cy + bar], radius=2, fill=color)
    draw.rounded_rectangle([cx - bar, cy - r, cx + bar, cy + r], radius=2, fill=color)


def _draw_arrow_back(draw, cx, cy, size, color):
    """Draw a < back arrow icon."""
    r = size // 2
    points = [(cx + r // 2, cy - r), (cx - r // 2, cy), (cx + r // 2, cy + r)]
    draw.line(points, fill=color, width=max(2, size // 8))


def _draw_close_x(draw, cx, cy, size, color):
    """Draw an X close icon."""
    r = size // 3
    lw = max(2, size // 8)
    draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=color, width=lw)
    draw.line([(cx - r, cy + r), (cx + r, cy - r)], fill=color, width=lw)


def _draw_bell(draw, cx, cy, size, color):
    """Draw a simplified bell/notification icon."""
    r = size // 2
    # Bell body (top arc + trapezoid)
    draw.pieslice([cx - r, cy - r, cx + r, cy + r // 2], start=180, end=0, fill=color)
    draw.rectangle([cx - r, cy - r // 4, cx + r, cy + r // 2], fill=color)
    # Bell bottom rim
    draw.rounded_rectangle([cx - r - 2, cy + r // 2 - 2, cx + r + 2, cy + r // 2 + 4],
                           radius=2, fill=color)
    # Clapper
    draw.ellipse([cx - 3, cy + r // 2 + 3, cx + 3, cy + r // 2 + 9], fill=color)


ICON_BUTTON_DRAWERS = [
    ("hamburger", _draw_hamburger),
    ("back_arrow", _draw_arrow_back),
    ("close_x", _draw_close_x),
    ("bell", _draw_bell),
]

# Shapes for icon-only nav bar (no text)
ICON_NAV_SHAPES = ["circle", "square", "triangle", "star", "heart"]


def _draw_element_in_slot(draw, w, h, slot_y_center, palette, rng, fonts, section_type):
    """Draw a single UI element centered at slot_y_center. Returns element dict or None."""

    if section_type == "button_row":
        btn_w = int(w * rng.uniform(0.25, 0.55))
        btn_h = int(h * rng.uniform(0.028, 0.042))
        x1 = rng.randint(int(w * 0.05), max(int(w * 0.05) + 1, w - btn_w - int(w * 0.05)))
        y1 = slot_y_center - btn_h // 2
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
        return {"bbox": [x1, y1, x1 + btn_w, y1 + btn_h], "type": "button", "text": text}

    elif section_type == "input_field":
        inp_w = int(w * rng.uniform(0.6, 0.88))
        inp_h = int(h * rng.uniform(0.028, 0.038))
        x1 = rng.randint(int(w * 0.05), max(int(w * 0.05) + 1, w - inp_w - int(w * 0.05)))
        y1 = slot_y_center - inp_h // 2

        bg = (255, 255, 255) if palette["bg"][0] > 128 else (45, 45, 45)
        outline = (189, 189, 189) if palette["bg"][0] > 128 else (100, 100, 100)
        draw.rounded_rectangle([x1, y1, x1 + inp_w, y1 + inp_h], radius=8, fill=bg, outline=outline, width=2)

        text = rng.choice(INPUT_LABELS)
        font = fonts["md"]
        text_color = (150, 150, 150) if palette["bg"][0] > 128 else (120, 120, 120)
        draw.text((x1 + 12, y1 + (inp_h - font.size) / 2), text, fill=text_color, font=font)
        return {"bbox": [x1, y1, x1 + inp_w, y1 + inp_h], "type": "input", "text": text}

    elif section_type == "label":
        text = rng.choice(NAV_LABELS)
        font = fonts[rng.choice(["md", "lg", "xl"])]
        x1 = rng.randint(int(w * 0.05), int(w * 0.3))
        y1 = slot_y_center - font.size // 2
        text_color = (33, 33, 33) if palette["bg"][0] > 128 else (220, 220, 220)
        draw.text((x1, y1), text, fill=text_color, font=font)

        tw = draw.textlength(text, font=font)
        return {"bbox": [x1, y1, x1 + int(tw), y1 + font.size], "type": "label", "text": text}

    elif section_type == "icon_row":
        num_icons = rng.randint(3, 6)
        icon_size = rng.randint(int(h * 0.02), int(h * 0.035))
        spacing = (w - int(w * 0.1)) // num_icons
        pick = rng.randint(0, num_icons - 1)
        result = None
        for ic in range(num_icons):
            cx = int(w * 0.05) + ic * spacing + spacing // 2
            cy = slot_y_center
            shape = rng.choice(ICON_SHAPES)
            color = palette[rng.choice(["primary", "accent", "danger"])]
            _draw_icon(draw, cx, cy, icon_size, shape, color)
            if ic == pick:
                result = {
                    "bbox": [cx - icon_size // 2, cy - icon_size // 2, cx + icon_size // 2, cy + icon_size // 2],
                    "type": "icon", "text": shape,
                }
        return result

    elif section_type == "card":
        card_w = int(w * rng.uniform(0.7, 0.9))
        card_h = int(h * rng.uniform(0.06, 0.1))
        x1 = (w - card_w) // 2
        y1 = slot_y_center - card_h // 2
        card_bg = (255, 255, 255) if palette["bg"][0] > 128 else (40, 40, 40)
        _draw_shadow_rect(draw, [x1, y1, x1 + card_w, y1 + card_h], card_bg, radius=16, shadow_offset=6)

        font = fonts["lg"]
        title = rng.choice(NAV_LABELS)
        text_color = (33, 33, 33) if palette["bg"][0] > 128 else (220, 220, 220)
        draw.text((x1 + 20, y1 + 12), title, fill=text_color, font=font)

        subtitle = rng.choice(["Tap to view", "Updated recently", "3 new items", "Active", "Pending"])
        draw.text((x1 + 20, y1 + 12 + font.size + 4), subtitle, fill=(120, 120, 120), font=fonts["sm"])
        return {"bbox": [x1, y1, x1 + card_w, y1 + card_h], "type": "card", "text": title}

    elif section_type == "toggle_row":
        x1 = int(w * 0.05)
        y1 = slot_y_center - int(h * 0.009)
        label_text = rng.choice(NAV_LABELS)
        font = fonts["md"]
        text_color = (33, 33, 33) if palette["bg"][0] > 128 else (220, 220, 220)
        draw.text((x1, y1), label_text, fill=text_color, font=font)

        tog_w = int(w * 0.1)
        tog_h = int(h * 0.018)
        tog_x = w - int(w * 0.05) - tog_w
        is_on = rng.random() < 0.5
        tog_color = palette["primary"] if is_on else (180, 180, 180)
        draw.rounded_rectangle([tog_x, y1, tog_x + tog_w, y1 + tog_h], radius=max(1, tog_h // 2), fill=tog_color)
        circle_r = tog_h // 2 - 3
        circle_x = tog_x + tog_w - circle_r - 5 if is_on else tog_x + circle_r + 5
        draw.ellipse([circle_x - circle_r, y1 + 3, circle_x + circle_r, y1 + tog_h - 3], fill=(255, 255, 255))
        return {"bbox": [tog_x, y1, tog_x + tog_w, y1 + tog_h], "type": "toggle", "text": label_text}

    elif section_type == "fab":
        # Floating action button — round colored circle with + icon
        fab_size = int(h * rng.uniform(0.032, 0.045))
        # FABs typically sit on the right side
        cx = rng.randint(int(w * 0.7), int(w * 0.9))
        cy = slot_y_center
        color = palette[rng.choice(["primary", "accent"])]
        r = fab_size // 2
        # Shadow
        draw.ellipse([cx - r + 3, cy - r + 3, cx + r + 3, cy + r + 3], fill=(180, 180, 180))
        # Circle
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        # Plus icon
        _draw_plus(draw, cx, cy, int(fab_size * 0.5), (255, 255, 255))
        return {
            "bbox": [cx - r, cy - r, cx + r, cy + r],
            "type": "fab", "text": "fab_plus",
        }

    elif section_type == "icon_button":
        # Standalone icon button (hamburger, back arrow, close X, bell)
        btn_size = int(h * rng.uniform(0.025, 0.038))
        # Can appear anywhere horizontally — left for hamburger/back, right for close/bell
        icon_name, draw_fn = rng.choice(ICON_BUTTON_DRAWERS)
        if icon_name in ("hamburger", "back_arrow"):
            cx = rng.randint(int(w * 0.05), int(w * 0.15))
        elif icon_name in ("close_x", "bell"):
            cx = rng.randint(int(w * 0.85), int(w * 0.95))
        else:
            cx = rng.randint(int(w * 0.05), int(w * 0.95))
        cy = slot_y_center

        # Optional circular bg (50% of the time)
        r = btn_size // 2
        if rng.random() < 0.5:
            bg_color = (240, 240, 240) if palette["bg"][0] > 128 else (50, 50, 50)
            draw.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4], fill=bg_color)

        icon_color = palette[rng.choice(["primary", "accent"])]
        draw_fn(draw, cx, cy, btn_size, icon_color)
        return {
            "bbox": [cx - r, cy - r, cx + r, cy + r],
            "type": "icon_button", "text": icon_name,
        }

    elif section_type == "icon_nav_bar":
        # Horizontal row of icon-only nav items (no text labels)
        num_items = rng.randint(3, 5)
        icon_size = rng.randint(int(h * 0.018), int(h * 0.028))
        total_w = w - int(w * 0.1)
        spacing = total_w // num_items
        pick = rng.randint(0, num_items - 1)
        result = None

        # Light bg strip behind the icons
        strip_h = int(icon_size * 2.5)
        strip_y = slot_y_center - strip_h // 2
        strip_bg = (245, 245, 245) if palette["bg"][0] > 128 else (35, 35, 35)
        draw.rectangle([0, strip_y, w, strip_y + strip_h], fill=strip_bg)

        for i in range(num_items):
            cx = int(w * 0.05) + i * spacing + spacing // 2
            cy = slot_y_center
            shape = rng.choice(ICON_NAV_SHAPES)
            color = palette["primary"] if i == pick else (170, 170, 170)
            _draw_icon(draw, cx, cy, icon_size, shape, color)
            if i == pick:
                result = {
                    "bbox": [cx - icon_size // 2, cy - icon_size // 2, cx + icon_size // 2, cy + icon_size // 2],
                    "type": "icon_nav", "text": shape,
                }
        return result

    return None


def _generate_screen(screen_size, palette, rng, fonts):
    """Generate a screen with elements distributed uniformly across the Y range."""
    w, h = screen_size
    elements = []

    img = Image.new("RGB", screen_size, palette["bg"])
    draw = ImageDraw.Draw(img)

    # Gradient background (30%)
    if rng.random() < 0.3:
        bg2 = tuple(max(0, min(255, c + rng.randint(-30, 30))) for c in palette["bg"])
        _draw_gradient_bg(img, palette["bg"], bg2)
        draw = ImageDraw.Draw(img)

    # Header bar (decorative)
    header_h = int(h * 0.05)
    draw.rectangle([0, 0, w, header_h], fill=palette["header"])
    title = rng.choice(["Home", "Settings", "Profile", "Dashboard", "Messages", "Search"])
    font_lg = fonts["lg"]
    tw = draw.textlength(title, font=font_lg)
    draw.text(((w - tw) / 2, header_h // 2 - font_lg.size // 2), title, fill=(255, 255, 255), font=font_lg)

    # --- Slot-based layout for uniform Y distribution ---
    y_min = int(h * 0.06)
    y_max = int(h * 0.90)

    num_slots = rng.randint(6, 12)
    slot_height = (y_max - y_min) / num_slots

    for slot_idx in range(num_slots):
        slot_top = y_min + slot_idx * slot_height
        slot_bottom = slot_top + slot_height
        slot_y_center = int(rng.uniform(slot_top + slot_height * 0.2, slot_bottom - slot_height * 0.2))

        section_type = rng.choices(SECTION_TYPES, weights=SECTION_WEIGHTS)[0]
        elem = _draw_element_in_slot(draw, w, h, slot_y_center, palette, rng, fonts, section_type)
        if elem is not None:
            elements.append(elem)

    # Bottom nav bar (50% of screens) — one item clickable
    if rng.random() < 0.5:
        nav_h = int(h * 0.06)
        nav_y = h - nav_h
        nav_bg = palette["header"]
        draw.rectangle([0, nav_y, w, h], fill=nav_bg)
        nav_items = rng.sample(NAV_LABELS[:8], min(5, len(NAV_LABELS[:8])))
        spacing = w // len(nav_items)
        pick_nav = rng.randint(0, len(nav_items) - 1)
        for i, label in enumerate(nav_items):
            cx = i * spacing + spacing // 2
            cy = nav_y + nav_h // 2
            font = fonts["sm"]
            tw_nav = draw.textlength(label, font=font)
            color = palette["primary"] if i == 0 else (150, 150, 150)
            draw.text((cx - tw_nav / 2, cy - font.size / 2), label, fill=color, font=font)

            if i == pick_nav:
                elem_w = int(tw_nav) + 20
                elem_h = nav_h - 10
                elements.append({
                    "bbox": [int(cx - elem_w / 2), nav_y + 5, int(cx + elem_w / 2), nav_y + 5 + elem_h],
                    "type": "nav_item",
                    "text": label,
                })

    # Noise overlay (10%)
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


def _redraw_element(draw, elem, palette, rng, fonts):
    """Redraw a single element on the target image."""
    x1, y1, x2, y2 = elem["bbox"]
    etype = elem["type"]

    if etype == "button":
        color = palette[rng.choice(["primary", "accent", "danger"])]
        draw.rounded_rectangle([x1, y1, x2, y2], radius=12, fill=color)
        font = fonts[rng.choice(["md", "lg"])]
        tw = draw.textlength(elem["text"], font=font)
        draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - font.size) / 2),
                  elem["text"], fill=(255, 255, 255), font=font)

    elif etype == "input":
        bg = (255, 255, 255) if palette["bg"][0] > 128 else (45, 45, 45)
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=bg, outline=(189, 189, 189), width=2)
        draw.text((x1 + 12, y1 + (y2 - y1 - fonts["md"].size) / 2),
                  elem["text"], fill=(150, 150, 150), font=fonts["md"])

    elif etype == "label":
        color = (33, 33, 33) if palette["bg"][0] > 128 else (220, 220, 220)
        draw.text((x1, y1), elem["text"], fill=color, font=fonts["md"])

    elif etype == "icon":
        color = palette[rng.choice(["primary", "accent"])]
        sz = max(10, x2 - x1)
        _draw_icon(draw, (x1 + x2) // 2, (y1 + y2) // 2, sz, elem["text"], color)

    elif etype == "card":
        card_bg = (255, 255, 255) if palette["bg"][0] > 128 else (40, 40, 40)
        draw.rounded_rectangle([x1, y1, x2, y2], radius=16, fill=card_bg)
        color = (33, 33, 33) if palette["bg"][0] > 128 else (220, 220, 220)
        draw.text((x1 + 20, y1 + 12), elem["text"], fill=color, font=fonts["lg"])

    elif etype == "toggle":
        is_on = rng.random() < 0.5
        color = palette["primary"] if is_on else (180, 180, 180)
        th = y2 - y1
        draw.rounded_rectangle([x1, y1, x2, y2], radius=max(1, th // 2), fill=color)

    elif etype == "nav_item":
        font = fonts["sm"]
        tw = draw.textlength(elem["text"], font=font)
        color = palette[rng.choice(["primary", "accent"])]
        draw.text((x1 + ((x2 - x1) - tw) / 2, y1 + ((y2 - y1) - font.size) / 2),
                  elem["text"], fill=color, font=font)

    elif etype == "fab":
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = (x2 - x1) // 2
        color = palette[rng.choice(["primary", "accent"])]
        draw.ellipse([cx - r + 3, cy - r + 3, cx + r + 3, cy + r + 3], fill=(180, 180, 180))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        _draw_plus(draw, cx, cy, int(r * 0.9), (255, 255, 255))

    elif etype == "icon_button":
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        btn_size = x2 - x1
        icon_name = elem["text"]
        drawer_map = dict(ICON_BUTTON_DRAWERS)
        if icon_name in drawer_map:
            if rng.random() < 0.5:
                r = btn_size // 2
                bg_color = (240, 240, 240) if palette["bg"][0] > 128 else (50, 50, 50)
                draw.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4], fill=bg_color)
            icon_color = palette[rng.choice(["primary", "accent"])]
            drawer_map[icon_name](draw, cx, cy, btn_size, icon_color)

    elif etype == "icon_nav":
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        sz = max(10, x2 - x1)
        color = palette["primary"]
        _draw_icon(draw, cx, cy, sz, elem["text"], color)


# All clickable element types
CLICKABLE_TYPES = {"button", "input", "card", "icon", "toggle", "nav_item",
                   "fab", "icon_button", "icon_nav"}


def _compute_distribution_stats(annotations):
    """Compute Y-distribution stats for source and target, clicks and scrolls."""
    src_click_ys = []
    tgt_click_ys = []
    src_scroll_ys = []
    tgt_scroll_ys = []

    for pair in annotations["pairs"]:
        src_h = pair["source"]["size"][1]
        tgt_h = pair["target"]["size"][1]
        for action in pair["actions"]:
            if action["type"] == "click":
                src_click_ys.append(action["source_coords"]["at"][1] / src_h)
                tgt_click_ys.append(action["target_coords"]["at"][1] / tgt_h)
            elif action["type"] == "scroll":
                src_scroll_ys.append(action["source_coords"]["from_arg"][1] / src_h)
                src_scroll_ys.append(action["source_coords"]["to_arg"][1] / src_h)
                tgt_scroll_ys.append(action["target_coords"]["from_arg"][1] / tgt_h)
                tgt_scroll_ys.append(action["target_coords"]["to_arg"][1] / tgt_h)

    def quartile_pcts(values):
        if not values:
            return {"q1_0_25": 0, "q2_25_50": 0, "q3_50_75": 0, "q4_75_100": 0}
        total = len(values)
        return {
            "q1_0_25": round(sum(1 for v in values if v < 0.25) / total * 100, 1),
            "q2_25_50": round(sum(1 for v in values if 0.25 <= v < 0.50) / total * 100, 1),
            "q3_50_75": round(sum(1 for v in values if 0.50 <= v < 0.75) / total * 100, 1),
            "q4_75_100": round(sum(1 for v in values if 0.75 <= v) / total * 100, 1),
        }

    def basic_stats(values):
        if not values:
            return {}
        values_sorted = sorted(values)
        n = len(values_sorted)
        return {
            "count": n,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "mean": round(sum(values) / n, 4),
            "median": round(values_sorted[n // 2], 4),
            "p10": round(values_sorted[int(n * 0.1)], 4),
            "p90": round(values_sorted[int(n * 0.9)], 4),
        }

    total_actions = sum(len(p["actions"]) for p in annotations["pairs"])
    total_clicks = sum(1 for p in annotations["pairs"] for a in p["actions"] if a["type"] == "click")
    total_scrolls = sum(1 for p in annotations["pairs"] for a in p["actions"] if a["type"] == "scroll")

    # Resolution distribution
    src_res_counts = {}
    tgt_res_counts = {}
    for pair in annotations["pairs"]:
        sk = "{}x{}".format(*pair["source"]["size"])
        tk = "{}x{}".format(*pair["target"]["size"])
        src_res_counts[sk] = src_res_counts.get(sk, 0) + 1
        tgt_res_counts[tk] = tgt_res_counts.get(tk, 0) + 1

    # Element type distribution
    elem_type_counts = {}
    for pair in annotations["pairs"]:
        for action in pair["actions"]:
            if action["type"] == "click":
                etype = action.get("element_type", "unknown")
                elem_type_counts[etype] = elem_type_counts.get(etype, 0) + 1

    return {
        "total_pairs": len(annotations["pairs"]),
        "total_actions": total_actions,
        "total_clicks": total_clicks,
        "total_scrolls": total_scrolls,
        "source_click_y_distribution": {
            "description": "Normalized Y (0=top, 1=bottom) of source click coordinates",
            "quartile_percentages": quartile_pcts(src_click_ys),
            "stats": basic_stats(src_click_ys),
        },
        "target_click_y_distribution": {
            "description": "Normalized Y (0=top, 1=bottom) of target click coordinates",
            "quartile_percentages": quartile_pcts(tgt_click_ys),
            "stats": basic_stats(tgt_click_ys),
        },
        "source_scroll_y_distribution": {
            "description": "Normalized Y of source scroll from/to coordinates",
            "quartile_percentages": quartile_pcts(src_scroll_ys),
            "stats": basic_stats(src_scroll_ys),
        },
        "target_scroll_y_distribution": {
            "description": "Normalized Y of target scroll from/to coordinates",
            "quartile_percentages": quartile_pcts(tgt_scroll_ys),
            "stats": basic_stats(tgt_scroll_ys),
        },
        "resolution_distribution": {
            "source": src_res_counts,
            "target": tgt_res_counts,
        },
        "click_element_types": elem_type_counts,
    }


def generate_dataset(output_dir: str, num_pairs: int, seed: int = 42):
    source_dir = os.path.join(output_dir, "source")
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    rng = random.Random(seed)
    annotations = {"pairs": []}

    for i in range(num_pairs):
        pair_id = "pair_{:05d}".format(i)

        # Pick random source and target devices
        src_device = rng.choice(ANDROID_DEVICES + IOS_DEVICES)
        tgt_device = rng.choice(ANDROID_DEVICES + IOS_DEVICES)
        # Ensure source != target resolution (allow same-OS, different device)
        while tgt_device["size"] == src_device["size"]:
            tgt_device = rng.choice(ANDROID_DEVICES + IOS_DEVICES)

        src_size = src_device["size"]
        tgt_size = tgt_device["size"]

        # Determine platform labels
        src_platform = "android" if src_device in ANDROID_DEVICES else "ios"
        tgt_platform = "android" if tgt_device in ANDROID_DEVICES else "ios"

        # Pick palette
        palette = rng.choice(PALETTES)

        # Resolution-scaled fonts
        src_fonts = _get_fonts(src_size[1])
        tgt_fonts = _get_fonts(tgt_size[1])

        # Generate source screen
        src_img, src_elems = _generate_screen(src_size, palette, rng, src_fonts)

        # Generate target screen with same elements at scaled positions
        tgt_img = Image.new("RGB", tgt_size, palette["bg"])
        tgt_draw = ImageDraw.Draw(tgt_img)

        if rng.random() < 0.3:
            bg2 = tuple(max(0, min(255, c + rng.randint(-30, 30))) for c in palette["bg"])
            _draw_gradient_bg(tgt_img, palette["bg"], bg2)
            tgt_draw = ImageDraw.Draw(tgt_img)

        jitter_rng = random.Random(seed + i + 50000)
        tgt_elems = []
        for elem in src_elems:
            scaled_bbox = _scale_bbox(elem["bbox"], src_size, tgt_size, jitter_rng)
            tgt_elems.append({**elem, "bbox": scaled_bbox})

        # Redraw elements on target
        for elem in tgt_elems:
            _redraw_element(tgt_draw, elem, palette, rng, tgt_fonts)

        src_img.save(os.path.join(source_dir, "{}.png".format(pair_id)))
        tgt_img.save(os.path.join(target_dir, "{}.png".format(pair_id)))

        # Generate actions
        actions = []
        clickable = [e for e in src_elems if e["type"] in CLICKABLE_TYPES]
        if clickable:
            num_clicks = min(rng.randint(1, 4), len(clickable))
            for elem in rng.sample(clickable, num_clicks):
                idx = src_elems.index(elem)
                src_center = _bbox_center(src_elems[idx]["bbox"])
                tgt_center = _bbox_center(tgt_elems[idx]["bbox"])
                actions.append({
                    "type": "click",
                    "element_type": elem["type"],
                    "source_coords": {"at": src_center},
                    "target_coords": {"at": tgt_center},
                })

        # Scroll actions — full Y range
        num_scrolls = rng.randint(0, 2)
        for _ in range(num_scrolls):
            src_x = rng.randint(int(src_size[0] * 0.3), int(src_size[0] * 0.7))
            src_from_y = rng.randint(int(src_size[1] * 0.1), int(src_size[1] * 0.9))
            scroll_dy = rng.randint(int(src_size[1] * 0.1), int(src_size[1] * 0.3))
            src_to_y = max(0, min(src_size[1], src_from_y + rng.choice([-1, 1]) * scroll_dy))

            sx = tgt_size[0] / src_size[0]
            sy = tgt_size[1] / src_size[1]
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
            if src_elems:
                elem = rng.choice(src_elems)
                idx = src_elems.index(elem)
            else:
                idx = 0
            src_center = _bbox_center(src_elems[idx]["bbox"]) if src_elems else [src_size[0] // 2, src_size[1] // 2]
            tgt_center = _bbox_center(tgt_elems[idx]["bbox"]) if tgt_elems else [tgt_size[0] // 2, tgt_size[1] // 2]
            actions.append({
                "type": "click",
                "element_type": src_elems[idx]["type"] if src_elems else "fallback",
                "source_coords": {"at": src_center},
                "target_coords": {"at": tgt_center},
            })

        annotations["pairs"].append({
            "id": pair_id,
            "source": {"image": "source/{}.png".format(pair_id), "platform": src_platform,
                        "device": src_device["name"], "size": list(src_size)},
            "target": {"image": "target/{}.png".format(pair_id), "platform": tgt_platform,
                        "device": tgt_device["name"], "size": list(tgt_size)},
            "actions": actions,
        })

        if (i + 1) % 500 == 0:
            print("  Generated {}/{} pairs".format(i + 1, num_pairs))

    # Compute and attach distribution stats
    stats = _compute_distribution_stats(annotations)
    annotations["distribution_stats"] = stats

    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f)

    total_actions = stats["total_actions"]
    print("\nGenerated {} pairs with {} total actions in {}".format(num_pairs, total_actions, output_dir))
    print("\n=== Y-Distribution Stats ===")
    print("Source click quartiles: {}".format(stats["source_click_y_distribution"]["quartile_percentages"]))
    print("Target click quartiles: {}".format(stats["target_click_y_distribution"]["quartile_percentages"]))
    print("Source scroll quartiles: {}".format(stats["source_scroll_y_distribution"]["quartile_percentages"]))
    print("Target scroll quartiles: {}".format(stats["target_scroll_y_distribution"]["quartile_percentages"]))
    print("\n=== Resolution Distribution ===")
    print("Source: {}".format(stats["resolution_distribution"]["source"]))
    print("Target: {}".format(stats["resolution_distribution"]["target"]))
    print("\n=== Click Element Types ===")
    print(stats["click_element_types"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/cross_match_v4")
    parser.add_argument("--num-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.num_pairs, args.seed)
