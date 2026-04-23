"""Synthetic data generation v5 for CrossMatch training.

Key improvements over v2:
  - Screen archetypes: home_screen, product_list, settings, form, chat
  - Real SVG icon packs (Phosphor + Simple Icons) on colored rounded-square backgrounds
  - Icon grid layouts mimicking real home screens
  - Product cards with image-like placeholders (noise/gradient fills)
  - Uniform Y-distribution: elements spread across full screen height (6-94%)
  - Complex wallpaper-like backgrounds (mesh gradients, subtle gradients)
  - Realistic status bar on all screens + bottom dock on home screens
  - Tiny icon labels under grid items
  - Same-image identity pairs (12% of dataset)
  - Varied screen sizes (8 Android + 8 iOS resolutions)
  - Optional slight blur for realism
  - Configurable archetype distribution

Usage:
    python -m cross_match.synthetic_v5 --output-dir data/cross_match_v5 --num-pairs 10000
    python -m cross_match.synthetic_v5 --output-dir data/cross_match_v5 --num-pairs 20 --preview
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import random
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    import ctypes
    try:
        ctypes.cdll.LoadLibrary("/opt/homebrew/lib/libcairo.2.dylib")
    except OSError:
        pass
    import cairosvg
    HAS_CAIROSVG = True
except (ImportError, OSError):
    import subprocess as _sp
    try:
        _t = _sp.run(["rsvg-convert", "--version"], capture_output=True, timeout=2)
        if _t.returncode == 0:
            class _RsvgFallback:
                @staticmethod
                def svg2png(bytestring=None, output_width=64, output_height=64, **kw):
                    r = _sp.run(["rsvg-convert", "-w", str(output_width), "-h", str(output_height), "-f", "png"],
                                input=bytestring, capture_output=True, timeout=5)
                    return r.stdout if r.returncode == 0 else None
            cairosvg = _RsvgFallback()
            HAS_CAIROSVG = True
            print("Using rsvg-convert CLI for SVG rendering")
        else:
            HAS_CAIROSVG = False
    except Exception:
        HAS_CAIROSVG = False

if not HAS_CAIROSVG:
    print("WARNING: No SVG renderer available. Icons will use letter-on-square fallback.")

# ─── Screen Sizes ───

ANDROID_SIZES = [
    (1080, 1920), (1080, 2340), (1080, 2400), (1440, 3040),
    (1440, 3200), (720, 1280), (720, 1600), (1080, 2160),
]
IOS_SIZES = [
    (1170, 2532), (1179, 2556), (1290, 2796), (1125, 2436),
    (828, 1792), (750, 1334), (1242, 2688), (1284, 2778),
]

DEFAULT_ARCHETYPE_WEIGHTS = {
    "home_screen": 30, "product_list": 20, "settings": 20, "form": 15, "chat": 15,
}
IDENTITY_PAIR_RATIO = 0.12

# ─── Palettes ───

PALETTES = [
    {"bg": (250, 250, 250), "primary": (33, 150, 243), "accent": (76, 175, 80), "danger": (244, 67, 54), "header": (33, 150, 243), "text": (33, 33, 33), "text_secondary": (120, 120, 120), "surface": (255, 255, 255)},
    {"bg": (18, 18, 18), "primary": (100, 181, 246), "accent": (129, 199, 132), "danger": (239, 154, 154), "header": (30, 30, 30), "text": (230, 230, 230), "text_secondary": (160, 160, 160), "surface": (40, 40, 40)},
    {"bg": (255, 248, 225), "primary": (255, 152, 0), "accent": (0, 150, 136), "danger": (211, 47, 47), "header": (255, 152, 0), "text": (50, 40, 20), "text_secondary": (130, 110, 80), "surface": (255, 255, 245)},
    {"bg": (232, 234, 246), "primary": (63, 81, 181), "accent": (0, 188, 212), "danger": (233, 30, 99), "header": (63, 81, 181), "text": (30, 30, 50), "text_secondary": (100, 100, 130), "surface": (245, 245, 255)},
    {"bg": (243, 229, 245), "primary": (156, 39, 176), "accent": (255, 87, 34), "danger": (244, 67, 54), "header": (156, 39, 176), "text": (40, 20, 50), "text_secondary": (120, 90, 130), "surface": (255, 245, 255)},
    {"bg": (240, 248, 240), "primary": (56, 142, 60), "accent": (255, 167, 38), "danger": (211, 47, 47), "header": (56, 142, 60), "text": (20, 40, 20), "text_secondary": (90, 120, 90), "surface": (250, 255, 250)},
    {"bg": (255, 243, 240), "primary": (230, 74, 25), "accent": (38, 166, 154), "danger": (198, 40, 40), "header": (230, 74, 25), "text": (50, 30, 20), "text_secondary": (140, 100, 80), "surface": (255, 250, 248)},
]

ICON_BG_COLORS = [
    (0, 122, 255), (52, 199, 89), (255, 59, 48), (255, 149, 0), (175, 82, 222),
    (88, 86, 214), (255, 45, 85), (90, 200, 250), (0, 199, 190), (255, 204, 0),
    (50, 173, 230), (76, 217, 100), (30, 30, 30), (60, 60, 60), (220, 60, 60),
    (60, 130, 200), (200, 80, 160), (120, 190, 80), (230, 160, 50), (80, 80, 180),
]

BUTTON_TEXTS = [
    "Sign In", "Submit", "Continue", "Next", "Save", "Cancel", "OK", "Delete",
    "Send", "Search", "Share", "Edit", "Done", "Add to Cart", "Buy Now",
    "Subscribe", "Download", "Play", "Follow", "Like", "Checkout", "Upgrade",
]
INPUT_LABELS = [
    "Email", "Password", "Username", "Search...", "Phone number", "Address",
    "First name", "Last name", "Card number", "Promo code", "Message...",
    "Enter amount", "CVV", "Expiry date", "Company name", "Website URL",
]
NAV_LABELS = [
    "Home", "Settings", "Profile", "Notifications", "Messages", "Search",
    "Favorites", "History", "Downloads", "Orders", "Cart", "Explore",
    "Discover", "Trending", "Activity", "Camera", "Library", "Feed",
]
APP_NAMES = [
    "Photos", "Camera", "Settings", "Mail", "Maps", "Music", "Notes", "Calendar",
    "Weather", "Clock", "Calculator", "Podcasts", "News", "Health", "Wallet",
    "Files", "Translate", "Compass", "Chrome", "Safari", "WhatsApp", "Telegram",
    "Signal", "Discord", "Slack", "Instagram", "Twitter", "YouTube", "Netflix",
    "Spotify", "TikTok", "Uber", "Amazon", "PayPal", "Zoom", "Notion", "GitHub",
    "Reddit", "Pinterest", "LinkedIn", "Firefox", "Reminders", "Shortcuts",
]
PRODUCT_NAMES = [
    "Galaxy S20 Ultra", "iPhone 15 Pro", "AirPods Max", "MacBook Air",
    "Pixel Watch", "Echo Dot", "Kindle", "iPad Mini", "Surface Pro",
    "Sony WH-1000XM5", "Running Shoes", "Leather Jacket", "Coffee Maker",
    "Desk Lamp", "Backpack Pro", "Wireless Charger", "USB-C Hub",
]
CHAT_MESSAGES = [
    "Hey, how are you?", "Sure, sounds good!", "Can you send me the file?",
    "I'll be there in 10", "Thanks!", "See you tomorrow", "What time?",
    "Awesome!", "Let me check", "On my way", "Got it", "Haha nice",
    "Can we reschedule?", "That works for me", "Just finished",
    "Look at this!", "Wanna grab lunch?", "Meeting at 3pm", "Running late",
]
CHAT_SENDERS = ["Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace"]

# ─── Icon Loading ───

_icon_cache = {}

def _find_icon_dirs():
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "icons")
    dirs = []
    for name, sub in [("phosphor_fill", "phosphor/core-main/assets/fill"),
                       ("phosphor_regular", "phosphor/core-main/assets/regular"),
                       ("simple_icons", "simple-icons/simple-icons-develop/icons")]:
        p = os.path.join(base, sub)
        if os.path.isdir(p):
            dirs.append((name, p))
    return dirs

def _load_icon_paths():
    paths = {}
    for name, dirpath in _find_icon_dirs():
        svgs = glob.glob(os.path.join(dirpath, "*.svg"))
        if svgs:
            paths[name] = svgs
    return paths

def _render_svg_to_image(svg_path, size, color=None):
    key = (svg_path, size, color)
    if key in _icon_cache:
        return _icon_cache[key].copy()
    if not HAS_CAIROSVG:
        return None
    try:
        with open(svg_path, "r") as f:
            svg = f.read()
        if color:
            import re
            hx = "#{:02x}{:02x}{:02x}".format(*color)
            svg = re.sub(r'fill="[^"]*"', 'fill="{}"'.format(hx), svg)
            svg = re.sub(r'stroke="[^"]*"', 'stroke="{}"'.format(hx), svg)
            if 'fill=' not in svg:
                svg = svg.replace('<svg ', '<svg fill="{}" '.format(hx), 1)
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=size, output_height=size)
        if not png:
            return None
        img = Image.open(io.BytesIO(png)).convert("RGBA")
        _icon_cache[key] = img.copy()
        return img
    except Exception:
        return None

# ─── Drawing Helpers ───

def _try_load_fonts():
    fonts = {}
    base_path = None
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(p):
            base_path = p
            break
    for name, size in [("xs", 18), ("sm", 24), ("md", 32), ("lg", 40), ("xl", 52), ("xxl", 64)]:
        try:
            fonts[name] = ImageFont.truetype(base_path, size) if base_path else ImageFont.load_default()
        except Exception:
            fonts[name] = ImageFont.load_default()
    fonts["md_bold"] = fonts["md"]
    fonts["lg_bold"] = fonts["lg"]
    return fonts

def _draw_mesh_gradient(img, rng, num_blobs=4):
    w, h = img.size
    overlay = Image.new("RGB", (w, h), (rng.randint(20, 60), rng.randint(20, 60), rng.randint(40, 80)))
    for _ in range(num_blobs):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(w // 3, w)
        c = (rng.randint(30, 255), rng.randint(30, 255), rng.randint(30, 255))
        blob = Image.new("RGB", (w, h), (0, 0, 0))
        ImageDraw.Draw(blob).ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
        blob = blob.filter(ImageFilter.GaussianBlur(radius=r // 2))
        overlay = Image.blend(overlay, blob, 0.35)
    img.paste(overlay)

def _apply_subtle_bg(img, palette, rng):
    if rng.random() < 0.3:
        w, h = img.size
        draw = ImageDraw.Draw(img)
        bg = palette["bg"]
        bg2 = tuple(max(0, min(255, c + rng.randint(-25, 25))) for c in bg)
        for i in range(h):
            t = i / h
            draw.line([(0, i), (w, i)], fill=tuple(int(bg[j] + (bg2[j] - bg[j]) * t) for j in range(3)))

def _draw_status_bar(draw, w, h, palette, fonts, rng):
    bar_h = int(h * 0.035)
    draw.rectangle([0, 0, w, bar_h], fill=palette.get("header", (0, 0, 0)))
    font = fonts["xs"]
    time_str = "{:02d}:{:02d}".format(rng.randint(8, 23), rng.randint(0, 59))
    draw.text((int(w * 0.04), (bar_h - font.size) // 2), time_str, fill=(255, 255, 255), font=font)
    bx = w - int(w * 0.08)
    by = bar_h // 2 - 6
    draw.rectangle([bx, by, bx + 24, by + 12], outline=(255, 255, 255), width=1)
    draw.rectangle([bx + 1, by + 1, bx + rng.randint(8, 22), by + 11], fill=(255, 255, 255))
    sx = bx - 35
    for i in range(4):
        bh = 4 + i * 3
        draw.rectangle([sx + i * 6, by + 12 - bh, sx + i * 6 + 4, by + 12], fill=(255, 255, 255))
    return bar_h

def _draw_app_icon(draw, x, y, size, bg_color, icon_paths, rng, img=None):
    radius = size // 4
    draw.rounded_rectangle([x, y, x + size, y + size], radius=radius, fill=bg_color)
    if rng.random() < 0.3:
        for i in range(size // 2):
            a = int(40 * (i / (size // 2)))
            gy = y + size // 2 + i
            c = tuple(max(0, bg_color[j] - a) for j in range(3))
            draw.line([(x, gy), (x + size, gy)], fill=c)
    svg_rendered = False
    if icon_paths and HAS_CAIROSVG:
        pack = rng.choice(list(icon_paths.keys()))
        svg_path = rng.choice(icon_paths[pack])
        isz = int(size * 0.55)
        icon_img = _render_svg_to_image(svg_path, isz, color=(255, 255, 255))
        if icon_img and img:
            img.paste(icon_img, (x + (size - isz) // 2, y + (size - isz) // 2), icon_img)
            svg_rendered = True
    if not svg_rendered:
        letter = chr(rng.randint(65, 90))
        fs = size // 2
        try:
            f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", fs)
        except Exception:
            f = ImageFont.load_default()
        tw = draw.textlength(letter, font=f)
        draw.text((x + (size - tw) // 2, y + (size - fs) // 2), letter, fill=(255, 255, 255), font=f)

def _draw_image_placeholder(draw, x1, y1, x2, y2, rng):
    w, h = x2 - x1, y2 - y1
    c1 = tuple(rng.randint(60, 200) for _ in range(3))
    c2 = tuple(rng.randint(60, 200) for _ in range(3))
    style = rng.choice(["gradient", "diagonal", "blocks", "radial"])
    if style == "gradient":
        for i in range(h):
            t = i / max(1, h)
            draw.line([(x1, y1 + i), (x2, y1 + i)], fill=tuple(int(c1[j] + (c2[j] - c1[j]) * t) for j in range(3)))
    elif style == "diagonal":
        draw.rectangle([x1, y1, x2, y2], fill=c1)
        draw.polygon([(x1, y2), (x2, y1), (x2, y2)], fill=c2)
    elif style == "blocks":
        bw, bh = max(1, w // rng.randint(3, 6)), max(1, h // rng.randint(3, 6))
        for by in range(y1, y2, bh):
            for bx in range(x1, x2, bw):
                draw.rectangle([bx, by, min(bx + bw, x2), min(by + bh, y2)],
                               fill=tuple(rng.randint(40, 220) for _ in range(3)))
    elif style == "radial":
        draw.rectangle([x1, y1, x2, y2], fill=c1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for r in range(min(w, h) // 2, 0, -3):
            t = r / (min(w, h) // 2)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(c2[j] + (c1[j] - c2[j]) * t) for j in range(3)))

def _draw_bottom_nav_text(draw, w, h, palette, fonts, rng, elements):
    nav_h = int(h * 0.055)
    nav_y = h - nav_h
    draw.rectangle([0, nav_y, w, h], fill=palette["surface"])
    draw.line([(0, nav_y), (w, nav_y)], fill=(200, 200, 200), width=1)
    items = rng.sample(NAV_LABELS[:8], min(5, len(NAV_LABELS[:8])))
    spacing = w // len(items)
    for i, label in enumerate(items):
        cx = i * spacing + spacing // 2
        f = fonts["xs"]
        tw = draw.textlength(label, font=f)
        color = palette["primary"] if i == 0 else palette["text_secondary"]
        tx, ty = cx - tw / 2, nav_y + (nav_h - f.size) / 2
        draw.text((tx, ty), label, fill=color, font=f)
        elements.append({"bbox": [int(tx), int(ty), int(tx + tw), int(ty + f.size)], "type": "label", "text": label})

# ─── Screen Archetype Generators ───

def _gen_home_screen(sz, pal, rng, fonts, icon_paths, img, content=None):
    w, h = sz
    draw = ImageDraw.Draw(img)
    elements = []
    _draw_mesh_gradient(img, rng)
    draw = ImageDraw.Draw(img)
    bar_h = _draw_status_bar(draw, w, h, {"header": (0, 0, 0)}, fonts, rng)

    cols = rng.choice([3, 4, 5])
    rows = rng.randint(3, 6)
    icon_size = int(w * 0.14)
    label_h = int(h * 0.02)
    cell_w = w // cols
    cell_h = icon_size + label_h + int(h * 0.02)
    grid_top = bar_h + int(h * 0.04)
    max_icons = cols * rows
    names = content["app_names"][:max_icons] if content and "app_names" in content else rng.sample(APP_NAMES, min(max_icons, len(APP_NAMES)))
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            if idx >= len(names):
                break
            cx = col * cell_w + cell_w // 2
            cy = grid_top + row * cell_h + icon_size // 2
            x1, y1 = cx - icon_size // 2, cy - icon_size // 2
            if y1 + icon_size + label_h > h - int(h * 0.1):
                break
            _draw_app_icon(draw, x1, y1, icon_size, rng.choice(ICON_BG_COLORS), icon_paths, rng, img=img)
            draw = ImageDraw.Draw(img)
            name = names[idx]
            f = fonts["xs"]
            tw = draw.textlength(name, font=f)
            if tw > cell_w - 4:
                name = name[:8] + ".."
                tw = draw.textlength(name, font=f)
            draw.text((cx - tw // 2, y1 + icon_size + 2), name, fill=(255, 255, 255), font=f)
            elements.append({"bbox": [x1, y1, x1 + icon_size, y1 + icon_size], "type": "icon", "text": names[idx]})
    # Bottom dock
    dock_h = int(h * 0.08)
    dock_y = h - dock_h
    draw.rectangle([0, dock_y, w, h], fill=pal["surface"])
    dock_n = 4
    d_icon = int(dock_h * 0.6)
    d_sp = w // (dock_n + 1)
    for i in range(dock_n):
        dx = d_sp * (i + 1) - d_icon // 2
        dy = dock_y + (dock_h - d_icon) // 2
        _draw_app_icon(draw, dx, dy, d_icon, rng.choice(ICON_BG_COLORS), icon_paths, rng, img=img)
        draw = ImageDraw.Draw(img)
        elements.append({"bbox": [dx, dy, dx + d_icon, dy + d_icon], "type": "icon", "text": rng.choice(APP_NAMES)})
    return elements

def _gen_product_list(sz, pal, rng, fonts, icon_paths, img, content=None):
    w, h = sz
    draw = ImageDraw.Draw(img)
    elements = []
    draw.rectangle([0, 0, w, h], fill=pal["bg"])
    _apply_subtle_bg(img, pal, rng)
    draw = ImageDraw.Draw(img)
    status_h = _draw_status_bar(draw, w, h, pal, fonts, rng)
    hdr_h = int(h * 0.05)
    draw.rectangle([0, status_h, w, status_h + hdr_h], fill=pal["header"])
    title = rng.choice(["Shop", "Products", "Store", "Market", "Deals"])
    draw.text((int(w * 0.35), status_h + (hdr_h - fonts["lg"].size) // 2), title, fill=(255, 255, 255), font=fonts["lg"])
    draw.text((w - int(w * 0.12), status_h + (hdr_h - fonts["sm"].size) // 2), "Cart", fill=(255, 255, 255), font=fonts["sm"])
    fy = status_h + hdr_h + int(h * 0.01)
    draw.text((w - int(w * 0.25), fy), "Filter & Sort", fill=pal["primary"], font=fonts["sm"])
    cnt = rng.randint(5, 50)
    draw.text((int(w * 0.1), fy), "{} Product(s)".format(cnt), fill=pal["text_secondary"], font=fonts["sm"])
    y = fy + int(h * 0.03)
    margin = int(w * 0.06)
    products = content["products"] if content and "products" in content else rng.sample(PRODUCT_NAMES, min(rng.randint(2, 4), len(PRODUCT_NAMES)))
    for pname in products:
        if y > h * 0.85:
            break
        cw = w - 2 * margin
        img_h = int(h * rng.uniform(0.12, 0.18))
        txt_h = int(h * 0.1)
        ch = img_h + txt_h
        draw.rounded_rectangle([margin, y, margin + cw, y + ch], radius=12, fill=pal["surface"], outline=(220, 220, 220), width=1)
        _draw_image_placeholder(draw, margin + 4, y + 4, margin + cw - 4, y + img_h, rng)
        hx, hy = margin + cw - int(w * 0.08), y + 10
        draw.text((hx, hy), "♡", fill=pal["text_secondary"], font=fonts["lg"])
        elements.append({"bbox": [hx, hy, hx + 40, hy + 40], "type": "icon", "text": "heart"})
        ny = y + img_h + 8
        draw.text((margin + 16, ny), pname, fill=pal["text"], font=fonts["md_bold"])
        py = ny + fonts["md_bold"].size + 4
        draw.text((margin + 16, py), "$ {}.00".format(rng.randint(9, 2999)), fill=pal["text"], font=fonts["lg_bold"])
        by = py + fonts["lg_bold"].size + 8
        bh = int(h * 0.03)
        bw = cw - 32
        draw.rounded_rectangle([margin + 16, by, margin + 16 + bw, by + bh], radius=6, fill=(30, 30, 30))
        btxt = rng.choice(["Add to cart", "Buy Now", "Add to bag"])
        tw = draw.textlength(btxt, font=fonts["md"])
        draw.text((margin + 16 + (bw - tw) // 2, by + (bh - fonts["md"].size) // 2), btxt, fill=(255, 255, 255), font=fonts["md"])
        elements.append({"bbox": [margin + 16, by, margin + 16 + bw, by + bh], "type": "button", "text": btxt})
        elements.append({"bbox": [margin, y, margin + cw, y + ch], "type": "card", "text": pname})
        y += ch + int(h * 0.02)
    _draw_bottom_nav_text(draw, w, h, pal, fonts, rng, elements)
    return elements

def _gen_settings(sz, pal, rng, fonts, icon_paths, img, content=None):
    w, h = sz
    draw = ImageDraw.Draw(img)
    elements = []
    draw.rectangle([0, 0, w, h], fill=pal["bg"])
    _apply_subtle_bg(img, pal, rng)
    draw = ImageDraw.Draw(img)
    status_h = _draw_status_bar(draw, w, h, pal, fonts, rng)
    hdr_h = int(h * 0.05)
    draw.rectangle([0, status_h, w, status_h + hdr_h], fill=pal["header"])
    draw.text((int(w * 0.35), status_h + (hdr_h - fonts["lg"].size) // 2), "Settings", fill=(255, 255, 255), font=fonts["lg"])
    y = status_h + hdr_h + int(h * 0.02)
    margin = int(w * 0.05)
    num_rows = rng.randint(8, 14)
    labels = content["labels"] if content and "labels" in content else rng.sample(NAV_LABELS, min(num_rows, len(NAV_LABELS)))
    avail = int(h * 0.90) - y
    row_h = max(int(h * 0.035), avail // len(labels))
    for label in labels:
        if y > h * 0.92:
            break
        draw.line([(margin, y), (w - margin, y)], fill=(200, 200, 200), width=1)
        y += 4
        ir = row_h // 3
        draw.ellipse([margin, y + row_h // 2 - ir, margin + ir * 2, y + row_h // 2 + ir], fill=rng.choice(ICON_BG_COLORS[:10]))
        tx = margin + ir * 2 + 16
        draw.text((tx, y + (row_h - fonts["md"].size) // 2), label, fill=pal["text"], font=fonts["md"])
        if rng.random() < 0.5:
            tw, th = int(w * 0.1), int(row_h * 0.5)
            togx = w - margin - tw
            togy = y + (row_h - th) // 2
            on = rng.random() < 0.5
            draw.rounded_rectangle([togx, togy, togx + tw, togy + th], radius=th // 2, fill=pal["primary"] if on else (180, 180, 180))
            cr = th // 2 - 2
            cx = togx + tw - cr - 3 if on else togx + cr + 3
            draw.ellipse([cx - cr, togy + 2, cx + cr, togy + th - 2], fill=(255, 255, 255))
            elements.append({"bbox": [togx, togy, togx + tw, togy + th], "type": "toggle", "text": label})
        else:
            draw.text((w - margin - 10, y + (row_h - fonts["sm"].size) // 2), ">", fill=pal["text_secondary"], font=fonts["sm"])
            elements.append({"bbox": [tx, y, w - margin, y + row_h], "type": "label", "text": label})
        y += row_h + int(h * 0.005)
    return elements

def _gen_form(sz, pal, rng, fonts, icon_paths, img, content=None):
    w, h = sz
    draw = ImageDraw.Draw(img)
    elements = []
    draw.rectangle([0, 0, w, h], fill=pal["bg"])
    _apply_subtle_bg(img, pal, rng)
    draw = ImageDraw.Draw(img)
    status_h = _draw_status_bar(draw, w, h, pal, fonts, rng)
    hdr_h = int(h * 0.05)
    draw.rectangle([0, status_h, w, status_h + hdr_h], fill=pal["header"])
    title = rng.choice(["Sign In", "Register", "Create Account", "Login", "Sign Up", "Checkout"])
    tw = draw.textlength(title, font=fonts["lg"])
    draw.text(((w - tw) // 2, status_h + (hdr_h - fonts["lg"].size) // 2), title, fill=(255, 255, 255), font=fonts["lg"])
    logo_y = status_h + hdr_h + int(h * 0.05)
    logo_s = int(w * 0.18)
    lx = (w - logo_s) // 2
    draw.rounded_rectangle([lx, logo_y, lx + logo_s, logo_y + logo_s], radius=logo_s // 4, fill=pal["primary"])
    draw.text((lx + logo_s // 3, logo_y + logo_s // 4), "A", fill=(255, 255, 255), font=fonts["xxl"])
    y = logo_y + logo_s + int(h * 0.03)
    margin = int(w * 0.08)
    for label in rng.sample(INPUT_LABELS, min(rng.randint(2, 5), len(INPUT_LABELS))):
        if y > h * 0.6:
            break
        iw, ih = w - 2 * margin, int(h * 0.035)
        draw.text((margin, y), label, fill=pal["text_secondary"], font=fonts["sm"])
        y += fonts["sm"].size + 4
        draw.rounded_rectangle([margin, y, margin + iw, y + ih], radius=8, fill=pal["surface"], outline=(180, 180, 180), width=2)
        draw.text((margin + 12, y + (ih - fonts["md"].size) // 2), "Enter " + label.lower(), fill=pal["text_secondary"], font=fonts["md"])
        elements.append({"bbox": [margin, y, margin + iw, y + ih], "type": "input", "text": label})
        y += ih + int(h * 0.02)
    bw, bh = w - 2 * margin, int(h * 0.04)
    by = y + int(h * 0.02)
    draw.rounded_rectangle([margin, by, margin + bw, by + bh], radius=12, fill=pal["primary"])
    btxt = rng.choice(["Sign In", "Submit", "Continue", "Create Account"])
    tw = draw.textlength(btxt, font=fonts["lg"])
    draw.text((margin + (bw - tw) // 2, by + (bh - fonts["lg"].size) // 2), btxt, fill=(255, 255, 255), font=fonts["lg"])
    elements.append({"bbox": [margin, by, margin + bw, by + bh], "type": "button", "text": btxt})
    ly = by + bh + int(h * 0.02)
    ltxt = rng.choice(["Forgot password?", "Need an account?", "Skip for now"])
    tw = draw.textlength(ltxt, font=fonts["sm"])
    draw.text(((w - tw) // 2, ly), ltxt, fill=pal["primary"], font=fonts["sm"])
    elements.append({"bbox": [int((w - tw) // 2), ly, int((w + tw) // 2), ly + fonts["sm"].size], "type": "label", "text": ltxt})
    # Bottom section for Y-coverage: social login + terms
    sy = int(h * rng.uniform(0.72, 0.82))
    draw.line([(margin, sy - int(h * 0.01)), (w - margin, sy - int(h * 0.01))], fill=(200, 200, 200), width=1)
    ortxt = "or continue with"
    tw = draw.textlength(ortxt, font=fonts["sm"])
    draw.text(((w - tw) // 2, sy - int(h * 0.01) - fonts["sm"].size // 2), ortxt, fill=pal["text_secondary"], font=fonts["sm"])
    socials = rng.sample(["Google", "Apple", "Facebook", "GitHub", "Twitter"], 3)
    sbw = (w - 2 * margin - 20) // 3
    sbh = int(h * 0.03)
    for si, sn in enumerate(socials):
        sx = margin + si * (sbw + 10)
        draw.rounded_rectangle([sx, sy, sx + sbw, sy + sbh], radius=8, fill=pal["surface"], outline=(180, 180, 180), width=1)
        stw = draw.textlength(sn, font=fonts["sm"])
        draw.text((sx + (sbw - stw) // 2, sy + (sbh - fonts["sm"].size) // 2), sn, fill=pal["text"], font=fonts["sm"])
        elements.append({"bbox": [sx, sy, sx + sbw, sy + sbh], "type": "button", "text": sn})
    ty = int(h * 0.92)
    ttxt = rng.choice(["By continuing you agree to our Terms", "Privacy Policy | Terms of Use", "Need help? Contact support"])
    tw = draw.textlength(ttxt, font=fonts["xs"])
    draw.text(((w - tw) // 2, ty), ttxt, fill=pal["text_secondary"], font=fonts["xs"])
    elements.append({"bbox": [int((w - tw) // 2), ty, int((w + tw) // 2), ty + fonts["xs"].size], "type": "label", "text": ttxt})
    return elements

def _gen_chat(sz, pal, rng, fonts, icon_paths, img, content=None):
    w, h = sz
    draw = ImageDraw.Draw(img)
    elements = []
    draw.rectangle([0, 0, w, h], fill=pal["bg"])
    _apply_subtle_bg(img, pal, rng)
    draw = ImageDraw.Draw(img)
    status_h = _draw_status_bar(draw, w, h, pal, fonts, rng)
    hdr_h = int(h * 0.05)
    draw.rectangle([0, status_h, w, status_h + hdr_h], fill=pal["header"])
    contact = rng.choice(CHAT_SENDERS)
    draw.text((int(w * 0.15), status_h + (hdr_h - fonts["lg"].size) // 2), contact, fill=(255, 255, 255), font=fonts["lg"])
    avr = hdr_h // 3
    draw.ellipse([int(w * 0.04), status_h + hdr_h // 2 - avr, int(w * 0.04) + 2 * avr, status_h + hdr_h // 2 + avr], fill=rng.choice(ICON_BG_COLORS))
    margin = int(w * 0.04)
    max_y = int(h * 0.88)
    content_top = status_h + hdr_h + int(h * 0.02)
    num_msgs = rng.randint(8, 18)
    avail_h = max_y - content_top
    msg_spacing = avail_h / max(num_msgs, 1)
    y = content_top
    msgs = rng.choices(CHAT_MESSAGES, k=num_msgs)
    for msg in msgs:
        if y > max_y:
            break
        is_mine = rng.random() < 0.45
        f = fonts["md"]
        tw = draw.textlength(msg, font=f)
        bw = min(int(tw) + 24, int(w * 0.65))
        bh = int(f.size * 1.5) + 12
        if is_mine:
            x1, bc, tc = w - margin - bw, pal["primary"], (255, 255, 255)
        else:
            x1, bc, tc = margin, pal["surface"], pal["text"]
        draw.rounded_rectangle([x1, y, x1 + bw, y + bh], radius=16, fill=bc)
        draw.text((x1 + 12, y + 8), msg, fill=tc, font=f)
        elements.append({"bbox": [x1, y, x1 + bw, y + bh], "type": "card", "text": msg})
        y += bh + max(int(msg_spacing - bh), int(h * 0.008))
    # Message input bar at bottom
    bar_y = h - int(h * 0.06)
    bar_h = int(h * 0.045)
    draw.rectangle([0, bar_y - 5, w, h], fill=pal["surface"])
    iw = int(w * 0.72)
    draw.rounded_rectangle([margin, bar_y, margin + iw, bar_y + bar_h], radius=bar_h // 2, fill=pal["bg"], outline=(180, 180, 180), width=1)
    draw.text((margin + 16, bar_y + (bar_h - fonts["md"].size) // 2), "Message...", fill=pal["text_secondary"], font=fonts["md"])
    elements.append({"bbox": [margin, bar_y, margin + iw, bar_y + bar_h], "type": "input", "text": "Message"})
    sx = margin + iw + 12
    ss = bar_h
    draw.ellipse([sx, bar_y, sx + ss, bar_y + ss], fill=pal["primary"])
    elements.append({"bbox": [sx, bar_y, sx + ss, bar_y + ss], "type": "button", "text": "Send"})
    return elements

# ─── Pair Generation ───

ARCHETYPE_GENERATORS = {
    "home_screen": _gen_home_screen, "product_list": _gen_product_list,
    "settings": _gen_settings, "form": _gen_form, "chat": _gen_chat,
}

def _scale_bbox(bbox, src_size, tgt_size, rng, jitter=12):
    sx, sy = tgt_size[0] / src_size[0], tgt_size[1] / src_size[1]
    jx, jy = rng.randint(-jitter, jitter), rng.randint(-jitter, jitter)
    return [max(0, int(bbox[0] * sx + jx)), max(0, int(bbox[1] * sy + jy)),
            min(tgt_size[0], int(bbox[2] * sx + jx)), min(tgt_size[1], int(bbox[3] * sy + jy))]

def _bbox_center(bbox):
    return [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]

def generate_dataset(output_dir, num_pairs, seed=42, archetype_weights=None, preview=False):
    src_dir = os.path.join(output_dir, "source")
    tgt_dir = os.path.join(output_dir, "target")
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)
    fonts = _try_load_fonts()
    icon_paths = _load_icon_paths()
    rng = random.Random(seed)
    total_icons = sum(len(v) for v in icon_paths.values())
    if total_icons:
        print("Loaded {} SVG icons from {} packs".format(total_icons, len(icon_paths)))
    else:
        print("WARNING: No icon packs found at data/icons/")
    weights = archetype_weights or DEFAULT_ARCHETYPE_WEIGHTS
    names = list(weights.keys())
    probs = [weights[k] for k in names]
    annotations = {"pairs": []}

    for i in range(num_pairs):
        pid = "pair_{:05d}".format(i)
        archetype = rng.choices(names, weights=probs)[0]
        gen = ARCHETYPE_GENERATORS[archetype]
        is_id = rng.random() < IDENTITY_PAIR_RATIO
        pal = rng.choice(PALETTES)
        src_sz = rng.choice(ANDROID_SIZES)
        tgt_sz = rng.choice(IOS_SIZES)

        # Generate source screen
        src_rng = random.Random(seed + i * 2)
        src_img = Image.new("RGB", src_sz, pal["bg"])
        src_elems = gen(src_sz, pal, src_rng, fonts, icon_paths, src_img)

        if rng.random() < 0.15:
            src_img = src_img.filter(ImageFilter.GaussianBlur(radius=0.8))

        if is_id:
            # Identity pair (12%): same image, same positions
            tgt_sz = src_sz
            tgt_img = src_img.copy()
            tgt_elems = [dict(e) for e in src_elems]
            pair_strategy = "identity"
        else:
            # Choose target strategy for position variability:
            #   proportional (45%): scale bbox by resolution ratio + jitter (teaches resolution mapping)
            #   independent  (40%): regenerate with same content, different layout (teaches semantic matching)
            #   shuffled     (15%): proportional positions but elements reordered (breaks position-only learning)
            strategy_roll = rng.random()
            tgt_pal = rng.choice(PALETTES)

            if strategy_roll < 0.45:
                # Proportional: scale + jitter (original approach)
                tgt_img = Image.new("RGB", tgt_sz, tgt_pal["bg"])
                tgt_rng = random.Random(seed + i * 2 + 1)
                gen(tgt_sz, tgt_pal, tgt_rng, fonts, icon_paths, tgt_img)
                jrng = random.Random(seed + i + 50000)
                tgt_elems = [{**e, "bbox": _scale_bbox(e["bbox"], src_sz, tgt_sz, jrng)} for e in src_elems]
                pair_strategy = "proportional"

            elif strategy_roll < 0.85:
                # Independent layout: same element content, different positions
                # Pre-select shared content from source elements
                content = {}
                src_names = [e["text"] for e in src_elems if e["type"] == "icon"]
                src_labels = [e["text"] for e in src_elems if e["type"] in ("label", "toggle")]
                src_products = [e["text"] for e in src_elems if e["type"] == "card"]
                if src_names: content["app_names"] = src_names
                if src_labels: content["labels"] = src_labels
                if src_products: content["products"] = src_products

                tgt_rng = random.Random(seed + i * 2 + 1)
                tgt_img = Image.new("RGB", tgt_sz, tgt_pal["bg"])
                tgt_elems_raw = gen(tgt_sz, tgt_pal, tgt_rng, fonts, icon_paths, tgt_img, content=content)

                # Match by semantic identity
                tgt_lookup = {}
                for e in tgt_elems_raw:
                    key = (e["type"], e["text"])
                    if key not in tgt_lookup:
                        tgt_lookup[key] = e["bbox"]
                tgt_elems = []
                for e in src_elems:
                    key = (e["type"], e["text"])
                    if key in tgt_lookup:
                        tgt_elems.append({**e, "bbox": tgt_lookup[key]})
                    else:
                        tgt_elems.append({**e, "bbox": None})
                pair_strategy = "independent"

            else:
                # Shuffled: proportional scaling but elements in different order
                tgt_img = Image.new("RGB", tgt_sz, tgt_pal["bg"])
                tgt_rng = random.Random(seed + i * 2 + 1)
                gen(tgt_sz, tgt_pal, tgt_rng, fonts, icon_paths, tgt_img)
                jrng = random.Random(seed + i + 50000)
                scaled = [{**e, "bbox": _scale_bbox(e["bbox"], src_sz, tgt_sz, jrng)} for e in src_elems]
                # Shuffle the scaled bboxes while keeping element identities
                bboxes = [e["bbox"] for e in scaled]
                jrng.shuffle(bboxes)
                tgt_elems = [{**e, "bbox": bb} for e, bb in zip(src_elems, bboxes)]
                pair_strategy = "shuffled"

            if rng.random() < 0.15:
                tgt_img = tgt_img.filter(ImageFilter.GaussianBlur(radius=0.8))

        src_img.save(os.path.join(src_dir, "{}.png".format(pid)))
        tgt_img.save(os.path.join(tgt_dir, "{}.png".format(pid)))

        actions = []
        # Only use elements that have a valid match on both source and target
        clickable = [j for j, e in enumerate(src_elems)
                     if e["type"] in ("button", "input", "card", "icon", "toggle")
                     and j < len(tgt_elems) and tgt_elems[j]["bbox"] is not None]
        if clickable:
            for idx in rng.sample(clickable, min(rng.randint(1, 3), len(clickable))):
                sc = _bbox_center(src_elems[idx]["bbox"])
                tc = _bbox_center(tgt_elems[idx]["bbox"])
                actions.append({"type": "click", "source_coords": {"at": sc}, "target_coords": {"at": tc}})
        if rng.random() < 0.4:
            sx = rng.randint(int(src_sz[0] * 0.3), int(src_sz[0] * 0.7))
            sfy = rng.randint(int(src_sz[1] * 0.2), int(src_sz[1] * 0.7))
            sdy = rng.randint(int(src_sz[1] * 0.1), int(src_sz[1] * 0.3))
            sty = max(0, min(src_sz[1], sfy + rng.choice([-1, 1]) * sdy))
            rx, ry = tgt_sz[0] / src_sz[0], tgt_sz[1] / src_sz[1]
            actions.append({"type": "scroll",
                            "source_coords": {"from_arg": [sx, sfy], "to_arg": [sx, sty]},
                            "target_coords": {"from_arg": [int(sx * rx), int(sfy * ry)], "to_arg": [int(sx * rx), int(sty * ry)]}})
        if not actions:
            cx, cy = src_sz[0] // 2, src_sz[1] // 2
            actions.append({"type": "click", "source_coords": {"at": [cx, cy]}, "target_coords": {"at": [cx, cy]}})

        # Compute Y-distribution metrics for this pair
        src_action_ys = []
        tgt_action_ys = []
        for a in actions:
            if a["type"] == "click":
                src_action_ys.append(a["source_coords"]["at"][1] / src_sz[1])
                tgt_action_ys.append(a["target_coords"]["at"][1] / tgt_sz[1])
            elif a["type"] == "scroll":
                src_action_ys.append(a["source_coords"]["from_arg"][1] / src_sz[1])
                src_action_ys.append(a["source_coords"]["to_arg"][1] / src_sz[1])
                tgt_action_ys.append(a["target_coords"]["from_arg"][1] / tgt_sz[1])
                tgt_action_ys.append(a["target_coords"]["to_arg"][1] / tgt_sz[1])

        src_elem_ys = [((e["bbox"][1] + e["bbox"][3]) / 2) / src_sz[1] for e in src_elems]
        tgt_elem_ys = [((e["bbox"][1] + e["bbox"][3]) / 2) / tgt_sz[1] for e in tgt_elems if e["bbox"] is not None]

        def _quartile_dist(ys):
            if not ys:
                return {"q1": 0, "q2": 0, "q3": 0, "q4": 0}
            return {
                "q1": round(sum(1 for y in ys if y < 0.25) / len(ys) * 100, 1),
                "q2": round(sum(1 for y in ys if 0.25 <= y < 0.50) / len(ys) * 100, 1),
                "q3": round(sum(1 for y in ys if 0.50 <= y < 0.75) / len(ys) * 100, 1),
                "q4": round(sum(1 for y in ys if y >= 0.75) / len(ys) * 100, 1),
            }

        y_dist = {
            "source_actions": {
                "count": len(src_action_ys),
                "min_y": round(min(src_action_ys), 3) if src_action_ys else None,
                "max_y": round(max(src_action_ys), 3) if src_action_ys else None,
                "quartiles_pct": _quartile_dist(src_action_ys),
            },
            "source_elements": {
                "count": len(src_elem_ys),
                "min_y": round(min(src_elem_ys), 3) if src_elem_ys else None,
                "max_y": round(max(src_elem_ys), 3) if src_elem_ys else None,
                "quartiles_pct": _quartile_dist(src_elem_ys),
            },
            "target_actions": {
                "count": len(tgt_action_ys),
                "min_y": round(min(tgt_action_ys), 3) if tgt_action_ys else None,
                "max_y": round(max(tgt_action_ys), 3) if tgt_action_ys else None,
                "quartiles_pct": _quartile_dist(tgt_action_ys),
            },
        }

        annotations["pairs"].append({
            "id": pid,
            "source": {"image": "source/{}.png".format(pid), "platform": "android", "size": list(src_sz)},
            "target": {"image": "target/{}.png".format(pid), "platform": "ios" if not is_id else "android", "size": list(tgt_sz)},
            "actions": actions, "archetype": archetype, "is_identity": is_id, "pair_strategy": pair_strategy,
            "y_distribution": y_dist,
        })
        if (i + 1) % 500 == 0 or preview:
            print("  Generated {}/{} [{}{}]".format(i + 1, num_pairs, archetype, " (identity)" if is_id else ""))

    total_actions = sum(len(p["actions"]) for p in annotations["pairs"])
    archetypes = {}
    for p in annotations["pairs"]:
        archetypes[p["archetype"]] = archetypes.get(p["archetype"], 0) + 1
    id_count = sum(1 for p in annotations["pairs"] if p.get("is_identity"))
    sizes = set()
    for p in annotations["pairs"]:
        sizes.add(tuple(p["source"]["size"]))
        sizes.add(tuple(p["target"]["size"]))

    # Aggregate Y-distribution across all pairs
    all_action_ys = []
    all_elem_ys = []
    per_archetype_ys = {}
    for p in annotations["pairs"]:
        yd = p.get("y_distribution", {})
        src_a = yd.get("source_actions", {})
        src_e = yd.get("source_elements", {})
        # Reconstruct individual Y values from actions for aggregate stats
        src_sz_p = p["source"]["size"]
        for a in p["actions"]:
            if a["type"] == "click":
                y_norm = a["source_coords"]["at"][1] / src_sz_p[1]
                all_action_ys.append(y_norm)
            elif a["type"] == "scroll":
                all_action_ys.append(a["source_coords"]["from_arg"][1] / src_sz_p[1])
                all_action_ys.append(a["source_coords"]["to_arg"][1] / src_sz_p[1])
        arch = p["archetype"]
        if arch not in per_archetype_ys:
            per_archetype_ys[arch] = []
        for a in p["actions"]:
            if a["type"] == "click":
                per_archetype_ys[arch].append(a["source_coords"]["at"][1] / src_sz_p[1])

    def _fmt_quartiles(ys):
        if not ys:
            return "no data"
        n = len(ys)
        q1 = sum(1 for y in ys if y < 0.25) / n * 100
        q2 = sum(1 for y in ys if 0.25 <= y < 0.50) / n * 100
        q3 = sum(1 for y in ys if 0.50 <= y < 0.75) / n * 100
        q4 = sum(1 for y in ys if y >= 0.75) / n * 100
        return "Q1(0-25%)={:.1f}% Q2(25-50%)={:.1f}% Q3(50-75%)={:.1f}% Q4(75-100%)={:.1f}%".format(q1, q2, q3, q4)

    # Add aggregate stats to annotations
    annotations["dataset_stats"] = {
        "num_pairs": num_pairs,
        "total_actions": total_actions,
        "identity_pairs": id_count,
        "archetypes": archetypes,
        "screen_sizes_used": len(sizes),
        "y_distribution_aggregate": {
            "total_action_coords": len(all_action_ys),
            "quartiles_pct": {
                "q1_0_25": round(sum(1 for y in all_action_ys if y < 0.25) / max(len(all_action_ys), 1) * 100, 1),
                "q2_25_50": round(sum(1 for y in all_action_ys if 0.25 <= y < 0.50) / max(len(all_action_ys), 1) * 100, 1),
                "q3_50_75": round(sum(1 for y in all_action_ys if 0.50 <= y < 0.75) / max(len(all_action_ys), 1) * 100, 1),
                "q4_75_100": round(sum(1 for y in all_action_ys if y >= 0.75) / max(len(all_action_ys), 1) * 100, 1),
            },
            "min_y": round(min(all_action_ys), 3) if all_action_ys else None,
            "max_y": round(max(all_action_ys), 3) if all_action_ys else None,
            "mean_y": round(sum(all_action_ys) / max(len(all_action_ys), 1), 3),
        },
        "y_distribution_per_archetype": {
            arch: _fmt_quartiles(ys) for arch, ys in per_archetype_ys.items()
        },
    }

    # Rewrite annotations with stats
    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=2 if preview else None)

    print("\n=== Dataset Summary ===")
    print("Pairs: {}  Actions: {}  Identity: {}".format(num_pairs, total_actions, id_count))
    print("Archetypes: {}".format(archetypes))
    print("Screen sizes used: {}".format(len(sizes)))
    print("\n--- Y-Distribution (action coords) ---")
    print("Overall: {}".format(_fmt_quartiles(all_action_ys)))
    for arch, ys in sorted(per_archetype_ys.items()):
        print("  {}: {}".format(arch, _fmt_quartiles(ys)))
    print("\nOutput: {}".format(output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrossMatch synthetic data v5")
    parser.add_argument("--output-dir", default="data/cross_match_v5")
    parser.add_argument("--num-pairs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--archetype-weights", type=str, default=None,
                        help="e.g. 'home_screen:40,settings:30,form:30'")
    args = parser.parse_args()
    weights = None
    if args.archetype_weights:
        weights = {}
        for kv in args.archetype_weights.split(","):
            k, v = kv.strip().split(":")
            weights[k.strip()] = int(v.strip())
    generate_dataset(args.output_dir, args.num_pairs, args.seed, weights, args.preview)
