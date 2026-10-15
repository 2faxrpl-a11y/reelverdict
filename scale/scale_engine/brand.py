"""SCALE brand kit: the growing-letter wordmark, icon, banner and thumbnails.

The wordmark is built from real glyph outlines (Inter Display, OFL), so the SVG
and every PNG come from the same geometry. Each letter is `GROWTH` times the
size of the one before it, sharing one baseline, so a tiny S scales up to a
towering E.

    python -m scale_engine.brand            # writes brand/out/*
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .spec import ROOT

BRAND = ROOT / "brand"
OUT = BRAND / "out"
FONT = BRAND / "fonts" / "InterDisplay-SemiBold.otf"
FONT_LIGHT = BRAND / "fonts" / "InterDisplay-Light.otf"
FONT_TEXT = BRAND / "fonts" / "Inter-Medium.otf"

WORD = "SCALE"
GROWTH = 1.55             # each letter is 1.55x the previous: E = 5.8x S
TRACKING = 0.07           # gap after each letter, fraction of that letter's size

INK = (244, 239, 230)     # warm white
VOID = (7, 8, 11)         # near-black
SPARK = (232, 166, 87)    # amber: the tiny S, where every zoom begins
DIM = (150, 146, 140)


# ---------------------------------------------------------------- geometry
def _flatten(ops, steps=24):
    """RecordingPen ops -> list of closed polygons (font units)."""
    polys, cur, pt = [], [], (0.0, 0.0)
    for op, args in ops:
        if op == "moveTo":
            if cur:
                polys.append(cur)
            cur, pt = [args[0]], args[0]
        elif op == "lineTo":
            cur.append(args[0])
            pt = args[0]
        elif op == "curveTo":
            p0 = pt
            for c1, c2, p3 in [args[i:i + 3] for i in range(0, len(args), 3)]:
                for k in range(1, steps + 1):
                    t = k / steps
                    a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3
                    cur.append((a * p0[0] + b * c1[0] + c * c2[0] + d * p3[0],
                                a * p0[1] + b * c1[1] + c * c2[1] + d * p3[1]))
                p0 = p3
            pt = p0
        elif op == "qCurveTo":
            pts = list(args)
            p0 = pt
            ctrl, end = pts[:-1], pts[-1]
            # implied on-curve points between consecutive controls
            segs = []
            for i, c in enumerate(ctrl):
                e = end if i == len(ctrl) - 1 else ((c[0] + ctrl[i + 1][0]) / 2, (c[1] + ctrl[i + 1][1]) / 2)
                segs.append((c, e))
            for c, e in segs:
                for k in range(1, steps + 1):
                    t = k / steps
                    cur.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t**2 * e[0],
                                (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t**2 * e[1]))
                p0 = e
            pt = p0
        elif op in ("closePath", "endPath"):
            if cur:
                polys.append(cur)
            cur = []
    if cur:
        polys.append(cur)
    return polys


def _area(p):
    return 0.5 * sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))


def wordmark_geometry(cap: float = 1.0, font: Path = FONT):
    """Letters as polygons in a y-down space; the E's cap height == `cap`.

    Returns (letters, width, height) where letters = [(char, [polys], sign_outer)].
    """
    f = TTFont(str(font))
    gs = f.getGlyphSet()
    cmap = f.getBestCmap()
    upm = f["head"].unitsPerEm
    cap_units = getattr(f["OS/2"], "sCapHeight", 0) or 0.7 * upm
    n = len(WORD)
    sizes = [GROWTH ** (i - (n - 1)) for i in range(n)]       # E = 1.0
    letters, x = [], 0.0
    for ch, s in zip(WORD, sizes):
        gname = cmap[ord(ch)]
        pen = RecordingPen()
        gs[gname].draw(pen)
        k = s * cap / cap_units
        polys = [[(x + px * k, cap - py * k) for px, py in poly] for poly in _flatten(pen.value)]
        lsb = f["hmtx"][gname][1]
        if not letters:                       # start the mark exactly at the S's ink
            dx = lsb * k
            polys = [[(px - dx, py) for px, py in p] for p in polys]
            x -= dx
        biggest = max(polys, key=lambda p: abs(_area(p)))
        letters.append((ch, polys, math.copysign(1, _area(biggest))))
        x += f["hmtx"][gname][0] * k + TRACKING * s * cap
    xmax = max(px for _, polys, _ in letters for p in polys for px, _ in p)
    ymin = min(py for _, polys, _ in letters for p in polys for _, py in p)
    ymax = max(py for _, polys, _ in letters for p in polys for _, py in p)
    return letters, xmax, ymax, ymin


def draw_wordmark(img: Image.Image, x0: float, y0: float, cap: float, ink=INK, spark=SPARK, ss: int = 4):
    """Paint the wordmark with its top-left at (x0, y0) and E cap height `cap` (pixels)."""
    letters, w, h, ymin = wordmark_geometry(cap)
    W, H = int(math.ceil(w)) + 4, int(math.ceil(h - ymin)) + 4
    for idx, (ch, polys, sign) in enumerate(letters):
        mask = Image.new("L", (W * ss, H * ss), 0)
        d = ImageDraw.Draw(mask)
        for p in sorted(polys, key=lambda p: -abs(_area(p))):
            fill = 255 if math.copysign(1, _area(p)) == sign else 0
            d.polygon([((px + 2) * ss, (py - ymin + 2) * ss) for px, py in p], fill=fill)
        mask = mask.resize((W, H), Image.LANCZOS)
        color = spark if (idx == 0 and spark) else ink
        layer = Image.new("RGBA", (W, H), color + (255,))
        layer.putalpha(mask)
        img.alpha_composite(layer, (int(round(x0 - 2)), int(round(y0 + ymin - 2))))
    return w, h - ymin


def wordmark_svg(ink="#F4EFE6", spark="#E8A657", bg: str | None = None, cap: float = 400) -> str:
    letters, w, h, ymin = wordmark_geometry(cap)
    pad = cap * 0.12
    vw, vh = w + 2 * pad, h - ymin + 2 * pad
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw:.1f} {vh:.1f}" width="{vw:.0f}" height="{vh:.0f}">',
           '<title>SCALE</title>']
    if bg:
        out.append(f'<rect width="100%" height="100%" fill="{bg}"/>')
    for idx, (ch, polys, _) in enumerate(letters):
        d = " ".join("M" + " L".join(f"{px + pad:.2f} {py - ymin + pad:.2f}" for px, py in p) + " Z" for p in polys)
        color = spark if (idx == 0 and spark) else ink
        out.append(f'<path fill="{color}" fill-rule="nonzero" d="{d}"/>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------- rasters
def logo_png(path: Path, ink, bg, spark=SPARK, cap: int = 600):
    letters, w, h, ymin = wordmark_geometry(cap)
    pad = int(cap * 0.18)
    W, H = int(w + 2 * pad), int(h - ymin + 2 * pad)
    img = Image.new("RGBA", (W, H), (bg + (255,)) if bg else (0, 0, 0, 0))
    draw_wordmark(img, pad, pad, cap, ink, spark)
    img.save(path)
    return img


def _starfield(w, h, seed=11, density=0.0007):
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w), np.float32)
    n = int(w * h * density)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    a[ys, xs] = np.clip(rng.pareto(2.5, n) * 0.35, 0, 1.4)
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(0.6))
    return np.asarray(img, np.float32) / 255 * 1.6


def _void(w, h, glow_xy=None, glow_r=None, seed=11):
    """Deep-space backdrop: near-black, faint stars, optional warm bloom."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.ones((h, w, 3), np.float32) * np.array(VOID, np.float32) / 255
    stars = _starfield(w, h, seed)[..., None] * np.array([0.85, 0.88, 1.0])
    base += stars * 0.55
    if glow_xy:
        gx, gy = glow_xy
        d = np.sqrt((x - gx) ** 2 + (y - gy) ** 2) / glow_r
        base += (np.exp(-d**2) * 0.22)[..., None] * np.array(SPARK, np.float32) / 255
        base += (np.exp(-d**2 * 4) * 0.12)[..., None]
    return Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")


def icon_png(path: Path, size: int = 800):
    """Profile picture: the full growing mark inside YouTube's circle crop.

    The tiny amber S reads as a spark at 48px while the A-L-E stair carries the
    shape; the concept survives even when the S is a single pixel.
    """
    img = _void(size, size, (size * 0.36, size * 0.62), size * 0.55, seed=5)
    letters, w, h, ymin = wordmark_geometry(1.0)
    usable = size * 0.74                       # inside the circle crop
    cap = usable / w
    mark_h = (h - ymin) * cap
    draw_wordmark(img, (size - w * cap) / 2, (size - mark_h) / 2 + size * 0.02, cap)
    img.convert("RGB").save(path)
    return img


def banner_png(path: Path, eye_img: Path | None = None):
    """2560x1440 banner. Everything that matters sits in the 1546x423 safe area
    (visible on every device); the outer field is cinematic atmosphere for TV/desktop."""
    W, H = 2560, 1440
    sw, sh = 1546, 423
    sx, sy = (W - sw) // 2, (H - sh) // 2
    img = _void(W, H, (W * 0.5, H * 0.52), W * 0.30, seed=21)
    cap = sh * 0.50
    letters, w, h, ymin = wordmark_geometry(cap)
    mx = sx + (sw - w) / 2
    my = sy + sh * 0.14
    # powers-of-ten rings radiating from the tiny S: the zoom, drawn once
    s_polys = letters[0][1]
    scx = mx + sum(px for p in s_polys for px, _ in p) / sum(len(p) for p in s_polys)
    scy = my + sum(py for p in s_polys for _, py in p) / sum(len(p) for p in s_polys) - ymin
    rings = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(rings)
    r, k = cap * 0.09, 0
    while r < W * 1.3:
        a = int(max(14, 90 - k * 8))
        col = SPARK if k == 0 else INK
        d.ellipse([scx - r, scy - r, scx + r, scy + r], outline=col + (a,), width=2 if k < 4 else 3)
        r *= 1.62
        k += 1
    img.alpha_composite(rings)
    draw_wordmark(img, mx, my, cap)
    f = ImageFont.truetype(str(FONT_LIGHT), int(sh * 0.085))
    tag = "ONE ZOOM.  EVERYTHING."
    d = ImageDraw.Draw(img)
    _spaced(d, (W / 2, my + (h - ymin) + sh * 0.13), tag, f, INK + (215,), spacing=0.32)
    f2 = ImageFont.truetype(str(FONT_TEXT), int(sh * 0.05))
    _spaced(d, (W / 2, sy + sh - sh * 0.04), "THE UNIVERSE, ONE ZOOM AT A TIME", f2, DIM + (200,), spacing=0.4)
    img.convert("RGB").save(path, quality=95)
    return img


def _spaced(d: ImageDraw.ImageDraw, center, text, font, fill, spacing=0.2):
    """Letter-spaced text centered on `center` (x center, y middle)."""
    size = font.size
    widths = [d.textlength(c, font=font) for c in text]
    total = sum(widths) + spacing * size * (len(text) - 1)
    x = center[0] - total / 2
    for c, cw in zip(text, widths):
        d.text((x, center[1]), c, font=font, fill=fill, anchor="lm")
        x += cw + spacing * size


def _cover(img: Image.Image, w: int, h: int, focus=(0.5, 0.5)) -> Image.Image:
    s = max(w / img.width, h / img.height)
    im = img.resize((int(math.ceil(img.width * s)), int(math.ceil(img.height * s))), Image.LANCZOS)
    x = int(min(max(0, focus[0] * im.width - w / 2), im.width - w))
    y = int(min(max(0, focus[1] * im.height - h / 2), im.height - h))
    return im.crop((x, y, x + w, y + h))


def _eye_with_universe(eye: Image.Image, inset: Image.Image, iris) -> Image.Image:
    """The hero image: the universe visible inside the pupil."""
    eye = eye.convert("RGB")
    cx, cy, r = iris[0] * eye.width, iris[1] * eye.height, iris[2] * eye.width
    d = int(r * 2)
    uni = _cover(inset.convert("RGB"), d, d)
    m = Image.new("L", (d * 4, d * 4), 0)
    ImageDraw.Draw(m).ellipse([0, 0, d * 4 - 1, d * 4 - 1], fill=255)
    m = m.resize((d, d), Image.LANCZOS).filter(ImageFilter.GaussianBlur(d * 0.06))
    # screen blend keeps the iris texture and adds the cosmos on top
    base = eye.crop((int(cx - r), int(cy - r), int(cx - r) + d, int(cy - r) + d))
    a = np.asarray(base, np.float32) / 255
    b = np.asarray(uni, np.float32) / 255
    scr = 1 - (1 - a) * (1 - b * 1.25)
    blended = Image.fromarray((np.clip(scr, 0, 1) * 255).astype(np.uint8))
    eye.paste(blended, (int(cx - r), int(cy - r)), m)
    return eye


def _fit_font(d, text, path, max_w, start):
    size = start
    while size > 10:
        f = ImageFont.truetype(str(path), size)
        if d.textlength(text, font=f) <= max_w:
            return f
        size -= 4
    return ImageFont.truetype(str(path), size)


def thumbnails(ep, layer_paths: list[Path], out_dir: Path) -> list[Path]:
    """1280x720 custom thumbnail (what the API uploads) + a 1080x1920 cover frame."""
    cfg = ep.raw.get("thumbnail", {})
    ids = [l.id for l in ep.layers]
    eye = Image.open(layer_paths[0])
    inset_id = cfg.get("inset_layer", ids[-1])
    inset = Image.open(layer_paths[ids.index(inset_id)])
    hero = _eye_with_universe(eye, inset, cfg.get("iris", (0.5, 0.5, 0.12)))
    head, kick, foot = cfg.get("headline", "SCALE"), cfg.get("kicker", ""), cfg.get("footer", "")
    made = []

    # ---- 16:9: hero on the left two-thirds, type on the right
    W, H = 1280, 720
    t = _void(W, H, (W * 0.72, H * 0.5), W * 0.35, seed=31)
    iris = cfg.get("iris", (0.5, 0.5, 0.12))
    hw = int(W * 0.68)
    r_px = iris[2] * hero.width
    box_w = r_px * 6.5                            # frame the eye tight: iris ~ 30% of the crop
    box_h = box_w * H / hw
    cx, cy = iris[0] * hero.width, iris[1] * hero.height
    hero_crop = hero.crop((int(cx - box_w * 0.52), int(cy - box_h / 2), int(cx + box_w * 0.48), int(cy + box_h / 2)))
    hero_crop = hero_crop.resize((hw, H), Image.LANCZOS)
    xs = np.arange(hw, dtype=np.float32)
    ramp = np.clip((hw - xs) / (hw * 0.32), 0, 1) ** 1.4
    fade = Image.fromarray((np.tile(ramp, (H, 1)) * 255).astype(np.uint8), "L")
    t.paste(hero_crop.convert("RGBA"), (0, 0), fade)
    d = ImageDraw.Draw(t)
    tx, tw = int(W * 0.60), int(W * 0.37)
    f_kick = _fit_font(d, kick, FONT_TEXT, tw, 44)
    f_head = _fit_font(d, head, FONT, tw, 190)
    f_foot = _fit_font(d, foot, FONT_TEXT, tw, 40)
    y = H * 0.30
    d.text((tx, y), kick, font=f_kick, fill=SPARK + (255,), anchor="ls")
    d.text((tx - 4, y + f_head.size * 0.95), head, font=f_head, fill=INK + (255,), anchor="ls",
           stroke_width=2, stroke_fill=VOID + (255,))
    d.text((tx, y + f_head.size * 0.95 + f_foot.size * 1.7), foot, font=f_foot, fill=INK + (225,), anchor="ls")
    lg_cap = 34
    letters, lw, lh, lymin = wordmark_geometry(lg_cap)
    draw_wordmark(t, W - lw - 40, H - (lh - lymin) - 34, lg_cap)
    p = out_dir / "thumbnail_1280x720.jpg"
    t.convert("RGB").save(p, quality=92)
    made.append(p)

    # ---- 9:16 cover
    W, H = 1080, 1920
    c = _cover(hero, W, H, focus=(iris[0], iris[1])).convert("RGBA")
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ds = ImageDraw.Draw(shade)
    for i in range(H // 3):
        a = int(200 * (1 - i / (H / 3)) ** 1.6)
        ds.line([(0, i), (W, i)], fill=VOID + (a,))
        ds.line([(0, H - 1 - i), (W, H - 1 - i)], fill=VOID + (a,))
    c.alpha_composite(shade)
    d = ImageDraw.Draw(c)
    f_head = _fit_font(d, head, FONT, int(W * 0.86), 240)
    d.text((W / 2, H * 0.17), kick, font=_fit_font(d, kick, FONT_TEXT, W * 0.8, 64), fill=SPARK + (255,), anchor="mm")
    d.text((W / 2, H * 0.17 + f_head.size * 0.85), head, font=f_head, fill=INK + (255,), anchor="mm")
    d.text((W / 2, H * 0.80), foot, font=_fit_font(d, foot, FONT_TEXT, W * 0.86, 60), fill=INK + (235,), anchor="mm")
    letters, lw, lh, lymin = wordmark_geometry(60)
    draw_wordmark(c, (W - lw) / 2, H * 0.86, 60)
    p = out_dir / "cover_1080x1920.jpg"
    c.convert("RGB").save(p, quality=92)
    made.append(p)
    return made


def build_all() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    (OUT / "scale_logo.svg").write_text(wordmark_svg(), encoding="utf-8")
    (OUT / "scale_logo_dark.svg").write_text(wordmark_svg(bg="#07080B"), encoding="utf-8")
    (OUT / "scale_logo_light.svg").write_text(wordmark_svg(ink="#0B0C10", spark="#C27A2C", bg="#F4EFE6"), encoding="utf-8")
    made += [OUT / "scale_logo.svg", OUT / "scale_logo_dark.svg", OUT / "scale_logo_light.svg"]
    logo_png(OUT / "scale_logo_transparent_white.png", INK, None)
    logo_png(OUT / "scale_logo_transparent_black.png", (11, 12, 16), None, spark=(194, 122, 44))
    logo_png(OUT / "scale_logo_dark.png", INK, VOID)
    logo_png(OUT / "scale_logo_light.png", (11, 12, 16), INK, spark=(194, 122, 44))
    icon_png(OUT / "scale_icon_800.png", 800)
    banner_png(OUT / "scale_banner_2560x1440.jpg")
    made += [OUT / n for n in ("scale_logo_transparent_white.png", "scale_logo_transparent_black.png",
                               "scale_logo_dark.png", "scale_logo_light.png", "scale_icon_800.png",
                               "scale_banner_2560x1440.jpg")]
    return made


if __name__ == "__main__":
    for p in build_all():
        print(p)
