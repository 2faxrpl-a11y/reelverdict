"""Procedural previz layers.

Every layer kind has a generator that paints a stand-in image in pure numpy, so
an episode renders end to end before any real stills exist. Each generator
places its contents to physical scale (it knows the layer's frame width in
meters), so the animatic doubles as a scale check. The cosmic layers are good
enough to ship; the human-scale ones (eye, face, room) are blocking art to be
swapped for real images.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .spec import Layer

AU = 1.495979e11
LY = 9.4607e15


# ---------------------------------------------------------------- helpers
def _resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((w, h), Image.BICUBIC))


def fbm(w: int, h: int, rng, base: int = 4, octaves: int = 5, gain: float = 0.5) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gx = base * 2**o
        gy = max(2, int(round(gx * h / w)))
        out += amp * _resize(rng.random((gy + 1, gx + 1)), w, h)
        total += amp
        amp *= gain
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def _box(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float32)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian approximation: three box passes per axis (works on HxW or HxWxC)."""
    if sigma <= 0.3:
        return a
    r = max(1, int(round((math.sqrt(12 * sigma**2 / 3 + 1) - 1) / 2)))
    out = a.astype(np.float32)
    for axis in (0, 1):
        for _ in range(3):
            out = _box(out, r, axis)
    return out


def bloom(img: np.ndarray, strength: float = 0.6, sigmas=(2, 8, 24)) -> np.ndarray:
    hot = np.clip(img - 0.55, 0, None)
    acc = np.zeros_like(img)
    for s in sigmas:
        acc += blur(hot, s)
    return img + strength * acc / len(sigmas)


def grid(w: int, h: int):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return x, y


def glow(img, x, y, radius, color, intensity=1.0, power=2.0):
    """Add a soft radial glow, computed only in its bounding window."""
    h, w = img.shape[:2]
    r = radius * min(400.0, (max(intensity, 1e-3) * 800) ** (1 / power))   # until the glow is invisible
    x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
    y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / max(radius, 1e-3)
    f = intensity / (1 + d**power)
    img[y0:y1, x0:x1] += f[..., None] * np.asarray(color, np.float32)


def splat(img, xs, ys, cols, sigma=0.8):
    """Accumulate point lights then soften them."""
    h, w = img.shape[:2]
    buf = np.zeros_like(img)
    xi, yi = xs.astype(int), ys.astype(int)
    m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    for c in range(3):
        np.add.at(buf[..., c], (yi[m], xi[m]), cols[m, c])
    img += blur(buf, sigma) * (2 * math.pi * sigma**2 if sigma > 0 else 1)


def starfield(img, rng, density=0.0009, bright=1.0):
    h, w = img.shape[:2]
    n = int(w * h * density)
    xs, ys = rng.random(n) * w, rng.random(n) * h
    mag = rng.pareto(2.2, n) * 0.08 * bright
    temp = rng.random(n)
    tint = np.stack([0.8 + 0.2 * temp, 0.85 + 0.1 * temp, 1.0 - 0.25 * temp], -1)
    splat(img, xs, ys, np.clip(mag, 0, 4)[:, None] * tint, sigma=0.7)


def tone(img: np.ndarray) -> Image.Image:
    img = bloom(img, 0.5)
    img = 1 - np.exp(-np.clip(img, 0, None) * 1.25)          # filmic shoulder
    return Image.fromarray((np.clip(img, 0, 1) ** (1 / 1.1) * 255).astype(np.uint8), "RGB")


def anchor_px(layer: Layer, w: int, h: int) -> tuple[float, float]:
    return layer.inner_at[0] * w, layer.inner_at[1] * h


def px(layer: Layer, w: int, meters: float) -> float:
    """Real-world length -> pixels on this layer."""
    return meters / layer.meters * w


# ---------------------------------------------------------------- human scale
def gen_eye(layer, w, h, rng):
    x, y = grid(w, h)
    cx, cy = w / 2, h / 2
    skin = np.array([0.55, 0.36, 0.28])
    n = fbm(w, h, rng, base=24, octaves=4)
    img = skin * (0.75 + 0.25 * n[..., None])
    img *= (0.55 + 0.45 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (0.9 * w) ** 2))[..., None]
    ew, eh = px(layer, w, 0.030) / 2, px(layer, w, 0.013) / 2
    u, v = (x - cx) / ew, (y - cy) / eh
    shape = np.clip(1 - u**2, 0, 1) ** 0.9
    almond = (np.abs(v) < shape) & (np.abs(u) < 1)
    lid = np.clip(1 - np.abs(v) / np.maximum(shape, 1e-3), 0, 1)
    veins = fbm(w, h, rng, base=40, octaves=4)
    sclera = np.array([0.93, 0.89, 0.86]) - (np.clip(veins - 0.7, 0, 1) * 1.6)[..., None] * np.array([0.0, 0.5, 0.5])
    sclera = sclera * (0.55 + 0.45 * lid[..., None] ** 0.35)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    th = np.arctan2(y - cy, x - cx)
    ri, rp = px(layer, w, 0.0118) / 2, px(layer, w, 0.0045) / 2
    fib = rng.random(720)
    fib = np.convolve(np.r_[fib, fib[:8]], np.ones(8) / 8, "same")[:720]
    f = fib[((th + math.pi) / (2 * math.pi) * 719).astype(int)]
    rn = r / ri
    outer = np.array([0.10, 0.36, 0.44])
    inner = np.array([0.78, 0.52, 0.18])
    mix = np.clip((rn - 0.35) / 0.4, 0, 1)[..., None]
    iris = inner * (1 - mix) + outer * mix
    iris = iris * (0.55 + 0.7 * f[..., None] * (0.6 + 0.4 * np.sin(rn * 40 + f * 6))[..., None])
    iris *= np.clip(1.25 - rn**6, 0.15, 1)[..., None]           # limbal ring
    eye = np.where((r < ri)[..., None], iris, sclera)
    eye = np.where((r < rp)[..., None], np.array([0.01, 0.01, 0.015]), eye)
    eye *= (0.35 + 0.65 * np.clip((v + 1.1) / 1.2, 0, 1) ** 0.6)[..., None]   # upper lid shadow
    img = np.where(almond[..., None], eye, img * (0.6 + 0.4 * (1 - np.exp(-((np.abs(v) - 1) ** 2) * 2)))[..., None])
    glow(img, cx - ri * 0.35, cy - ri * 0.4, ri * 0.08, (1.0, 1.0, 1.0), 2.0, 4)
    return tone(img * 1.1)


def _wood(w, h, rng, plank_px, base=(0.32, 0.20, 0.12)):
    x, y = grid(w, h)
    grain = fbm(w, h, rng, base=6, octaves=5)
    planks = (y // plank_px).astype(int)
    shade = rng.random(planks.max() + 2)[planks]
    seam = (np.mod(y, plank_px) < max(1, plank_px * 0.04)).astype(np.float32)
    streak = 0.5 + 0.5 * np.sin(x / (plank_px * 0.9) + grain * 9)
    wood = np.array(base) * (0.7 + 0.25 * shade[..., None] + 0.15 * streak[..., None])
    return wood * (1 - 0.6 * seam[..., None])


def gen_face(layer, w, h, rng):
    x, y = grid(w, h)
    img = _wood(w, h, rng, plank_px=px(layer, w, 0.12))
    ax, ay = anchor_px(layer, w, h)
    eye_gap = px(layer, w, 0.064)
    fx, fy = ax + eye_gap / 2, ay + px(layer, w, 0.025)
    hw, hh = px(layer, w, 0.155) / 2, px(layer, w, 0.225) / 2
    d = ((x - fx) / hw) ** 2 + ((y - fy) / hh) ** 2
    hair = (((x - fx) / (hw * 1.12)) ** 2 + ((y - fy + hh * 0.12) / (hh * 1.05)) ** 2) < 1
    img = np.where(hair[..., None], np.array([0.05, 0.035, 0.03]) * (0.7 + 0.6 * fbm(w, h, rng, 30, 3))[..., None], img)
    skin = np.array([0.62, 0.42, 0.33]) * (0.8 + 0.2 * fbm(w, h, rng, 20, 4))[..., None]
    light = np.clip(1.1 - 0.6 * (x - fx) / hw, 0.4, 1.2)[..., None]
    img = np.where((d < 1)[..., None], skin * light * (1 - 0.35 * d[..., None] ** 2), img)
    for ex in (ax, ax + eye_gap):
        e = ((x - ex) / px(layer, w, 0.015)) ** 2 + ((y - ay) / px(layer, w, 0.0065)) ** 2
        img = np.where((e < 1)[..., None], np.array([0.75, 0.70, 0.66]) * (1 - 0.5 * e[..., None]), img)
        img *= (1 - 0.35 * np.exp(-((x - ex) ** 2 + (y - ay + px(layer, w, 0.01)) ** 2) / px(layer, w, 0.02) ** 2))[..., None]
    nose = np.exp(-(((x - fx) / px(layer, w, 0.012)) ** 2 + ((y - fy - hh * 0.25) / px(layer, w, 0.02)) ** 2))
    mouth = np.exp(-(((x - fx) / px(layer, w, 0.025)) ** 2 + ((y - fy - hh * 0.55) / px(layer, w, 0.005)) ** 2))
    img *= (1 - 0.25 * nose - 0.4 * mouth)[..., None]
    glow(img, w * 0.15, h * 0.2, w * 0.4, (0.35, 0.22, 0.10), 0.5, 2)
    return tone(img)


def gen_room(layer, w, h, rng):
    x, y = grid(w, h)
    img = _wood(w, h, rng, plank_px=px(layer, w, 0.15)) * 0.8
    ax, ay = anchor_px(layer, w, h)
    rug = (np.abs(x - ax) < px(layer, w, 0.9)) & (np.abs(y - ay - px(layer, w, 0.5)) < px(layer, w, 1.3))
    img = np.where(rug[..., None], np.array([0.30, 0.10, 0.10]) * (0.7 + 0.5 * fbm(w, h, rng, 60, 3))[..., None], img)
    bed = (np.abs(x - w * 0.8) < px(layer, w, 0.75)) & (np.abs(y - h * 0.3) < px(layer, w, 1.0))
    img = np.where(bed[..., None], np.array([0.50, 0.52, 0.58]) * (0.8 + 0.3 * fbm(w, h, rng, 12, 3))[..., None], img)
    # person lying on the floor, head at the anchor
    hr = px(layer, w, 0.11)
    head = (x - ax) ** 2 + (y - ay) ** 2 < hr**2
    torso = (np.abs(x - ax) < px(layer, w, 0.2)) & (y > ay + hr * 0.6) & (y < ay + px(layer, w, 0.85))
    legs = (np.abs(np.abs(x - ax) - px(layer, w, 0.09)) < px(layer, w, 0.07)) & (y > ay + px(layer, w, 0.8)) & (y < ay + px(layer, w, 1.7))
    img = np.where(torso[..., None] | legs[..., None], np.array([0.12, 0.14, 0.2]), img)
    img = np.where(head[..., None], np.array([0.08, 0.05, 0.04]), img)
    for _ in range(9):   # scattered books
        bx, by = rng.random() * w, rng.random() * h
        bw, bh = px(layer, w, 0.15), px(layer, w, 0.22)
        b = (np.abs(x - bx) < bw / 2) & (np.abs(y - by) < bh / 2)
        img = np.where(b[..., None], rng.random(3) * 0.5, img)
    img *= 0.35
    glow(img, w * 0.18, h * 0.62, px(layer, w, 0.6), (1.0, 0.62, 0.28), 1.2, 1.6)
    glow(img, ax, ay, px(layer, w, 1.4), (0.7, 0.45, 0.25), 0.25, 2)
    return tone(img)


def gen_rooftop(layer, w, h, rng):
    x, y = grid(w, h)
    n = fbm(w, h, rng, base=10, octaves=6)
    img = np.array([0.10, 0.10, 0.11]) * (0.6 + 0.6 * n[..., None])
    puddle = fbm(w, h, rng, base=5, octaves=4) > 0.72
    img = np.where(puddle[..., None], np.array([0.20, 0.17, 0.22]), img)
    m = px(layer, w, 0.6)
    edge = (x < m) | (x > w - m) | (y < h * 0.12) | (y > h * 0.88)
    img = np.where(edge[..., None], np.array([0.03, 0.03, 0.035]), img)
    for _ in range(6):
        bx, by = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
        if abs(bx - w / 2) < w * 0.15 and abs(by - h / 2) < w * 0.15:
            continue
        s = px(layer, w, rng.uniform(1.2, 2.5))
        b = (np.abs(x - bx) < s) & (np.abs(y - by) < s * 0.7)
        img = np.where(b[..., None], np.array([0.22, 0.23, 0.25]) * (0.7 + 0.3 * n[..., None]), img)
    sw = px(layer, w, 4.0)
    frame = (np.abs(x - w / 2) < sw * 0.55) & (np.abs(y - h / 2) < sw * 0.55 * 16 / 9)
    img = np.where(frame[..., None], np.array([0.05, 0.05, 0.05]), img)
    img *= 0.6
    glow(img, w / 2, h / 2, sw * 0.9, (1.0, 0.6, 0.3), 0.5, 2)
    return tone(img)


# ---------------------------------------------------------------- planetary
def gen_night_city(layer, w, h, rng):
    p = layer.params
    x, y = grid(w, h)
    cx, cy = w / 2, h / 2
    rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / w
    dens = fbm(w, h, rng, base=6, octaves=6) * np.exp(-rr / p.get("falloff", 1.0)) * p.get("density", 0.7)
    land = np.ones((h, w), np.float32)
    if p.get("coast"):
        c = fbm(w, h, rng, base=3, octaves=6)
        land = ((x - cx) / w + (c - 0.5) * 0.6 < 0.18).astype(np.float32)
        land = blur(land, 1.5)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([0.010, 0.012, 0.02]) * (1 - land[..., None])
    img += np.array([0.015, 0.013, 0.012]) * land[..., None]
    sodium = np.array([1.0, 0.62, 0.25])
    blocks = p.get("blocks", 0)
    if blocks:
        step = w / blocks
        ang = rng.uniform(-0.08, 0.08)
        xr = (x - cx) * math.cos(ang) - (y - cy) * math.sin(ang)
        yr = (x - cx) * math.sin(ang) + (y - cy) * math.cos(ang)
        lw = max(0.8, step * 0.05)
        streets = np.maximum(np.exp(-(np.mod(xr, step) - step / 2) ** 2 / lw**2),
                             np.exp(-(np.mod(yr, step) - step / 2) ** 2 / lw**2))
        img += sodium * (streets * (0.15 + dens) * land)[..., None] * 0.45
    n = int(w * h * 0.02 * p.get("density", 0.7))
    xs, ys = rng.random(n) * w, rng.random(n) * h
    keep = rng.random(n) < dens[ys.astype(int), xs.astype(int)] * land[ys.astype(int), xs.astype(int)] * 1.6
    xs, ys = xs[keep], ys[keep]
    cols = np.where(rng.random(len(xs))[:, None] < 0.8, sodium, np.array([0.8, 0.9, 1.0])) * rng.pareto(3, len(xs))[:, None] * 0.25
    splat(img, xs, ys, cols, sigma=0.7)
    for _ in range(int(p.get("roads", 7))):          # radial highways
        a = rng.uniform(0, 2 * math.pi)
        t = np.linspace(0, w * 1.2, 3000)
        wob = np.cumsum(rng.normal(0, 0.0015, 3000))
        hx, hy = cx + t * np.cos(a + wob), cy + t * np.sin(a + wob)
        cols = np.tile(sodium * 0.25, (3000, 1)) * np.exp(-t / w * 2)[:, None]
        splat(img, hx, hy, cols, sigma=0.6)
    img += blur(img, w * 0.01) * p.get("glow", 1.0) * 1.5
    glow(img, cx, cy, w * 0.04, sodium, 0.25, 2)
    return tone(img * 1.4)


def _globe(layer, w, h, rng, night=True):
    x, y = grid(w, h)
    cx, cy = w / 2, h / 2
    R = px(layer, w, 6.371e6)
    nx, ny = (x - cx) / R, (y - cy) / R
    r2 = nx**2 + ny**2
    inside = r2 < 1
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    img = np.zeros((h, w, 3), np.float32)
    starfield(img, rng, 0.0006, 0.6)
    # texture lookup on the sphere (equirect-ish so it curves properly)
    tex_w = 2048
    tex = fbm(tex_w, tex_w // 2, rng, base=4, octaves=7)
    cloud = fbm(tex_w, tex_w // 2, rng, base=6, octaves=6)
    lit = fbm(tex_w, tex_w // 2, rng, base=24, octaves=4)
    lon = np.arctan2(nx, nz)
    lat = np.arcsin(np.clip(ny, -1, 1))
    ti = ((lon / math.pi + 1) / 2 * (tex_w - 1) * 0.5 + tex_w * 0.25).astype(int) % tex_w
    tj = ((lat / math.pi + 0.5) * (tex_w // 2 - 1)).astype(int)
    h_ = tex[tj, ti]
    h_ = h_ + 0.25 * np.exp(-r2 * 30)                    # guarantee land under the city
    land = h_ > 0.52
    cl = np.clip((cloud[tj, ti] - 0.55) * 3, 0, 1)
    ocean = np.array([0.02, 0.07, 0.18])
    ground = np.array([0.16, 0.18, 0.10]) * (0.6 + 0.6 * h_[..., None])
    day = np.where(land[..., None], ground, ocean)
    day = day * (1 - cl[..., None]) + np.array([0.85, 0.88, 0.92]) * cl[..., None]
    sun = np.array([0.82, -0.25, -0.12]) if night else np.array([0.3, -0.3, 0.9])
    sun = sun / np.linalg.norm(sun)
    ndl = nx * sun[0] + ny * sun[1] + nz * sun[2]
    dayf = np.clip(ndl * 3 + 0.15, 0, 1)
    lights = (land & (lit[tj, ti] > 0.62)).astype(np.float32) * (lit[tj, ti] - 0.62) * 6
    lights += np.exp(-r2 * 900) * 2.5                     # our city at the sub-camera point
    lights *= (1 - dayf) * (1 - 0.7 * cl)
    surf = day * dayf[..., None] * 1.3 + np.array([1.0, 0.6, 0.25]) * lights[..., None] * 0.6
    surf = surf * (0.3 + 0.7 * nz[..., None] ** 0.4)
    img = np.where(inside[..., None], surf, img)
    rim = np.exp(-((np.sqrt(r2) - 1) * R / max(2.0, R * 0.025)) ** 2)
    rim *= np.clip(ndl + 0.5, 0, 1)
    img += np.array([0.25, 0.5, 1.0]) * rim[..., None] * 0.8
    return tone(img)


def gen_globe(layer, w, h, rng):
    return _globe(layer, w, h, rng, night=layer.params.get("night", True))


def gen_earth_moon(layer, w, h, rng):
    img = np.zeros((h, w, 3), np.float32)
    starfield(img, rng, 0.0009, 0.8)
    cx, cy = w / 2, h / 2
    re = max(1.2, px(layer, w, 6.371e6))
    glow(img, cx, cy, re, (0.35, 0.55, 1.0), 2.0, 6)
    d = px(layer, w, 3.844e8)
    a = math.radians(-38)
    x, y = grid(w, h)
    ring = np.exp(-((np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - d) / 1.0) ** 2) * 0.03
    img += ring[..., None] * np.array([0.6, 0.7, 1.0])
    glow(img, cx + d * math.cos(a), cy + d * math.sin(a), max(1.0, px(layer, w, 1.737e6)), (0.75, 0.75, 0.72), 1.4, 6)
    return tone(img)


def gen_solar(layer, w, h, rng):
    img = np.zeros((h, w, 3), np.float32)
    starfield(img, rng, 0.0012, 0.8)
    cx, cy = w / 2, h / 2
    x, y = grid(w, h)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    sun_r = max(1.5, layer.params.get("sun_px", 0.01) * w)
    glow(img, cx, cy, sun_r, (1.0, 0.85, 0.55), 3.0, 2.2)
    glow(img, cx, cy, sun_r * 8, (1.0, 0.6, 0.3), 0.25, 2)
    colors = [(0.7, 0.65, 0.6), (0.95, 0.85, 0.6), (0.4, 0.6, 1.0), (0.95, 0.5, 0.35)]
    for k, au in enumerate(layer.params.get("orbits", [1.0])):
        d = px(layer, w, au * AU)
        img += (np.exp(-((r - d) / 1.0) ** 2) * 0.05)[..., None] * np.array([0.6, 0.7, 0.9])
        a = 0.0 if abs(au - 1.0) < 1e-6 else rng.uniform(0, 2 * math.pi)
        glow(img, cx + d * math.cos(a), cy + d * math.sin(a), 1.2, colors[k % 4], 1.2, 4)
    return tone(img)


# ---------------------------------------------------------------- cosmic
def gen_stars(layer, w, h, rng):
    img = np.zeros((h, w, 3), np.float32)
    dust = fbm(w, h, rng, base=3, octaves=7)
    img += (np.clip(dust - 0.45, 0, 1) ** 1.5)[..., None] * np.array([0.10, 0.05, 0.12])
    starfield(img, rng, 0.003, 1.3)
    glow(img, w / 2, h / 2, 2.0, (1.0, 0.9, 0.6), 1.6, 3)
    return tone(img)


def _spiral(size: int, rng, n=220_000, arms=2, pitch=0.22, color_core=(1.0, 0.8, 0.5)) -> np.ndarray:
    img = np.zeros((size, size, 3), np.float32)
    c = size / 2
    r = rng.exponential(0.18, n) * size * 0.5
    r = r[r < size * 0.5]
    m = len(r)
    arm = rng.integers(0, arms, m) * (2 * math.pi / arms)
    th = np.log(np.maximum(r, 1) / (size * 0.02)) / math.tan(pitch) + arm + rng.normal(0, 0.35, m)
    th += rng.normal(0, 1, m) * np.exp(-r / (size * 0.06))           # bulge is round
    xs, ys = c + r * np.cos(th), c + r * np.sin(th)
    t = np.clip(r / (size * 0.25), 0, 1)[:, None]
    cols = (np.array(color_core) * (1 - t) + np.array([0.45, 0.6, 1.0]) * t) * 0.06
    pink = rng.random(m) < 0.02
    cols[pink] = np.array([1.0, 0.35, 0.55]) * 0.25
    splat(img, xs, ys, cols, sigma=0.6)
    img += blur(img, size * 0.01) * 1.5 + blur(img, size * 0.04) * 1.0
    glow(img, c, c, size * 0.03, color_core, 0.9, 2)
    return img


def gen_galaxy(layer, w, h, rng):
    img = np.zeros((h, w, 3), np.float32)
    starfield(img, rng, 0.0004, 0.5)
    d = int(px(layer, w, 1.0e5 * LY) * 1.35)
    s = _spiral(d, rng, n=600_000, arms=2)
    x0, y0 = w // 2 - d // 2, h // 2 - d // 2
    ys0, ys1 = max(0, y0), min(h, y0 + d)
    xs0, xs1 = max(0, x0), min(w, x0 + d)
    img[ys0:ys1, xs0:xs1] += s[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    ax, ay = anchor_px(layer, w, h)
    glow(img, ax, ay, 1.5, (1.0, 0.9, 0.6), 0.6, 3)
    return tone(img)


def _paste(img, s, cx, cy):
    h, w = img.shape[:2]
    d = s.shape[0]
    x0, y0 = int(cx - d / 2), int(cy - d / 2)
    ys0, ys1 = max(0, y0), min(h, y0 + d)
    xs0, xs1 = max(0, x0), min(w, x0 + d)
    if ys0 < ys1 and xs0 < xs1:
        img[ys0:ys1, xs0:xs1] += s[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]


def gen_local_group(layer, w, h, rng):
    img = np.zeros((h, w, 3), np.float32)
    haze = fbm(w, h, rng, base=3, octaves=6)
    img += (haze**3)[..., None] * np.array([0.03, 0.02, 0.05])
    starfield(img, rng, 0.0002, 0.4)
    ax, ay = anchor_px(layer, w, h)
    mw = max(8, int(px(layer, w, 1.0e5 * LY) * 1.35))
    _paste(img, _spiral(mw, rng, n=60_000) * 0.35, ax, ay)
    an = max(10, int(px(layer, w, 2.2e5 * LY) * 1.35))
    d = px(layer, w, 2.5e6 * LY)
    _paste(img, _spiral(an, rng, n=90_000, pitch=0.3) * 0.4, ax + d * 0.44, ay - d * 0.9)
    for _ in range(14):
        dx, dy = rng.normal(0, w * 0.25, 2)
        glow(img, ax + dx, ay + dy, rng.uniform(1, 4), (0.8, 0.75, 0.9), 0.3, 2)
    return tone(img)


def _web(w, h, rng, seeds: int, width: float) -> np.ndarray:
    sw, sh = w // 2, h // 2
    pts = rng.random((seeds, 2)) * [sw, sh]
    x, y = grid(sw, sh)
    d1 = np.full((sh, sw), np.inf, np.float32)
    d2 = np.full((sh, sw), np.inf, np.float32)
    for px_, py_ in pts:
        d = np.sqrt((x - px_) ** 2 + (y - py_) ** 2)
        nd2 = np.where(d < d1, d1, np.minimum(d2, d))
        d1 = np.minimum(d1, d)
        d2 = nd2
    edge = np.exp(-((d2 - d1) / width) ** 2)
    return _resize(edge, w, h)


def gen_cosmic_web(layer, w, h, rng):
    img = np.zeros((h, w, 3), np.float32)
    e = _web(w, h, rng, seeds=90, width=w * 0.006)
    mod = fbm(w, h, rng, base=8, octaves=5)
    fil = e * np.clip(mod - 0.2, 0, 1) ** 2.5 * 2.5
    node = np.clip(blur(fil, w * 0.01) - 0.3, 0, None) * 2
    img += fil[..., None] * np.array([0.30, 0.18, 0.50]) * 0.45
    img += node[..., None] * np.array([1.0, 0.75, 0.4]) * 0.5
    n = int(w * h * 0.01)
    xs, ys = rng.random(n) * w, rng.random(n) * h
    keep = rng.random(n) < fil[ys.astype(int), xs.astype(int)] * 0.8
    splat(img, xs[keep], ys[keep], np.tile([0.9, 0.8, 1.0], (keep.sum(), 1)) * 0.25, sigma=0.6)
    img += blur(img, w * 0.006) * 1.2
    glow(img, w / 2, h / 2, w * 0.005, (1.0, 0.8, 0.5), 0.35, 2)
    return tone(img)


def gen_observable(layer, w, h, rng):
    x, y = grid(w, h)
    cx, cy = w / 2, h / 2
    R = px(layer, w, 4.4e26)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / R
    img = np.zeros((h, w, 3), np.float32)
    e = _web(w, h, rng, seeds=500, width=w * 0.0025) * (0.4 + fbm(w, h, rng, base=16, octaves=4))
    inside = np.clip(1 - r, 0, 1) ** 0.25 * (r < 1)
    img += (e * inside)[..., None] * np.array([0.40, 0.28, 0.60]) * 0.7
    rim = np.exp(-((r - 1) / 0.035) ** 2)
    cmb = fbm(w, h, rng, base=20, octaves=4)
    img += (rim * (0.6 + 0.6 * cmb))[..., None] * np.array([1.0, 0.45, 0.2]) * 0.9
    img += (np.exp(-((r - 0.97) / 0.12) ** 2) * (r < 1))[..., None] * np.array([0.4, 0.15, 0.1]) * 0.3
    glow(img, cx, cy, w * 0.004, (1.0, 0.85, 0.6), 1.0, 2)
    return tone(img)


GENERATORS = {
    "eye": gen_eye, "face": gen_face, "room": gen_room, "rooftop": gen_rooftop,
    "night_city": gen_night_city, "globe": gen_globe, "earth_moon": gen_earth_moon,
    "solar": gen_solar, "stars": gen_stars, "galaxy": gen_galaxy,
    "local_group": gen_local_group, "cosmic_web": gen_cosmic_web, "observable": gen_observable,
}


def generate(layer: Layer, w: int, h: int, seed: int) -> Image.Image:
    if layer.kind not in GENERATORS:
        raise KeyError(f"no placeholder generator for kind '{layer.kind}' (have: {', '.join(GENERATORS)})")
    rng = np.random.default_rng(seed)
    return GENERATORS[layer.kind](layer, w, h, rng)
