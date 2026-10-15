"""Code-built cinematic layer art.

Every layer is painted from math: height fields lit with real normals, soft
shadows, strand-by-strand hair and lashes, light transport faked with glows and
bloom. Each generator works in real-world units (mm, m, km, AU, ly) so the art
is to scale, and receives the image of the layer inside it (`inner`) so the
seams between scales carry the same content.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .placeholders import (AU, LY, _resize, _spiral, _web, blur, fbm, glow, grid, splat,
                           starfield)

# ---------------------------------------------------------------- shared tools


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def finish(img: np.ndarray, bloom_amt=0.55, exposure=1.0, sat=1.0) -> Image.Image:
    """Bloom -> filmic tone map -> mild saturation, returns 8-bit RGB."""
    img = np.nan_to_num(np.clip(img * exposure, 0, None))
    hot = np.clip(img - 0.6, 0, None)
    h, w = img.shape[:2]
    acc = np.zeros_like(img)
    for s in (w * 0.002, w * 0.008, w * 0.03):
        acc += blur(hot, s)
    img = img + bloom_amt * acc / 3
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14            # ACES approximation
    img = np.clip((img * (a * img + b)) / (img * (c * img + d) + e), 0, 1)
    if sat != 1.0:
        lum = img.mean(-1, keepdims=True)
        img = np.clip(lum + (img - lum) * sat, 0, 1)
    return Image.fromarray((img ** (1 / 1.08) * 255).astype(np.uint8), "RGB")


def normals(hgt: np.ndarray, scale: float):
    gy, gx = np.gradient(hgt * scale)
    n = np.stack([-gx, -gy, np.ones_like(hgt)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def lambert(n, light, wrap=0.0):
    L = np.asarray(light, np.float32)
    L = L / np.linalg.norm(L)
    d = (n * L).sum(-1)
    return np.clip((d + wrap) / (1 + wrap), 0, 1)


def spec(n, light, power=40.0):
    L = np.asarray(light, np.float32)
    L = L / np.linalg.norm(L)
    H = L + np.array([0, 0, 1.0])
    H = H / np.linalg.norm(H)
    return np.clip((n * H).sum(-1), 0, 1) ** power


def shadow(mask: np.ndarray, dx: float, dy: float, soft: float) -> np.ndarray:
    """Soft drop shadow of `mask` offset by (dx, dy) pixels; 0 = no shadow, 1 = full."""
    m = np.roll(np.roll(mask.astype(np.float32), int(round(dy)), 0), int(round(dx)), 1)
    return np.clip(blur(m, soft), 0, 1)


def strokes(w, h, lines, ss=2) -> np.ndarray:
    """Draw tapered polylines into a coverage mask. lines = [(pts, w0, w1)] in pixels."""
    im = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(im)
    for pts, w0, w1 in lines:
        n = len(pts) - 1
        for i in range(n):
            wid = max(1, int(round((w0 + (w1 - w0) * i / max(1, n - 1)) * ss)))
            d.line([(pts[i][0] * ss, pts[i][1] * ss), (pts[i + 1][0] * ss, pts[i + 1][1] * ss)], fill=255, width=wid)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255


def bezier(p0, p1, p2, n=10):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2).tolist()


def embed(img: np.ndarray, inner: Image.Image | None, cx, cy, wpx, mask: np.ndarray | None = None, gamma=2.2):
    """Paste the inner layer's image at true scale (wpx wide, centered on cx, cy)."""
    if inner is None or wpx < 3:
        return img
    h, w = img.shape[:2]
    hpx = wpx * inner.height / inner.width
    ip = inner.convert("RGB").resize((max(1, int(round(wpx))), max(1, int(round(hpx)))), Image.LANCZOS)
    arr = (np.asarray(ip, np.float32) / 255) ** gamma
    x0, y0 = int(round(cx - ip.width / 2)), int(round(cy - ip.height / 2))
    xs0, ys0, xs1, ys1 = max(0, x0), max(0, y0), min(w, x0 + ip.width), min(h, y0 + ip.height)
    if xs0 >= xs1 or ys0 >= ys1:
        return img
    if mask is None:
        mask = np.ones((h, w), np.float32)
    m = mask[ys0:ys1, xs0:xs1, None]
    img[ys0:ys1, xs0:xs1] = img[ys0:ys1, xs0:xs1] * (1 - m) + arr[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] * m
    return img


def soft_rect(w, h, cx, cy, rw, rh, feather):
    x, y = grid(w, h)
    dx = rw / 2 - np.abs(x - cx)
    dy = rh / 2 - np.abs(y - cy)
    return smooth(0, feather, np.minimum(dx, dy))


def unpack(inner):
    return inner if inner else (None, None)


def lin(c):
    """sRGB-ish color -> linear working space."""
    return np.asarray(c, np.float32) ** 2.2


def wood_floor(w, h, rng, plank_px, tone=(0.40, 0.24, 0.13), vertical=True):
    x, y = grid(w, h)
    u, v = (x, y) if vertical else (y, x)
    idx = (u // plank_px).astype(int)
    off = rng.random(idx.max() + 2) * 10 * plank_px
    seg_len = plank_px * rng.uniform(6, 11)
    seg = ((v + off[idx]) // seg_len).astype(int)
    key = (idx * 131 + seg) % 997
    tint = rng.random(1000)[key]
    grain_n = fbm(w, h, rng, base=max(4, int(w / plank_px * 0.6)), octaves=5)
    stretched = _resize(rng.random((max(8, int(h / plank_px * 3)), max(8, int(w / plank_px * 40)))), w, h) if vertical else \
        _resize(rng.random((max(8, int(h / plank_px * 40)), max(8, int(w / plank_px * 3)))), w, h)
    rings = 0.5 + 0.5 * np.sin((u % plank_px) / plank_px * 9 + grain_n * 14 + idx[..., ] * 1.7)
    base = lin(tone) * (0.65 + 0.5 * tint[..., None])
    col = base * (0.75 + 0.18 * rings[..., None] + 0.22 * stretched[..., None])
    seam_u = np.minimum(u % plank_px, plank_px - u % plank_px)
    seam_v = np.minimum((v + off[idx]) % seg_len, seg_len - (v + off[idx]) % seg_len)
    seams = np.maximum(np.exp(-(seam_u / max(0.8, plank_px * 0.02)) ** 2), np.exp(-(seam_v / max(0.8, plank_px * 0.02)) ** 2))
    col *= (1 - 0.7 * seams)[..., None]
    sheen = fbm(w, h, rng, base=3, octaves=3)
    return col, sheen


# ---------------------------------------------------------------- the eye


def gen_eye(layer, w, h, rng, inner=None):
    mm = w / (layer.meters * 1000)                       # pixels per millimetre
    x, y = grid(w, h)
    X, Y = (x - w / 2) / mm, (y - h / 2) / mm
    Yt = Y + 0.05 * X                                     # slight canthal tilt
    ew = 14.5
    Xs = np.clip(1 - ((X - 1.0) / ew) ** 2, 0, 1)
    up = -5.3 * Xs ** 0.62
    lo = 4.1 * np.clip(1 - (X / ew) ** 2, 0, 1) ** 0.85 + 0.25
    opening = (Yt > up) & (Yt < lo) & (np.abs(X) < ew)
    d_up, d_lo = Yt - up, lo - Yt

    # --- skin height field: eyeball under the lids, crease, brow bone, cheek
    hgt = 6.5 * np.exp(-((X / 14) ** 2 + (Y / 10.5) ** 2))
    crease_y = up - 4.2 - 1.5 * Xs
    hgt -= 0.9 * np.exp(-((Yt - crease_y) / 0.9) ** 2) * (np.abs(X) < ew + 3)
    hgt += 3.0 * np.exp(-((Y + 19) / 5) ** 2)
    hgt += 1.2 * np.exp(-((Y - 13) / 5) ** 2)
    hgt -= 1.2 * np.exp(-((Yt - lo - 1.6) / 1.1) ** 2) * np.exp(-(X / 16) ** 2)    # under-eye hollow
    hgt += 0.12 * fbm(w, h, rng, base=60, octaves=3)
    n = normals(blur(hgt, 0.35 * mm), mm)                 # height in mm -> slope per mm
    key = (-0.55, -0.45, 0.70)
    diff = lambert(n, key, wrap=0.35)
    pores = fbm(w, h, rng, base=int(50 / 0.35), octaves=2)
    mott = fbm(w, h, rng, base=10, octaves=5)
    skin_c = lin((0.78, 0.55, 0.44)) * (0.85 + 0.25 * mott[..., None])
    skin_c = skin_c * (0.9 + 0.15 * pores[..., None])
    sss = lin((0.85, 0.35, 0.25))
    skin = skin_c * (0.04 + 0.95 * diff[..., None] ** 1.3) + sss * (diff * (1 - diff))[..., None] * 0.35
    skin += 0.10 * spec(n, key, 30)[..., None] * (0.5 + pores[..., None])
    lidm = np.exp(-(np.clip(-d_lo, 0, None) / 0.7) ** 2) * (Yt > lo) * (np.abs(X) < ew)    # pink lower lid margin
    skin = skin * (1 - 0.6 * lidm[..., None]) + lin((0.75, 0.42, 0.40)) * 0.8 * lidm[..., None]

    # --- eyeball
    R2 = X**2 + Y**2
    ball = np.sqrt(np.clip(1 - R2 / 12.5**2, 0, 1))
    ridge = 1 - np.abs(2 * fbm(w, h, rng, base=26, octaves=5) - 1)
    veins = np.clip((ridge - 0.9) * 6, 0, 1) * smooth(6, 13, np.abs(X)) * 0.6
    sclera = lin((0.93, 0.89, 0.86)) * (0.45 + 0.55 * ball[..., None])
    sclera = sclera * (1 - veins[..., None] * np.array([0.05, 0.55, 0.55]))
    car = np.exp(-(((X - (ew - 1.6)) / 1.6) ** 2 + ((Y - 0.9) / 1.4) ** 2))
    sclera = sclera * (1 - car[..., None]) + lin((0.80, 0.45, 0.45)) * car[..., None]

    # --- iris
    Ri = 6.0
    r = np.sqrt(R2) / Ri
    th = np.arctan2(Y, X)
    Nt = 1024

    def ptex(gt, gr):
        g = rng.random((gr, gt))
        g = np.concatenate([g, g[:, :1]], 1)
        return _resize(g, Nt + int(Nt / gt), 256)[:, :Nt]
    fib = 0.55 * ptex(320, 5) + 0.30 * ptex(640, 9) + 0.15 * ptex(160, 3)
    fib = (fib - fib.min()) / (fib.max() - fib.min())
    ti = ((th + math.pi) / (2 * math.pi) * (Nt - 1)).astype(int)
    ri = (np.clip(r, 0, 0.999) * 255).astype(int)
    f = fib[ri, ti]
    coll_r = 0.56 + 0.05 * (ptex(24, 2)[0][ti] - 0.5) * 2
    rp = 2.25 / Ri
    stroma = lin((0.32, 0.47, 0.55)) * (0.55 + 0.9 * f[..., None] ** 1.5)
    stroma += lin((0.75, 0.82, 0.86)) * (np.clip(f - 0.72, 0, 1) * 2.2)[..., None]
    inner_c = lin((0.62, 0.40, 0.16)) * (0.5 + 0.9 * f[..., None])
    cm = smooth(coll_r - 0.04, coll_r + 0.04, r)[..., None]
    iris = inner_c * (1 - cm) + stroma * cm
    iris *= (1 - 0.55 * np.exp(-((r - coll_r) / 0.025) ** 2))[..., None] ** 0.6
    for rr in (0.74, 0.86):                                        # contraction furrows
        iris *= (1 - 0.25 * np.exp(-((r - rr) / 0.012) ** 2))[..., None]
    crypt = np.zeros((h, w), np.float32)
    for _ in range(26):
        a = rng.uniform(-math.pi, math.pi)
        rc = rng.uniform(0.47, 0.84)
        dth = np.angle(np.exp(1j * (th - a)))
        crypt += np.exp(-((dth * rc / 0.05) ** 2 + ((r - rc) / 0.035) ** 2))
    iris *= (1 - 0.55 * np.clip(crypt, 0, 1))[..., None]
    iris *= (1 - 0.85 * smooth(0.84, 1.0, r))[..., None]          # limbal ring
    ruff = np.exp(-((r - rp - 0.02) / 0.018) ** 2)
    iris = iris * (1 - ruff[..., None]) + lin((0.22, 0.12, 0.06)) * ruff[..., None]
    pupil = smooth(rp + 0.006, rp - 0.006, r)
    iris = iris * (1 - pupil[..., None]) + lin((0.008, 0.008, 0.012)) * pupil[..., None]
    in_iris = smooth(1.005, 0.985, r)
    eye = sclera * (1 - in_iris[..., None]) + iris * in_iris[..., None]
    # lids and lashes shade the ball
    ao = (1 - 0.75 * np.exp(-np.clip(d_up, 0, None) / 1.5)) * (1 - 0.35 * np.exp(-np.clip(d_lo, 0, None) / 0.7))
    eye *= ao[..., None]
    eye *= (0.55 + 0.6 * lambert(np.stack([X / 12.5, Y / 12.5, ball], -1), key, 0.4))[..., None]

    # --- cornea reflections
    cornea = smooth(6.4, 5.9, np.sqrt(R2))
    env = np.exp(-(((X + 2.5) / 4.5) ** 2 + ((Y + 3.5) / 3.0) ** 2)) * 0.10
    sb_x, sb_y = -1.9, -1.7                                         # softbox catchlight with window bars
    sup = (np.abs(X - sb_x) / 0.95) ** 4 + (np.abs(Y - sb_y) / 0.7) ** 4        # rounded softbox
    box = smooth(1.15, 0.75, sup)
    bars = (1 - 0.7 * np.exp(-((X - sb_x) / 0.05) ** 2)) * (1 - 0.7 * np.exp(-((Y - sb_y) / 0.05) ** 2))
    catch = blur(box * bars, mm * 0.04) * 2.6
    small = np.exp(-(((X - 2.6) / 0.28) ** 2 + ((Y - 2.1) / 0.22) ** 2)) * 1.6
    eye += ((env + catch + small) * cornea)[..., None] * lin((1.0, 0.97, 0.92))
    tear = np.exp(-((Yt - (lo - 0.28)) / 0.13) ** 2) * (np.abs(X) < ew - 1.2) * (0.4 + fbm(w, h, rng, 70, 2))
    eye += (tear * 0.35)[..., None]

    opening_soft = blur(opening.astype(np.float32), mm * 0.12)
    img = skin * (1 - opening_soft[..., None]) + eye * opening_soft[..., None]

    # --- lashes and brow, strand by strand
    lines = []
    for i in range(130):
        t = rng.uniform(-0.93, 0.88)
        bx = t * ew + rng.normal(0, 0.15)
        by = -5.3 * np.clip(1 - ((bx - 1.0) / ew) ** 2, 0, 1) ** 0.62 - 0.05 * bx - 0.15
        out = bx / ew
        L = (7.5 - 3.5 * max(0, out) ** 2) * rng.uniform(0.75, 1.15)
        ang = math.atan2(-1.0, out * 1.4) + rng.normal(0, 0.12)
        dx, dy = math.cos(ang), math.sin(ang)
        curl = 0.35 * L
        p0 = (w / 2 + bx * mm, h / 2 + by * mm)
        p1 = (p0[0] + dx * L * 0.5 * mm - dy * curl * 0.2 * mm * np.sign(out + 1e-3), p0[1] + dy * L * 0.5 * mm)
        p2 = (p0[0] + dx * L * mm + out * curl * mm * 0.5, p0[1] + dy * L * mm * 0.75 - curl * 0.25 * mm)
        lines.append((bezier(p0, p1, p2, 9), 0.16 * mm, 0.03 * mm))
    for i in range(55):
        t = rng.uniform(-0.85, 0.75)
        bx = t * ew
        by = 4.1 * np.clip(1 - (bx / ew) ** 2, 0, 1) ** 0.85 + 0.45 - 0.05 * bx
        L = rng.uniform(2.0, 3.8)
        ang = math.atan2(1.0, t * 1.2) + rng.normal(0, 0.15)
        p0 = (w / 2 + bx * mm, h / 2 + by * mm)
        p2 = (p0[0] + math.cos(ang) * L * mm, p0[1] + math.sin(ang) * L * mm)
        lines.append((bezier(p0, ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2 + 0.3 * mm), p2, 6), 0.09 * mm, 0.02 * mm))
    for i in range(900):                                            # brow
        bx = rng.uniform(-24, 22)
        by = -21 - 3.5 * math.exp(-((bx + 2) / 14) ** 2) + rng.normal(0, 1.3) + 0.04 * bx
        L = rng.uniform(2.5, 5.0)
        ang = math.radians(-12 if bx < 6 else -28) + rng.normal(0, 0.18) + (0.4 if bx > 14 else 0)
        p0 = (w / 2 + bx * mm, h / 2 + by * mm)
        p2 = (p0[0] - math.cos(ang) * L * mm, p0[1] + math.sin(ang) * L * mm)
        lines.append((bezier(p0, ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2 - 0.4 * mm), p2, 5), 0.10 * mm, 0.03 * mm))
    hair = strokes(w, h, lines, ss=2)
    lash_shadow = blur(np.roll(hair, int(0.9 * mm), 0), 0.6 * mm) * opening_soft
    img *= (1 - 0.6 * lash_shadow)[..., None]
    img = img * (1 - 0.92 * hair[..., None]) + lin((0.035, 0.022, 0.018)) * 0.92 * hair[..., None]
    img += (hair * 0.04)[..., None] * lin((0.9, 0.7, 0.5))                 # rim light on hairs
    # warm lamp falloff + cool fill on the shadow side
    img *= (0.62 + 0.45 * np.exp(-(((X + 6) / 40) ** 2 + ((Y + 4) / 40) ** 2)))[..., None]
    img += lin((0.08, 0.12, 0.20)) * 0.12 * (1 - diff[..., None]) * (1 - opening_soft[..., None])
    return finish(img * 1.15, bloom_amt=0.45)


# ---------------------------------------------------------------- the face


def gen_face(layer, w, h, rng, inner=None):
    cm = w / (layer.meters * 100)
    x, y = grid(w, h)
    ax, ay = layer.inner_at[0] * w, layer.inner_at[1] * h
    gap = 6.4 * cm
    hx, hy = ax + gap / 2, ay + 1.2 * cm                          # head center
    X, Y = (x - hx) / cm, (y - hy) / cm                            # cm, head-centered
    ex = gap / 2 / cm
    ey = (ay - hy) / cm

    floor, sheen = wood_floor(w, h, rng, plank_px=12 * cm, tone=(0.36, 0.21, 0.12))
    # head height field
    head = np.clip(1 - (X / 7.4) ** 2 - ((Y + 0.6) / 10.6) ** 2, 0, 1)
    hgt = 8.5 * np.sqrt(head)
    for sx in (-ex, ex):
        hgt -= 1.3 * np.exp(-(((X - sx) / 1.9) ** 2 + ((Y - ey - 0.2) / 1.4) ** 2))       # sockets
        hgt += 0.7 * np.exp(-(((X - sx) / 1.1) ** 2 + ((Y - ey) / 0.7) ** 2))             # eyeballs
        hgt += 0.7 * np.exp(-(((X - sx * 1.25) / 1.8) ** 2 + ((Y - ey - 2.6) / 1.2) ** 2))  # cheekbones
    hgt += 0.7 * np.exp(-((X / 4.5) ** 2 + ((Y - ey + 1.9) / 0.8) ** 2))                  # brow ridge
    t = np.clip((Y - ey) / 4.6, 0, 1)
    hgt += (0.6 + 2.0 * t**1.4) * np.exp(-(X / (0.75 + 0.6 * t)) ** 2) * ((Y - ey) > -0.8) * ((Y - ey) < 5.2)
    hgt += 0.8 * np.exp(-((X / 1.5) ** 2 + ((Y - ey - 4.9) / 0.7) ** 2))                  # nose tip
    hgt += 0.45 * np.exp(-((X / 2.3) ** 2 + ((Y - ey - 7.0) / 0.45) ** 2))                # upper lip
    hgt += 0.55 * np.exp(-((X / 2.1) ** 2 + ((Y - ey - 7.9) / 0.55) ** 2))                # lower lip
    hgt += 0.6 * np.exp(-((X / 2.4) ** 2 + ((Y - ey - 10.2) / 1.3) ** 2))                 # chin
    n = normals(blur(hgt, 0.3 * cm), cm)
    key, fill = (-0.6, -0.35, 0.75), (0.6, 0.2, 0.5)
    d = lambert(n, key, 0.45)
    df = lambert(n, fill, 0.1)
    mott = fbm(w, h, rng, base=24, octaves=5)
    pores = fbm(w, h, rng, base=int(w / (0.06 * cm)), octaves=1)
    skin_c = lin((0.80, 0.56, 0.45)) * (0.88 + 0.2 * mott[..., None]) * (0.95 + 0.08 * pores[..., None])
    skin = skin_c * (0.03 + 1.25 * d[..., None] ** 1.4) + lin((0.25, 0.35, 0.55)) * 0.08 * df[..., None]
    skin += lin((0.9, 0.3, 0.2)) * (d * (1 - d))[..., None] * 0.35
    skin += 0.09 * spec(n, key, 25)[..., None]
    nost = sum(np.exp(-(((X - s) / 0.45) ** 2 + ((Y - ey - 5.4) / 0.3) ** 2)) for s in (-0.8, 0.8))
    mouth = np.exp(-((X / 2.2) ** 2 + ((Y - ey - 7.45) / 0.09) ** 2)) * (np.abs(X) < 2.4)
    lips = (np.exp(-((X / 2.2) ** 2 + ((Y - ey - 7.0) / 0.45) ** 2)) + np.exp(-((X / 2.0) ** 2 + ((Y - ey - 7.9) / 0.55) ** 2)))
    skin = skin * (1 - 0.15 * np.clip(lips, 0, 1)[..., None]) + lin((0.62, 0.30, 0.28)) * 0.25 * np.clip(lips, 0, 1)[..., None] * d[..., None]
    skin *= (1 - 0.8 * nost - 0.7 * mouth)[..., None]
    lidcrease = sum(np.exp(-(((X - s) / 1.6) ** 2 + ((Y - ey + 0.75) / 0.12) ** 2)) for s in (-ex, ex))
    skin *= (1 - 0.35 * lidcrease)[..., None]

    # neck and shirt below the chin
    neck = (np.abs(X) < 4.6) & (Y > 6) & (Y < 17)
    neck_h = np.sqrt(np.clip(1 - (X / 4.6) ** 2, 0, 1)) * neck
    shirt = (Y > 15.5 + 0.02 * X**2) & (np.abs(X) < 22)
    folds = fbm(w, h, rng, base=7, octaves=5)
    shirt_c = lin((0.20, 0.24, 0.34)) * (0.4 + 1.0 * folds[..., None]) * (0.4 + 0.6 * np.clip(1 - (X / 26) ** 2, 0, 1)[..., None])
    img = floor * (0.25 + 0.75 * np.exp(-(((x - w * 0.1) / (w * 0.9)) ** 2))[..., None])
    img += lin((0.6, 0.45, 0.3)) * 0.06 * sheen[..., None]
    body_m = np.clip(head * 50, 0, 1)
    contact = shadow(np.maximum(body_m, neck | shirt), 0.8 * cm, 0.5 * cm, 1.6 * cm)
    img *= (1 - 0.75 * contact)[..., None]
    img = np.where(shirt[..., None], shirt_c, img)
    neck_c = skin_c * (0.08 + 0.7 * neck_h[..., None]) * (0.6 + 0.4 * smooth(17, 8, Y)[..., None])
    img = np.where((neck & ~shirt)[..., None], neck_c, img)
    face_m = smooth(0.0, 0.02, head)
    img = img * (1 - face_m[..., None]) + skin * face_m[..., None]

    # the eyes: the real eye layer at true scale (and its mirror)
    inner, il = unpack(inner)
    eye_w = (il.meters / layer.meters) * w if il else 0
    mm_px = eye_w / 50.0 if eye_w else cm / 10
    for sx, flip in ((ax, False), (ax + gap, True)):
        # carry the eye layer's whole neighborhood (lid, lashes, brow, skin) so both scales agree
        em = np.exp(-(((x - sx) / (21 * mm_px)) ** 6 + ((y - ay + 9 * mm_px) / (17 * mm_px)) ** 6))
        im = inner.transpose(Image.FLIP_LEFT_RIGHT) if (inner is not None and flip) else inner
        img = embed(img, im, sx, ay, eye_w, mask=em)

    # hair fanned out on the floor around the crown
    lines = []
    for i in range(1800):
        a = rng.uniform(-math.pi * 0.98, -math.pi * 0.02)
        a2 = a + rng.normal(0, 0.25)
        r0 = rng.uniform(0.75, 0.98)
        sx0 = hx + math.cos(a) * 7.4 * cm * r0
        sy0 = hy - 0.6 * cm + math.sin(a) * 10.6 * cm * r0
        L = rng.uniform(6, 18) * cm
        p2 = (sx0 + math.cos(a2) * L, sy0 + math.sin(a2) * L * 0.8)
        p1 = (sx0 + math.cos(a) * L * 0.5 + rng.normal(0, 1.2 * cm), sy0 + math.sin(a) * L * 0.4 + rng.normal(0, 1.2 * cm))
        lines.append((bezier((sx0, sy0), p1, p2, 12), 0.06 * cm, 0.02 * cm))
    for i in range(500):                                    # hairline over the forehead
        a = rng.uniform(-math.pi * 0.85, -math.pi * 0.15)
        sx0 = hx + math.cos(a) * 6.6 * cm
        sy0 = hy - 0.6 * cm + math.sin(a) * 9.4 * cm
        p2 = (sx0 + math.cos(a) * 3.0 * cm, sy0 + math.sin(a) * 3.0 * cm)
        lines.append((bezier((sx0, sy0), ((sx0 + p2[0]) / 2 + rng.normal(0, 0.3 * cm), (sy0 + p2[1]) / 2), p2, 6), 0.05 * cm, 0.02 * cm))
    hm = strokes(w, h, lines, ss=1)
    hmass = np.clip(blur(hm, 0.3 * cm) * 1.4, 0, 1) * 0.85 + hm * 0.15
    hair_c = lin((0.06, 0.035, 0.025)) * (0.6 + 0.8 * fbm(w, h, rng, base=60, octaves=2)[..., None])
    hair_sheen = np.clip(hm - 0.6, 0, 1) * np.exp(-(((x - w * 0.25) / (w * 0.5)) ** 2))
    img *= (1 - 0.55 * blur(hmass, 1.2 * cm))[..., None]
    img = img * (1 - hmass[..., None]) + hair_c * hmass[..., None] + lin((0.75, 0.55, 0.35)) * 0.25 * hair_sheen[..., None]
    # lamp: warm key from the left, darkness to the right
    img *= (0.25 + 0.95 * np.exp(-(((x + w * 0.2) / (w * 1.1)) ** 2 + ((y - h * 0.3) / (h * 0.9)) ** 2)))[..., None]
    return finish(img * 1.3, bloom_amt=0.35, sat=0.85)


# ---------------------------------------------------------------- the room


def _capsule(X, Y, p0, p1, r):
    p0, p1 = np.asarray(p0), np.asarray(p1)
    d = p1 - p0
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / (d @ d), 0, 1)
    dist = np.sqrt((X - p0[0] - t * d[0]) ** 2 + (Y - p0[1] - t * d[1]) ** 2)
    return np.sqrt(np.clip(1 - (dist / r) ** 2, 0, 1)) * r


def gen_room(layer, w, h, rng, inner=None):
    m = w / layer.meters
    x, y = grid(w, h)
    X, Y = x / m, y / m                                   # metres from the top-left
    ax, ay = layer.inner_at[0] * layer.meters, layer.inner_at[1] * layer.meters * h / w
    floor, sheen = wood_floor(w, h, rng, plank_px=0.14 * m, tone=(0.42, 0.26, 0.15))
    img = floor.copy()
    # rug under the person: border bands + medallion pattern
    rx0, rx1, ry0, ry1 = ax - 1.0, ax + 1.0, ay - 0.55, ay + 2.2
    in_rug = (X > rx0) & (X < rx1) & (Y > ry0) & (Y < ry1)
    du = np.minimum(np.minimum(X - rx0, rx1 - X), np.minimum(Y - ry0, ry1 - Y))
    pat = (0.5 + 0.5 * np.sin(X * 40) * np.sin(Y * 40)) * (0.5 + 0.5 * np.cos(np.hypot(X - ax, (Y - (ry0 + ry1) / 2) * 0.7) * 18))
    rug = lin((0.42, 0.08, 0.08)) * (0.6 + 0.5 * pat[..., None])
    rug = np.where((du < 0.16)[..., None], lin((0.12, 0.10, 0.25)) * (0.7 + 0.4 * (np.sin(X * 120 + Y * 120) > 0)[..., None]), rug)
    rug = np.where(((du > 0.16) & (du < 0.19))[..., None], lin((0.75, 0.6, 0.35)), rug)
    rug *= (0.85 + 0.3 * fbm(w, h, rng, base=int(w / (0.01 * m)), octaves=1)[..., None])
    img = np.where(in_rug[..., None], rug, img)
    # furniture: (x, y, w, h, height m, color)
    furn = [
        (2.85, 0.35, 1.55, 2.05, 0.55, (0.62, 0.64, 0.70)),     # bed
        (0.15, 0.20, 1.20, 0.55, 0.75, (0.30, 0.20, 0.12)),     # desk
        (0.15, 5.6, 0.45, 1.4, 1.9, (0.25, 0.16, 0.10)),        # bookshelf
        (3.2, 5.4, 0.7, 0.7, 0.45, (0.20, 0.30, 0.22)),         # armchair
    ]
    occ = np.zeros((h, w), np.float32)
    lamp = (ax - 0.85, ay - 0.35)
    for fx, fy, fw, fh, fz, col in furn:
        mk = (X > fx) & (X < fx + fw) & (Y > fy) & (Y < fy + fh)
        sh = shadow(mk, (fx + fw / 2 - lamp[0]) * 0.18 * fz * m, (fy + fh / 2 - lamp[1]) * 0.18 * fz * m, 0.12 * m)
        img *= (1 - 0.6 * sh)[..., None]
        edge = np.minimum(np.minimum(X - fx, fx + fw - X), np.minimum(Y - fy, fy + fh - Y))
        top = lin(col) * (0.7 + 0.35 * fbm(w, h, rng, base=40, octaves=3)[..., None]) * (0.7 + 0.3 * smooth(0, 0.05, edge)[..., None])
        img = np.where(mk[..., None], top, img)
        occ = np.maximum(occ, mk)
    # bed details: pillows + duvet folds
    pil = (np.abs(X - 3.25) < 0.32) & (np.abs(Y - 0.65) < 0.2) | (np.abs(X - 4.0) < 0.32) & (np.abs(Y - 0.65) < 0.2)
    img = np.where(pil[..., None], lin((0.85, 0.85, 0.88)) * (0.8 + 0.2 * fbm(w, h, rng, 30, 3))[..., None], img)
    folds = (0.5 + 0.5 * np.sin(Y * 22 + 3 * fbm(w, h, rng, 6, 4))) * ((X > 2.85) & (X < 4.4) & (Y > 1.0) & (Y < 2.4))
    img *= (1 - 0.18 * folds)[..., None]
    # laptop on the desk: cool screen glow
    lap = (np.abs(X - 0.7) < 0.17) & (np.abs(Y - 0.45) < 0.12)
    img = np.where(lap[..., None], lin((0.55, 0.7, 1.0)) * 1.4, img)
    # books on the shelf
    for k in range(18):
        by0 = 5.62 + k * 0.077
        bk = (X > 0.18) & (X < 0.55) & (Y > by0) & (Y < by0 + 0.06)
        img = np.where(bk[..., None], lin(tuple(rng.uniform(0.15, 0.6, 3))), img)
    # plant: leaf blobs
    for k in range(40):
        a, rr = rng.uniform(0, 6.28), rng.uniform(0.05, 0.35)
        px_, py_ = 3.6 + rr * math.cos(a), 3.6 + rr * math.sin(a)
        lf = ((X - px_) ** 2 / 0.012 + (Y - py_) ** 2 / 0.004) < 1
        img = np.where(lf[..., None], lin((0.12, 0.30, 0.12)) * rng.uniform(0.6, 1.3), img)

    # the person, lying on the rug: height-field capsules
    body = np.zeros((h, w), np.float32)
    parts = [
        ((ax, ay - 0.02), (ax, ay + 0.02), 0.105, (0.78, 0.56, 0.45)),           # head
        ((ax, ay + 0.28), (ax, ay + 0.80), 0.19, (0.30, 0.36, 0.52)),           # torso
        ((ax - 0.22, ay + 0.25), (ax - 0.30, ay + 0.85), 0.055, (0.30, 0.36, 0.52)),
        ((ax + 0.22, ay + 0.25), (ax + 0.33, ay + 0.83), 0.055, (0.30, 0.36, 0.52)),
        ((ax - 0.09, ay + 0.85), (ax - 0.14, ay + 1.62), 0.075, (0.28, 0.33, 0.45)),
        ((ax + 0.09, ay + 0.85), (ax + 0.16, ay + 1.60), 0.075, (0.28, 0.33, 0.45)),
    ]
    cols = np.zeros((h, w, 3), np.float32)
    for p0, p1, rr, col in parts:
        c = _capsule(X, Y, p0, p1, rr)
        sel = c > body
        cols = np.where(sel[..., None], lin(col), cols)
        body = np.maximum(body, c)
    hands = [(ax - 0.31, ay + 0.9), (ax + 0.34, ay + 0.88)]
    for hx_, hy_ in hands:
        c = np.sqrt(np.clip(1 - ((X - hx_) ** 2 + (Y - hy_) ** 2) / 0.05**2, 0, 1)) * 0.05
        cols = np.where((c > body)[..., None], lin((0.78, 0.56, 0.45)), cols)
        body = np.maximum(body, c)
    bm = body > 0
    nb = normals(body, m * 1.0)
    shade_b = 0.2 + 1.0 * lambert(nb, (lamp[0] - ax, lamp[1] - ay - 0.4, 0.8), 0.3)
    cloth = 0.8 + 0.4 * fbm(w, h, rng, base=int(w / (0.04 * m)), octaves=3)
    img *= (1 - 0.7 * shadow(bm, 0.05 * m, 0.04 * m, 0.05 * m) * ~bm)[..., None]
    img = np.where(bm[..., None], cols * shade_b[..., None] * cloth[..., None], img)
    # hair spread around the head
    hair = np.exp(-(((X - ax) / 0.2) ** 2 + ((Y - ay + 0.12) / 0.16) ** 2)) * (Y < ay + 0.02)
    hair_m = np.clip(hair * 2.5 * (0.6 + 0.6 * fbm(w, h, rng, int(w / (0.02 * m)), 2)), 0, 1)
    img = img * (1 - hair_m[..., None]) + lin((0.05, 0.03, 0.02)) * hair_m[..., None]
    # true-scale face from the layer inside
    inner, il = unpack(inner)
    fw = il.meters / layer.meters * w if il else 0
    fm = np.exp(-(((x - ax * m) / (0.11 * m)) ** 4 + ((y - ay * m) / (0.15 * m)) ** 4))
    img = embed(img, inner, ax * m, ay * m, fw, mask=fm * 0.9)

    # walls (cut-away tops) and a window throwing moonlight
    wall = (X < 0.12) | (X > layer.meters - 0.12) | (Y < 0.12) | (Y > h / m - 0.12)
    img = np.where(wall[..., None], lin((0.55, 0.52, 0.50)) * 0.55, img)
    img *= (1 - 0.6 * shadow(wall, 0, 0, 0.15 * m) * ~wall)[..., None]
    moon = ((X > 0.12) & (X < 1.6) & (Y > 2.4) & (Y < 4.6)).astype(np.float32)
    moon = blur(moon * (np.sin((Y - 2.4) * 9) > -0.9) * (np.sin((X - 0.12) * 6.5) > -0.95), 0.03 * m)
    win = (X < 0.12) & (Y > 2.6) & (Y < 4.4)
    img = np.where(win[..., None], lin((0.5, 0.65, 0.9)) * 0.8, img)
    light = 0.13 + 1.6 * np.exp(-(((X - lamp[0]) ** 2 + (Y - lamp[1]) ** 2) / 1.6**2)) ** 1.2
    img = img * light[..., None] * lin((1.0, 0.78, 0.55)) + img * moon[..., None] * lin((0.45, 0.6, 1.0)) * 0.9
    img += 0.05 * sheen[..., None] * light[..., None] * ~bm[..., None]
    glow(img, lamp[0] * m, lamp[1] * m, 0.12 * m, lin((1.0, 0.75, 0.45)), 2.5, 2.2)
    lr = np.hypot(X - lamp[0], Y - lamp[1])
    shade_ring = (lr < 0.17) & (lr > 0.13)
    img = np.where((lr <= 0.13)[..., None], lin((1.0, 0.85, 0.62)) * 1.6, img)
    img = np.where(shade_ring[..., None], lin((0.9, 0.75, 0.55)) * 0.9, img)
    glow(img, 0.7 * m, 0.45 * m, 0.25 * m, lin((0.5, 0.65, 1.0)), 0.4, 2)
    return finish(img * 1.1, bloom_amt=0.5)


# ---------------------------------------------------------------- the rooftop


def gen_rooftop(layer, w, h, rng, inner=None):
    m = w / layer.meters
    x, y = grid(w, h)
    X, Y = (x - w / 2) / m, (y - h / 2) / m
    H = h / m
    # street far below, outside the roof
    img = lin((0.03, 0.03, 0.04)) * (0.6 + 0.6 * fbm(w, h, rng, 30, 4)[..., None])
    rw, rh = 26, 46
    roof = (np.abs(X) < rw / 2) & (np.abs(Y) < rh / 2)
    for sx in (-1, 1):                                    # streetlights + cars below
        for k in range(10):
            ly = -H / 2 + (k + 0.5) * H / 10
            glow(img, w / 2 + sx * (rw / 2 + 4.5) * m, h / 2 + ly * m, 1.4 * m, lin((1.0, 0.6, 0.25)), 0.8, 2)
        lane = np.exp(-((X - sx * (rw / 2 + 3.5)) / 0.15) ** 2) * (fbm(w, h, rng, 4, 3) > 0.55)
        img += lane[..., None] * (lin((1.0, 0.15, 0.1)) if sx < 0 else lin((1.0, 0.95, 0.85))) * 0.6
    # membrane panels, gravel, stains
    gravel = fbm(w, h, rng, base=int(w / (0.08 * m)), octaves=2)
    stains = fbm(w, h, rng, base=6, octaves=6)
    seams = np.maximum(np.exp(-(((X + 50) % 3.0 - 1.5) / 0.03) ** 2), np.exp(-(((Y + 50) % 6.0 - 3.0) / 0.03) ** 2))
    surf = lin((0.40, 0.39, 0.42)) * (0.6 + 0.35 * gravel[..., None]) * (0.7 + 0.5 * stains[..., None]) * (1 - 0.4 * seams[..., None])
    puddle = smooth(0.70, 0.74, fbm(w, h, rng, base=5, octaves=5))
    surf = surf * (1 - puddle[..., None] * 0.6) + lin((0.12, 0.14, 0.20)) * puddle[..., None] * 0.6
    img = np.where(roof[..., None], surf, img)
    edge = roof & ((np.abs(X) > rw / 2 - 0.45) | (np.abs(Y) > rh / 2 - 0.45))
    img = np.where(edge[..., None], lin((0.45, 0.43, 0.42)), img)
    img *= (1 - 0.5 * shadow(edge, 0.25 * m, 0.25 * m, 0.4 * m) * (roof & ~edge))[..., None]
    occ = np.zeros((h, w), bool)
    # HVAC units, vents, a water tank
    units = [(-8, -14, 3.2, 2.2), (6.5, -16, 2.6, 2.6), (-7.5, 12, 4.2, 2.4), (7, 15, 2.2, 3.4), (8, -2, 1.6, 1.6)]
    for ux, uy, uw, uh in units:
        mk = (np.abs(X - ux) < uw / 2) & (np.abs(Y - uy) < uh / 2)
        img *= (1 - 0.65 * shadow(mk, 0.7 * m, 0.9 * m, 0.35 * m) * ~mk)[..., None]
        fan = np.hypot(X - ux, Y - uy) < min(uw, uh) * 0.32
        grille = 0.5 + 0.5 * np.sin(np.hypot(X - ux, Y - uy) * 40)
        edge_d = np.minimum(uw / 2 - np.abs(X - ux), uh / 2 - np.abs(Y - uy))
        top = lin((0.48, 0.49, 0.52)) * (0.8 + 0.2 * fbm(w, h, rng, 50, 2)[..., None]) * (0.6 + 0.4 * smooth(0, 0.25, edge_d)[..., None])
        top = np.where(fan[..., None], lin((0.12, 0.12, 0.13)) * (0.6 + 0.6 * grille[..., None]), top)
        img = np.where(mk[..., None], top, img)
        occ |= mk
    tank = np.hypot(X + 9, Y + 1) < 2.0
    img *= (1 - 0.6 * shadow(tank, 1.0 * m, 1.2 * m, 0.5 * m) * ~tank)[..., None]
    tank_n = np.sqrt(np.clip(1 - ((X + 9) ** 2 + (Y + 1) ** 2) / 4, 0, 1))
    img = np.where(tank[..., None], lin((0.35, 0.28, 0.22)) * (0.4 + 0.6 * tank_n[..., None]) * (0.8 + 0.3 * np.sin(np.arctan2(Y + 1, X + 9) * 24)[..., None] * 0.3), img)
    # pipes
    for (p0, p1) in [((-8, -12.9), (-8, 10.8)), ((-5.3, 12), (6, 12)), ((6.5, -14.7), (6.5, -4))]:
        c = _capsule(X, Y, p0, p1, 0.11)
        img *= (1 - 0.5 * shadow(c > 0, 0.15 * m, 0.2 * m, 0.1 * m) * (c <= 0))[..., None]
        img = np.where((c > 0)[..., None], lin((0.42, 0.40, 0.38)) * (0.3 + 0.7 * (c / 0.11))[..., None], img)
    # the skylight: the room below shows through, glowing
    sw, sh = 4.0 * 1.06, 4.0 * h / w * 1.06
    frame = (np.abs(X) < sw / 2 + 0.25) & (np.abs(Y) < sh / 2 + 0.25)
    glass = (np.abs(X) < sw / 2) & (np.abs(Y) < sh / 2)
    img *= (1 - 0.5 * shadow(frame, 0.2 * m, 0.25 * m, 0.25 * m) * ~frame)[..., None]
    img = np.where(frame[..., None], lin((0.6, 0.6, 0.62)) * 0.5, img)
    inner, il = unpack(inner)
    img = embed(img, inner, w / 2, h / 2, (il.meters if il else 4.0) / layer.meters * w,
                mask=glass.astype(np.float32))
    mullion = glass & ((np.abs(X) < 0.05) | (np.abs(Y) < 0.05))
    img = np.where(mullion[..., None], lin((0.5, 0.5, 0.52)) * 0.4, img)
    refl = glass * np.clip(0.12 * (1 - np.abs((X + Y * 0.6) / 2.5)), 0, 0.12)
    img += refl[..., None] * lin((0.6, 0.7, 0.9))
    # night light: skylight spill, city sky glow, cool ambient
    spill = np.exp(-((X / 4.5) ** 2 + (Y / 6) ** 2))
    amb = 0.75
    lit = img * (amb + 1.2 * spill * ~glass)[..., None] * np.where(glass[..., None], 1.0, lin((0.75, 0.82, 1.0)))
    lit += puddle[..., None] * roof[..., None] * lin((0.55, 0.62, 0.85)) * 0.03 * (1 + 3 * spill[..., None])
    return finish(lit * 1.25, bloom_amt=0.55)


# ---------------------------------------------------------------- the city at night


def gen_night_city(layer, w, h, rng, inner=None):
    p = layer.params
    m = w / layer.meters                          # px per metre
    x, y = grid(w, h)
    cx, cy = w / 2, h / 2
    X, Y = (x - cx) / m, (y - cy) / m
    rr = np.hypot(X, Y) / (layer.meters * 0.5)
    sodium, led, red = lin((1.0, 0.58, 0.22)), lin((0.85, 0.92, 1.0)), lin((1.0, 0.12, 0.08))
    warp = fbm(w, h, rng, base=3, octaves=4)
    dens = fbm(w, h, rng, base=5, octaves=6) * 0.6 + 0.4
    dens *= np.exp(-rr * (1.6 / max(0.25, p.get("falloff", 1.0)))) * p.get("density", 0.8) * 1.4
    land = np.ones((h, w), np.float32)
    if p.get("coast"):
        c = fbm(w, h, rng, base=3, octaves=7)
        land = smooth(-0.01, 0.01, 0.15 - (X / layer.meters + (c - 0.5) * 0.55))
    river = np.exp(-((X / layer.meters - 0.12 + 0.18 * (warp - 0.5) - 0.08 * np.sin(Y / layer.meters * 9)) / 0.012) ** 2) \
        if layer.meters < 50_000 else np.zeros((h, w), np.float32)
    img = lin((0.012, 0.012, 0.016)) * np.ones((h, w, 3), np.float32)
    img += lin((0.03, 0.035, 0.045)) * (fbm(w, h, rng, 40, 3) * land)[..., None]
    img[..., 2] += (1 - land) * 0.004

    spacing = 110.0                                    # metres between streets
    sp_px = spacing * m
    ang = 0.12
    Xr = X * math.cos(ang) - Y * math.sin(ang) + 30 * (warp - 0.5) * (layer.meters > 2000)
    Yr = X * math.sin(ang) + Y * math.cos(ang)
    if sp_px >= 6:
        # streets, buildings and windows of light at street-level scales
        du = np.abs((Xr + spacing / 2) % spacing - spacing / 2)
        dv = np.abs((Yr + spacing / 2) % spacing - spacing / 2)
        sw = 7.0
        street = np.exp(-(np.minimum(du, dv) / (sw / 2)) ** 2)
        bi = ((Xr // spacing) * 7 + (Yr // spacing) * 13).astype(int) % 997
        sub = rng.integers(2, 5, 1000)[bi]
        lx = (Xr % spacing) / spacing * sub
        ly = (Yr % spacing) / spacing * sub
        bldg_id = (bi * 31 + lx.astype(int) * 5 + ly.astype(int)) % 997
        bheight = rng.random(1000)[bldg_id]
        bedge = np.minimum(np.minimum(lx % 1, 1 - lx % 1), np.minimum(ly % 1, 1 - ly % 1)) * spacing / sub
        roof_c = lin((0.08, 0.08, 0.09)) * (0.5 + 0.9 * bheight[..., None]) * (0.8 + 0.3 * fbm(w, h, rng, int(w / max(2, 2 * m)) if m > 0.5 else 8, 2)[..., None])
        roof_c *= smooth(0.5, 2.5, bedge)[..., None] * 0.8 + 0.2
        bmask = (np.minimum(du, dv) > sw * 0.75)
        img = np.where(bmask[..., None], roof_c, img)
        sh = (1 - smooth(0, 8, bedge)) * bmask * (bheight > 0.5)
        img *= (1 - 0.4 * sh)[..., None]
        # rooftop details: lit skylights / aviation lights
        for k in range(int(layer.meters**2 / 900 * p.get("density", 0.7))):
            px_, py_ = rng.uniform(-0.5, 0.5) * w, rng.uniform(-0.5, 0.5) * h
            glow(img, cx + px_, cy + py_, max(0.6, 0.6 * m), sodium if rng.random() < 0.7 else led, 0.35, 3)
        img += lin((0.05, 0.05, 0.06)) * street[..., None] * 0.5
        img += sodium * (street * (0.3 + dens))[..., None] * 0.035
        # streetlights at intervals and car light trails
        n_l = int((w * h) / sp_px**2 * 60)
        t = rng.random(n_l)
        along = rng.uniform(-0.5, 0.5, n_l) * math.hypot(w, h) / m
        reach = int(math.hypot(w, h) / m / 2 / spacing) + 2
        lane = (rng.integers(-reach, reach, n_l) + 0.5) * spacing
        horiz = rng.random(n_l) < 0.5
        Xs = np.where(horiz, along, lane + rng.choice([-1, 1], n_l) * sw * 0.45)
        Ys = np.where(horiz, lane + rng.choice([-1, 1], n_l) * sw * 0.45, along)
        xs = (Xs * math.cos(-ang) - Ys * math.sin(-ang)) * m + cx
        ys = (Xs * math.sin(-ang) + Ys * math.cos(-ang)) * m + cy
        cols = np.where((t < 0.55)[:, None], sodium * 1.2, np.where((t < 0.8)[:, None], red * 0.9, led * 1.1))
        k = 0.9 if m > 1 else 0.6
        splat(img, xs, ys, cols * k * (0.4 + rng.random(n_l)[:, None]), sigma=max(0.6, 1.6 * m))
        # the building at the very center is the rooftop we came from
        glow(img, cx, cy, max(1.0, 3 * m), sodium, 0.8, 2)
    else:
        # beyond street scale: light density following roads and the grid
        n = int(w * h * 0.22 * p.get("density", 0.7))
        xs, ys = rng.random(n) * w, rng.random(n) * h
        yi, xi = ys.astype(int), xs.astype(int)
        gridness = np.exp(-(np.minimum(np.abs((Xr + spacing / 2) % spacing - spacing / 2),
                                       np.abs((Yr + spacing / 2) % spacing - spacing / 2)) / 25) ** 2)
        keep = rng.random(n) < (dens[yi, xi] * land[yi, xi] * (0.6 + 0.6 * gridness[yi, xi])) * 1.2
        xs, ys = xs[keep], ys[keep]
        cols = np.where(rng.random(len(xs))[:, None] < 0.82, sodium, led) * (rng.pareto(2.5, len(xs))[:, None] * 0.35 + 0.10)
        splat(img, xs, ys, cols, sigma=0.75)
        img += sodium * (blur(dens * land, w * 0.01) * 0.10)[..., None]          # urban sky glow
    # highways: smooth curves radiating out, dotted with traffic
    for k in range(int(p.get("roads", 9)) if m < 1 else 0):      # at street scale the streets are the roads
        a = rng.uniform(0, 2 * math.pi)
        tt = np.linspace(0, 1, 4000)
        R = tt * max(w, h) * 0.9
        bend = rng.normal(0, 0.25) * tt**2 + rng.normal(0, 0.08) * np.sin(tt * 6)
        hx_, hy_ = cx + R * np.cos(a + bend), cy + R * np.sin(a + bend)
        step = max(w, h) * 0.9 / 4000
        cols = np.tile(sodium * 0.035 * step, (4000, 1)) * np.exp(-tt * 1.8)[:, None] * (0.4 + rng.random((4000, 1)))
        splat(img, hx_, hy_, cols, sigma=max(0.6, min(1.5, 1.2 * m)))
    img *= (1 - 0.85 * river)[..., None]
    img += river[..., None] * blur(img, w * 0.004) * 0.6                      # reflections in the water
    haze = blur(img, w * 0.02)
    img += haze * p.get("glow", 1.0) * 0.45 + lin((0.35, 0.22, 0.12)) * (dens * 0.03)[..., None]
    if p.get("coast"):                                     # moonlit ocean glint
        g = fbm(w, h, rng, base=60, octaves=2)
        img += ((1 - land) * np.clip(g - 0.6, 0, 1) * 0.15 * np.exp(-((X / layer.meters - 0.35) / 0.15) ** 2))[..., None] * lin((0.6, 0.7, 0.9))
    inner, il = unpack(inner)
    if il is not None and il.kind != "night_city" and il.meters / layer.meters * w >= 8:
        iw = il.meters / layer.meters * w
        img = embed(img, inner, cx, cy, iw, mask=soft_rect(w, h, cx, cy, iw * 0.8, iw * 0.8 * h / w, iw * 0.15))
    return finish(img * 1.6, bloom_amt=0.5, sat=1.05)


# ---------------------------------------------------------------- Earth


def gen_globe(layer, w, h, rng, inner=None):
    x, y = grid(w, h)
    cx, cy = w / 2, h / 2
    R = 6.371e6 / layer.meters * w
    nx, ny = (x - cx) / R, (y - cy) / R
    r2 = nx**2 + ny**2
    inside = r2 < 1
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    img = np.zeros((h, w, 3), np.float32)
    starfield(img, rng, 0.0005, 0.5)
    img = img ** 2.2
    tw = 2048
    lon = np.arctan2(nx, nz)
    lat = np.arcsin(np.clip(ny, -1, 1))
    ti = ((lon / (2 * math.pi) + 0.5) * (tw - 1)).astype(int) % tw
    tj = ((lat / math.pi + 0.5) * (tw // 2 - 1)).astype(int)
    warp = fbm(tw, tw // 2, rng, base=3, octaves=4)
    hmap = fbm(tw, tw // 2, rng, base=3, octaves=8)
    hmap = 0.6 * hmap + 0.4 * _resize(hmap, tw, tw // 2)
    jj, ii = np.mgrid[0:tw // 2, 0:tw]
    hmap = hmap[np.clip((jj + (warp - 0.5) * 120).astype(int), 0, tw // 2 - 1), (ii + (warp - 0.5) * 240).astype(int) % tw]
    big = R > w                                                   # close orbit: paint detail at screen resolution
    if big:
        def tex(base, octv):
            return fbm(w, h, rng, base=base, octaves=octv)
        hmap_s = 0.6 * tex(3, 8) + 0.4 * tex(9, 6)
    hh = (hmap_s if big else hmap[tj, ti]) + 0.3 * np.exp(-r2 * 25)   # our continent under the camera
    land = smooth(0.53, 0.55, hh)
    latf = np.abs(lat) / (math.pi / 2)
    desert = smooth(0.15, 0.35, latf) * smooth(0.55, 0.4, latf) * fbm(tw, tw // 2, rng, 6, 4)[tj, ti]
    if big:
        desert = desert * 0 + 0.3 * smooth(0.5, 0.7, fbm(w, h, rng, 5, 5))
    ground = lin((0.13, 0.20, 0.09)) * (1 - desert[..., None]) + lin((0.55, 0.42, 0.26)) * desert[..., None]
    ground = ground * (0.7 + 0.6 * (hh[..., None] - 0.55))
    ice = smooth(0.78, 0.86, latf)
    ocean = lin((0.015, 0.06, 0.16)) * (0.7 + 0.3 * smooth(0.55, 0.40, hh))[..., None]
    surf = ocean * (1 - land[..., None]) + ground * land[..., None]
    surf = surf * (1 - ice[..., None]) + lin((0.9, 0.93, 0.97)) * ice[..., None]
    cl = 0.55 * fbm(tw, tw // 2, rng, base=10, octaves=6) + 0.45 * fbm(tw, tw // 2, rng, base=28, octaves=4)
    swirl = fbm(tw, tw // 2, rng, base=2, octaves=3)
    cl = cl[np.clip((jj + (swirl - 0.5) * 80).astype(int), 0, tw // 2 - 1), (ii + (swirl - 0.5) * 300).astype(int) % tw]
    clouds = smooth(0.50, 0.66, (0.55 * tex(6, 7) + 0.45 * tex(20, 5)) if big else cl[tj, ti]) * 0.9
    sun = np.array([0.88, -0.30, layer.params.get("sun_z", -0.18)]) if layer.params.get("night", True) else np.array([0.3, -0.3, 0.9])
    sun = sun / np.linalg.norm(sun)
    ndl = nx * sun[0] + ny * sun[1] + nz * sun[2]
    day = smooth(-0.05, 0.25, ndl)
    twilight = np.exp(-((ndl - 0.02) / 0.035) ** 2)
    shade = surf * (1 - clouds[..., None] * 0.9) + lin((0.80, 0.82, 0.86)) * clouds[..., None] * 0.55
    lit = shade * (day[..., None] * 0.9) + lin((1.0, 0.55, 0.25)) * (twilight * 0.10)[..., None] * (0.3 + clouds[..., None])
    glint = np.clip(nx * (sun[0] + 0) + ny * sun[1] + nz * (sun[2] + 1), 0, None)
    glint = (glint / max(1e-3, np.linalg.norm(sun + np.array([0, 0, 1])))) ** 600
    lit += (glint * (1 - land) * (1 - clouds) * day)[..., None] * 0.6
    citytex = 0.5 * fbm(tw, tw // 2, rng, base=40, octaves=3) + 0.5 * fbm(tw, tw // 2, rng, base=220, octaves=2)
    coastal = np.exp(-((hh - 0.56) / 0.04) ** 2) + 0.5 * smooth(0.55, 0.65, hh)
    lo_t = 0.52 if big else 0.56
    lights = np.zeros((h, w), np.float32)
    lights += np.exp(-r2 * R**2 / 64) * 2.0                       # our city, a bright point under the camera
    lights *= (1 - day) * (1 - 0.75 * clouds)
    lit += lin((1.0, 0.62, 0.28)) * lights[..., None] * 0.9
    if True:                                    # city lights as speckles, never a stretched texture
        cd = fbm(w, h, rng, 24, 4) if big else citytex[tj, ti]
        dens = land * (1 - day) * smooth(0.45, 0.75, cd) * (0.35 + coastal) * (1 - 0.7 * clouds) * inside
        n = int(w * h * 0.08)
        xs, ys = rng.random(n) * w, rng.random(n) * h
        keep = rng.random(n) < dens[ys.astype(int), xs.astype(int)]
        k = int(keep.sum())
        pts = np.zeros_like(lit)
        splat(pts, xs[keep], ys[keep], np.tile(lin((1.0, 0.62, 0.28)), (k, 1)) * (rng.pareto(2.5, k)[:, None] * 0.25 + 0.06), sigma=0.7)
        lit += pts + blur(pts, w * 0.006) * 1.5
    lit += shade * (1 - day[..., None]) * 0.035 * lin((0.6, 0.7, 1.0))   # moonlit night side
    lit *= (0.25 + 0.75 * nz[..., None] ** 0.35)
    img = np.where(inside[..., None], lit, img)
    rim = np.exp(-((np.sqrt(r2) - 1) * R / max(2.0, R * 0.02)) ** 2) * (np.sqrt(r2) > 0.97)
    atm = smooth(-0.35, 0.3, ndl)
    img += rim[..., None] * (lin((0.3, 0.55, 1.0)) * atm[..., None] * 1.6 + lin((1.0, 0.5, 0.2)) * (twilight * 1.2)[..., None])
    img += (smooth(0.9, 1.0, np.sqrt(r2)) * inside * atm)[..., None] * lin((0.3, 0.5, 1.0)) * 0.5
    return finish(img * 1.1, bloom_amt=0.5)


def gen_earth_moon(layer, w, h, rng, inner=None):
    img = np.zeros((h, w, 3), np.float32)
    starfield(img, rng, 0.0009, 0.8)
    img = img ** 2.2
    milky_band(img, rng, 1.0)
    cx, cy = w / 2, h / 2
    x, y = grid(w, h)
    re = max(1.5, 6.371e6 / layer.meters * w)
    sun = np.array([0.88, -0.30])
    for (px_, py_, rad, col) in [(cx, cy, re, (0.35, 0.55, 1.0)),
                                 (cx + 0.3 * w, cy - 0.24 * w, max(1.0, 1.737e6 / layer.meters * w), (0.75, 0.74, 0.70))]:
        d = np.hypot(x - px_, y - py_) / rad
        disk = smooth(1.05, 0.95, d)
        nzz = np.sqrt(np.clip(1 - d**2, 0, 1))
        lam = np.clip(((x - px_) / rad * sun[0] + (y - py_) / rad * sun[1] + nzz * 0.1), 0, 1)
        img += (disk * (0.08 + lam * 1.6))[..., None] * lin(col)
        glow(img, px_, py_, rad * 1.3, lin(col), 0.25, 2.5)
    d = np.hypot(x - cx, y - cy)
    ring = np.exp(-((d - 3.844e8 / layer.meters * w) / 1.0) ** 2) * 0.035
    img += ring[..., None] * lin((0.6, 0.7, 1.0))
    return finish(img * 1.2, bloom_amt=0.7)


def milky_band(img, rng, strength=1.0, angle=-0.55):
    """A Milky Way band behind deep-space layers: dense faint stars, glow and dust lanes."""
    h, w = img.shape[:2]
    x, y = grid(w, h)
    d = ((x - w / 2) * math.sin(angle) + (y - h / 2) * math.cos(angle)) / w
    band = np.exp(-(d / 0.16) ** 2)
    neb = fbm(w, h, rng, base=4, octaves=7)
    lanes = smooth(0.5, 0.75, fbm(w, h, rng, base=7, octaves=6)) * np.exp(-(d / 0.05) ** 2)
    glow_ = band * (0.4 + neb) * (1 - 0.8 * lanes)
    img += glow_[..., None] * (lin((0.55, 0.50, 0.62)) * 0.16 + lin((0.9, 0.6, 0.4)) * 0.08 * neb[..., None]) * strength
    n = int(w * h * 0.05 * strength)
    xs, ys = rng.random(n) * w, rng.random(n) * h
    keep = rng.random(n) < band[ys.astype(int), xs.astype(int)] * (1 - 0.8 * lanes[ys.astype(int), xs.astype(int)])
    k = int(keep.sum())
    temp = rng.random(k)[:, None]
    splat(img, xs[keep], ys[keep], (lin((1.0, 0.85, 0.7)) * temp + lin((0.7, 0.8, 1.0)) * (1 - temp)) * rng.pareto(3, k)[:, None] * 0.28, sigma=0.55)


def _sun(img, cx, cy, core, w, intensity=1.0, flare=True):
    """Sun with corona, diffraction spikes and an anamorphic streak."""
    glow(img, cx, cy, core, lin((1.0, 0.92, 0.75)), 6.0 * intensity, 2.4)
    if not flare:                 # nested layers: flares come from the frame-level bloom instead
        return
    glow(img, cx, cy, core * 5, lin((1.0, 0.65, 0.35)), 0.6 * intensity, 2.0)
    h, ww = img.shape[:2]
    x, y = grid(ww, h)
    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy) + 1e-3
    a = np.arctan2(dy, dx)
    spikes = sum(np.exp(-((np.angle(np.exp(1j * (a - k * math.pi / 3 - 0.3))) * r) / (core * 0.35)) ** 2) for k in range(6))
    img += (spikes * np.exp(-r / (w * 0.18)) * 0.5 * intensity)[..., None] * lin((1.0, 0.85, 0.65))
    streak = np.exp(-(dy / (core * 0.5)) ** 2) * np.exp(-np.abs(dx) / (w * 0.45))
    img += (streak * 0.35 * intensity)[..., None] * lin((0.55, 0.7, 1.0))


def gen_solar(layer, w, h, rng, inner=None):
    img = np.zeros((h, w, 3), np.float32)
    starfield(img, rng, 0.0012, 0.8)
    img = img ** 2.2
    milky_band(img, rng, 1.0)
    cx, cy = w / 2, h / 2
    x, y = grid(w, h)
    r = np.hypot(x - cx, y - cy)
    pxau = AU / layer.meters * w
    belt_r = np.array([2.2, 3.3]) * pxau
    if belt_r[1] < w:                                            # asteroid belt
        n = int(min(30000, 0.35 * math.pi * (belt_r[1] ** 2 - belt_r[0] ** 2)))
        rr = rng.uniform(*belt_r, n)
        a = rng.uniform(0, 2 * math.pi, n)
        splat(img, cx + rr * np.cos(a), cy + rr * np.sin(a), np.tile(lin((0.5, 0.45, 0.4)) * 0.05, (n, 1)), sigma=0.6)
    if 30 * pxau < w * 2:                                        # Kuiper belt haze
        img += (np.exp(-((r - 42 * pxau) / (8 * pxau)) ** 2) * 0.02)[..., None] * lin((0.6, 0.6, 0.8))
    colors = [(0.7, 0.65, 0.6), (0.95, 0.85, 0.6), (0.4, 0.6, 1.0), (0.95, 0.5, 0.35), (0.9, 0.75, 0.55),
              (0.95, 0.85, 0.6), (0.6, 0.85, 0.95), (0.4, 0.55, 1.0)]
    for k, au in enumerate(layer.params.get("orbits", [1.0])):
        d = au * pxau
        img += (np.exp(-((r - d) / 0.9) ** 2) * 0.05)[..., None] * lin((0.55, 0.65, 0.9))
        a = 0.0 if abs(au - 1.0) < 1e-6 else rng.uniform(0, 2 * math.pi)
        glow(img, cx + d * math.cos(a), cy + d * math.sin(a), 1.3, lin(colors[k % len(colors)]), 1.5, 3)
    _sun(img, cx, cy, max(2.0, layer.params.get("sun_px", 0.01) * w), w, 1.0, flare=False)
    return finish(img, bloom_amt=0.7)


def gen_stars(layer, w, h, rng, inner=None):
    img = np.zeros((h, w, 3), np.float32)
    dust = fbm(w, h, rng, base=3, octaves=8)
    dust2 = fbm(w, h, rng, base=5, octaves=7)
    img += (smooth(0.45, 0.9, dust) ** 2)[..., None] * lin((0.25, 0.10, 0.30)) * 0.25
    img += (smooth(0.5, 0.9, dust2) ** 2)[..., None] * lin((0.10, 0.20, 0.40)) * 0.2
    img *= (1 - 0.7 * smooth(0.55, 0.8, fbm(w, h, rng, 6, 6)))[..., None]       # dark lanes
    starfield(img, rng, 0.004, 3.0)
    milky_band(img, rng, 1.4, angle=0.4)
    for _ in range(14):                                       # bright neighbors with spikes
        sx, sy = rng.uniform(0.05, 0.95) * w, rng.uniform(0.05, 0.95) * h
        _sun(img, sx, sy, rng.uniform(1.0, 2.2), w * 0.3, rng.uniform(0.15, 0.4))
    _sun(img, w / 2, h / 2, 1.6, w * 0.3, 0.45, flare=False)
    return finish(img, bloom_amt=0.6)


def gen_galaxy(layer, w, h, rng, inner=None):
    img = np.zeros((h, w, 3), np.float32)
    starfield(img, rng, 0.0004, 0.5)
    d = int(1.0e5 * LY / layer.meters * w * 1.4)
    s = _spiral(d, rng, n=1_400_000, arms=2, pitch=0.24, scale=0.32)
    yy, xx = np.mgrid[0:d, 0:d].astype(np.float32)
    rr = np.hypot(xx - d / 2, yy - d / 2) / (d / 2)
    th = np.arctan2(yy - d / 2, xx - d / 2)
    phase = np.log(np.maximum(rr, 1e-3) / 0.04) / math.tan(0.24)
    lanes = np.cos(2 * (th - phase) + 0.9)
    dustn = fbm(d, d, rng, base=10, octaves=5)
    dust_m = smooth(0.4, 0.95, lanes) * smooth(0.06, 0.2, rr) * smooth(0.95, 0.5, rr) * (0.5 + dustn)
    s *= (1 - 0.75 * np.clip(dust_m, 0, 1))[..., None]
    s += (np.exp(-(rr / 0.09) ** 2) * 1.2)[..., None] * np.array([1.0, 0.78, 0.5])       # bulge
    bar = np.exp(-(((xx - d / 2) * 0.94 + (yy - d / 2) * 0.34) / (d * 0.10)) ** 2 - (((yy - d / 2) * 0.94 - (xx - d / 2) * 0.34) / (d * 0.03)) ** 2)
    s += bar[..., None] * np.array([1.0, 0.8, 0.55]) * 0.5
    s = (s / max(1e-6, np.percentile(s, 99.0))) ** 1.1 * 0.8
    x0, y0 = w // 2 - d // 2, h // 2 - d // 2
    ys0, ys1, xs0, xs1 = max(0, y0), min(h, y0 + d), max(0, x0), min(w, x0 + d)
    img[ys0:ys1, xs0:xs1] += s[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    ax, ay = layer.inner_at[0] * w, layer.inner_at[1] * h
    glow(img, ax, ay, 1.4, lin((1.0, 0.9, 0.6)), 0.4, 3)
    return finish(img * 1.1, bloom_amt=0.6, sat=1.1)


def gen_local_group(layer, w, h, rng, inner=None):
    img = np.zeros((h, w, 3), np.float32)
    haze = fbm(w, h, rng, base=3, octaves=7)
    img += (haze**3)[..., None] * lin((0.10, 0.06, 0.16)) * 0.5
    starfield(img, rng, 0.0002, 0.4)
    img = img ** 1.5
    ax, ay = layer.inner_at[0] * w, layer.inner_at[1] * h
    mw = max(8, int(1.0e5 * LY / layer.meters * w * 1.4))
    from .placeholders import _paste

    def gal(d, n, pitch=0.24, k=0.5):
        g = _spiral(d, rng, n=n, pitch=pitch)
        return (g / max(1e-6, np.percentile(g, 99.0))) ** 1.1 * k * 2.5
    _paste(img, gal(mw, 80_000), ax, ay)
    an = max(10, int(2.2e5 * LY / layer.meters * w * 1.4))
    dd = 2.5e6 * LY / layer.meters * w
    _paste(img, gal(an, 120_000, 0.3, 0.6), ax + dd * 0.44, ay - dd * 0.9)
    _paste(img, gal(int(an * 0.4), 30_000, 0.35, 0.35), ax + dd * 0.15, ay - dd * 0.55)
    for _ in range(24):
        dx, dy = rng.normal(0, w * 0.22, 2)
        glow(img, ax + dx, ay + dy, rng.uniform(1, 3.5), lin((0.85, 0.8, 0.95)), 0.25, 2)
    return finish(img * 1.1, bloom_amt=0.6)


def _cosmic(w, h, rng, seeds, width, warp_amt):
    """Warped multi-scale Voronoi filaments: a density field that reads as the cosmic web."""
    e1 = np.clip(_web(w, h, rng, seeds=seeds, width=width), 0, 1)
    e2 = np.clip(_web(w, h, rng, seeds=seeds * 4, width=width * 0.6), 0, 1)
    wx = (fbm(w, h, rng, base=4, octaves=4) - 0.5) * warp_amt
    wy = (fbm(w, h, rng, base=4, octaves=4) - 0.5) * warp_amt
    x, y = grid(w, h)
    xi = np.clip((x + wx).astype(int), 0, w - 1)
    yi = np.clip((y + wy).astype(int), 0, h - 1)
    e = 0.7 * e1[yi, xi] + 0.45 * e2[yi, xi] * e1[yi, xi] ** 0.3
    mod = fbm(w, h, rng, base=10, octaves=5)
    return np.clip(e * (0.35 + mod) ** 2, 0, None)


def gen_cosmic_web(layer, w, h, rng, inner=None):
    img = np.zeros((h, w, 3), np.float32)
    fil = _cosmic(w, h, rng, seeds=70, width=w * 0.006, warp_amt=w * 0.06)
    fil = fil / max(1e-6, np.percentile(fil, 99.5))
    node = np.clip(blur(fil, w * 0.012) - 0.3, 0, None) * 1.2
    img += (fil ** 1.6)[..., None] * lin((0.40, 0.25, 0.85)) * 0.30
    img += (fil ** 4)[..., None] * lin((0.55, 0.65, 1.0)) * 0.25
    x_, y_ = grid(w, h)
    calm = 1 - 0.85 * np.exp(-(((x_ - w / 2) ** 2 + (y_ - h / 2) ** 2) / (w * 0.06) ** 2))
    node *= calm
    fil *= 0.4 + 0.6 * calm
    img += node[..., None] * lin((1.0, 0.70, 0.35)) * 0.6
    n = int(w * h * 0.03)
    xs, ys = rng.random(n) * w, rng.random(n) * h
    keep = rng.random(n) < np.clip(fil[ys.astype(int), xs.astype(int)], 0, 1) * 0.7
    k = keep.sum()
    splat(img, xs[keep], ys[keep], np.tile(lin((0.95, 0.88, 1.0)), (k, 1)) * rng.pareto(3, k)[:, None] * 0.12, sigma=0.6)
    img += blur(img, w * 0.008) * 0.6
    glow(img, w / 2, h / 2, w * 0.002, lin((1.0, 0.8, 0.5)), 0.3, 2)
    return finish(img * 1.2, bloom_amt=0.6, sat=1.15)


def gen_observable(layer, w, h, rng, inner=None):
    x, y = grid(w, h)
    cx, cy = w / 2, h / 2
    R = 4.4e26 / layer.meters * w
    r = np.hypot(x - cx, y - cy) / R
    img = np.zeros((h, w, 3), np.float32)
    fil = _cosmic(w, h, rng, seeds=420, width=w * 0.0028, warp_amt=w * 0.02)
    inside = (r < 1).astype(np.float32)
    dens = 0.5 + 0.8 * r**2                                       # looking back in time: denser toward the edge
    img += (fil * inside * dens)[..., None] * lin((0.38, 0.26, 0.70)) * 0.9
    cmb = fbm(w, h, rng, base=40, octaves=4)
    cmb2 = fbm(w, h, rng, base=12, octaves=4)
    edge = np.exp(-((r - 1) / 0.03) ** 2)
    edge_col = lin((1.0, 0.42, 0.18)) * (0.6 + 0.8 * cmb[..., None]) + lin((0.3, 0.45, 1.0)) * (smooth(0.6, 0.8, cmb2) * 0.5)[..., None]
    img += edge[..., None] * edge_col * 1.2
    img += (np.exp(-((r - 0.96) / 0.12) ** 2) * inside)[..., None] * lin((0.5, 0.18, 0.12)) * 0.35
    img += (np.exp(-((r - 1.0) / 0.25) ** 2) * (1 - inside))[..., None] * lin((0.3, 0.1, 0.08)) * 0.2
    glow(img, cx, cy, w * 0.003, lin((1.0, 0.85, 0.6)), 1.0, 2)
    return finish(img * 1.3, bloom_amt=0.7)


GENERATORS = {
    "eye": gen_eye, "face": gen_face, "room": gen_room, "rooftop": gen_rooftop,
    "night_city": gen_night_city, "globe": gen_globe, "earth_moon": gen_earth_moon,
    "solar": gen_solar, "stars": gen_stars, "galaxy": gen_galaxy,
    "local_group": gen_local_group, "cosmic_web": gen_cosmic_web, "observable": gen_observable,
}

from .face import gen_face as _realistic_face  # noqa: E402  (face.py builds on the helpers above)

GENERATORS["face"] = _realistic_face
