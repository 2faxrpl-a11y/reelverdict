"""A realistic, code-built human face, seen from above (lying on a floor, looking up).

Pipeline: an anatomical height field in centimetres -> normals -> key light with
ray-marched self-shadowing -> texture-space subsurface scattering (red light
diffuses furthest under skin) -> cavity occlusion -> two-lobe specular on an
oil map -> cool rim light -> strand hair with clumping and anisotropic sheen.
The eye layer itself is embedded at both eyes, colour-matched to the face.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .art import bezier, blur, embed, finish, lin, normals, smooth, unpack, wood_floor
from .placeholders import fbm, grid


def _g(X, Y, x0, y0, sx, sy, amp, rot=0.0):
    """Anisotropic gaussian bump (cm)."""
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        dx, dy = (X - x0) * c + (Y - y0) * s, -(X - x0) * s + (Y - y0) * c
    else:
        dx, dy = X - x0, Y - y0
    return amp * np.exp(-((dx / sx) ** 2 + (dy / sy) ** 2))


def _curve_dist(X, Y, p0, p1, p2, n=40):
    """Distance from each pixel to a quadratic curve (coarse, vectorised over samples)."""
    pts = np.asarray(bezier(p0, p1, p2, n))
    d = np.full(X.shape, np.inf, np.float32)
    for px, py in pts:
        d = np.minimum(d, (X - px) ** 2 + (Y - py) ** 2)
    return np.sqrt(d)


def _shift(a, dx, dy):
    """Sample a[y + dy, x + dx] with edge clamping (integer offsets)."""
    h, w = a.shape
    xs = np.clip(np.arange(w) + dx, 0, w - 1)
    ys = np.clip(np.arange(h) + dy, 0, h - 1)
    return a[ys][:, xs]


def ray_shadow(hgt_px: np.ndarray, L, steps=28, step_px=3.0) -> np.ndarray:
    """1 = lit, 0 = shadowed. hgt_px is height in pixel units; L is the light direction."""
    L = np.asarray(L, np.float32) / np.linalg.norm(L)
    lxy = math.hypot(L[0], L[1]) + 1e-6
    rise = L[2] / lxy                                   # height gained per pixel travelled toward the light
    ux, uy = L[0] / lxy, L[1] / lxy
    occ = np.zeros_like(hgt_px)
    for k in range(1, steps + 1):
        t = k * step_px
        other = _shift(hgt_px, int(round(ux * t)), int(round(uy * t)))
        occ = np.maximum(occ, np.clip((other - (hgt_px + t * rise)) / (step_px * 1.5), 0, 1))
    return 1 - occ


def gen_face(layer, w, h, rng, inner=None):
    cm = w / (layer.meters * 100)                       # pixels per centimetre
    x, y = grid(w, h)
    ax, ay = layer.inner_at[0] * w, layer.inner_at[1] * h
    gap = 6.3 * cm
    cx_px = ax + gap / 2                                # face midline
    X, Y = (x - cx_px) / cm, (y - ay) / cm              # cm; Y = 0 on the eye line
    ex = gap / 2 / cm
    S = (-1, 1)

    # ---------------- head silhouette: cranium, cheekbones, jaw taper, chin
    ys_ = np.array([-12.2, -10, -6, -2, 1, 4, 7, 9.5, 11.2, 12.3, 12.8])
    ws_ = np.array([0.0, 5.8, 7.5, 7.7, 7.5, 7.0, 6.2, 5.1, 3.9, 1.8, 0.0])
    yf = np.linspace(-13, 13, 1041)
    wf = np.convolve(np.pad(np.interp(yf, ys_, ws_), 60, mode="edge"), np.hanning(121) / np.hanning(121).sum(), "same")[60:-60]
    half = np.interp(Y, yf, wf)                          # smoothed profile: no facets on the jaw
    inside = np.clip(1 - (X / np.maximum(half, 1e-3)) ** 2, 0, 1) * (half > 0)
    vcap = np.clip(1 - ((Y - 0.2) / 12.3) ** 2, 0, 1)
    hgt = 8.6 * np.sqrt(inside) * vcap ** 0.35

    # ---------------- anatomy (cm)
    for s in S:
        hgt += _g(X, Y, s * 3.2, -1.95, 2.3, 0.65, 0.55, rot=s * 0.12)        # brow ridge
        hgt += _g(X, Y, s * 3.2, -0.15, 1.9, 1.35, -1.55)                      # orbit
        hgt += _g(X, Y, s * 1.75, 0.05, 0.75, 1.0, -0.55)                      # inner orbit
        hgt += _g(X, Y, s * 3.2, 0.0, 1.25, 0.78, 0.95)                        # eyeball under lids
        hgt += _g(X, Y, s * 3.25, -0.78, 1.35, 0.13, -0.22)                    # upper lid crease
        hgt += _g(X, Y, s * 3.2, 0.95, 1.35, 0.32, 0.14)                       # lower lid
        hgt += _g(X, Y, s * 2.45, 1.45, 1.0, 0.28, -0.22, rot=s * 0.5)         # tear trough
        hgt += _g(X, Y, s * 4.7, 2.1, 1.8, 1.25, 0.85)                         # cheekbone
        hgt += _g(X, Y, s * 4.9, 5.3, 1.35, 1.9, -0.45)                        # cheek hollow
        hgt += _g(X, Y, s * 6.4, -1.6, 1.1, 2.0, -0.5)                         # temple
        hgt += _g(X, Y, s * 5.4, 7.6, 1.1, 2.0, 0.25)                          # jaw angle
        hgt += _g(X, Y, s * 1.15, 5.0, 0.62, 0.55, 0.62)                       # nostril wing
        hgt += _g(X, Y, s * 0.62, 5.42, 0.32, 0.17, -0.55)                     # nostril
        hgt += _g(X, Y, s * 0.36, 6.25, 0.16, 0.62, 0.08)                      # philtrum column
        d = _curve_dist(X, Y, (s * 1.65, 5.25), (s * 2.6, 6.6), (s * 2.55, 8.3))  # nasolabial fold
        hgt += -0.32 * np.exp(-(d / 0.22) ** 2) + 0.18 * np.exp(-((d - 0.5) / 0.45) ** 2) * (X * s > 1.8)
    hgt += _g(X, Y, 0, -1.55, 1.1, 0.8, 0.25)                                  # glabella
    hgt += _g(X, Y, 0, -0.75, 0.9, 0.45, -0.25)                                # nasion
    t = np.clip((Y + 0.6) / 5.2, 0, 1)                                          # nose bridge: grows to the tip
    ridge = (0.40 + 1.35 * t ** 1.25) * np.exp(-(X / (0.55 + 0.5 * t)) ** 2) * smooth(-1.3, -0.5, Y) * smooth(5.3, 4.4, Y)
    hgt += blur(ridge.astype(np.float32), 0.15 * cm)
    hgt += _g(X, Y, 0, 4.75, 0.85, 0.72, 0.6)                                  # nose tip
    hgt += _g(X, Y, 0, 6.2, 1.9, 0.95, 0.55)                                   # maxilla under the nose
    hgt += _g(X, Y, 0, 5.35, 0.28, 0.25, 0.1)                                  # columella
    hgt += _g(X, Y, 0, 6.25, 0.16, 0.62, -0.07)                                # philtrum groove
    ym = 7.45                                                                   # mouth line
    cupid = 0.14 * np.exp(-((np.abs(X) - 0.42) / 0.28) ** 2) - 0.05 * np.exp(-(X / 0.18) ** 2)
    up_edge = ym - 0.55 - cupid
    lip_w = np.clip(1 - (X / 2.35) ** 2, 0, 1)
    upper = (Y > up_edge) & (Y < ym) & (lip_w > 0)
    lower_edge = ym + 0.78 * lip_w ** 0.7
    lower = (Y >= ym) & (Y < lower_edge) & (lip_w > 0)
    hgt += blur((upper * 0.32 * lip_w ** 0.5 + lower * 0.42 * lip_w ** 0.5).astype(np.float32), 0.12 * cm)
    hgt += -0.28 * np.exp(-((Y - ym) / 0.07) ** 2) * (np.abs(X) < 2.45)        # mouth line
    for s in S:
        hgt += _g(X, Y, s * 2.4, ym + 0.05, 0.3, 0.3, -0.25)                   # mouth corners
    hgt += _g(X, Y, 0, 8.95, 1.6, 0.35, -0.28)                                 # mentolabial sulcus
    hgt += _g(X, Y, 0, 10.2, 1.75, 1.1, 0.55)                                  # chin
    # ears, partly tucked under hair
    ear = np.zeros_like(hgt)
    for s in S:
        e = np.clip(1 - ((X - s * 7.7) / 0.95) ** 2 - ((Y - 1.8) / 2.7) ** 2, 0, 1)
        helix = np.exp(-((np.sqrt(((X - s * 7.7) / 0.95) ** 2 + ((Y - 1.8) / 2.7) ** 2) - 0.8) / 0.12) ** 2)
        ear = np.maximum(ear, 2.6 * np.sqrt(e) + 0.25 * helix * (e > 0) - 0.5 * _g(X, Y, s * 7.65, 2.2, 0.35, 0.8, 1.0))
    hgt = np.maximum(hgt, ear)
    hgt = blur(hgt, 0.12 * cm)
    pores = fbm(w, h, rng, base=int(w / (0.07 * cm)), octaves=2)
    fine = fbm(w, h, rng, base=int(w / (0.25 * cm)), octaves=2)
    skin_mask = smooth(0.0, 0.25, hgt) * (hgt > 0.05)
    hgt_d = hgt + (0.018 * pores + 0.016 * fine) * skin_mask                   # micro relief

    # ---------------- lighting
    key = np.array([-0.55, -0.40, 0.85])                # warm lamp, upper left, fairly high
    rim_dir = np.array([0.85, -0.25, 0.35])             # cool moonlight from the right
    n = normals(hgt_d, cm)
    nd = (n * (key / np.linalg.norm(key))).sum(-1)
    sh = ray_shadow(hgt * cm, key, steps=30, step_px=max(1.5, 0.12 * cm))
    sh = np.clip(blur(sh, 0.22 * cm), 0, 1)
    E = np.clip((nd + 0.18) / 1.18, 0, 1) * sh          # wrapped diffuse irradiance
    # subsurface: red scatters furthest, blue least
    Er, Eg, Eb = blur(E, 0.32 * cm), blur(E, 0.14 * cm), blur(E, 0.05 * cm)
    E3 = np.stack([0.55 * Er + 0.45 * E, 0.6 * Eg + 0.4 * E, 0.7 * Eb + 0.3 * E], -1)
    cav = np.clip(1 - 0.7 * np.clip(blur(hgt, 0.6 * cm) - hgt, 0, None), 0.6, 1)   # crease occlusion

    # albedo: base tone, flush on cheeks/nose/ears/lips, darker under the eyes
    mott = fbm(w, h, rng, base=18, octaves=5)
    freck = np.clip(fbm(w, h, rng, base=int(w / (0.35 * cm)), octaves=1) - 0.72, 0, 1) * 2.0
    alb = lin((0.78, 0.60, 0.52)) * (0.9 + 0.16 * mott[..., None]) * (1 - 0.18 * freck[..., None])
    flush = sum(_g(X, Y, s * 4.1, 3.2, 2.2, 1.6, 1.6) for s in S) + _g(X, Y, 0, 4.8, 1.0, 0.9, 1.4) + \
        sum(_g(X, Y, s * 7.7, 1.8, 1.0, 2.6, 1.0) for s in S)
    alb = alb * (1 - 0.18 * flush[..., None]) + lin((0.80, 0.42, 0.38)) * 0.18 * flush[..., None]
    under = sum(_g(X, Y, s * 3.1, 1.25, 1.4, 0.45, 1.0) for s in S)
    alb = alb * (1 - 0.22 * under[..., None]) + lin((0.45, 0.33, 0.36)) * 0.22 * under[..., None]
    lipm = blur((upper | lower).astype(np.float32), 0.06 * cm)
    lip_lines = 0.5 + 0.5 * np.sin(X * 38 + fbm(w, h, rng, 30, 2) * 4)
    alb = alb * (1 - 0.8 * lipm[..., None]) + lin((0.66, 0.42, 0.40)) * 0.8 * (0.85 + 0.15 * lip_lines[..., None]) * lipm[..., None]
    alb *= (0.94 + 0.08 * pores[..., None])

    # specular: oily T-zone and lips, sharp + broad lobes, fresnel boost
    oil = np.clip(_g(X, Y, 0, -4.5, 3.5, 2.5, 1.0) + _g(X, Y, 0, 3.0, 1.0, 2.6, 1.0) +
                  _g(X, Y, 0, 10.0, 1.6, 1.0, 0.7) + lipm * 1.2, 0, 1) * (0.6 + 0.4 * pores)
    Lk = key / np.linalg.norm(key)
    Hv = (Lk + np.array([0, 0, 1.0]))
    Hv /= np.linalg.norm(Hv)
    nh = np.clip((n * Hv).sum(-1), 0, 1)
    fres = 0.04 + 0.96 * (1 - n[..., 2]) ** 5
    spec = (0.20 * oil * nh ** 80 + 0.018 * nh ** 12) * sh * (1 + 2 * fres)
    rim = np.clip((n * (rim_dir / np.linalg.norm(rim_dir))).sum(-1), 0, 1) ** 2 * (1 - n[..., 2]) ** 1.5

    warm, cool = lin((1.0, 0.86, 0.72)), lin((0.55, 0.70, 1.0))
    bounce = np.clip(0.5 + 0.5 * n[..., 2], 0, 1)[..., None] * lin((0.85, 0.66, 0.55))   # warm floor/room bounce
    up_fill = np.clip(-n[..., 1], 0, 1)[..., None] * 0.10 * lin((0.9, 0.7, 0.55))   # floor bounce into downward faces
    skin = alb * (E3 * warm * 1.40 + 0.12 * bounce + up_fill + 0.03 * cool) * cav[..., None]
    skin += spec[..., None] * lin((1.0, 0.92, 0.85))
    skin += rim[..., None] * alb * cool * 0.9
    # low-key: a slash of lamplight across the eyes, the rest of the face falls into shadow
    slash = np.exp(-(((Y + 0.4 + 0.12 * X) / 3.2) ** 2)) * (0.6 + 0.4 * np.exp(-((X + 4) / 9) ** 2))
    skin *= (0.22 + 0.95 * slash)[..., None]

    # ---------------- the eye layer, colour-matched into both orbits
    inner_img, il = unpack(inner)
    if inner_img is not None:
        eye_w = il.meters / layer.meters * w
        mm_px = eye_w / 50.0
        for sxp, flip in ((ax, False), (ax + gap, True)):
            im = inner_img.transpose(Image.FLIP_LEFT_RIGHT) if flip else inner_img
            ew_i, eh_i = int(round(eye_w)), int(round(eye_w * im.height / im.width))
            patch = (np.asarray(im.convert("RGB").resize((ew_i, eh_i), Image.LANCZOS), np.float32) / 255) ** 2.2
            x0, y0 = int(round(sxp - ew_i / 2)), int(round(ay - eh_i / 2))
            xs0, ys0, xs1, ys1 = max(0, x0), max(0, y0), min(w, x0 + ew_i), min(h, y0 + eh_i)
            region = skin[ys0:ys1, xs0:xs1]
            pp = patch[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
            gain = np.clip((blur(region, 0.6 * cm) + 1e-3) / (blur(pp, 0.6 * cm) + 1e-3), 0.15, 3.0)
            yy, xx = np.mgrid[ys0:ys1, xs0:xs1].astype(np.float32)
            opening = np.exp(-(((xx - sxp) / (14 * mm_px)) ** 2 + ((yy - ay) / (5.5 * mm_px)) ** 2))
            g = gain ** (1 - 0.55 * opening[..., None])          # the eye keeps some of its own light
            m = np.exp(-(((xx - sxp) / (17.5 * mm_px)) ** 6 + ((yy - ay + 1 * mm_px) / (9.5 * mm_px)) ** 6))[..., None]
            skin[ys0:ys1, xs0:xs1] = region * (1 - m) + pp * g * m

    # ---------------- neck, shirt, floor
    floor, sheen = wood_floor(w, h, rng, plank_px=12 * cm, tone=(0.36, 0.21, 0.12))
    img = floor * (0.18 + 0.7 * np.exp(-(((X + 12) / 26) ** 2 + ((Y + 8) / 34) ** 2)))[..., None]
    img += lin((0.6, 0.45, 0.3)) * 0.04 * sheen[..., None]
    neck_hw = 4.3 + 0.25 * np.clip(Y - 12, 0, None)                  # neck widens into the trapezius
    neck = (np.abs(X) < neck_hw) & (Y > 8) & (Y < 21)
    neck_h = 4.2 * np.sqrt(np.clip(1 - (X / neck_hw) ** 2, 0, 1)) * neck
    collar = 17.2 + 0.9 * (np.abs(X) / 5.0) ** 2                       # crew neckline
    shoulder = 18.5 + 0.05 * np.abs(X) ** 1.6
    shirt = (Y > np.minimum(collar, shoulder + 3)) & (Y > shoulder - 2.0 - 0 * X)
    folds = blur(fbm(w, h, rng, base=6, octaves=5) * 2.4, 0.3 * cm) + 4.0 * np.sqrt(np.clip(1 - (X / 30) ** 2, 0, 1))
    body_h = np.where(shirt, folds, 0) + neck_h * ~shirt
    occ_all = np.maximum(hgt * skin_mask, body_h)
    contact = blur((occ_all > 0.05).astype(np.float32), 1.6 * cm)
    img *= (1 - 0.75 * np.clip(contact * 1.6, 0, 1))[..., None]
    nb = normals(blur(body_h, 0.2 * cm), cm)
    body_l = np.clip(((nb * Lk).sum(-1) + 0.2) / 1.2, 0, 1)
    knit = 0.85 + 0.15 * np.sin(Y * cm * 0.9) * np.sin(X * cm * 0.25)
    shirt_c = lin((0.16, 0.19, 0.27)) * (0.15 + 1.1 * body_l[..., None]) * knit[..., None]
    shirt_c += lin((0.55, 0.7, 1.0)) * 0.05 * np.clip((nb * rim_dir / np.linalg.norm(rim_dir)).sum(-1), 0, 1)[..., None]
    neck_c = alb * (0.06 + 1.1 * body_l[..., None]) * lin((1.0, 0.82, 0.65)) * smooth(19, 10, Y)[..., None]
    neck_c *= (1 - 0.6 * smooth(9, 11.5, Y) * np.exp(-(X / 3.5) ** 2))[..., None]   # under-jaw shadow
    img = np.where(shirt[..., None], shirt_c, img)
    img = np.where((neck & ~shirt)[..., None], neck_c, img)
    fm = smooth(0.02, 0.35, hgt)
    img_floor = img
    skin_layer = skin

    # ---------------- hair: clumped strands fanned on the floor, a soft hairline on the forehead
    hairline = -5.6 - 1.4 * np.abs(X / 6.0) ** 2.2 + 0.25 * np.sin(X * 2.1)
    lines, vals = [], []
    key_ang = math.atan2(key[1], key[0])
    for c in range(140):                              # clumps of long hair spread on the floor
        a = rng.uniform(-math.pi * 0.98, -math.pi * 0.02) if rng.random() < 0.7 else \
            rng.choice([-1, 1]) * rng.uniform(0.0, 0.55) + (0 if rng.random() < 0.5 else -math.pi)
        hx0 = 7.0 * math.cos(a) * rng.uniform(0.8, 0.98)
        hy0 = -0.6 + 10.6 * math.sin(a) * rng.uniform(0.85, 0.98)
        flow = math.atan2(hy0 + 3.0, hx0)              # hair falls away from the crown
        L = rng.uniform(10, 24)
        bend = rng.normal(0, 0.25)
        for k in range(55):
            jit = rng.normal(0, 0.28, 2)
            p0 = (cx_px + (hx0 + jit[0]) * cm, ay + (hy0 + jit[1]) * cm)
            f2 = flow + bend + rng.normal(0, 0.06)
            Lk_ = L * rng.uniform(0.6, 1.0)
            p1 = (p0[0] + math.cos(flow) * Lk_ * 0.5 * cm, p0[1] + math.sin(flow) * Lk_ * 0.5 * cm)
            p2 = (p0[0] + math.cos(f2) * Lk_ * cm, p0[1] + math.sin(f2) * Lk_ * cm)
            lines.append((bezier(p0, p1, p2, 16), 0.03 * cm, 0.012 * cm))
            vals.append(0.15 + 0.85 * abs(math.sin(flow - key_ang)) ** 8)  # sheen across the strand
    for k in range(1400):                             # side hair over the temples, falling outward past the ears
        sx_ = rng.choice([-1, 1])
        hy_ = rng.uniform(-8.0, 1.2)
        hx_ = sx_ * rng.uniform(5.8, 7.4)
        ang = math.atan2(rng.normal(-0.2, 0.3), sx_ * 1.0)
        L = rng.uniform(5, 13)
        p0 = (cx_px + hx_ * cm, ay + hy_ * cm)
        p2 = (p0[0] + math.cos(ang) * L * cm, p0[1] + math.sin(ang) * L * cm)
        p1 = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2 + rng.normal(0, 0.4) * cm)
        lines.append((bezier(p0, p1, p2, 10), 0.03 * cm, 0.014 * cm))
        vals.append(0.2 + 0.8 * abs(math.sin(ang - key_ang)) ** 8)
    for k in range(3200):                             # strands sweeping back from the hairline over the crown
        hx_ = rng.uniform(-7.3, 7.3)
        hy_ = -5.6 - 1.4 * abs(hx_ / 6.0) ** 2.2 + rng.uniform(-0.05, 1.2) * (rng.random() < 0.15) + rng.uniform(-0.05, 0.25)
        ang = math.atan2(-1.0, hx_ * 0.30) + rng.normal(0, 0.10)
        L = rng.uniform(6, 15)
        p0 = (cx_px + hx_ * cm, ay + hy_ * cm)
        p2 = (p0[0] + math.cos(ang) * L * cm, p0[1] + math.sin(ang) * L * cm)
        p1 = ((p0[0] + p2[0]) / 2 + rng.normal(0, 0.25) * cm, (p0[1] + p2[1]) / 2)
        lines.append((bezier(p0, p1, p2, 10), 0.028 * cm, 0.014 * cm))
        vals.append(0.2 + 0.8 * abs(math.sin(ang - key_ang)) ** 8)
    def draw(group):
        cov = Image.new("L", (w, h), 0)
        shine = Image.new("L", (w, h), 0)
        dc, ds = ImageDraw.Draw(cov), ImageDraw.Draw(shine)
        for (pts, w0, w1), v in group:
            for i in range(len(pts) - 1):
                wid = max(1, int(round(w0 + (w1 - w0) * i / max(1, len(pts) - 2))))
                seg = [tuple(pts[i]), tuple(pts[i + 1])]
                dc.line(seg, fill=255, width=wid)
                ds.line(seg, fill=int(255 * v), width=wid)
        return np.asarray(cov, np.float32) / 255, np.asarray(shine, np.float32) / 255

    pairs = list(zip(lines, vals))
    n_floor = len(pairs) - 4600
    lamp = np.exp(-(((X + 9) / 20) ** 2 + ((Y + 6) / 22) ** 2))
    hair_base = lin((0.06, 0.04, 0.03)) * (0.7 + 0.6 * fbm(w, h, rng, base=40, octaves=3)[..., None])

    def shade_hair(hm, hs, k_sheen=0.30):
        c = hair_base + lin((0.85, 0.62, 0.40)) * (hs * hm * k_sheen * (0.3 + lamp))[..., None] \
            + lin((0.5, 0.65, 1.0)) * (hs * hm * 0.05)[..., None]
        return c * (0.45 + 0.55 * hm[..., None])

    # 1) hair lying on the floor, under the head
    hm, hs = draw(pairs[:n_floor])
    fmass = np.clip(blur(hm, 0.12 * cm) * 1.3, 0, 1) * smooth(0.0, 0.6, blur(hm, 1.2 * cm) * 3)
    head_m = (inside > 0).astype(np.float32)
    halo = np.clip(blur(head_m, 2.2 * cm) * 2.2, 0, 1) * smooth(5.0, 1.5, Y) * (1 - head_m)   # hair mat hugging the skull
    fmass = np.maximum(fmass, halo * (0.75 + 0.25 * blur(hm, 0.3 * cm)))
    img_floor = img_floor * (1 - 0.5 * np.clip(blur(fmass, 0.8 * cm), 0, 1) * (1 - fmass))[..., None]
    img = img_floor * (1 - fmass[..., None]) + shade_hair(hm, hs) * fmass[..., None]
    # 2) the head on top of it
    img = img * (1 - fm[..., None]) + skin_layer * fm[..., None]
    # 3) eyebrows, strand by strand, where the eye layer has its brow
    brow_lines = []
    for s_ in S:
        for k in range(420):
            u = rng.uniform(-2.3, 2.1)                                 # cm along the brow, nose side first
            bx = s_ * (ex - u)
            by = -2.1 - 0.35 * math.exp(-((u + 0.2) / 1.4) ** 2) + rng.normal(0, 0.12) + 0.04 * u
            ang = math.radians(-14 if u < 0.7 else 18) + rng.normal(0, 0.18)
            L = rng.uniform(0.3, 0.6)
            dx_, dy_ = -s_ * math.cos(ang) * L, -math.sin(ang) * L * 0.6
            p0 = (cx_px + bx * cm, ay + by * cm)
            p2 = (p0[0] + dx_ * cm, p0[1] + dy_ * cm)
            brow_lines.append((bezier(p0, ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2 - 0.05 * cm), p2, 5), 0.022 * cm, 0.01 * cm))
    from .art import strokes
    bm = strokes(w, h, brow_lines, ss=2)
    img = img * (1 - 0.85 * bm[..., None]) + lin((0.07, 0.045, 0.035)) * 0.85 * bm[..., None]
    # 4) crown hair over the top of the head, soft hairline
    hm2, hs2 = draw(pairs[n_floor:])
    face_oval = np.sqrt((X / 6.1) ** 2 + ((Y - 2.6) / 9.6) ** 2)          # the bare face; hair frames it
    sides = smooth(0.96, 1.06, face_oval) * smooth(3.5, 2.0, Y)
    above = np.maximum(smooth(0.2, -1.2, Y - hairline) * (Y < 0), sides) * (inside > 0)
    cmass = np.clip(blur(hm2, 0.08 * cm) * 1.15 + above * 0.9, 0, 1)
    img *= (1 - 0.45 * np.clip(blur(cmass, 0.35 * cm), 0, 1) * (1 - cmass))[..., None]
    texture = 0.5 + 0.5 * blur(fbm(w, h, rng, base=int(w / (0.4 * cm)), octaves=2), 0.05 * cm)
    crown = shade_hair(np.maximum(hm2, above * 0.75 * texture), np.maximum(hs2, above * 0.2 * texture))
    img = img * (1 - cmass[..., None]) + crown * cmass[..., None]
    return finish(img * 1.35, bloom_amt=0.3, sat=0.82)
