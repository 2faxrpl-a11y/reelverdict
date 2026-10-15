"""The zoom compositor.

For a camera index z the frame is built from the layer stack: the smallest
layer that fully covers the view is drawn opaque as the base, then every
smaller layer that is still at least a couple of pixels wide is drawn on top
with a feathered edge. Each layer keeps a mip pyramid so heavy shrinking never
aliases, and only the on-screen box of each layer is resampled.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .spec import Episode, load_episode
from .timeline import Camera, Caption, format_scale


# ---------------------------------------------------------------- layers
def _fit_aspect(img: Image.Image, aspect: float) -> Image.Image:
    w, h = img.size
    if abs(h / w - aspect) < 1e-3:
        return img
    if h / w > aspect:                       # too tall: crop height
        nh = int(round(w * aspect))
        y0 = (h - nh) // 2
        return img.crop((0, y0, w, y0 + nh))
    nw = int(round(h / aspect))
    x0 = (w - nw) // 2
    return img.crop((x0, 0, x0 + nw, h))


def _feather_mask(w: int, h: int, feather: float) -> Image.Image:
    f = max(1.0, feather * w)
    x = np.minimum(np.arange(w), np.arange(w)[::-1]) / f
    y = np.minimum(np.arange(h), np.arange(h)[::-1]) / f
    m = np.clip(np.minimum(x[None, :], y[:, None]), 0, 1)
    m = m * m * (3 - 2 * m)                  # smoothstep
    return Image.fromarray((m * 255).astype(np.uint8), "L")


SPACE_KINDS = {"earth_moon", "solar", "stars", "galaxy", "local_group", "cosmic_web", "observable"}
SAME_LIGHT = {"eye", "face", "room"}          # one lamp-lit scene: low frequencies may be matched fully


class LayerImage:
    def __init__(self, img: Image.Image, feather: float):
        rgba = img.convert("RGBA")
        rgba.putalpha(_feather_mask(img.width, img.height, feather))
        self.opaque = [img]
        self.soft = [rgba]
        while self.opaque[-1].width > 32:
            w, h = self.opaque[-1].width // 2, self.opaque[-1].height // 2
            self.opaque.append(self.opaque[-1].resize((w, h), Image.BOX))
            self.soft.append(self.soft[-1].resize((w, h), Image.BOX))

    def level(self, out_w: float) -> int:
        s = self.opaque[0].width / max(out_w, 1e-6)
        if s <= 1.5:
            return 0
        return min(len(self.opaque) - 1, int(math.floor(math.log2(s / 1.5))) + 1)


def _load(path: Path, aspect: float, max_w: int) -> Image.Image:
    img = _fit_aspect(Image.open(path).convert("RGB"), aspect)
    if img.width > max_w:
        img = img.resize((max_w, int(round(max_w * aspect))), Image.LANCZOS)
    return img


def seam_match(inner: Image.Image, outer: Image.Image, frac: float, at: tuple[float, float],
               reach: float = 0.5, full: bool = False) -> Image.Image:
    """Shift the inner image's low frequencies toward the outer image near its edges.

    `frac` is the inner layer's width as a fraction of the outer's, `at` its
    center in the outer image. The correction fades to zero over `reach` of the
    way to the inner image's center, so its subject is never touched.
    """
    gw, gh = 12, max(2, int(round(12 * inner.height / inner.width)))
    ow, oh = outer.size
    bw, bh = frac * ow, frac * oh
    box = (at[0] * ow - bw / 2, at[1] * oh - bh / 2, at[0] * ow + bw / 2, at[1] * oh + bh / 2)
    target = np.asarray(outer.resize((gw, gh), Image.BOX, box=box), np.float32)
    have = np.asarray(inner.resize((gw, gh), Image.BOX), np.float32)
    diff = np.clip(target - have, -70, 70)
    w, h = inner.size
    up = np.stack([np.asarray(Image.fromarray(diff[..., c], "F").resize((w, h), Image.BICUBIC)) for c in range(3)], -1)
    x = np.minimum(np.arange(w), np.arange(w)[::-1]) / (w / 2)
    y = np.minimum(np.arange(h), np.arange(h)[::-1]) / (h / 2)
    d = np.minimum(x[None, :], y[:, None])
    wgt = np.clip(1 - d / reach, 0, 1)
    wgt = wgt * wgt * (3 - 2 * wgt)
    if full:                                 # same-light scenes: match everything but the subject at the center
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rc = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (w / 2))
        core = np.clip((rc - 0.35) / 0.3, 0, 1)
        wgt = np.maximum(wgt, core * core * (3 - 2 * core))
    out = np.asarray(inner, np.float32) + up * wgt[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


class Compositor:
    def __init__(self, ep: Episode, layer_paths: list[Path], width: int, height: int):
        self.ep, self.W, self.H = ep, width, height
        look = ep.cfg["look"]
        self.bg = tuple(look.get("background", [0, 0, 0]))
        self.fade = look.get("magnify_fade", [6.0, 20.0])
        self.feather = look["feather"]
        max_w = max(1024, int(width * 2))
        imgs = [_load(p, ep.aspect, max_w) for p in layer_paths]
        if look.get("seam_match", True):
            for j in range(len(imgs) - 2, -1, -1):   # each against the untouched outer image
                L, O = ep.layers[j], ep.layers[j + 1]
                imgs[j] = seam_match(imgs[j], imgs[j + 1], L.meters / O.meters, O.inner_at,
                                     full=L.kind in SAME_LIGHT and O.kind in SAME_LIGHT)
        self.layers = [LayerImage(im, look["feather"]) for im in imgs]

    def _rect(self, j: int, v: float, cx: float, cy: float):
        L = self.ep.layers[j]
        asp = self.ep.aspect
        sx = self.W / v
        left = (L.center[0] - L.width / 2 - (cx - v / 2)) * sx
        top = (L.center[1] - L.width * asp / 2 - (cy - v * asp / 2)) * sx
        size = L.width * sx
        return left, top, size, size * asp

    def frame(self, z: float) -> Image.Image:
        v, cx, cy = self.ep.view(z)
        W, H = self.W, self.H
        rects = [self._rect(j, v, cx, cy) for j in range(len(self.layers))]
        # A layer becomes the opaque base only once its un-feathered core covers
        # the screen, so the hand-off between soft overlay and base never pops.
        eps = 0.5
        fe = self.feather
        base = None
        for j, (l, t, w, h) in enumerate(rects):
            m = fe * w
            if l + m <= eps and t + m <= eps and l + w - m >= W - eps and t + h - m >= H - eps:
                base = j
                break
        out = Image.new("RGBA", (W, H), self.bg + (255,))
        if base is not None:
            self._draw(out, base, rects[base], opaque=True)
            start = base - 1
        else:
            start = len(self.layers) - 1
        for j in range(start, -1, -1):
            if rects[j][2] < 1:
                break
            self._draw(out, j, rects[j], opaque=False)
        return out.convert("RGB")

    def _draw(self, out: Image.Image, j: int, rect, opaque: bool):
        l, t, w, h = rect
        W, H = self.W, self.H
        x0, y0 = max(0, int(math.floor(l))), max(0, int(math.floor(t)))
        x1, y1 = min(W, int(math.ceil(l + w))), min(H, int(math.ceil(t + h)))
        if x1 <= x0 or y1 <= y0:
            return
        li = self.layers[j]
        k = li.level(w)
        src = (li.opaque if opaque else li.soft)[k]
        a = src.width / w
        e = src.height / h
        data = (a, 0, (x0 - l) * a, 0, e, (y0 - t) * e)
        resample = Image.BICUBIC if a < 1 else Image.BILINEAR
        patch = src.transform((x1 - x0, y1 - y0), Image.AFFINE, data, resample=resample)
        if opaque:
            # a base blown up far past its resolution dissolves to the background
            mag = w / li.opaque[0].width
            lo, hi = self.fade
            f = min(1.0, max(0.0, (mag - lo) / (hi - lo)))
            if self.ep.layers[j].kind not in SPACE_KINDS:
                f = 0.0                      # human-scale layers soften instead of going dark
            if f > 0:
                patch = Image.blend(patch, Image.new("RGB", patch.size, self.bg), f * f * (3 - 2 * f))
            out.paste(patch.convert("RGBA"), (x0, y0))
        else:
            out.alpha_composite(patch, (x0, y0))


# ---------------------------------------------------------------- overlays
def _font(paths: list[str], size: int):
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


class Overlay:
    def __init__(self, ep: Episode, captions: list[Caption], width: int, height: int, seed: int = 3):
        self.ep, self.W, self.H = ep, width, height
        cfg = ep.cfg
        self.look, self.hud, self.capcfg = cfg["look"], cfg["hud"], cfg["captions"]
        self.caps = captions
        self.f_cap = _font(cfg["fonts"]["caption"], int(self.capcfg["size"] * width))
        self.f_hud = _font(cfg["fonts"]["hud"], int(self.hud["size"] * width))
        self.f_hud_small = _font(cfg["fonts"]["hud"], int(self.hud["size"] * width * 0.55))
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        r = np.sqrt(((x - width / 2) / (width / 2)) ** 2 + ((y - height / 2) / (height / 2)) ** 2) / math.sqrt(2)
        self.vig = (1 - self.look["vignette"] * np.clip(r, 0, 1) ** 2.2)[..., None]
        mrng = np.random.default_rng(seed + 11)
        n_m = int(self.look.get("dust", {}).get("count", 240))
        self._motes = list(zip(mrng.uniform(0, 2 * math.pi, n_m), mrng.random(n_m), mrng.pareto(2.0, n_m) * 0.4 + 0.25))
        self.credit = cfg.get("credit", {})
        self._init_credits(width, height)
        rng = np.random.default_rng(seed)
        self.grain = [(rng.standard_normal((height, width, 1)) * 255 * self.look["grain"]).astype(np.float32)
                      for _ in range(6)]

    def grade(self, arr: np.ndarray, frame_idx: int) -> np.ndarray:
        """Film finish: bloom + halation, chromatic aberration, split-tone grade, vignette, grain."""
        lk = self.look
        out = arr.astype(np.float32)
        W, H = self.W, self.H
        if lk.get("bloom", 0) > 0 or lk.get("halation", 0) > 0:
            small = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).reduce(4)
            hot = np.clip(np.asarray(small, np.float32) - 150, 0, None) * (255 / 105)
            hot_img = Image.fromarray(np.clip(hot, 0, 255).astype(np.uint8))
            b1 = np.asarray(hot_img.filter(ImageFilter.GaussianBlur(W / 4 * 0.006)).resize((W, H), Image.BILINEAR), np.float32)
            b2 = np.asarray(hot_img.filter(ImageFilter.GaussianBlur(W / 4 * 0.03)).resize((W, H), Image.BILINEAR), np.float32)
            out += lk.get("bloom", 0) * (0.6 * b1 + 0.4 * b2)
            out += lk.get("halation", 0) * b2 * np.array([1.0, 0.35, 0.15], np.float32)
        ca = lk.get("aberration", 0)
        if ca > 0:
            im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
            r, g, b = im.split()

            def scaled(ch, f):
                cw, chh = int(round(W * f)), int(round(H * f))
                big = ch.resize((cw, chh), Image.BILINEAR)
                x0, y0 = (cw - W) // 2, (chh - H) // 2
                if f >= 1:
                    return big.crop((x0, y0, x0 + W, y0 + H))
                canvas = Image.new("L", (W, H), 0)
                canvas.paste(big, (-x0, -y0))
                return canvas
            out = np.stack([np.asarray(scaled(r, 1 + ca), np.float32), np.asarray(g, np.float32),
                            np.asarray(scaled(b, 1 - ca), np.float32)], -1)
        tone = lk.get("split_tone", 0)
        if tone > 0:                                            # teal shadows, amber highlights
            l = out.mean(-1, keepdims=True) / 255
            out += tone * 255 * ((1 - l) ** 2 * np.array([-0.02, 0.01, 0.035]) + l**2 * np.array([0.03, 0.01, -0.03]))
        out *= self.vig
        if lk["grain"] > 0:
            out += self.grain[frame_idx % len(self.grain)]
        return np.clip(out, 0, 255).astype(np.uint8)

    def dust(self, arr: np.ndarray, z: float, z_prev: float, n_layers: int) -> np.ndarray:
        """Scale dust: motes that drift toward the center as we zoom out, and streak on fast moves."""
        cfg = self.look.get("dust", {})
        if not cfg or not cfg.get("enabled", True):
            return arr
        W, H = self.W, self.H
        speed = abs(z - z_prev)
        space = min(1.0, max(0.0, (z - 3.5) / 2.0))
        a_base = cfg.get("opacity", 0.5) * max(space, min(1.0, speed * 4))
        if a_base < 0.01:
            return arr
        layer = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(layer)
        rate, span = cfg.get("rate", 0.55), math.log(1.6 * H / (0.02 * W))
        rmin = 0.02 * W
        cx, cy = W / 2, H / 2
        for th, ph, b in self._motes:
            u = (ph - z * rate) % 1.0
            u0 = (ph - z_prev * rate) % 1.0
            r = rmin * math.exp(span * u)
            r0 = rmin * math.exp(span * u0)
            if abs(u - u0) > 0.5:                      # wrapped this frame: no streak across the wrap
                r0 = r
            r0 = r + max(-0.35 * H, min(0.35 * H, r0 - r))
            fade = math.sin(math.pi * u)
            a = int(255 * a_base * b * fade)
            if a < 4:
                continue
            c, s_ = math.cos(th), math.sin(th)
            wd = 1 if abs(r0 - r) < 3 else 2
            if abs(r0 - r) < 1.5:
                d.ellipse([cx + c * r - 1.2, cy + s_ * r - 1.2, cx + c * r + 1.2, cy + s_ * r + 1.2], fill=a)
            else:
                d.line([(cx + c * r0, cy + s_ * r0), (cx + c * r, cy + s_ * r)], fill=a, width=wd)
        m = np.asarray(layer.filter(ImageFilter.GaussianBlur(0.6)), np.float32)[..., None] / 255
        return arr + m * np.array([255, 236, 210], np.float32) * 0.9

    def _init_credits(self, W: int, H: int):
        from .brand import FONT_LIGHT, FONT_TEXT, OUT, build_all
        logo_path = OUT / "scale_logo_transparent_white.png"
        if not logo_path.exists():
            build_all()
        self.logo = Image.open(logo_path).convert("RGBA")
        self.f_credit = ImageFont.truetype(str(FONT_TEXT), max(10, int(W * self.credit.get("watermark", {}).get("size", 0.026))))
        self.f_end = ImageFont.truetype(str(FONT_LIGHT), max(10, int(W * 0.040)))
        self.f_cta = ImageFont.truetype(str(FONT_TEXT), max(10, int(W * 0.030)))
        self._logo_cache = {}

    def _logo(self, width: int) -> Image.Image:
        if width not in self._logo_cache:
            h = int(round(self.logo.height * width / self.logo.width))
            self._logo_cache[width] = self.logo.resize((width, h), Image.LANCZOS)
        return self._logo_cache[width]

    def _paste_alpha(self, img: Image.Image, logo: Image.Image, cx: float, cy: float, a: float):
        if a <= 0.01:
            return
        lg = logo
        if a < 1:
            lg = logo.copy()
            lg.putalpha(lg.getchannel("A").point(lambda v: int(v * a)))
        img.alpha_composite(lg, (int(cx - lg.width / 2), int(cy - lg.height / 2)))

    def credits(self, img: Image.Image, t: float, story_end: float, total: float) -> Image.Image:
        """Ajay Haynes credit: watermark throughout, intro lockup over the hook, end card."""
        c = self.credit
        if not c:
            return img
        W, H = self.W, self.H
        img = img.convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        intro = c.get("intro", {})
        wm = c.get("watermark", {})
        in_intro = intro.get("enabled") and t < intro.get("start", 0) + intro.get("duration", 0) + 0.4
        if wm.get("enabled") and not in_intro and t < story_end:
            a = wm.get("opacity", 0.5) * min(1.0, (t - intro.get("duration", 0)) / 0.6 if intro.get("enabled") else 1.0)
            a *= min(1.0, max(0.0, (story_end - t) / 0.5))
            if a > 0.01:
                d.text((wm["x"] * W, wm["y"] * H), c["name"], font=self.f_credit, anchor="lm",
                       fill=(255, 255, 255, int(255 * a)))
        if intro.get("enabled"):
            t0, dur = intro.get("start", 0), intro.get("duration", 2.5)
            a = min(1.0, max(0.0, (t - t0) / 0.35), max(0.0, (t0 + dur - t) / 0.45))
            if a > 0:
                lw = int(W * 0.30)
                lg = self._logo(lw)
                y = intro.get("y", 0.22) * H
                self._paste_alpha(layer, lg, W / 2, y, a)
                d.text((W / 2, y + lg.height * 0.62), intro.get("text", "by " + c["name"]), font=self.f_credit,
                       anchor="mm", fill=(255, 255, 255, int(220 * a)))
        img = Image.alpha_composite(img, layer)
        end = c.get("end_card", {})
        if end.get("enabled") and t >= story_end:
            u = t - story_end
            black = Image.new("RGBA", img.size, (4, 5, 7, int(255 * min(1.0, u / 0.7))))
            img = Image.alpha_composite(img, black)
            card = Image.new("RGBA", img.size, (0, 0, 0, 0))
            dc = ImageDraw.Draw(card)
            a = min(1.0, max(0.0, (u - 0.45) / 0.6), max(0.0, (total - t) / 0.35))
            lg = self._logo(int(W * 0.62))
            self._paste_alpha(card, lg, W / 2, H * 0.44, a)
            dc.text((W / 2, H * 0.44 + lg.height * 0.75), "Created by " + c["name"], font=self.f_end,
                    anchor="mm", fill=(244, 239, 230, int(235 * a)))
            a2 = min(1.0, max(0.0, (u - 1.1) / 0.6), max(0.0, (total - t) / 0.35))
            if end.get("cta"):
                dc.text((W / 2, H * 0.44 + lg.height * 0.75 + W * 0.085), end["cta"], font=self.f_cta,
                        anchor="mm", fill=(232, 166, 87, int(230 * a2)))
            img = Image.alpha_composite(img, card)
        return img.convert("RGB")

    def text(self, img: Image.Image, t: float, z: float, total: float) -> Image.Image:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        W, H = self.W, self.H
        if self.hud["enabled"]:
            a = min(1.0, t / 0.8, max(0.0, (total - t) / 0.8))
            human, power = format_scale(self.ep.meters_at(z))
            y = self.hud["y"] * H
            d.text((W / 2, y), human, font=self.f_hud, fill=(255, 255, 255, int(235 * a)), anchor="mm")
            d.text((W / 2, y + self.hud["size"] * W * 0.95), power, font=self.f_hud_small,
                   fill=(255, 255, 255, int(150 * a)), anchor="mm")
            i = int(round(z))
            if 0 <= i < len(self.ep.layers):
                la = max(0.0, 1 - abs(z - i) / self.hud["label_fade"]) * a
                if la > 0.01:
                    d.text((W / 2, y + self.hud["size"] * W * 1.75), self.ep.layers[i].label.upper(),
                           font=self.f_hud_small, fill=(255, 220, 170, int(200 * la)), anchor="mm")
        if self.capcfg["enabled"]:
            for c in self.caps:
                if c.start <= t < c.end:
                    fade = min(1.0, (t - c.start) / 0.08, (c.end - t) / 0.08)
                    sw = max(2, int(self.capcfg["size"] * W * 0.07))
                    d.text((W / 2, self.capcfg["y"] * H), c.text, font=self.f_cap, anchor="mm",
                           fill=(255, 255, 255, int(255 * fade)), stroke_width=sw,
                           stroke_fill=(0, 0, 0, int(170 * fade)))
                    break
        return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


# ---------------------------------------------------------------- worker
_W: dict = {}


def _init_worker(job: dict):
    ep = load_episode(job["episode"])
    ep.cfg["look"].update(job.get("look", {}))
    _W["ep"] = ep
    _W["comp"] = Compositor(ep, [Path(p) for p in job["layers"]], job["width"], job["height"])
    _W["over"] = Overlay(ep, job["captions"], job["width"], job["height"])
    _W["cam"] = Camera(job["keys"])
    _W["job"] = job


def render_frame(i: int) -> bytes:
    job, cam, comp, over = _W["job"], _W["cam"], _W["comp"], _W["over"]
    fps = job["fps"]
    t = i / fps
    dz = cam.speed(t) / fps
    n = int(min(job["blur_max"], max(1, math.ceil(dz * 6))))
    if n == 1:
        arr = np.asarray(comp.frame(cam(t)), np.float32)
    else:
        acc = None
        for k in range(n):
            tk = t + (k / (n - 1) - 0.5) * 0.5 / fps          # 180-degree shutter
            a = np.asarray(comp.frame(cam(tk)), np.float32)
            acc = a if acc is None else acc + a
        arr = acc / n
    arr = over.dust(arr, cam(t), cam(t - 1 / fps), len(_W["ep"].layers))
    img = Image.fromarray(over.grade(arr, i))
    story_end = job.get("story_end", job["total"])
    if t < story_end:
        img = over.text(img, t, cam(t), story_end)
    img = over.credits(img, t, story_end, job["total"])
    return img.tobytes()


def _encoder_args(cfg: dict, preview: bool) -> list[str]:
    codec = cfg["output"].get("codec", "auto")
    if codec == "auto":
        codec = "libx264"
        if platform.system() == "Darwin":
            enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
            if "h264_videotoolbox" in enc:
                codec = "h264_videotoolbox"
    if codec == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", cfg["output"].get("vt_bitrate", "18M"), "-profile:v", "high"]
    return ["-c:v", "libx264", "-preset", "veryfast" if preview else "slow",
            "-crf", "26" if preview else str(cfg["output"].get("crf", 19)),
            "-maxrate", "8M" if preview else "40M", "-bufsize", "16M" if preview else "80M"]


def encode(job: dict, audio: Path, out: Path, preview: bool, progress=None) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found (brew install ffmpeg)")
    ep_cfg = job["cfg"]
    n_frames = int(math.ceil(job["total"] * job["fps"]))
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{job['width']}x{job['height']}", "-r", str(job["fps"]), "-i", "-",
           "-i", str(audio),
           *_encoder_args(ep_cfg, preview), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    workers = job["workers"]
    try:
        if workers <= 1:
            _init_worker(job)
            frames = (render_frame(i) for i in range(n_frames))
            pool = None
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(workers, initializer=_init_worker, initargs=(job,))
            frames = pool.imap(render_frame, range(n_frames), chunksize=2)
        for i, buf in enumerate(frames):
            proc.stdin.write(buf)
            if progress:
                progress(i + 1, n_frames)
        if pool:
            pool.close()
            pool.join()
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")
    return out


def still(job: dict, t: float) -> Image.Image:
    if _W.get("job") is not job:
        _init_worker(job)
    i = int(round(t * job["fps"]))
    return Image.frombytes("RGB", (job["width"], job["height"]), render_frame(i))
