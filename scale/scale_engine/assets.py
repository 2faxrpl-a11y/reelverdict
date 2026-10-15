"""Layer image resolution, placeholder caching, prompt sheets and outpaint prep.

Lookup order for a layer image:
  1. `asset:` on the layer (relative to assets/, e.g. library/earth_night.png)
  2. assets/<episode id>/<layer id>.(png|jpg|jpeg|webp)
  3. a procedural previz placeholder, generated once and cached in build/
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from . import placeholders
from .spec import ROOT, Episode, Layer

ASSETS = ROOT / "assets"
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def real_asset(ep: Episode, layer: Layer) -> Path | None:
    if layer.asset:
        p = ASSETS / layer.asset
        if not p.exists():
            raise FileNotFoundError(f"layer '{layer.id}': asset {p} not found")
        return p
    for ext in EXTS:
        p = ASSETS / ep.id / f"{layer.id}{ext}"
        if p.exists():
            return p
    return None


def placeholder_path(ep: Episode, layer: Layer, width: int) -> Path:
    key = json.dumps([layer.kind, layer.meters, layer.inner_at, layer.params, width,
                      placeholders.__file__ and Path(placeholders.__file__).stat().st_mtime], default=str)
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    return ROOT / "build" / "cache" / "previz" / f"{ep.id}_{layer.id}_{width}_{h}.png"


def resolve(ep: Episode, width: int, log=print) -> tuple[list[Path], list[str]]:
    """Return one image per layer plus the ids of layers still on placeholders."""
    paths, previz = [], []
    for i, layer in enumerate(ep.layers):
        p = real_asset(ep, layer)
        if p is None:
            p = placeholder_path(ep, layer, width)
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                log(f"  previz  {layer.id:<14} ({layer.kind})")
                h = int(round(width * ep.aspect))
                placeholders.generate(layer, width, h, seed=1000 + i).save(p)
            previz.append(layer.id)
        paths.append(p)
    return paths, previz


def prompt_sheet(ep: Episode) -> str:
    """Markdown brief for producing the real stills, innermost first."""
    out = [f"# {ep.id} — {ep.title}", "",
           "Make the stills **innermost first**, then build each outer layer by outpainting the one",
           "before it (`python -m scale_engine outpaint-prep`), so every zoom seam lines up exactly.",
           f"All images 9:16 vertical, at least 1440x2560. Save as `assets/{ep.id}/<layer id>.png`.", "",
           f"**Global style:** {ep.style}", ""]
    for i, l in enumerate(ep.layers):
        out.append(f"## {i:02d} · `{l.id}` — {l.label} ({l.meters:.3g} m across)")
        out.append("")
        out.append(f"{l.prompt} — {ep.style}")
        if i > 0:
            inner = ep.layers[i - 1]
            frac = inner.meters / l.meters
            u, v = l.inner_at
            out.append("")
            out.append(f"*Continuity:* the previous layer (`{inner.id}`) fills a {frac * 100:.2g}%-wide box "
                       f"centered at ({u:.2f}, {v:.2f}) of this frame. Outpaint from it rather than generating fresh.")
        out.append("")
    return "\n".join(out)


def outpaint_prep(ep: Episode, layer_id: str, width: int = 1440) -> tuple[Path, Path]:
    """Canvas + mask for generating layer `layer_id` from the layer just inside it.

    The inner image is shrunk to its true scale and placed at `inner_at`; the
    mask is white where the outpainting model should invent new content.
    """
    j = ep.layer_index(layer_id)
    if j == 0:
        raise ValueError("the innermost layer has nothing inside it; generate it directly")
    inner, outer = ep.layers[j - 1], ep.layers[j]
    src = real_asset(ep, inner)
    if src is None:
        raise FileNotFoundError(f"make assets/{ep.id}/{inner.id}.png first")
    h = int(round(width * ep.aspect))
    frac = inner.meters / outer.meters
    iw, ih = max(2, int(round(width * frac))), max(2, int(round(h * frac)))
    img = Image.open(src).convert("RGB").resize((iw, ih), Image.LANCZOS)
    cx, cy = outer.inner_at[0] * width, outer.inner_at[1] * h
    x0, y0 = int(round(cx - iw / 2)), int(round(cy - ih / 2))
    canvas = Image.new("RGB", (width, h), (0, 0, 0))
    canvas.paste(img, (x0, y0))
    mask = Image.new("L", (width, h), 255)
    ImageDraw.Draw(mask).rectangle([x0 + 2, y0 + 2, x0 + iw - 3, y0 + ih - 3], fill=0)
    mask = mask.filter(ImageFilter.GaussianBlur(2))
    out_dir = ROOT / "build" / ep.id / "outpaint"
    out_dir.mkdir(parents=True, exist_ok=True)
    cp, mp_ = out_dir / f"{layer_id}_canvas.png", out_dir / f"{layer_id}_mask.png"
    canvas.save(cp)
    mask.save(mp_)
    return cp, mp_
