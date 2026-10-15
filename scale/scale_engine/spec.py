"""Load series/cast/episode YAML and derive the zoom geometry.

Geometry model
--------------
Every layer is a 9:16 image covering a real-world width `meters`. Layers are
ordered innermost (index 0) to outermost. World units are layer-0 widths, so
layer 0 has width 1 and center (0, 0). Layer j+1 places layer j at its
`inner_at` point, which fixes layer j+1's center in world space.

The camera position is a continuous index z: z == i shows layer i exactly
filling the frame. Between layers the view width is interpolated in log space,
and the view center moves linearly with view width, which is the same as
zooming about a fixed point, so motion stays smooth across layer boundaries.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Layer:
    id: str
    meters: float
    label: str = ""
    kind: str = "stars"
    prompt: str = ""
    inner_at: tuple[float, float] = (0.5, 0.5)
    params: dict = field(default_factory=dict)
    asset: str | None = None
    # derived
    width: float = 1.0                      # world units (layer-0 widths)
    center: tuple[float, float] = (0.0, 0.0)


@dataclass
class Beat:
    id: str
    layer: str
    text: str
    narrator: str | None = None
    move: str = "smooth"                    # smooth | snap
    arrive: float = 0.5
    offset: float = 0.0
    hold: float | None = None
    pre: float | None = None
    snap: float = 1.2


@dataclass
class Episode:
    path: Path
    id: str
    slug: str
    title: str
    direction: str
    narrator: str
    style: str
    layers: list[Layer]
    beats: list[Beat]
    camera: dict
    cfg: dict                               # merged series config
    cast: dict
    raw: dict

    # ---- geometry -------------------------------------------------------
    @property
    def aspect(self) -> float:
        return self.cfg["output"]["height"] / self.cfg["output"]["width"]

    def layer_index(self, layer_id: str) -> int:
        for i, l in enumerate(self.layers):
            if l.id == layer_id:
                return i
        raise KeyError(f"{self.id}: beat references unknown layer '{layer_id}'")

    def view(self, z: float) -> tuple[float, float, float]:
        """Camera at continuous index z -> (view_width, center_x, center_y)."""
        L = self.layers
        n = len(L)
        if n == 1:
            return L[0].width * math.exp(z), L[0].center[0], L[0].center[1]
        if z <= 0:
            r = L[1].width / L[0].width
            return L[0].width * r ** z, L[0].center[0], L[0].center[1]
        if z >= n - 1:
            r = L[-1].width / L[-2].width
            return L[-1].width * r ** (z - (n - 1)), L[-1].center[0], L[-1].center[1]
        i = int(math.floor(z))
        f = z - i
        a, b = L[i], L[i + 1]
        v = a.width * (b.width / a.width) ** f
        t = (v - a.width) / (b.width - a.width)
        cx = a.center[0] + (b.center[0] - a.center[0]) * t
        cy = a.center[1] + (b.center[1] - a.center[1]) * t
        return v, cx, cy

    def meters_at(self, z: float) -> float:
        v, _, _ = self.view(z)
        return v * self.layers[0].meters / self.layers[0].width

    def narrator_for(self, beat: Beat) -> dict:
        nid = beat.narrator or self.narrator
        if nid not in self.cast["narrators"]:
            raise KeyError(f"{self.id}: unknown narrator '{nid}'")
        return {"id": nid, **self.cast["narrators"][nid]}


def load_series() -> tuple[dict, dict]:
    return load_yaml(CONFIG_DIR / "series.yaml"), load_yaml(CONFIG_DIR / "cast.yaml")


def load_episode(path: str | Path) -> Episode:
    path = Path(path).resolve()
    raw = load_yaml(path)
    series, cast = load_series()
    cfg = _deep_merge(series, {k: raw[k] for k in ("output", "look", "hud", "captions", "audio") if k in raw})

    layers = []
    for d in raw["layers"]:
        layers.append(Layer(
            id=d["id"], meters=float(d["meters"]), label=d.get("label", d["id"]),
            kind=d.get("kind", "stars"), prompt=(d.get("prompt") or "").strip(),
            inner_at=tuple(d.get("inner_at", (0.5, 0.5))), params=d.get("params") or {},
            asset=d.get("asset"),
        ))
    beats = [Beat(**{k: v for k, v in b.items()}) for b in raw["beats"]]

    ep = Episode(
        path=path, id=raw["id"], slug=raw.get("slug", raw["id"]), title=raw.get("title", raw["id"]),
        direction=raw.get("direction", "out"), narrator=raw.get("narrator", "guide"),
        style=(raw.get("style") or "").strip(), layers=layers, beats=beats,
        camera=raw.get("camera") or {}, cfg=cfg, cast=cast, raw=raw,
    )
    _derive_geometry(ep)
    validate(ep)
    return ep


def _derive_geometry(ep: Episode) -> None:
    asp = ep.aspect
    base = ep.layers[0].meters
    ep.layers[0].width = 1.0
    ep.layers[0].center = (0.0, 0.0)
    for j in range(1, len(ep.layers)):
        inner, outer = ep.layers[j - 1], ep.layers[j]
        outer.width = outer.meters / base
        u, v = outer.inner_at
        outer.center = (
            inner.center[0] - (u - 0.5) * outer.width,
            inner.center[1] - (v - 0.5) * outer.width * asp,
        )


def validate(ep: Episode) -> list[str]:
    """Raise on hard errors, return soft warnings."""
    warnings = []
    ids = [l.id for l in ep.layers]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{ep.id}: duplicate layer ids")
    for a, b in zip(ep.layers, ep.layers[1:]):
        if b.meters <= a.meters:
            raise ValueError(f"{ep.id}: layer '{b.id}' must be wider than '{a.id}' (layers run inner -> outer)")
        r = b.meters / a.meters
        if r > 40 and a.kind not in ("earth_moon", "solar", "stars", "galaxy", "local_group", "cosmic_web"):
            warnings.append(f"layer '{a.id}' -> '{b.id}' jumps {r:.0f}x; detailed subjects read best under ~15x")
    for b in ep.beats:
        ep.layer_index(b.layer)
        ep.narrator_for(b)
        if b.move not in ("smooth", "snap"):
            raise ValueError(f"{ep.id}: beat '{b.id}' move must be smooth or snap")
    if ep.narrator not in ep.cast["narrators"]:
        raise KeyError(f"{ep.id}: unknown narrator '{ep.narrator}'")
    return warnings
