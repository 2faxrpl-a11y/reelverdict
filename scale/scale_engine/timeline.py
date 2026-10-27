"""Turn narration timing into a camera curve and caption cues.

Narration drives everything: each beat's audio length sets when the camera
lands on that beat's layer. Smooth stretches use a monotone cubic (PCHIP)
through the landing points, so the zoom never stops or overshoots; snap beats
use an exponential ease for a whip move.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .spec import Episode


@dataclass
class Cue:
    beat_id: str
    narrator: str
    text: str
    audio: str | None
    start: float          # beat start
    speech_start: float
    speech_end: float
    end: float


@dataclass
class Key:
    t: float
    z: float
    mode: str = "smooth"  # how we arrive at this key from the previous one


def layout(ep: Episode, durations: list[float], audio_paths: list[str | None]) -> tuple[list[Cue], list[Key], float]:
    cues, keys = [], []
    t = 0.0
    z_first = ep.layer_index(ep.beats[0].layer) + ep.beats[0].offset
    z_prev = z_first + float(ep.camera.get("start_offset", 0.0))
    keys.append(Key(0.0, z_prev))
    for beat, dur, path in zip(ep.beats, durations, audio_paths):
        nar = ep.narrator_for(beat)
        z_target = ep.layer_index(beat.layer) + beat.offset
        if beat.move == "snap":
            pre = beat.pre if beat.pre is not None else beat.snap + 0.1
        else:
            pre = beat.pre if beat.pre is not None else nar.get("pre", 0.3)
        hold = beat.hold if beat.hold is not None else nar.get("gap", 0.5)
        start = t
        s0 = start + pre
        s1 = s0 + dur
        end = s1 + hold
        cues.append(Cue(beat.id, nar["id"], beat.text, path, start, s0, s1, end))
        if beat.move == "snap":
            keys.append(Key(start, z_prev, "smooth"))
            keys.append(Key(start + beat.snap, z_target, "snap"))
        else:
            keys.append(Key(s0 + beat.arrive * dur, z_target, "smooth"))
        z_prev = z_target
        t = end
    total = t
    keys.append(Key(total, z_prev + float(ep.camera.get("end_offset", 0.0)), "smooth"))
    # merge keys that collide in time (keep the later one's intent)
    clean: list[Key] = []
    for k in keys:
        if clean and k.t - clean[-1].t < 1e-3:
            clean[-1] = Key(clean[-1].t, k.z, k.mode)
        else:
            clean.append(k)
    return cues, clean, total


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    if n < 2:
        return np.zeros(n)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(n)
    if n == 2:
        m[:] = d[0]
        return m
    for k in range(1, n - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def end(h0, h1, d0, d1):
        s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(s) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(s) > abs(3 * d0):
            return 3 * d0
        return s

    m[0] = end(h[0], h[1], d[0], d[1])
    m[-1] = end(h[-1], h[-2], d[-1], d[-2])
    return m


def _ease_snap(u: float) -> float:
    # exponential in-out: lingers for a beat, whips, then settles hard
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    if u < 0.5:
        return 0.5 * 2 ** (20 * u - 10)
    return 1 - 0.5 * 2 ** (-20 * u + 10)


class Camera:
    """z(t) built from keys; smooth runs are PCHIP, snap segments are eased."""

    def __init__(self, keys: list[Key]):
        self.keys = keys
        self.t = np.array([k.t for k in keys])
        self.z = np.array([k.z for k in keys])
        # split into runs of consecutive smooth segments
        self.runs = []  # (i0, i1, slopes)
        i0 = 0
        for i in range(1, len(keys)):
            if keys[i].mode == "snap":
                if i - 1 > i0:
                    self._add_run(i0, i - 1)
                i0 = i
        if len(keys) - 1 > i0:
            self._add_run(i0, len(keys) - 1)

    def _add_run(self, i0: int, i1: int):
        x, y = self.t[i0:i1 + 1], self.z[i0:i1 + 1]
        self.runs.append((i0, i1, _pchip_slopes(x, y)))

    def __call__(self, t: float) -> float:
        if t <= self.t[0]:
            return float(self.z[0])
        if t >= self.t[-1]:
            return float(self.z[-1])
        i = int(np.searchsorted(self.t, t, side="right") - 1)
        if self.keys[i + 1].mode == "snap":
            u = (t - self.t[i]) / (self.t[i + 1] - self.t[i])
            return float(self.z[i] + (self.z[i + 1] - self.z[i]) * _ease_snap(u))
        for i0, i1, m in self.runs:
            if i0 <= i < i1:
                k = i - i0
                h = self.t[i + 1] - self.t[i]
                s = (t - self.t[i]) / h
                h00 = 2 * s**3 - 3 * s**2 + 1
                h10 = s**3 - 2 * s**2 + s
                h01 = -2 * s**3 + 3 * s**2
                h11 = s**3 - s**2
                return float(h00 * self.z[i] + h10 * h * m[k] + h01 * self.z[i + 1] + h11 * h * m[k + 1])
        return float(self.z[i])

    def speed(self, t: float, dt: float = 1 / 60) -> float:
        """|dz/dt| in layers per second."""
        return abs(self(t + dt / 2) - self(t - dt / 2)) / dt


@dataclass
class Caption:
    start: float
    end: float
    text: str


def captions(cues: list[Cue], max_words: int, max_chars: int) -> list[Caption]:
    out = []
    for c in cues:
        words = c.text.split()
        chunks, cur = [], []
        for w in words:
            trial = " ".join(cur + [w])
            if cur and (len(cur) >= max_words or len(trial) > max_chars):
                chunks.append(" ".join(cur))
                cur = [w]
            else:
                cur.append(w)
            if w[-1] in ".,;:!?" and cur:     # break at punctuation
                chunks.append(" ".join(cur))
                cur = []
        if cur:
            chunks.append(" ".join(cur))
        weights = [len(ch) + 4 for ch in chunks]
        total = sum(weights)
        span = c.speech_end - c.speech_start
        t = c.speech_start
        for ch, wgt in zip(chunks, weights):
            d = span * wgt / total
            out.append(Caption(t, t + d, ch))
            t += d
    return out


def to_srt(caps: list[Caption]) -> str:
    def ts(x: float) -> str:
        ms = int(round(x * 1000))
        h, ms = divmod(ms, 3600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, c in enumerate(caps, 1):
        lines += [str(i), f"{ts(c.start)} --> {ts(c.end)}", c.text, ""]
    return "\n".join(lines)


def format_scale(meters: float) -> tuple[str, str]:
    """Human readout + power-of-ten readout for a frame width in meters."""
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    exp = int(math.floor(math.log10(meters) + 1e-9))
    power = f"10{str(exp).translate(sup)} m"
    LY, AU = 9.4607e15, 1.495979e11

    def num(x: float) -> str:
        if x >= 100:
            return f"{x:,.0f}"
        if x >= 10:
            return f"{x:.0f}"
        return f"{x:.1f}".rstrip("0").rstrip(".")

    def big(x: float, unit: str) -> str:
        for div, word in ((1e9, "billion"), (1e6, "million")):
            if x >= div:
                return f"{num(x / div)} {word} {unit}"
        return f"{num(x)} {unit}"

    if meters < 1e-3:
        human = f"{num(meters * 1e6)} µm"
    elif meters < 1e-2:
        human = f"{num(meters * 1e3)} mm"
    elif meters < 1:
        human = f"{num(meters * 100)} cm"
    elif meters < 1e3:
        human = f"{num(meters)} m"
    elif meters < 0.05 * AU:
        human = big(meters / 1e3, "km")
    elif meters < 0.5 * LY:
        human = f"{num(meters / AU)} AU"
    else:
        human = big(meters / LY, "light-years")
    return human, power
