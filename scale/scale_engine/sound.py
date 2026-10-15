"""Audio: narration placement, a procedural score that follows the camera, and the mix.

The score is code-built like the visuals: a low drone whose upper partials open
up as the camera reaches bigger scales, an "air" layer that rises with zoom
speed, and a whoosh plus sub impact on snap beats. Drop a music file into the
episode (`music: path`) to replace the drone; air and SFX still apply.
"""
from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import numpy as np

from .timeline import Camera, Cue


def read_wav(path: Path, sr: int) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == sr, f"{path}: expected {sr} Hz"
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        if w.getnchannels() == 2:
            a = a.reshape(-1, 2).mean(1)
    return a


def write_wav(path: Path, a: np.ndarray, sr: int) -> None:
    a = np.clip(a, -1, 1)
    if a.ndim == 1:
        a = a[:, None]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(a.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((a * 32767).astype(np.int16).tobytes())


def db(x: float) -> float:
    return 10 ** (x / 20)


def _env_track(fn, total: float, rate: int = 100) -> np.ndarray:
    t = np.arange(int(total * rate) + 2) / rate
    return t, np.array([fn(x) for x in t], np.float32)


def _upsample(t_ctrl, v_ctrl, n, sr):
    return np.interp(np.arange(n) / sr, t_ctrl, v_ctrl).astype(np.float32)


def _bandnoise(n, sr, lo, hi, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    shape = np.exp(-0.5 * (np.log(np.maximum(f, 1) / math.sqrt(lo * hi)) / (0.5 * math.log(hi / lo))) ** 2)
    out = np.fft.irfft(spec * shape, n)
    return (out / (np.abs(out).max() + 1e-9)).astype(np.float32)


def drone(n, sr, scale_ctrl, rng):
    t = np.arange(n) / sr
    out = np.zeros((n, 2), np.float32)
    base = 55.0
    for k, (ratio, amp) in enumerate([(1, 1.0), (1.5, 0.55), (2, 0.5), (3, 0.25), (4, 0.18)]):
        for ch, det in enumerate((-0.12, 0.12)):
            lfo = 0.75 + 0.25 * np.sin(2 * math.pi * (0.05 + 0.013 * k) * t + k + ch)
            out[:, ch] += amp * lfo * np.sin(2 * math.pi * (base * ratio + det) * t + rng.random() * 6)
    # shimmer: opens up with scale (0 at the eye, 1 at the edge of everything)
    for ratio in (6, 8, 9, 12):
        for ch in range(2):
            out[:, ch] += 0.09 * scale_ctrl * np.sin(2 * math.pi * base * ratio * t * (1 + 0.002 * ch) + rng.random() * 6) \
                * (0.6 + 0.4 * np.sin(2 * math.pi * 0.11 * t * ratio / 6 + ch))
    return out / np.abs(out).max()


def whoosh(sr, dur, rng):
    n = int(sr * dur)
    noise = rng.standard_normal(n).astype(np.float32)
    y = np.zeros(n, np.float32)
    acc = 0.0
    for i in range(n):        # swept one-pole low-pass, short enough for a Python loop
        u = i / n
        fc = 300 + 7000 * (math.sin(math.pi * u) ** 2)
        a = 1 - math.exp(-2 * math.pi * fc / sr)
        acc += a * (noise[i] - acc)
        y[i] = acc
    env = np.sin(np.linspace(0, math.pi, n)) ** 1.5
    y = y * env
    return y / (np.abs(y).max() + 1e-9)


def impact(sr, rng, dur=2.2):
    n = int(sr * dur)
    t = np.arange(n) / sr
    f = 32 + 40 * np.exp(-t * 6)
    phase = 2 * math.pi * np.cumsum(f) / sr
    y = np.sin(phase) * np.exp(-t * 2.2)
    y[: int(sr * 0.02)] += rng.standard_normal(int(sr * 0.02)) * np.exp(-np.arange(int(sr * 0.02)) / (sr * 0.004)) * 0.5
    return (y / np.abs(y).max()).astype(np.float32)


def build_mix(cues: list[Cue], cam: Camera, total: float, n_layers: int, cfg: dict,
              out_path: Path, snaps: list[tuple[float, float]], music: Path | None = None, seed: int = 7) -> Path:
    a = cfg["audio"]
    sr = a["sample_rate"]
    rng = np.random.default_rng(seed)
    tail = 1.5
    n = int((total + tail) * sr)
    voice = np.zeros(n, np.float32)
    for c in cues:
        if not c.audio:
            continue
        clip = read_wav(Path(c.audio), sr)
        i = int(c.speech_start * sr)
        j = min(n, i + len(clip))
        voice[i:j] += clip[: j - i]

    t_c, z_c = _env_track(cam, total + tail)
    scale_ctrl = _upsample(t_c, np.clip(z_c / max(1, n_layers - 1), 0, 1), n, sr)
    _, sp = _env_track(lambda x: min(cam.speed(x), 6.0), total + tail)
    speed = _upsample(t_c, np.clip(sp / 1.5, 0, 1), n, sr)

    if music and Path(music).exists():
        tmp = out_path.with_name("music_tmp.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", str(music),
                        "-t", f"{total + tail:.3f}", "-ac", "2", "-ar", str(sr), str(tmp)], check=True)
        with wave.open(str(tmp), "rb") as w:
            bed = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32).reshape(-1, 2) / 32768
        tmp.unlink(missing_ok=True)
        bed = np.pad(bed, ((0, max(0, n - len(bed))), (0, 0)))[:n]
    else:
        bed = drone(n, sr, scale_ctrl, rng)
    fade = np.clip(np.minimum(np.arange(n) / (sr * 1.5), (n - np.arange(n)) / (sr * tail)), 0, 1)
    bed *= fade[:, None]

    air = np.stack([_bandnoise(n, sr, 400, 5000, rng), _bandnoise(n, sr, 400, 5000, rng)], -1)
    air *= (0.08 + speed)[:, None] * fade[:, None]

    sfx = np.zeros(n, np.float32)
    for s0, s1 in snaps:
        w_ = whoosh(sr, (s1 - s0) + 0.6, rng)
        i = max(0, int((s0 - 0.3) * sr))
        j = min(n, i + len(w_))
        sfx[i:j] += w_[: j - i]
        hit = impact(sr, rng)
        i = int(s1 * sr)
        j = min(n, i + len(hit))
        sfx[i:j] += hit[: j - i] * 0.9

    # sidechain duck: bed and air dip under the voice
    blk = sr // 100
    m = len(voice) // blk
    rms = np.sqrt((voice[: m * blk].reshape(m, blk) ** 2).mean(1) + 1e-12)
    gate = np.clip((20 * np.log10(rms) + 50) / 20, 0, 1)
    # attack fast, release slow
    g = np.zeros_like(gate)
    for k in range(1, len(gate)):
        g[k] = gate[k] if gate[k] > g[k - 1] else g[k - 1] * 0.97 + gate[k] * 0.03
    duck = db(a["duck_db"]) ** _upsample(np.arange(m) * blk / sr, g, n, sr)

    mix = (voice[:, None] * np.ones((1, 2))
           + bed * db(a["bed_gain_db"]) * duck[:, None]
           + air * db(a["air_gain_db"]) * duck[:, None]
           + sfx[:, None] * db(a["sfx_gain_db"]))
    raw = out_path.with_name("mix_raw.wav")
    write_wav(raw, mix / max(1.0, np.abs(mix).max()), sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                    "-af", f"loudnorm=I={a['target_lufs']}:TP={a['true_peak']}:LRA=11",
                    "-ar", str(sr), "-ac", "2", str(out_path)], check=True)
    raw.unlink(missing_ok=True)
    return out_path
