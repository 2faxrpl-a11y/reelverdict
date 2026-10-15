"""Narration: ElevenLabs (production), macOS `say` (free scratch), or silent (animatic).

Every line is cached by a hash of provider + voice settings + text, so re-renders
never re-bill a line that hasn't changed. All outputs are normalised to
48 kHz mono 16-bit WAV.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import wave
from pathlib import Path

import numpy as np

from .spec import ROOT

CACHE = ROOT / "build" / "cache" / "tts"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _to_wav(src: Path, dst: Path, sr: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(sr),
         "-sample_fmt", "s16", str(dst)],
        check=True,
    )


def _silent(text: str, narrator: dict, dst: Path, sr: int) -> None:
    words = max(1, len(text.split()))
    secs = words / narrator.get("pace_wpm", 140) * 60.0
    secs += 0.25 * text.count(",") + 0.4 * sum(text.count(c) for c in ".?!")
    n = int(secs * sr)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.zeros(n, dtype=np.int16).tobytes())


def _say(text: str, narrator: dict, dst: Path, sr: int) -> None:
    if not shutil.which("say"):
        raise RuntimeError("`say` is macOS-only. Use --tts silent here, or --tts elevenlabs.")
    s = narrator.get("say", {})
    aiff = dst.with_suffix(".aiff")
    cmd = ["say", "-o", str(aiff)]
    if s.get("voice"):
        cmd += ["-v", s["voice"]]
    if s.get("rate"):
        cmd += ["-r", str(s["rate"])]
    subprocess.run(cmd + [text], check=True)
    _to_wav(aiff, dst, sr)
    aiff.unlink(missing_ok=True)


def _elevenlabs(text: str, narrator: dict, dst: Path, sr: int, cfg: dict) -> None:
    el = narrator.get("elevenlabs", {})
    voice = el.get("voice_id")
    if not voice or voice == "REPLACE_ME":
        raise RuntimeError(f"{narrator['name']}: set elevenlabs.voice_id in config/cast.yaml")
    key = os.environ.get(cfg.get("api_key_env", "ELEVENLABS_API_KEY"))
    if not key:
        raise RuntimeError(f"export {cfg.get('api_key_env', 'ELEVENLABS_API_KEY')}=... first")
    body = {
        "text": text,
        "model_id": cfg.get("model_id", "eleven_multilingual_v2"),
        "voice_settings": {k: el[k] for k in ("stability", "similarity_boost", "style", "speed") if k in el}
        | {"use_speaker_boost": True},
    }
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
           f"?output_format={cfg.get('output_format', 'mp3_44100_192')}")
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
        "xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
    mp3 = dst.with_suffix(".mp3")
    with urllib.request.urlopen(req, timeout=120) as r:
        mp3.write_bytes(r.read())
    _to_wav(mp3, dst, sr)
    mp3.unlink(missing_ok=True)


def synth(text: str, narrator: dict, provider: str, series_cfg: dict) -> Path:
    sr = series_cfg["audio"]["sample_rate"]
    settings = {"silent": narrator.get("pace_wpm"), "say": narrator.get("say"),
                "elevenlabs": [narrator.get("elevenlabs"), series_cfg["tts"]["elevenlabs"].get("model_id")]}
    key = json.dumps([provider, narrator["id"], settings.get(provider), text, sr], sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"{provider}_{narrator['id']}_{h}.wav"
    if dst.exists():
        return dst
    tmp = dst.with_name(dst.stem + ".part.wav")
    if provider == "silent":
        _silent(text, narrator, tmp, sr)
    elif provider == "say":
        _say(text, narrator, tmp, sr)
    elif provider == "elevenlabs":
        _elevenlabs(text, narrator, tmp, sr, series_cfg["tts"]["elevenlabs"])
    else:
        raise ValueError(f"unknown tts provider '{provider}'")
    tmp.replace(dst)
    return dst
