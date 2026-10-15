"""Episode build: narration -> timeline -> score/mix -> frames -> mp4 + upload metadata."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from . import assets, sound, tts
from .render import encode, still
from .spec import ROOT, Episode, load_episode, validate
from .timeline import Camera, captions, layout, to_srt


def _log(msg: str) -> None:
    print(msg, flush=True)


def prepare(ep: Episode, provider: str, preview: bool, workers: int | None = None) -> dict:
    cfg = ep.cfg
    out = cfg["output"]
    scale = out["preview_scale"] if preview else 1.0
    width = int(round(out["width"] * scale / 2)) * 2
    height = int(round(out["height"] * scale / 2)) * 2
    fps = out["preview_fps"] if preview else out["fps"]

    _log(f"narration  ({provider})")
    paths, durs = [], []
    for b in ep.beats:
        nar = ep.narrator_for(b)
        p = tts.synth(b.text, nar, provider, cfg)
        paths.append(str(p))
        durs.append(tts.wav_duration(p))
        _log(f"  {b.id:<8} {nar['id']:<8} {durs[-1]:5.2f}s  {b.text}")

    cues, keys, story_end = layout(ep, durs, paths)
    end_card = cfg.get("credit", {}).get("end_card", {})
    total = story_end + (end_card.get("duration", 0) if end_card.get("enabled") else 0)
    caps = captions(cues, cfg["captions"]["max_words"], cfg["captions"]["max_chars"])

    _log("layers")
    layer_w = max(1024, width * 2) if not preview else max(720, width * 2)
    layer_paths, previz = assets.resolve(ep, min(1440, layer_w), _log)

    if workers is None:
        w = out.get("workers", "auto")
        workers = min(4, os.cpu_count() or 1) if w == "auto" else int(w)
    return {
        "episode": str(ep.path), "cfg": cfg, "width": width, "height": height, "fps": fps,
        "total": total, "story_end": story_end, "keys": keys, "captions": caps, "cues": cues,
        "layers": [str(p) for p in layer_paths], "previz": previz,
        "blur_max": ep.cfg["look"]["motion_blur_max_samples"] if not preview else 4,
        "workers": workers, "provider": provider,
    }


def build(path: str, provider: str | None = None, preview: bool = False, workers: int | None = None,
          out_name: str | None = None) -> Path:
    t0 = time.time()
    ep = load_episode(path)
    for w in validate(ep):
        _log(f"warning: {w}")
    provider = provider or ep.cfg["tts"]["default_provider"]
    job = prepare(ep, provider, preview, workers)
    bdir = ROOT / "build" / ep.id
    bdir.mkdir(parents=True, exist_ok=True)

    cam = Camera(job["keys"])
    snaps = []
    for k0, k1 in zip(job["keys"], job["keys"][1:]):
        if k1.mode == "snap":
            snaps.append((k0.t, k1.t))
    _log(f"score + mix  ({job['total']:.1f}s)")
    music = ep.raw.get("music")
    pm = bdir / "postmortem"
    (pm / "lines").mkdir(parents=True, exist_ok=True)
    mix = sound.build_mix(job["cues"], cam, job["total"], len(ep.layers), ep.cfg, bdir / "mix.wav",
                          snaps, Path(ROOT / music) if music else None, voice_stem=pm / "narration_stem.wav")
    save_postmortem(ep, job, pm)

    (bdir / "captions.srt").write_text(to_srt(job["captions"]), encoding="utf-8")
    (bdir / "timeline.json").write_text(json.dumps({
        "total": job["total"],
        "beats": [c.__dict__ for c in job["cues"]],
        "camera": [k.__dict__ for k in job["keys"]],
        "previz_layers": job["previz"],
    }, indent=2), encoding="utf-8")
    write_upload_meta(ep, job, bdir / "upload.json")
    from .brand import thumbnails
    for p in thumbnails(ep, [Path(x) for x in job["layers"]], bdir):
        _log(f"thumbnail  {p}")

    tag = ("_preview" if preview else "") + ("_previz" if job["previz"] else "") + \
          ("" if provider != "silent" else "_silent")
    out = bdir / (out_name or f"{ep.id}_{ep.slug}{tag}.mp4")
    _log(f"render  {job['width']}x{job['height']} @ {job['fps']}fps, {job['workers']} workers")
    last = [0.0]

    def progress(i, n):
        now = time.time()
        if now - last[0] > 2 or i == n:
            last[0] = now
            el = now - t0
            sys.stdout.write(f"\r  frame {i}/{n}  ({i / n:5.1%})  {el:5.0f}s elapsed")
            sys.stdout.flush()
    encode(job, mix, out, preview, progress)
    _log("")
    if job["previz"]:
        _log(f"note: {len(job['previz'])} layer(s) still on previz art: {', '.join(job['previz'])}")
    _log(f"done  {out}  ({time.time() - t0:.0f}s)")
    return out


def save_postmortem(ep: Episode, job: dict, pm: Path) -> None:
    """Keep the exact narration that shipped: every line, the placed stem, and the script with timings."""
    import shutil
    rows = ["| # | beat | narrator | starts | length | line |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(job["cues"], 1):
        if c.audio:
            shutil.copy2(c.audio, pm / "lines" / f"{i:02d}_{c.beat_id}.wav")
        rows.append(f"| {i} | {c.beat_id} | {c.narrator} | {c.speech_start:.2f}s | "
                    f"{c.speech_end - c.speech_start:.2f}s | {c.text} |")
    leads = sorted({c.narrator for c in job["cues"]})
    voices = {n: {k: v for k, v in ep.cast["narrators"][n].items() if k in ("name", "elevenlabs", "say", "pace_wpm", "pre", "gap")}
              for n in leads}
    md = [f"# {ep.id} narration postmortem", "",
          f"- provider: `{job['provider']}`",
          f"- story length: {job['story_end']:.2f}s (+ end card, total {job['total']:.2f}s)",
          f"- voices: {', '.join(v['name'] for v in voices.values())}", "",
          "## Lines", "", *rows, "",
          "## Voice settings used", "", "```json", json.dumps(voices, indent=2), "```", "",
          "## Notes after publishing", "", "- retention dip at:", "- line to rewrite:", "- pacing:", ""]
    (pm / "script.md").write_text("\n".join(md), encoding="utf-8")


def write_upload_meta(ep: Episode, job: dict, path: Path) -> None:
    series = ep.cfg["series"]
    yt = ep.raw.get("youtube", {})
    narrators = sorted({c.narrator for c in job["cues"]})
    title = yt.get("title", ep.title)[:100]
    desc = yt.get("description") or " ".join(c.text for c in job["cues"])
    tags_line = " ".join(yt.get("hashtags", ["#shorts"]))
    desc = f"{desc.strip()}\n\n{tags_line}"[:5000]
    tags, used = [], 0
    for t in dict.fromkeys(series.get("default_tags", []) + yt.get("tags", []) + ep.raw.get("tags", [])):
        if used + len(t) + 1 > 480:          # YouTube caps tags at 500 characters in total
            break
        tags.append(t)
        used += len(t) + 1
    meta = {
        "title": title, "description": desc, "tags": tags,
        "category_id": ep.cfg.get("youtube", {}).get("category_id", "28"),
        "episode": ep.id,
        "narrators": [ep.cast["narrators"][n]["name"] for n in narrators],
        "duration_s": round(job["total"], 2),
        "direction": ep.direction,
        "previz_layers": job["previz"],
        "provider": job["provider"],
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def stills(path: str, times: list[float] | None, provider: str = "silent", sheet: bool = True) -> Path:
    ep = load_episode(path)
    job = prepare(ep, provider, preview=True, workers=1)
    if not times:
        n = 12
        times = [job["total"] * (i + 0.5) / n for i in range(n)]
    bdir = ROOT / "build" / ep.id / "stills"
    bdir.mkdir(parents=True, exist_ok=True)
    imgs = []
    for t in times:
        im = still(job, t)
        im.save(bdir / f"t{t:06.2f}.png")
        imgs.append(im)
    if not sheet:
        return bdir
    from PIL import Image
    cols = 6
    rows = (len(imgs) + cols - 1) // cols
    tw, th = imgs[0].width // 2, imgs[0].height // 2
    s = Image.new("RGB", (cols * tw, rows * th))
    for k, im in enumerate(imgs):
        s.paste(im.resize((tw, th)), ((k % cols) * tw, (k // cols) * th))
    out = bdir / "contact_sheet.png"
    s.save(out)
    return out
