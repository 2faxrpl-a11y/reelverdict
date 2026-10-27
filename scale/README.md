# SCALE — production pipeline

Faceless, code-built cinematic Shorts. Each episode is **one continuous zoom**, from the impossibly small to the impossibly huge or the other way round. One YAML file per episode, one command to an upload-ready 1080×1920 MP4.

```
episode.yaml ──► narration (ElevenLabs / macOS say / silent, cached per line)
             ──► timeline (each line's length decides where the camera lands)
             ──► score + mix (procedural drone, motion-reactive air, snap whoosh/impact, ducking, -14 LUFS)
             ──► zoom compositor (layer stack, mip-mapped, feathered, seam-matched, motion blur)
             ──► grade + HUD scale readout + captions ──► ffmpeg ──► mp4 + srt + upload.json
```

## Launching the channel and publishing

```bash
python -m scale_engine.brand                     # logo (svg/png, dark/light/transparent), icon, banner
python -m scale_engine.youtube checklist         # the browser-only steps: create channel, handle, avatar
python -m scale_engine.youtube auth              # reuses INVISIBLE 589's OAuth client; pick SCALE on consent
python -m scale_engine build ep001 --tts elevenlabs --publish   # render, then post automatically
python -m scale_engine.youtube launch ep001      # or: branding + upload + thumbnail + playlist in one go
```

The full step list is in `brand/CHANNEL_SETUP.md`. Publishing refuses previews and silent renders. It also refuses renders that still use previz art, unless you pass `--allow-previz`.
Every build credits **Ajay Haynes** three ways: a watermark, an intro lockup over the hook, and an end card. It also archives the exact narration that shipped in `build/<ep>/postmortem/`: each line's WAV, the voice-only stem, and `script.md` with timings and voice settings.

## Setup (Mac)

```bash
cd scale
./setup_mac.sh            # brew ffmpeg + python 3.13 venv + deps + tests
source .venv/bin/activate
```

## Daily commands

| Command | What it does |
|---|---|
| `python -m scale_engine build ep001 --preview --tts say` | Fast 540×960 animatic with a free macOS scratch voice |
| `python -m scale_engine build ep001 --tts elevenlabs` | Full 1080×1920 / 30 fps master |
| `python -m scale_engine stills ep001 --at 5 20 40` | Single frames and a contact sheet, in seconds |
| `python -m scale_engine prompts ep001` | Brief for making the real stills (`build/ep001/prompts.md`) |
| `python -m scale_engine outpaint-prep ep001 face` | Canvas and mask to outpaint `face` from `eye` |
| `python -m scale_engine lint` | Checks every episode and the cast rotation |
| `python -m scale_engine cast --tone cosmic` | Narrator history and who should lead next |
| `python -m scale_engine new deep-dive --direction in` | Scaffolds the next episode with a suggested narrator |
| `python -m scale_engine batch` | Builds every episode |

Outputs go to `build/<episode>/`: the MP4, `mix.wav`, `captions.srt`, `timeline.json` and `upload.json` (title, description, tags).
File names carry `_previz` while any layer is still procedural art, and `_silent` when there is no voice track, so a draft can't be posted by mistake.

## How an episode works

- **Layers** (innermost → outermost) are 9:16 stills, each with a real-world width in `meters`. `inner_at` says where the previous layer sits inside this one, so the camera drifts naturally (eye → face center, Earth → Sun).
- **Beats** are narration lines. Each beat names the `layer` the camera lands on while it is spoken. Timing comes from the real audio, so swapping narrators or rewriting a line re-times the whole zoom.
- `move: snap` makes a whip move (exponential ease, up to 8-sample motion blur, whoosh plus sub impact). Episode one uses it for the snap back to the eye.
- **Zoom in vs out** is only beat order. List beats from the biggest layer to the smallest for a dive-in episode.

## The art: code-built and cinematic by default

`config/series.yaml → art.mode: code`. Every layer is painted from math in `scale_engine/art.py`:
- **Human scale** is lit from height fields with real surface normals. The eye has strand-by-strand lashes and brow, iris fibers and crypts, and a softbox catchlight. The face and room add soft shadows, lamp and moonlight, and wood-grain floors.
- **City layers** have street grids, building tops, streetlight pools and car lights. Wider views show urban sky glow and highways.
- **Space layers:** Earth with atmosphere, clouds and night-side city lights; the Sun and planets; a Milky Way band with dust lanes; a spiral galaxy with a bar and dust lanes; the cosmic web; and the observable universe with its cosmic-background edge.
- **Each layer receives the image inside it.** For example, the face is painted with the eye layer's own eye, so neighboring scales share content. The compositor then matches lighting across seams.
- **Every frame gets a film finish:** bloom, halation, chromatic aberration, a teal/amber split tone, vignette and grain. Scale dust drifts with the zoom and streaks on fast moves.

Real photos or renders still slot in at any time: drop `assets/<ep>/<layer>.png` and that layer switches over. Set `art.mode: previz` if you want generated layers flagged as placeholders.

## Swapping in real stills (optional)

For real stills, work **innermost first** and outpaint outward. Then every seam lines up by construction:

1. Generate `assets/ep001/eye.png` (9:16, ≥1440×2560) from the prompt in `prompts.md`.
2. `outpaint-prep ep001 face` writes `build/ep001/outpaint/face_canvas.png` + `face_mask.png`. The eye is already shrunk to true scale at the right spot. Outpaint it in any tool (Higgsfield, Photoshop Generative Expand, ComfyUI) and save it as `assets/ep001/face.png`.
3. Repeat outward. Reusable layers (Earth, Milky Way, cosmic web) go in `assets/library/` and are referenced with `asset: library/earth_night.png`, so later episodes reuse them for free.

The engine prefers a real file over previz layer by layer, so you can swap images in one at a time.

## Cast

`config/cast.yaml` holds the four standing narrators: **awe** (anchor), **guide** (workhorse), **pulse** (momentum) and **whisper** (rare). Each has ElevenLabs settings, a macOS `say` fallback, and its own pacing (`pre`/`gap`). One narrator leads an episode. Setting `narrator:` on a beat hands off to a second voice for special episodes. `lint` warns when a voice leads more than 2 episodes in a row, or when The Whisper goes above 10% of episodes.

## Tuning

`config/series.yaml` holds resolution, codec (VideoToolbox on Mac), grade (vignette, grain), HUD, captions, audio targets and fonts. Any episode can override those sections.

## Episode 001 — The Eye

18 layers from 5 cm to 9×10²⁶ m: eye → face → room → rooftop → city → Earth → solar system → Milky Way → cosmic web → observable universe, then a snap back to the eye. Narrated by The Awe Voice, about 65–70 s with timing estimates. Measure again once the real voice track exists: if the cut needs to be under 60 s, lower `gap` for awe or trim a line.
