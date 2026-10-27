# SCALE: handoff brief for the next agent

**Owner:** Ajay Haynes · **Repo:** `2faxrpl-a11y/reelverdict`, branch `claude/scale-pipeline`, folder `scale/` (draft PR #1)
**Status (Oct 2026):** the pipeline is complete and episode 001 is rendered and narrated. **The YouTube channel does not exist yet. Nothing has been posted.**

---

## 1. What SCALE is

A faceless, cinematic, vertical YouTube Shorts series. Every episode is **one continuous zoom** from the impossibly small to the impossibly huge, or the reverse. It is a separate brand from the owner's other channel, INVISIBLE 589, but is managed under the same Google login.

- **Narration:** a standing cast of 4 voices, with one voice per episode, rotating across episodes.
  - **The Awe Voice:** slow, warm, reverent. The anchor.
  - **The Guide:** clear and curious. The workhorse.
  - **The Pulse:** sharp and urgent. Momentum.
  - **The Whisper:** intimate. Rare, at most 10% of episodes.

  A two-voice handoff happens only in special episodes.
- **Credit:** "Ajay Haynes" on every video, in three places: a burned-in watermark, an intro lockup over the hook, and an end card.
- **Built to scale to hundreds of episodes:** one YAML file per episode, one command per render.

## 2. What exists and works

| Piece | Where | Notes |
|---|---|---|
| Render engine | `scale/scale_engine/` | Python 3.13, numpy, Pillow, ffmpeg. Runs on macOS and Linux. |
| Episode 001 "The Eye" | `scale/episodes/ep001_the_eye.yaml` | Zooms out from an eye to the observable universe, then snaps back. 18 layers, 12 beats, 58.8 s. |
| Code-built cinematic art | `scale_engine/art.py`, `face.py` | Every layer is painted from math, so no image credits are needed. |
| Narration | `scale_engine/tts.py` | **Kokoro**, a free offline neural voice and the default. The Awe Voice is `bm_george` at speed 0.86. ElevenLabs is an optional upgrade. |
| Score and mix | `scale_engine/sound.py` | Procedural drone, air layer and snap whoosh/impact. Ducks under the voice, normalized to -14 LUFS. |
| Film finish | `render.py` | Bloom, halation, chromatic aberration, split tone, grain, and dust/streaks that move with the zoom. |
| Credits | `config/series.yaml → credit` | Watermark, intro lockup and end card. |
| Brand kit | `scale/brand/out/` | Growing-letter logo (SVG/PNG, dark/light/transparent), 800 px avatar, 2560×1440 banner. |
| Thumbnail | generated per episode | 1280×720 plus a 9:16 cover: the universe inside the pupil. |
| YouTube automation | `scale_engine/youtube.py`, `config/youtube.yaml` | OAuth with re-consent, then branding, banner, upload, thumbnail and playlist. Verifies the token is bound to SCALE before writing anything. |
| Narration archive | `build/<ep>/postmortem/` | Each line's WAV, the voice-only stem, and `script.md` with timings. |
| Channel copy | `scale/brand/CHANNEL_SETUP.md` | About text, upload defaults, handle candidates. |
| Tests | `scale/tests/` | `python -m unittest discover tests` |

**Already rendered (not in git; build output):** `build/ep001/ep001_the-eye.mp4` (master), `ep001_review_1080p.mp4`, `thumbnail_1280x720.jpg`, `cover_1080x1920.jpg`. These are bundled in `SCALE_handoff_bundle.zip`, which the owner has. Running `python -m scale_engine build ep001` recreates everything.

## 3. Commands

```bash
cd scale && ./setup_mac.sh && source .venv/bin/activate
python -m scale_engine lint                         # validate episodes and cast rotation
python -m scale_engine stills ep001 --at 5 20 40    # quick frame check (seconds)
python -m scale_engine build ep001 --preview        # 540x960 draft, ~4 min
python -m scale_engine build ep001                  # 1080x1920 master, ~12-20 min
python -m scale_engine build ep001 --publish        # render, then upload to SCALE
python -m scale_engine new <slug> --direction in    # scaffold the next episode
python -m scale_engine cast --tone cosmic           # who should narrate next
python -m scale_engine.youtube handles | auth | brand | publish ep001 | launch ep001
python -m scale_engine.brand                        # regenerate logo, icon, banner
```

## 4. What is blocked, and what needs the owner

The YouTube Data API **cannot** create a channel, claim a handle, or set a profile picture. Those happen in a browser, signed in as the owner. Steps that need the owner's hands:

1. **Create the channel:** youtube.com/channel_switcher → *Create a channel* → name it **SCALE**. This makes a Brand Account under the same login.
2. **Handle:** `@scale` is almost certainly taken. Run `python -m scale_engine.youtube handles` to check the rest, then claim the first free one at youtube.com/handle. Preferred fallback: `@scalezoom`.
3. **Profile picture:** upload `brand/out/scale_icon_800.png` in Studio → Customization.
4. **Phone verification** at youtube.com/verify. The owner has to enter the SMS code. Without it, custom thumbnails fail.
5. **OAuth consent:** `python -m scale_engine.youtube auth` opens Google's consent screen. The owner picks **SCALE**, not INVISIBLE 589. It reuses INVISIBLE 589's OAuth client: set `client_secrets` in `config/youtube.yaml` to that `client_secret*.json`.
6. **API audit caveat:** if the Cloud project hasn't passed YouTube's audit, API uploads are forced to private. `youtube.py` detects this and warns. In that case, publish manually in Studio.

## 5. Quality bar and known weak spots in ep001

- The face is code-built and realistic but still recognizably CG. It's dark, moving, and on screen about 2 seconds.
- There are two intentionally near-black stretches (stars → galaxy, galaxy → cosmic web). Real space is empty; the dust streaks keep them moving.
- **Motion is a camera zoom over still layers.** Nothing inside a layer moves yet. This is the biggest upgrade opportunity (see §6).

## 6. Recommended roadmap: "fully cinematic motion"

Do these in order. Each one is a self-contained engine change, and every future episode benefits.

1. **Living layers (highest impact).** Make each layer animate over time, not just scale:
   - the eye blinks and its pupil dilates
   - city traffic flows and windows flicker
   - clouds drift and Earth rotates
   - the Sun's corona shimmers and stars twinkle
   - the galaxy turns slowly and the cosmic web glows

   Implementation: generators take `t` and render low-cost animated overlays (particles, offsets, masks) on top of a cached base, not a full repaint per frame.
2. **2.5D parallax.** Generators also output a depth map. The renderer offsets depth planes slightly during the zoom (foreground hair or brows, mid face, background floor), which gives real depth.
3. **Camera language.** Add a subtle roll during long pulls, gentle handheld drift, speed ramps on the beat, and rack focus: shallow depth of field on the subject that opens up in space.
4. **Volumetrics and light:** god rays through the skylight, atmospheric haze when entering Earth's atmosphere, lens flares only in screen space.
5. **Music:** a licensed or AI-generated score with risers that hit at the scale jumps. Keep the procedural bed as fallback.
6. **Optional hero shots when there's budget:** generate real stills (innermost first, then outpaint outward) or short image-to-video clips for the eye, face and city. Drop them in as `assets/<ep>/<layer>.png`; the engine uses them automatically.
7. **Optional heavy path:** Blender (bpy) scenes for the human-scale layers, for true 3D with ray tracing.

## 7. Episode slate (first 10)

Rotation: Awe → Guide → Pulse → Guide → Whisper (special) → Pulse → Awe → Guide → Pulse → two-voice special.

| # | Title | Direction | Narrator | Arc |
|---|---|---|---|---|
| 001 | The Eye ✅ rendered | out | Awe | eye → observable universe → snap back |
| 002 | Into Your Fingertip | in | Guide | hand → skin → cell → DNA → atom → quark |
| 003 | Falling Into a Black Hole | in | Pulse | Milky Way → Sagittarius A* → event horizon |
| 004 | One Raindrop | in | Guide | storm → drop → water molecule → proton |
| 005 | Your Heartbeat | in | Whisper | chest → heart → muscle cell → mitochondrion → ATP synthase |
| 006 | The Tallest Thing | out | Pulse | ant → you → Burj Khalifa → Everest → Olympus Mons → Sun → VY Canis Majoris |
| 007 | Carnival From Space | out | Awe | one sequin on a mas costume → masquerader → band on the road → Port of Spain → Trinidad → Caribbean → Earth |
| 008 | The Smallest Thing | in | Guide | atom → nucleus → proton → quark → Planck length |
| 009 | How Far Is a Second of Light | out | Pulse | camera flash → Moon (1.3 s) → Sun (8 min) → Voyager → Proxima |
| 010 | Back to the Eye | in | Awe + Whisper | universe → eye, the mirror of ep001 |

Cadence: 3 Shorts per week. Write each episode's facts carefully; it's a science channel and comments will check them.

## 8. Non-negotiables

- Faceless, vertical 1080×1920, under 60 s preferred, hook in the first line, emotional landing at the end.
- "Ajay Haynes" credit on every video (watermark, intro, end card).
- Never post previews, silent renders or placeholder art. The engine's guards enforce this; don't bypass them.
- Ask the owner before anything public or irreversible: channel creation, publishing, deleting.
- Archive each episode's narration in `build/<ep>/postmortem/`.
