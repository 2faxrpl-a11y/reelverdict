# SCALE: channel launch sheet

Everything the API can do runs through `python -m scale_engine.youtube launch ep001`.
The steps below are the ones YouTube only allows in a browser. Do them once, in this order.

## 1. Create the channel (Brand Account, ~1 min)
1. Sign in to YouTube with the same Google account as INVISIBLE 589.
2. Go to https://www.youtube.com/channel_switcher, click **Create a channel**, and name it **SCALE**.
   This creates a separate Brand Account channel with its own subscribers and analytics, under the same login.

## 2. Handle
Run `python -m scale_engine.youtube handles`. It checks the candidates in `config/youtube.yaml` in order:
`@scale`, `@scalezoom`, `@scale.zoom`, `@thescale`, `@zoomscale`, `@scaleuniverse`…
Then, while switched to SCALE, go to https://www.youtube.com/handle and claim the first free one.
(`@scale` is almost certainly taken; `@scalezoom` is the preferred fallback.)

## 3. Profile picture and banner
- **Picture:** Studio → Customization → Branding → Picture → `brand/out/scale_icon_800.png`. This is web-only; the API can't set it.
- **Banner:** applied automatically by `launch`. To set it by hand, use `brand/out/scale_banner_2560x1440.jpg`. All text sits inside the 1546×423 safe area.
- **Video watermark** (optional): Studio → Branding → Video watermark → `brand/out/scale_logo_transparent_white.png`.

## 4. Verify the channel
Go to https://www.youtube.com/verify and verify by phone. Without this, YouTube rejects custom thumbnails and uploads longer than 15 minutes.

## 5. Upload defaults (Studio → Settings → Upload defaults)
| Field | Value |
|---|---|
| Title | `SCALE · ` |
| Description | `One continuous zoom, from the impossibly small to the impossibly huge.` ⏎ `Created by Ajay Haynes.` ⏎ `#shorts #space #universe` |
| Tags | SCALE, zoom, universe, space, powers of ten, scale of the universe, cinematic shorts |
| Visibility | Public |
| Category | Science & Technology |
| Video language | English |
| Caption certification | None (original content, no TV broadcast) |
| License | Standard YouTube License |
| Allow embedding | On |
| Comments | On, "Hold potentially inappropriate comments for review" |
| Show how many viewers like this video | On |
| Altered or synthetic content | No by default. Say Yes only for episodes that show realistic AI-generated people or events. A stylised zoom with an AI narrator doesn't need it, and `synthetic_media` in `config/series.yaml` controls it per upload. |

Also under **Settings → Channel → Basic info**: Country = United States. Under **Advanced settings**: audience = "No, it's not made for kids".

## 6. Launch
```bash
python -m scale_engine.youtube auth          # consent screen: pick SCALE, approve both scopes
python -m scale_engine build ep001 --tts elevenlabs
python -m scale_engine.youtube launch ep001  # banner + about + keywords + upload + thumbnail + playlist
```
`launch` prints the channel ID, the @handle and the video URL, and saves them to `build/ep001/published.json`.

## About text (already in config/youtube.yaml)
> One continuous zoom. From the impossibly small to the impossibly huge, and back again.
>
> Every SCALE episode is a single unbroken journey across the sizes of reality: from one human eye out past cities, planets and galaxies to the edge of everything we can ever see, or from your fingertip down through cells, molecules and atoms into the quantum foam. No cuts, no faces, just the zoom and a voice to guide you.
>
> Some episodes pull back until Earth is a pixel. Some dive in until a grain of sand becomes a universe. All of them end in the same place: somewhere you never realized you were standing.
>
> New zooms regularly. Subscribe and see how far it goes.
>
> Created by Ajay Haynes.
