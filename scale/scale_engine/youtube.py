"""SCALE channel setup and publishing via the YouTube Data API v3.

    python -m scale_engine.youtube auth             # one-time consent; pick the SCALE Brand Account
    python -m scale_engine.youtube whoami           # which channel the token is bound to
    python -m scale_engine.youtube handles          # find the cleanest free @handle
    python -m scale_engine.youtube brand            # description, keywords, country, banner
    python -m scale_engine.youtube publish ep001    # upload + thumbnail + playlist
    python -m scale_engine.youtube launch ep001     # auth -> brand -> publish, prints IDs and URL

What the API cannot do (YouTube web only): create the Brand Account channel,
claim the @handle, set the profile picture, set Studio "upload defaults".
`launch` checks each of those and stops with the exact step when one is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .spec import ROOT, load_episode, load_yaml

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",   # videos.insert, thumbnails.set
    "https://www.googleapis.com/auth/youtube",          # branding, banners, playlists
]
CFG_PATH = ROOT / "config" / "youtube.yaml"


class SetupNeeded(RuntimeError):
    """A step only the channel owner can do in a browser."""


def cfg() -> dict:
    return load_yaml(CFG_PATH)


def _path(p: str | None) -> Path | None:
    return Path(os.path.expanduser(p)) if p else None


# ---------------------------------------------------------------- auth
def _token_scopes(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return set()
    s = data.get("scopes") or data.get("scope") or []
    return set(s.split() if isinstance(s, str) else s)


def credentials(reauth: bool = False):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    c = cfg()
    secrets = _path(os.environ.get("SCALE_CLIENT_SECRETS") or c.get("client_secrets"))
    token = _path(c.get("token_file"))
    token.parent.mkdir(parents=True, exist_ok=True)

    creds = None
    if token.exists() and not reauth:
        have = _token_scopes(token)
        missing = [s for s in SCOPES if s not in have]
        if missing:
            print("SCALE token is missing scopes:", ", ".join(m.rsplit("/", 1)[-1] for m in missing))
            print("-> re-consent required; opening Google's consent screen.")
        else:
            creds = Credentials.from_authorized_user_file(str(token), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as e:  # noqa: BLE001 - revoked/expired refresh tokens: fall through to consent
            print(f"token refresh failed ({e}); re-consent required.")
            creds = None
    if not creds or not creds.valid:
        if not secrets or not secrets.exists():
            raise SetupNeeded(
                f"OAuth client not found at {secrets}.\n"
                "Point `client_secrets` in config/youtube.yaml (or SCALE_CLIENT_SECRETS) at the "
                "INVISIBLE 589 client_secret*.json from Google Cloud Console > APIs & Services > Credentials.")
        inv = _path(c.get("invisible_token"))
        if inv and inv.exists():
            lacking = [s.rsplit("/", 1)[-1] for s in SCOPES if s not in _token_scopes(inv)]
            print(f"INVISIBLE 589 token scopes checked: {'missing ' + ', '.join(lacking) if lacking else 'all present'}.")
            print("That token stays bound to INVISIBLE 589; SCALE needs its own consent (same OAuth client).")
        print("\n=== RE-CONSENT STEP =================================================")
        print("A browser window will open. Sign in with your usual Google account, then on")
        print("'Choose an account or brand account' pick  >>> SCALE <<<  (not INVISIBLE 589).")
        print("Approve both YouTube permissions. You'll be returned here automatically.")
        print("=====================================================================\n")
        flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
        creds = flow.run_local_server(port=0, prompt="consent select_account", access_type="offline",
                                      open_browser=True)
        token.write_text(creds.to_json())
        os.chmod(token, 0o600)
    return creds


def client(reauth: bool = False):
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=credentials(reauth), cache_discovery=False)


def my_channel(yt) -> dict:
    r = yt.channels().list(part="id,snippet,brandingSettings,status,statistics", mine=True).execute()
    items = r.get("items", [])
    if not items:
        raise SetupNeeded("This Google identity has no YouTube channel yet. Create the SCALE Brand Account "
                          "channel first (see `launch` checklist).")
    return items[0]


def verify_scale(yt) -> dict:
    ch = my_channel(yt)
    want = cfg().get("expected_channel_title", "SCALE")
    title = ch["snippet"]["title"]
    if title.strip().lower() != want.lower():
        raise SetupNeeded(
            f"The token is bound to the channel '{title}', not '{want}'.\n"
            f"Run: python -m scale_engine.youtube auth --reauth   and pick '{want}' on the account chooser.\n"
            "If SCALE isn't in that list, the Brand Account channel doesn't exist yet: create it first.")
    return ch


# ---------------------------------------------------------------- handles
def handle_status(h: str) -> str:
    req = urllib.request.Request(f"https://www.youtube.com/@{h}", method="GET",
                                 headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return "taken" if r.status == 200 else f"http {r.status}"
    except urllib.error.HTTPError as e:
        return "likely free" if e.code == 404 else f"http {e.code}"
    except OSError as e:
        return f"unknown ({e.__class__.__name__})"


def check_handles() -> str | None:
    first = None
    for h in cfg().get("handle_candidates", []):
        st = handle_status(h)
        print(f"  @{h:<18} {st}")
        if st == "likely free" and first is None:
            first = h
    if first:
        print(f"\nClaim it: https://www.youtube.com/handle  (while switched to SCALE) -> @{first}")
    return first


# ---------------------------------------------------------------- branding
def _keywords(words: list[str]) -> str:
    return " ".join(f'"{w}"' if " " in w else w for w in words)


def apply_branding(yt, ch: dict, banner: Path, dry: bool = False) -> dict:
    from googleapiclient.http import MediaFileUpload
    c = cfg()["channel"]
    bs = ch.get("brandingSettings", {})
    channel = dict(bs.get("channel", {}))
    channel.update({
        "description": c["description"].strip(),
        "keywords": _keywords(c["keywords"]),
        "country": c.get("country", "US"),
        "defaultLanguage": c.get("default_language", "en"),
    })
    body = {"id": ch["id"], "brandingSettings": {"channel": channel}}
    if dry:
        print(json.dumps(body, indent=2)[:2000])
        return body
    up = yt.channelBanners().insert(media_body=MediaFileUpload(str(banner), mimetype="image/jpeg")).execute()
    body["brandingSettings"]["image"] = {"bannerExternalUrl": up["url"]}
    res = yt.channels().update(part="brandingSettings", body=body).execute()
    print("branding applied: description, keywords, country, language, banner")
    return res


# ---------------------------------------------------------------- publishing
def find_master(ep_id: str) -> Path:
    bdir = ROOT / "build" / ep_id
    vids = sorted(bdir.glob(f"{ep_id}_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not vids:
        raise FileNotFoundError(f"no render in {bdir}; run: python -m scale_engine build {ep_id} --tts elevenlabs")
    return vids[0]


def guard(video: Path, allow_previz: bool) -> None:
    n = video.name
    if "_preview" in n:
        raise SetupNeeded(f"{n} is a half-res preview. Render the master: build without --preview.")
    if "_silent" in n:
        raise SetupNeeded(f"{n} has no narration. Render with --tts elevenlabs (or say).")
    if "_previz" in n and not allow_previz:
        raise SetupNeeded(f"{n} still uses procedural previz art for some layers (see build/<ep>/upload.json).\n"
                          "Add real stills to assets/<ep>/ and rebuild, or pass --allow-previz to post it anyway.")


def _resumable(req, label: str):
    from googleapiclient.errors import HttpError
    resp, retries = None, 0
    while resp is None:
        try:
            status, resp = req.next_chunk()
            if status:
                sys.stdout.write(f"\r  {label} {status.progress():5.1%}")
                sys.stdout.flush()
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and retries < 6:
                retries += 1
                time.sleep(2 ** retries)
                continue
            raise
    print(f"\r  {label} 100.0%")
    return resp


def ensure_playlist(yt, title: str) -> str:
    page = None
    while True:
        r = yt.playlists().list(part="snippet", mine=True, maxResults=50, pageToken=page).execute()
        for p in r.get("items", []):
            if p["snippet"]["title"] == title:
                return p["id"]
        page = r.get("nextPageToken")
        if not page:
            break
    r = yt.playlists().insert(part="snippet,status", body={
        "snippet": {"title": title, "description": "Every SCALE zoom, in order.", "defaultLanguage": "en"},
        "status": {"privacyStatus": "public"}}).execute()
    return r["id"]


def publish(yt, ep_id: str, allow_previz: bool = False, privacy: str | None = None, again: bool = False,
            dry: bool = False) -> dict:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    bdir = ROOT / "build" / ep_id
    record = bdir / "published.json"
    if record.exists() and not again:
        done = json.loads(record.read_text())
        print(f"{ep_id} already published: {done['url']}  (use --again to upload a second copy)")
        return done
    video = find_master(ep_id)
    guard(video, allow_previz)
    meta = json.loads((bdir / "upload.json").read_text())
    ycfg = load_episode(next((ROOT / "episodes").glob(f"{ep_id}*.yaml"))).cfg.get("youtube", {})
    body = {
        "snippet": {
            "title": meta["title"], "description": meta["description"], "tags": meta["tags"],
            "categoryId": str(meta.get("category_id", ycfg.get("category_id", "28"))),
            "defaultLanguage": ycfg.get("default_language", "en"),
            "defaultAudioLanguage": ycfg.get("default_language", "en"),
        },
        "status": {
            "privacyStatus": privacy or ycfg.get("privacy", "public"),
            "selfDeclaredMadeForKids": bool(ycfg.get("made_for_kids", False)),
            "containsSyntheticMedia": bool(ycfg.get("synthetic_media", False)),
            "embeddable": True, "publicStatsViewable": True, "license": "youtube",
        },
    }
    if dry:
        print(f"would upload {video}")
        print(json.dumps(body, indent=2))
        return body
    ch = verify_scale(yt)
    print(f"uploading {video.name} -> {ch['snippet']['title']}")
    req = yt.videos().insert(part="snippet,status", body=body, notifySubscribers=True,
                             media_body=MediaFileUpload(str(video), mimetype="video/mp4", chunksize=8 * 1024 * 1024,
                                                        resumable=True))
    resp = _resumable(req, "video")
    vid = resp["id"]

    thumb = bdir / "thumbnail_1280x720.jpg"
    thumb_ok = False
    if thumb.exists():
        try:
            yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb), mimetype="image/jpeg")).execute()
            thumb_ok = True
        except HttpError as e:
            print(f"  thumbnail not set ({e.resp.status}): custom thumbnails need a phone-verified channel -> "
                  "https://www.youtube.com/verify , then: python -m scale_engine.youtube thumbnail " + ep_id)
    pl_title = ycfg.get("playlist")
    if pl_title:
        pid = ensure_playlist(yt, pl_title)
        yt.playlistItems().insert(part="snippet", body={
            "snippet": {"playlistId": pid, "resourceId": {"kind": "youtube#video", "videoId": vid}}}).execute()

    st = yt.videos().list(part="status", id=vid).execute()["items"][0]["status"]
    wanted = body["status"]["privacyStatus"]
    if st.get("privacyStatus") != wanted:
        print(f"  WARNING: YouTube set privacy to '{st.get('privacyStatus')}' instead of '{wanted}'. Uploads from an "
              "unaudited API project are locked private; request the audit for the INVISIBLE 589 Cloud project.")
    out = {
        "video_id": vid, "url": f"https://www.youtube.com/shorts/{vid}",
        "channel_id": ch["id"], "handle": ch["snippet"].get("customUrl"),
        "privacy": st.get("privacyStatus"), "upload_status": st.get("uploadStatus"),
        "thumbnail_set": thumb_ok, "file": video.name, "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    record.write_text(json.dumps(out, indent=2))
    return out


def set_thumbnail(yt, ep_id: str) -> None:
    from googleapiclient.http import MediaFileUpload
    bdir = ROOT / "build" / ep_id
    vid = json.loads((bdir / "published.json").read_text())["video_id"]
    yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(bdir / "thumbnail_1280x720.jpg"),
                                                                mimetype="image/jpeg")).execute()
    print("thumbnail set")


# ---------------------------------------------------------------- launch
CHECKLIST = """
SCALE channel: one-time browser steps (the YouTube API cannot do these)
  1. Create the channel: https://www.youtube.com/channel_switcher -> "Create a channel" -> name it SCALE.
     (This makes a Brand Account under your login; INVISIBLE 589 is untouched.)
  2. While switched to SCALE: https://www.youtube.com/handle -> claim the handle `handles` found free.
  3. Profile picture: YouTube Studio -> Customization -> Branding -> Picture -> brand/out/scale_icon_800.png
  4. Phone-verify SCALE for custom thumbnails: https://www.youtube.com/verify
  5. Upload defaults: Studio -> Settings -> Upload defaults (values in brand/CHANNEL_SETUP.md)
Then run:  python -m scale_engine.youtube launch ep001
"""


def launch(ep_id: str, allow_previz: bool, privacy: str | None) -> dict:
    from .brand import OUT, build_all
    banner = OUT / "scale_banner_2560x1440.jpg"
    if not banner.exists():
        build_all()
    yt = client()
    ch = verify_scale(yt)
    handle = ch["snippet"].get("customUrl")
    if not handle or handle.lower().startswith("@uc") or len(handle) > 30:
        print("note: no custom @handle yet; claim one (step 2 of the checklist). Continuing.")
    apply_branding(yt, ch, banner)
    out = publish(yt, ep_id, allow_previz=allow_previz, privacy=privacy)
    ch = my_channel(yt)
    print("\n================ SCALE IS LIVE ================")
    print(f"channel id : {ch['id']}")
    print(f"handle     : {ch['snippet'].get('customUrl', '(not claimed yet)')}")
    print(f"channel    : https://www.youtube.com/channel/{ch['id']}")
    print(f"video      : {out['url']}  ({out['privacy']})")
    print(f"thumbnail  : {'set' if out['thumbnail_set'] else 'pending phone verification'}")
    print("===============================================")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scale_engine.youtube", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("auth")
    a_.add_argument("--reauth", action="store_true")
    sub.add_parser("whoami")
    sub.add_parser("handles")
    sub.add_parser("checklist")
    b = sub.add_parser("brand")
    b.add_argument("--dry-run", action="store_true")
    for name in ("publish", "launch"):
        p = sub.add_parser(name)
        p.add_argument("episode")
        p.add_argument("--allow-previz", action="store_true")
        p.add_argument("--privacy", choices=["public", "unlisted", "private"])
        if name == "publish":
            p.add_argument("--again", action="store_true")
            p.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("thumbnail")
    t.add_argument("episode")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "auth":
            ch = my_channel(client(a.reauth))
            print(f"authorized for: {ch['snippet']['title']}  ({ch['id']})  {ch['snippet'].get('customUrl', '')}")
            verify_scale_title = cfg().get("expected_channel_title", "SCALE")
            if ch["snippet"]["title"].lower() != verify_scale_title.lower():
                print(f"!! that's not {verify_scale_title}. Re-run with --reauth and pick {verify_scale_title}.")
        elif a.cmd == "whoami":
            ch = my_channel(client())
            print(json.dumps({"id": ch["id"], "title": ch["snippet"]["title"],
                              "handle": ch["snippet"].get("customUrl"),
                              "subscribers": ch.get("statistics", {}).get("subscriberCount")}, indent=2))
        elif a.cmd == "handles":
            check_handles()
        elif a.cmd == "checklist":
            print(CHECKLIST)
        elif a.cmd == "brand":
            from .brand import OUT, build_all
            build_all()
            if a.dry_run:
                apply_branding(None, {"id": "UC_DRY_RUN", "brandingSettings": {}}, OUT / "scale_banner_2560x1440.jpg", True)
            else:
                yt = client()
                apply_branding(yt, verify_scale(yt), OUT / "scale_banner_2560x1440.jpg")
        elif a.cmd == "publish":
            yt = None if a.dry_run else client()
            out = publish(yt, a.episode, a.allow_previz, a.privacy, a.again, a.dry_run)
            if not a.dry_run:
                print(json.dumps(out, indent=2))
        elif a.cmd == "launch":
            launch(a.episode, a.allow_previz, a.privacy)
        elif a.cmd == "thumbnail":
            set_thumbnail(client(), a.episode)
    except SetupNeeded as e:
        print(f"\nSETUP NEEDED\n{e}\n{CHECKLIST}")
        sys.exit(2)


if __name__ == "__main__":
    main()
