"""SCALE command line.  Run from the scale/ folder:  python -m scale_engine <command> ...

  build    EP [--preview] [--tts elevenlabs|say|silent] [--publish]   render (and post) an episode
  batch    [--preview] [--tts ...]                        build every episode in episodes/
  stills   EP [--at 1.5 10 30]                            frames + contact sheet for quick checks
  previz   EP                                             generate placeholder layer art only
  prompts  EP                                             write the image-generation brief
  outpaint-prep EP LAYER                                  canvas + mask to outpaint LAYER from the one inside it
  lint     [EP ...]                                       validate episodes and cast rotation
  cast     [--tone cosmic]                                rotation history + who should narrate next
  new      SLUG --direction in|out [--narrator awe]       scaffold the next episode file
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import assets, cast
from .spec import ROOT, load_episode, validate


def _ep(arg: str) -> str:
    p = Path(arg)
    if p.exists():
        return str(p)
    hits = sorted((ROOT / "episodes").glob(f"{arg}*.yaml"))
    if not hits:
        sys.exit(f"no episode matches '{arg}'")
    return str(hits[0])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scale", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("episode")
    b.add_argument("--preview", action="store_true", help="half-res, 24fps, fast encode")
    b.add_argument("--tts", choices=["kokoro", "elevenlabs", "say", "silent"])
    b.add_argument("--workers", type=int)
    b.add_argument("--publish", action="store_true", help="upload to the SCALE channel when the render finishes")
    b.add_argument("--allow-previz", action="store_true", help="with --publish: allow layers still on previz art")

    ba = sub.add_parser("batch")
    ba.add_argument("--preview", action="store_true")
    ba.add_argument("--tts", choices=["kokoro", "elevenlabs", "say", "silent"])

    s = sub.add_parser("stills")
    s.add_argument("episode")
    s.add_argument("--at", type=float, nargs="*")
    s.add_argument("--tts", default="silent", choices=["kokoro", "elevenlabs", "say", "silent"])

    pv = sub.add_parser("previz")
    pv.add_argument("episode")
    pv.add_argument("--width", type=int, default=1440)

    pr = sub.add_parser("prompts")
    pr.add_argument("episode")

    op = sub.add_parser("outpaint-prep")
    op.add_argument("episode")
    op.add_argument("layer")

    li = sub.add_parser("lint")
    li.add_argument("episodes", nargs="*")

    c = sub.add_parser("cast")
    c.add_argument("--tone")

    n = sub.add_parser("new")
    n.add_argument("slug")
    n.add_argument("--direction", choices=["in", "out"], required=True)
    n.add_argument("--narrator")

    a = ap.parse_args(argv)

    if a.cmd == "build":
        from .pipeline import build
        out = build(_ep(a.episode), a.tts, a.preview, a.workers)
        if a.publish:
            from . import youtube
            eid = out.parent.name
            try:
                res = youtube.publish(youtube.client(), eid, allow_previz=a.allow_previz)
            except youtube.SetupNeeded as e:
                sys.exit(f"\nnot published: {e}\n{youtube.CHECKLIST}")
            print(json.dumps(res, indent=2))
    elif a.cmd == "batch":
        from .pipeline import build
        for p in sorted((ROOT / "episodes").glob("*.yaml")):
            print(f"\n=== {p.name}")
            build(str(p), a.tts, a.preview)
    elif a.cmd == "stills":
        from .pipeline import stills
        print(stills(_ep(a.episode), a.at, a.tts))
    elif a.cmd == "previz":
        ep = load_episode(_ep(a.episode))
        paths, previz = assets.resolve(ep, a.width)
        for p in paths:
            print(p)
    elif a.cmd == "prompts":
        ep = load_episode(_ep(a.episode))
        out = ROOT / "build" / ep.id / "prompts.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(assets.prompt_sheet(ep), encoding="utf-8")
        print(out)
    elif a.cmd == "outpaint-prep":
        ep = load_episode(_ep(a.episode))
        for p in assets.outpaint_prep(ep, a.layer):
            print(p)
    elif a.cmd == "lint":
        files = [_ep(e) for e in a.episodes] or [str(p) for p in sorted((ROOT / "episodes").glob("*.yaml"))]
        bad = 0
        for f in files:
            try:
                ep = load_episode(f)
                ws = validate(ep)
                print(f"ok    {ep.id}  {len(ep.layers)} layers, {len(ep.beats)} beats, lead: {ep.narrator}")
                for w in ws:
                    print(f"      warning: {w}")
            except Exception as e:  # noqa: BLE001 - report every broken file, keep going
                bad += 1
                print(f"FAIL  {f}: {e}")
        for w in cast.check():
            print(f"cast  warning: {w}")
        sys.exit(1 if bad else 0)
    elif a.cmd == "cast":
        rows = cast.history()
        for eid, lead, extra in rows:
            print(f"{eid:<8} {lead:<8} {'+ ' + ', '.join(extra) if extra else ''}")
        for w in cast.check(rows):
            print(f"warning: {w}")
        print("\nnext episode, best fit first:")
        for nid, name in cast.suggest(a.tone, rows):
            print(f"  {nid:<8} {name}")
    elif a.cmd == "new":
        existing = sorted((ROOT / "episodes").glob("ep*.yaml"))
        nums = [int(m.group(1)) for p in existing if (m := re.match(r"ep(\d+)", p.stem))]
        num = (max(nums) + 1) if nums else 1
        eid = f"ep{num:03d}"
        narrator = a.narrator or cast.suggest()[0][0]
        slug = re.sub(r"[^a-z0-9]+", "-", a.slug.lower()).strip("-")
        out = ROOT / "episodes" / f"{eid}_{slug.replace('-', '_')}.yaml"
        tmpl = (ROOT / "episodes" / "_template.yaml.txt").read_text(encoding="utf-8")
        out.write_text(tmpl.format(id=eid, slug=slug, direction=a.direction, narrator=narrator), encoding="utf-8")
        print(f"{out}  (narrator: {narrator})")


if __name__ == "__main__":
    main()
