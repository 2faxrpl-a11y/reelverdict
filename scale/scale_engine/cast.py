"""Narrator rotation across the series."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .spec import ROOT, load_series, load_yaml

EPISODES = ROOT / "episodes"


def history() -> list[tuple[str, str, list[str]]]:
    """(episode id, lead narrator, extra handoff voices) for every episode, in id order."""
    rows = []
    for p in sorted(EPISODES.glob("*.yaml")):
        d = load_yaml(p)
        lead = d.get("narrator", "guide")
        extra = sorted({b["narrator"] for b in d.get("beats", []) if b.get("narrator") and b["narrator"] != lead})
        rows.append((d.get("id", p.stem), lead, extra))
    return rows


def check(rows=None) -> list[str]:
    _, cast = load_series()
    rot = cast.get("rotation", {})
    rows = rows if rows is not None else history()
    warns = []
    run, prev = 0, None
    for eid, lead, _ in rows:
        run = run + 1 if lead == prev else 1
        prev = lead
        if run > rot.get("max_consecutive", 2):
            warns.append(f"{eid}: {lead} leads {run} episodes in a row")
    if rows:
        share = sum(1 for r in rows if r[1] == "whisper") / len(rows)
        if share > rot.get("whisper_max_share", 0.1) and len(rows) >= 5:
            warns.append(f"The Whisper leads {share:.0%} of episodes; keep it rare")
    return warns


def suggest(tone: str | None = None, rows=None) -> list[tuple[str, str]]:
    """Rank narrators for the next episode: matching tone first, then least recently used."""
    _, cast = load_series()
    nars = cast["narrators"]
    rot = cast.get("rotation", {})
    rows = rows if rows is not None else history()
    recent = [r[1] for r in rows[-rot.get("window", 10):]]
    counts = Counter(recent)
    last = recent[-1] if recent else None
    run = 0
    for n in reversed(recent):
        if n != last:
            break
        run += 1

    def score(nid):
        n = nars[nid]
        s = counts[nid] * 1.0
        if nid == "whisper":
            s += 3
        if nid == last and run >= rot.get("max_consecutive", 2):
            s += 100
        if tone and (tone in n.get("use_for", []) or tone in n.get("tone", "")):
            s -= 5
        return s

    ranked = sorted(nars, key=score)
    return [(nid, nars[nid]["name"]) for nid in ranked]
