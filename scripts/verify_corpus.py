#!/usr/bin/env python3
"""Check a corpus against YouTube's own captions and report what looks wrong.

    python scripts/verify_corpus.py --all
    python scripts/verify_corpus.py --source-id 3 --show 40

Whisper is good but not right, and its mistakes are invisible from inside: a
word it mishears is stored confidently under the wrong spelling, so the corpus
looks complete and simply never produces that word. "rupees" was transcribed as
"rubies" -- the most quotable word in that corpus, unreachable by typing it,
and nothing anywhere said so.

This compares the stored labels against YouTube's captions, matched up by
timestamp, and reports two things:

  missed     a caption word with no clip anywhere near it -- speech the
             transcription skipped, so those words are simply absent
  disagree   both systems heard something at the same moment and spelled it
             differently -- one of them is wrong, and it is worth a look

YouTube's captions are NOT ground truth. Where a channel offers uploader-written
subtitles those would be, but none of the channels here do; what is available is
Google's speech recognition, which has its own failures. The value is that it is
a *different* system from Whisper, so the two agreeing means something and the
two disagreeing is worth a human deciding. Nothing here edits the corpus --
scripts/correct.py does that, deliberately, from a transcript you trust.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

from app.database import active, get_db, init_db

# How far apart two timings may be and still be considered the same word.
# Caption timings are coarser than forced-aligned clip boundaries, and the two
# drift against each other over a long video, so this is generous on purpose --
# it decides "did anyone say anything here", not where the word starts.
_TOL = 0.6

# A disagreement is only reported when the two are this close. Where the
# transcripts genuinely diverge -- a sung passage, a stretch of music -- every
# caption word in the region pairs with whatever clip happens to be nearby and
# invents dozens of conflicts that mean nothing. A real mishearing sits almost
# exactly on top of the word it replaced.
_CONFLICT_TOL = 0.25


def _cache_path(video_id: str) -> str:
    # Derived and re-fetchable, so it lives outside the packed members: a
    # bundle should not carry a copy of someone else's captions around.
    d = os.path.join(active()["dir"], "captions")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{video_id}.json")


def fetch_captions(video_id: str, cookies_from_browser: str | None = None,
                   refresh: bool = False) -> list[tuple[str, float]] | None:
    """[(word, start_seconds)] from YouTube's captions, or None if there are none."""
    cache = _cache_path(video_id)
    if os.path.exists(cache) and not refresh:
        with open(cache, encoding="utf-8") as f:
            return [(w, t) for w, t in json.load(f)]

    import yt_dlp
    opts: dict = {"quiet": True, "no_warnings": True, "skip_download": True}
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False)
    except Exception as exc:
        print(f"    could not reach YouTube: {str(exc).splitlines()[-1][:90]}")
        return None
    if not info:
        return None

    # Uploader-written subtitles first: those are a real transcript. Otherwise
    # the automatic ones, which are a second opinion rather than an answer.
    tracks = (info.get("subtitles") or {})
    manual = any(k.startswith("en") for k in tracks)
    if not manual:
        tracks = (info.get("automatic_captions") or {})
    lang = next((k for k in tracks if k.startswith("en")), None)
    if not lang:
        return None
    fmt = next((f for f in tracks[lang] if f.get("ext") == "json3"), None)
    if not fmt:
        return None

    req = urllib.request.Request(fmt["url"], headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")

    words: list[tuple[str, float]] = []
    for event in json.loads(raw).get("events", []):
        base = event.get("tStartMs", 0)
        for seg in event.get("segs") or []:
            # Exactly what scripts/ingest.transcribe does to Whisper's words.
            # Without it "it's" and "its" are a disagreement, and apostrophes
            # alone accounted for more reported conflicts than every real
            # mishearing put together.
            w = re.sub(r"[^\w]", "", seg.get("utf8", "")).lower()
            if w:
                words.append((w, (base + seg.get("tOffsetMs", 0)) / 1000.0))

    words.sort(key=lambda x: x[1])
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(words, f)
    return words


def compare(caption: list[tuple[str, float]],
            corpus: list[tuple[str, float, float]]) -> dict:
    """Match caption words to corpus clips by time; report misses and conflicts."""
    missed: list[tuple[str, float]] = []
    disagree: list[tuple[str, str, float]] = []
    agreed = 0
    drifted = 0        # something was said here, but too far off to pair up

    ci = 0
    for word, t in caption:
        # Walk the corpus forward to the clips overlapping this moment. Both
        # lists are in time order, so this stays linear rather than searching
        # the whole corpus per caption word.
        while ci < len(corpus) and corpus[ci][2] < t - _TOL:
            ci += 1
        window = []
        j = ci
        while j < len(corpus) and corpus[j][1] <= t + _TOL:
            window.append(corpus[j])
            j += 1
        if not window:
            missed.append((word, t))
        elif any(w == word for w, _s, _e in window):
            agreed += 1
        else:
            # Nearest by start time is the one it was most likely meant to be.
            near = min(window, key=lambda c: abs(c[1] - t))
            if abs(near[1] - t) <= _CONFLICT_TOL:
                disagree.append((word, near[0], t))
            else:
                drifted += 1
    return {"agreed": agreed, "missed": missed, "disagree": disagree,
            "drifted": drifted}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-id", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--show", type=int, default=15, metavar="N",
                    help="How many examples of each kind to print (default 15)")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-fetch captions instead of using the cached copy")
    ap.add_argument("--cookies-from-browser", default=None, metavar="BROWSER")
    args = ap.parse_args()

    if args.source_id is None and not args.all:
        return int(bool(sys.stderr.write(
            "Nothing to check: pass --source-id N or --all.\n"))) or 2

    init_db()
    with get_db() as conn:
        q = "SELECT id, video_id, title FROM sources"
        p: list = []
        if args.source_id is not None:
            q += " WHERE id=?"
            p.append(args.source_id)
        sources = [dict(r) for r in conn.execute(q + " ORDER BY id", p)]

    if not sources:
        return int(bool(sys.stderr.write("No such source.\n"))) or 1

    print(f"Checking {len(sources)} source(s) in '{active()['slug']}' "
          f"against YouTube captions\n" + "-" * 62)

    totals = Counter()
    suspects: Counter = Counter()
    for src in sources:
        print(f"\nsource {src['id']:3d}  {(src['title'] or src['video_id'])[:46]}")
        caption = fetch_captions(src["video_id"], args.cookies_from_browser,
                                 args.refresh)
        if not caption:
            print("    no English captions published for this video")
            continue

        with get_db() as conn:
            corpus = [(r["word"], r["start_time"], r["end_time"]) for r in conn.execute(
                "SELECT word, start_time, end_time FROM word_clips "
                "WHERE source_id=? ORDER BY start_time", (src["id"],))]
        if not corpus:
            print("    no clips stored for this source")
            continue

        r = compare(caption, corpus)
        n = len(caption)
        totals["caption"] += n
        totals["clips"] += len(corpus)
        totals["agreed"] += r["agreed"]
        totals["missed"] += len(r["missed"])
        totals["disagree"] += len(r["disagree"])
        print(f"    captions {n:>6}   clips {len(corpus):>6}   "
              f"agree {r['agreed'] * 100 // max(n, 1)}%   "
              f"missed {len(r['missed'])}   disagree {len(r['disagree'])}"
              + (f"   drifted {r['drifted']}" if r["drifted"] else ""))

        for word, said, _t in r["disagree"]:
            suspects[(said, word)] += 1
        if r["missed"][:args.show]:
            print("    missed  : " + ", ".join(w for w, _t in r["missed"][:args.show]))
        if r["disagree"][:args.show]:
            print("    disagree: " + ", ".join(
                f"{said}->{heard}" for heard, said, _t in r["disagree"][:args.show]))

    if not totals["caption"]:
        print("\nNothing to compare against.")
        return 0

    print("\n" + "-" * 62)
    print(f"{totals['caption']} caption words vs {totals['clips']} clips: "
          f"{totals['agreed'] * 100 // totals['caption']}% agree, "
          f"{totals['missed']} missed, {totals['disagree']} disagree")
    if suspects:
        # Repeats matter far more than one-offs. A word both systems disagree
        # about every single time it is said is a systematic mishearing -- the
        # "rubies" case -- while a single disagreement is usually just the two
        # of them splitting a phrase differently.
        print("\nmost repeated disagreements  (stored -> caption):")
        for (said, heard), count in suspects.most_common(20):
            if count > 1:
                print(f"  {count:>4}x  {said:<18} -> {heard}")
    print("\nNothing was changed. To act on these, put a transcript you trust in\n"
          "transcripts/<video_id>.txt and run scripts/correct.py --all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
