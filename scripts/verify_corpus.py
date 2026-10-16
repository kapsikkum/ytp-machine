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
import math
import os
import re
import sys
import urllib.request
from urllib.parse import urlencode
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

from app.database import active, artifact_filename, get_db, init_db

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
    return os.path.join(d, artifact_filename(video_id, ".json"))


def _caption_words(data: dict) -> list[tuple[str, float]]:
    """Keep words in multiword caption segments distinct, at their shared time."""
    words: list[tuple[str, float]] = []
    for event in data.get("events", []):
        if not isinstance(event, dict):
            continue
        for segment in event.get("segs") or []:
            if not isinstance(segment, dict) or not isinstance(segment.get("utf8"), str):
                continue
            try:
                time = (float(event.get("tStartMs", 0)) + float(segment.get("tOffsetMs", 0))) / 1000
            except (TypeError, ValueError):
                continue
            if not math.isfinite(time) or time < 0:
                continue
            for token in segment["utf8"].split():
                word = re.sub(r"[^\w]", "", token).lower()
                if word:
                    words.append((word, time))
    return sorted(words, key=lambda row: row[1])


def fetch_captions(video_id: str, cookies_from_browser: str | None = None,
                   refresh: bool = False) -> list[tuple[str, float]] | None:
    """[(word, start_seconds)] from YouTube's captions, or None if there are none."""
    cache = _cache_path(video_id)
    if os.path.exists(cache) and not refresh:
        try:
            with open(cache, encoding="utf-8") as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                raise ValueError("invalid caption cache")
            words = [(w, float(t)) for w, t in rows]
            if any(not isinstance(w, str) or not math.isfinite(t) or t < 0 for w, t in words):
                raise ValueError("invalid caption cache")
            return words
        except (ValueError, TypeError, OSError):
            pass                      # derived cache: refresh interrupted/bad data

    import yt_dlp
    opts: dict = {"quiet": True, "no_warnings": True, "skip_download": True}
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(
                "https://www.youtube.com/watch?" + urlencode({"v": video_id}), download=False)
    except Exception as exc:
        lines = str(exc).strip().splitlines()
        print(f"    could not reach YouTube: {(lines[-1] if lines else type(exc).__name__)[:90]}")
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
    limit = 32 << 20
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("caption response is too large")
    words = _caption_words(json.loads(raw.decode("utf-8")))
    fd, temporary = tempfile.mkstemp(prefix=".captions-", suffix=".json", dir=os.path.dirname(cache))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(words, f)
        os.replace(temporary, cache)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
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


def write_template(words, per_corpus: bool = False) -> str:
    """Append the unpronounceable words to the corpus's CSV, commented out.

    Commented, because a wrong pronunciation is worse than none: it does not
    report the word missing, it says something else confidently. Uncomment a
    line once you have filled in how it actually sounds.
    """
    from app.phonemes import global_dict_path, user_dict_path
    # The global file by default: most of what turns up is not specific to a
    # speaker, and a per-corpus copy would have to be maintained again for
    # every channel ingested.
    path = user_dict_path() if per_corpus else global_dict_path()
    existing = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                head = line.lstrip("# ").split(",", 1)[0].strip().lower()
                if head:
                    existing.add(head)

    new = [(w, n) for w, n in words if w not in existing]
    if not new:
        return path
    fresh = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        if fresh:
            f.write("# How to say the words no dictionary has. Two columns:\n"
                    "#\n"
                    "#   nug,N AH G       ARPAbet, if you know it\n"
                    "#   wii,wee          or just a word that already sounds right\n"
                    "#   mcnug,mick nug   several words are fine\n"
                    "#   usb,=letters     read it out letter by letter\n"
                    "#   hevexum,=skip    leave it unsayable on purpose\n"
                    "#\n"
                    "# The second form is the one to use -- nobody should have to\n"
                    "# learn ARPAbet to say that \"wii\" rhymes with \"wee\".\n")
        f.write("\n# Added by verify_corpus.py --unspliceable --write-template.\n"
                "# Uncomment and fill in the ones worth saying.\n")
        for word, n in new:
            f.write(f"#{word},   # {n} clip{'s' if n != 1 else ''}\n")
    return path


def report_unspliceable(show: int, write: bool = False,
                        per_corpus: bool = False) -> int:
    """Stored words with no pronunciation, worst first.

    A word with no pronunciation is skipped by the splice index entirely: it
    still matches as a whole word, so nothing looks wrong, but every clip of it
    is unusable as splice material and the word cannot be built if the corpus
    lacks a recording. That is invisible from the outside, which is why it is
    worth printing.

    No list of exceptions can ever be complete -- speakers invent words, and a
    dictionary cannot hold names or coinages -- so the maintainable form of
    "exhaustive" is this: measure your own corpus and add what actually turns
    up. Most of what remains here is genuinely unknowable, and outside strict
    mode the splicer guesses a pronunciation from spelling anyway.
    """
    from collections import Counter
    from app.phonemes import word_to_phonemes

    init_db()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT word, count(*) FROM word_clips GROUP BY word").fetchall()
    total_clips = sum(n for _w, n in rows)
    bad = Counter({w: n for w, n in rows if not word_to_phonemes(w)})
    stranded = sum(bad.values())

    print(f"corpus '{active()['slug']}': {len(rows)} distinct words, "
          f"{total_clips} clips")
    if not bad:
        print("every stored word has a pronunciation -- nothing is stranded.")
        return 0
    print(f"{len(bad)} words have no pronunciation, stranding {stranded} clips "
          f"({stranded * 100 // max(total_clips, 1)}% of the corpus)\n")
    for word, n in bad.most_common(show):
        print(f"  {n:>4}  {word}")
    if len(bad) > show:
        print(f"  … and {len(bad) - show} more (--show N for more)")
    if write:
        path = write_template(bad.most_common(), per_corpus)
        print(f"\nwritten to {path} -- uncomment the ones worth saying.")
    else:
        print("\nTo say any of these, put them in pronunciations.csv "
              "(--write-template starts\nit for you; --per-corpus to keep them "
              "with this corpus instead of\nsharing them). Names and coinages "
              "are fine to leave: outside strict\nmode the splicer guesses "
              "them from spelling.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-id", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--show", type=int, default=15, metavar="N",
                    help="How many examples of each kind to print (default 15)")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-fetch captions instead of using the cached copy")
    ap.add_argument("--cookies-from-browser", default=None, metavar="BROWSER")
    ap.add_argument("--write-template", action="store_true",
                    help="With --unspliceable, write the words into the corpus's "
                         "pronunciations.csv for you to fill in. Never overwrites "
                         "entries you have already made.")
    ap.add_argument("--per-corpus", action="store_true",
                    help="With --write-template, write into this corpus's own "
                         "dictionary instead of the global one. Use it for a "
                         "speaker's own coinages, which travel with the bundle.")
    ap.add_argument("--unspliceable", action="store_true",
                    help="Skip the caption check and list stored words that have "
                         "no pronunciation. Needs no network.")
    args = ap.parse_args()

    if args.unspliceable:
        return report_unspliceable(args.show, args.write_template,
                                   args.per_corpus)

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
