#!/usr/bin/env python3
"""
Detect Michael Rosen's non-verbal noises (clicks, spews, pops, blows).

Whisper only transcribes words, so the iconic mouth-noises live in the gaps
between words.  This scans those gaps for energetic bursts (vs. silence),
trims each to the burst, classifies it by duration (short -> "click", longer
-> "spew") and stores it in the noise_clips table.  Everything is also tagged
"noise" so you can type any of: noise, click, spew.

Usage:
    python scripts/find_noises.py --source-id 16        # dry run, one source
    python scripts/find_noises.py --apply               # all sources, write
"""
import argparse
import os
import subprocess
import tempfile
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db, get_db, resolve_path

_SR = 16000
_MIN_GAP   = 0.14   # ignore tiny gaps
_PEAK_FLOOR = 550   # int16 RMS a burst must exceed to count as a vocalisation
_WIN        = 0.01  # 10 ms analysis window
_CLICK_MAX  = 0.18  # bursts shorter than this are "click", else "spew"


def _load(path):
    import numpy as np
    w = wave.open(path, "rb")
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype("float32")
    w.close()
    return d


def _bursts_in_gap(source_file, g_start, g_end):
    """Return list of (abs_start, abs_end, peak) energetic bursts in a gap."""
    import numpy as np
    tmp = os.path.join(tempfile.gettempdir(), f"_nz_{os.getpid()}.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{g_start:.4f}", "-t", f"{g_end - g_start:.4f}",
             "-i", resolve_path(source_file), "-ac", "1", "-ar", str(_SR), tmp],
            capture_output=True,
        )
        d = _load(tmp)
    finally:
        try: os.remove(tmp)
        except OSError: pass
    if len(d) < int(_WIN * _SR):
        return []

    win = int(_WIN * _SR)
    env = np.array([np.sqrt(np.mean(d[k:k + win] ** 2)) for k in range(0, len(d), win)])
    if not len(env) or env.max() < _PEAK_FLOOR:
        return []

    thr = max(_PEAK_FLOOR * 0.6, env.max() * 0.30)
    bursts = []
    i = 0
    while i < len(env):
        if env[i] >= thr:
            j = i
            while j < len(env) and env[j] >= thr * 0.6:
                j += 1
            peak = float(env[i:j].max())
            # pad the burst a touch so the attack/decay isn't clipped
            a = g_start + max(0, i - 1) * _WIN
            b = g_start + min(len(env), j + 1) * _WIN
            if peak >= _PEAK_FLOOR and (b - a) >= 0.03:
                bursts.append((a, b, peak))
            i = j
        else:
            i += 1
    return bursts


def _duration(source_file) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", resolve_path(source_file)],
        capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def find_in_source(sid, apply, label=None, min_dur=0.0):
    with get_db() as conn:
        src = conn.execute("SELECT source_file FROM sources WHERE id=?", (sid,)).fetchone()
        words = [dict(r) for r in conn.execute(
            "SELECT start_time, end_time FROM word_clips WHERE source_id=? ORDER BY start_time",
            (sid,)).fetchall()]
    if not src or len(words) < 2:
        return 0
    sf = src["source_file"]

    # Every stretch of audio no word claims. The pairwise gaps are the obvious
    # ones, but a video that opens or closes on a noise has it outside every
    # pair -- and opening on one is common, because a speaker clears their
    # throat before the first word rather than after it.
    gaps = [(a["end_time"] + 0.02, b["start_time"] - 0.02)
            for a, b in zip(words, words[1:])]
    gaps.insert(0, (0.0, words[0]["start_time"] - 0.02))
    end = _duration(sf)
    if end > words[-1]["end_time"]:
        gaps.append((words[-1]["end_time"] + 0.02, end))

    rows = []
    for g_start, g_end in gaps:
        if g_end - g_start < _MIN_GAP:
            continue
        for (st, en, peak) in _bursts_in_gap(sf, g_start, g_end):
            # A breath, a lip smack and the plosive at the front of a word are
            # all bursts, and all short. When you are after one particular
            # sustained noise -- a hum, a groan -- length is what separates it
            # from the dozen incidental ones in the same recording.
            if (en - st) < min_dur:
                continue
            kind = label if label else ("click" if (en - st) < _CLICK_MAX else "spew")
            rows.append((sid, kind, round(st, 4), round(en, 4), sf))

    if apply and rows:
        with get_db() as conn:
            conn.executemany(
                "INSERT INTO noise_clips (source_id, word, start_time, end_time, source_file) "
                "VALUES (?, ?, ?, ?, ?)", rows)
    clicks = sum(1 for r in rows if r[1] == "click")
    print(f"  source {sid:3d}: {len(rows)} noises ({clicks} click, {len(rows)-clicks} spew)")
    return len(rows)


def main():
    # The noises only live in two poems: Hot Food (clicks) and Fridge (the
    # spew).  Scanning every source caught breaths/junk, so the build is curated.
    CURATED = [("Akwm2UZJ34o", "click"), ("JlfJx1aDmR4", "spew")]

    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--source-id", type=int, default=None,
                    help="Scan this source instead of the curated list. Every "
                         "corpus has its own noises and the curated pair names "
                         "two Michael Rosen videos, so this is what any other "
                         "speaker needs.")
    ap.add_argument("--min-dur", type=float, default=0.0, metavar="SEC",
                    help="Ignore bursts shorter than this. Breaths and lip "
                         "smacks are brief; a sustained noise worth naming is "
                         "not.")
    ap.add_argument("--label", default=None, metavar="WORD",
                    help="What to call what it finds, e.g. mmm. Without this "
                         "they are named by duration: click, or spew.")
    args = ap.parse_args()
    init_db()

    with get_db() as conn:
        if args.apply:
            # Scoped to the source when one is named. Wiping the table for a
            # single-source run would throw away every other source's noises
            # to add one video's.
            if args.source_id is not None:
                conn.execute("DELETE FROM noise_clips WHERE source_id=?",
                             (args.source_id,))
            else:
                conn.execute("DELETE FROM noise_clips")
        if args.source_id is not None:
            targets = [(args.source_id, args.label)]
        else:
            targets = []
            for vid, label in CURATED:
                r = conn.execute("SELECT id, title FROM sources WHERE source_file LIKE ?",
                                 (f"%{vid}%",)).fetchone()
                if r:
                    targets.append((r["id"], args.label or label))

    print(f"Noise build{'  (DRY RUN)' if not args.apply else ''}")
    total = 0
    for sid, label in targets:
        total += find_in_source(sid, args.apply, label, args.min_dur)
    print(f"\nTotal: {total} noise clips {'stored' if args.apply else '(dry run)'}.")
    if args.apply:
        try:
            from app.generate import invalidate_cache
            invalidate_cache()
        except Exception:
            pass
        print("Call POST /api/reload so the server picks up the new noises.")


if __name__ == "__main__":
    main()
