#!/usr/bin/env python3
"""Describe how every word clip sounds, so sentences can be put together in one tone.

    python scripts/measure_tone.py
    python scripts/measure_tone.py --source-id 3 --redo

Measures each clip's pitch, where its pitch starts and ends, how much it
moves, its loudness against its own recording, its pace and its brightness,
then groups the whole corpus into tone categories. See app/prosody.py for
what the generator does with them.

Run at build time, after the boundaries are final: everything is measured
inside the clip's stored times. No model is involved -- pitch is the same
YIN the YTPMV sampler uses -- so it is minutes, not hours, and it runs
anywhere. The results live in the corpus database and travel in its bundle.

Resumable. Clips that already have a measurement are skipped unless --redo;
the categories are always recomputed over every measured clip, because a new
source moves the averages they are drawn from.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[union-attr]
    except Exception:
        pass

import numpy as np                                               # noqa: E402

from app import prosody                                          # noqa: E402
from app.database import active, get_db, init_db, resolve_path  # noqa: E402
from app.phonemes import word_to_phonemes                       # noqa: E402


def _decode(path: str) -> np.ndarray | None:
    """The whole source at prosody.SR, mono, in one pass."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path,
                        "-ac", "1", "-ar", str(prosody.SR), "-f", "f32le", "-"],
                       capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    return np.frombuffer(r.stdout[: len(r.stdout) // 4 * 4], dtype=np.float32)


def _phones(word: str) -> int:
    return len(word_to_phonemes(word) or [])


def measure(source_id: int | None = None, redo: bool = False) -> tuple[int, int]:
    """Measure clips without a row (every clip, with *redo*). Returns (measured, failed sources)."""
    with get_db() as conn:
        q = "SELECT id, source_file FROM sources" + (" WHERE id=?" if source_id is not None else "") + " ORDER BY id"
        sources = conn.execute(q, (() if source_id is None else (source_id,))).fetchall()
    done = failed = 0
    for src in sources:
        with get_db() as conn:
            clips = [dict(r) for r in conn.execute(
                "SELECT id, word, start_time, end_time FROM word_clips WHERE source_id=? "
                + ("" if redo else "AND id NOT IN (SELECT clip_id FROM clip_tone) ")
                + "ORDER BY start_time", (src["id"],))]
        if not clips:
            continue
        x = _decode(resolve_path(src["source_file"]))
        if x is None:
            print(f"  source {src['id']}: could not decode, skipped")
            failed += 1
            continue
        rows = prosody.measure_source(x, clips, _phones)
        with get_db() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO clip_tone (clip_id, " + ", ".join(prosody.FEATURES)
                + ", category) VALUES (?" + ", ?" * len(prosody.FEATURES) + ", NULL)",
                [(cid, *(row[f] for f in prosody.FEATURES)) for cid, row in rows.items()])
        done += len(rows)
        print(f"  source {src['id']}: {len(rows)}/{len(clips)} clips")
    return done, failed


def categorise() -> list[tuple[str, int]]:
    """Recompute the corpus statistics and categories over every measured clip."""
    with get_db() as conn:
        # Rows for clips that no longer exist would still pull the averages.
        conn.execute("DELETE FROM clip_tone WHERE clip_id NOT IN (SELECT id FROM word_clips)")
        found = conn.execute("SELECT clip_id, " + ", ".join(prosody.FEATURES) + " FROM clip_tone").fetchall()
    rows = {int(r[0]): {f: r[i + 1] for i, f in enumerate(prosody.FEATURES)} for r in found}
    if not rows:
        return []
    stats = prosody.statistics(list(rows.values()))
    ids = list(rows)
    labels, centres = prosody.categorise([prosody.standard(rows[i], stats) for i in ids])
    names = [prosody.name(c) for c in centres]
    sizes = np.bincount(labels, minlength=len(centres))
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('tone_stats', ?)",
                     (json.dumps(stats),))
        conn.execute("DELETE FROM tone_categories")
        conn.executemany("INSERT INTO tone_categories (id, name, centroid, size) VALUES (?, ?, ?, ?)",
                         [(j, names[j], json.dumps([round(float(v), 4) for v in centres[j]]), int(sizes[j]))
                          for j in range(len(centres))])
        conn.executemany("UPDATE clip_tone SET category=? WHERE clip_id=?",
                         [(int(l), i) for i, l in zip(ids, labels)])
    try:
        from app.generate import invalidate_cache
        invalidate_cache(alignments=False)
    except Exception:
        pass
    return [(names[j], int(sizes[j])) for j in range(len(centres))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source-id", type=int)
    ap.add_argument("--redo", action="store_true", help="measure clips that already have a measurement")
    args = ap.parse_args()

    init_db()
    print(f"Measuring tone in {active()['slug']}")
    t0 = time.time()
    done, failed = measure(args.source_id, args.redo)
    print(f"{done} clips measured in {time.time() - t0:.0f}s")
    for j, (name, size) in enumerate(categorise()):
        print(f"  category {j}: {name:30s} {size} clips")
    if failed:
        sys.exit(f"{failed} source(s) could not be decoded")


if __name__ == "__main__":
    main()
