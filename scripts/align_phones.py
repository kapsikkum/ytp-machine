#!/usr/bin/env python3
"""Find where each phoneme of each clip actually is, and write it down.

    python scripts/align_phones.py --all
    python scripts/align_phones.py --source-id 3 --limit 200

The splicer cuts sub-word units by phoneme -- the "at" out of "fat", the Z off
the end of "calls". To do that it needs a time for each phoneme, and without
this pass it gets one by aligning the word's *letters* and guessing which
letters make which sound. Measured on a real corpus that guess disagrees with
the phoneme count for a third of words, and where it is wrong it is wrong in
one direction: on twelve Michael Rosen clips the final phoneme was placed
between 8ms and 135ms early, so asking for the K of "like" returned the tail
of the diphthong with it.

So the phonemes are aligned against the audio instead. Not recognised -- what
was said is already known -- forced: the dictionary supplies the sequence and
the model places it.

Run at build time, never on the server: the model is 1.2GB and the server's
container is capped below that. The times are stored per clip and travel in
the bundle, so the server reads them and loads nothing.

Resumable. Clips that already have times are skipped unless --redo.
"""

from __future__ import annotations

import argparse
import array
import json
import os
import subprocess
import sys
import tempfile
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[union-attr]
    except Exception:
        pass

from app.database import active, get_db, init_db, resolve_path  # noqa: E402
from app.phonemes import word_to_phonemes                       # noqa: E402

# The aligner is given a little air on each side, because a stored boundary is
# usually late on the onset -- the sound it is supposed to start on often
# begins before it. Times are shifted back afterwards so they stay relative to
# the clip's own start_time, and the first phoneme may legitimately come out
# slightly negative.
_PAD = 0.06


def _wav(path: str, start: float, dur: float):
    import torch
    tmp = os.path.join(tempfile.gettempdir(), f"_pa_{os.getpid()}.wav")
    try:
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.4f}",
                            "-t", f"{dur:.4f}", "-i", path,
                            "-ac", "1", "-ar", "16000", tmp],
                           capture_output=True)
        if r.returncode != 0 or not os.path.exists(tmp):
            return None
        with wave.open(tmp, "rb") as w:
            raw = w.readframes(w.getnframes())
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    a = array.array("h")
    a.frombytes(raw[: len(raw) // 2 * 2])
    if not a:
        return None
    return torch.tensor([v / 32768.0 for v in a], dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="every source")
    ap.add_argument("--source-id", type=int, help="just this source")
    ap.add_argument("--limit", type=int, help="stop after N clips")
    ap.add_argument("--redo", action="store_true",
                    help="realign clips that already have times")
    args = ap.parse_args()
    if not args.all and args.source_id is None:
        ap.error("pass --all or --source-id")

    from app import phone_align
    if not phone_align.available():
        print("Needs torch, torchaudio and transformers:\n"
              "  pip install -r requirements.txt", file=sys.stderr)
        return 1

    init_db()
    print(f"corpus: {active()['slug']}")

    where = "" if args.source_id is None else " AND source_id = ?"
    params: tuple = () if args.source_id is None else (args.source_id,)
    if not args.redo:
        where += " AND (phones IS NULL OR phones = '')"
    with get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            f"SELECT id, word, start_time, end_time, source_file FROM word_clips "
            f"WHERE end_time > start_time{where} ORDER BY source_id, start_time",
            params)]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("nothing to align")
        return 0

    print(f"aligning {len(rows)} clip(s) — loading the model…", flush=True)
    started = time.time()
    done = skipped = failed = 0

    for i, clip in enumerate(rows, start=1):
        phones = word_to_phonemes(clip["word"])
        if not phones:
            skipped += 1
            continue
        pad = min(_PAD, clip["start_time"])
        wav = _wav(resolve_path(clip["source_file"]),
                   clip["start_time"] - pad,
                   (clip["end_time"] - clip["start_time"]) + pad + _PAD)
        if wav is None:
            failed += 1
            continue
        got = phone_align.align(wav, phones)
        if not got:
            failed += 1
            continue
        # Back into the clip's own frame of reference.
        body = json.dumps([[p, round(a - pad, 4), round(b - pad, 4)]
                           for p, a, b in got])
        with get_db() as conn:
            conn.execute("UPDATE word_clips SET phones=? WHERE id=?",
                         (body, clip["id"]))
        done += 1

        if i % 50 == 0 or i == len(rows):
            rate = i / max(1e-6, time.time() - started)
            left = (len(rows) - i) / rate
            print(f"  {i}/{len(rows)}  {rate:.1f} clips/s  "
                  f"~{left / 60:.0f} min left", flush=True)

    try:
        from app.generate import invalidate_cache
        invalidate_cache()
    except Exception:
        pass

    print(f"\naligned {done}, no pronunciation {skipped}, could not align {failed}"
          f"  in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
