#!/usr/bin/env python3
"""The whole pipeline, on one real video.

    python tests/test_end_to_end.py            # ~2 minutes
    python tests/test_end_to_end.py --model base --keep

Downloads a single 63-second video, transcribes it, refines the boundaries,
generates sentences from it, and checks the videos that come out are real and
the right length.

Not part of CI: it needs the network, ffmpeg, torch and a Whisper model, none
of which the check job has. Run it before shipping anything that touches
ingest, clip selection or the ffmpeg builder -- it is the only test that
exercises those together, and every bug that has escaped so far escaped
*between* the parts rather than inside one.

Nothing here depends on what Whisper happens to hear: the test sentences are
assembled from whatever words the corpus ends up containing, so it asserts the
pipeline works rather than that a particular poem was transcribed a particular
way.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

# Short, one speaker, clear diction, and still up. A long video would make the
# test a chore to run, which is the surest way to stop it being run.
VIDEO = "https://www.youtube.com/watch?v=Akwm2UZJ34o"

# A word clip is at least this long and a sentence cannot be shorter than its
# words. Deliberately loose: this is checking that audio came out at all, not
# grading the pacing.
_MIN_PER_WORD = 0.05
_MAX_PER_WORD = 3.0

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(f"  {msg}")


def duration(path: str) -> float | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny",
                    help="Whisper model (default tiny -- accuracy is not what "
                         "is under test here, and tiny keeps this to minutes)")
    ap.add_argument("--keep", action="store_true",
                    help="Leave the corpus behind for inspection")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("SKIP: ffmpeg is not on PATH")
        return 0

    work = tempfile.mkdtemp(prefix="ytp-e2e-")
    os.environ["MRS_DATA_DIR"] = work
    os.environ["MRS_CORPUS"] = "e2e"
    # Imported only now: app.database reads MRS_DATA_DIR at import.
    from app.database import get_db, init_db
    from scripts.ingest import ingest

    print(f"corpus in {work}\n{'-' * 58}")
    started = time.time()

    # 1) Ingest -------------------------------------------------------------
    try:
        ingest(VIDEO, model_name=args.model, max_height=480, normalise=True)
    except Exception as exc:
        fail(f"ingest raised {type(exc).__name__}: {exc}")
        print("\n".join(failures))
        return 1

    init_db()
    with get_db() as conn:
        clips = conn.execute("SELECT count(*) FROM word_clips").fetchone()[0]
        words = [r[0] for r in conn.execute(
            "SELECT word FROM word_clips GROUP BY word HAVING count(*) >= 1 "
            "ORDER BY count(*) DESC LIMIT 40")]
        sources = conn.execute("SELECT count(*) FROM sources").fetchone()[0]
        stored = conn.execute("SELECT source_file FROM word_clips LIMIT 1").fetchone()[0]
    print(f"ingested: {sources} source, {clips} clips, {len(words)} sampled words")

    if sources != 1:
        fail(f"expected 1 source, got {sources}")
    if clips < 20:
        fail(f"only {clips} clips from a 63-second video -- transcription "
             f"produced almost nothing")
    # The path has to be relative or the corpus only works on this machine.
    # This is exactly what shipped broken: an absolute Windows path stored by
    # realign, which resolved nowhere on the server.
    if not str(stored).startswith("downloads/"):
        fail(f"clip path is not relative to the corpus: {stored!r}")

    # 2) Refine boundaries ---------------------------------------------------
    try:
        from scripts.refine_boundaries import refine_source, _load_model
        _load_model(None)
        updated, total, _median = refine_source(1, apply=True)
        print(f"refined : {updated}/{total} clips")
        if updated == 0:
            fail("boundary refinement updated nothing")
    except Exception as exc:
        fail(f"refine raised {type(exc).__name__}: {exc}")

    # 3) Generate ------------------------------------------------------------
    from app.generate import generate_video, invalidate_cache
    invalidate_cache()

    if len(words) < 4:
        fail("too few distinct words to build a sentence from")
    else:
        lengths = {}
        for n in (2, 5):
            text = " ".join(words[:n])
            try:
                result = generate_video(text)
            except Exception as exc:
                fail(f"generating {n} words raised {type(exc).__name__}: {exc}")
                continue

            if result["missing"]:
                fail(f"{n}-word line reported its own corpus's words missing: "
                     f"{result['missing']}")
            url = result.get("video_url")
            if not url:
                fail(f"{n}-word line produced no video")
                continue
            path = os.path.join(ROOT, url.lstrip("/"))
            if not os.path.exists(path):
                fail(f"{n}-word line: {url} does not exist on disk")
                continue

            dur = duration(path)
            size = os.path.getsize(path)
            print(f"generated {n} words: {dur:.2f}s, {size / 1000:.0f} KB")
            if dur is None or dur <= 0:
                fail(f"{n}-word line produced an unplayable video")
                continue
            lengths[n] = dur
            if dur < n * _MIN_PER_WORD:
                fail(f"{n}-word line is {dur:.2f}s -- too short to contain "
                     f"{n} words")
            if dur > n * _MAX_PER_WORD:
                fail(f"{n}-word line is {dur:.2f}s -- implausibly long for "
                     f"{n} words")
            if size < 1000:
                fail(f"{n}-word line is {size} bytes -- effectively empty")

        # The structural check: more words must mean more video. A builder that
        # silently drops segments still produces a plausible-looking file, and
        # only the comparison catches it.
        if len(lengths) == 2 and lengths[5] <= lengths[2]:
            fail(f"5 words ({lengths[5]:.2f}s) is not longer than 2 "
                 f"({lengths[2]:.2f}s) -- segments are being dropped")

    print(f"{'-' * 58}\nfinished in {time.time() - started:.0f}s")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"corpus kept at {work}")

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        print("\n".join(failures))
        return 1
    print("\nok: ingest, refine and generate all work on a real video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
