#!/usr/bin/env python3
"""
Re-transcribe already-downloaded videos using stable-whisper for accurate
word-level timestamps, replacing the original (plain Whisper) entries in DB.

No re-downloading needed — reads from the downloads/ folder directly.

Usage:
    python scripts/realign.py                     # all sources
    python scripts/realign.py --model small       # better accuracy
    python scripts/realign.py --source-id 3       # single source by DB id
    python scripts/realign.py --skip-errors
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db, get_db
from scripts.ingest import transcribe


def realign_source(source: dict, model_name: str, device: str | None = None) -> int:
    """Replace word_clips for one source. Returns number of new clips stored."""
    from app.database import resolve_path
    # Stored paths are relative to the corpus directory, not to wherever this
    # was run from -- without resolving, every source looks missing as soon as
    # the corpus is not the current directory.
    source_file = resolve_path(source["source_file"])
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Download not found: {source_file}")

    words = transcribe(source_file, model_name, device)
    print(f"  {len(words)} words transcribed")

    with get_db() as conn:
        # Remove old clips for this source
        conn.execute("DELETE FROM word_clips WHERE source_id = ?", (source["id"],))
        # Insert new clips
        conn.executemany(
            "INSERT INTO word_clips (source_id, word, start_time, end_time, source_file) "
            "VALUES (?, ?, ?, ?, ?)",
            [(source["id"], w["word"], w["start"], w["end"], source_file) for w in words],
        )

    return len(words)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-align word timestamps using stable-whisper."
    )
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--source-id", type=int, default=None,
                        help="Only re-align this DB source id")
    parser.add_argument("--start-from-id", type=int, default=None,
                        metavar="N", help="Skip sources with id < N")
    parser.add_argument("--end-at-id", type=int, default=None,
                        metavar="N", help="Stop after source id N (inclusive)")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Continue if one source fails")
    from app.device import add_argument as _device_arg
    _device_arg(parser)
    args = parser.parse_args()

    init_db()

    with get_db() as conn:
        if args.source_id is not None:
            sources = conn.execute(
                "SELECT * FROM sources WHERE id = ?", (args.source_id,)
            ).fetchall()
        else:
            q = "SELECT * FROM sources WHERE 1=1"
            params: list = []
            if args.start_from_id is not None:
                q += " AND id >= ?"
                params.append(args.start_from_id)
            if args.end_at_id is not None:
                q += " AND id <= ?"
                params.append(args.end_at_id)
            q += " ORDER BY id"
            sources = conn.execute(q, params).fetchall()

    if not sources:
        sys.exit("No sources found in database.")

    print(f"Re-aligning {len(sources)} source(s) with stable-whisper "
          f"(model: {args.model})\n{'-'*60}")

    ok = 0
    for i, src in enumerate(sources, 1):
        s = dict(src)
        print(f"\n[{i}/{len(sources)}] {s['title'] or s['video_id']}")
        print(f"  {s['source_file']}")
        try:
            n = realign_source(s, args.model, args.device)
            print(f"  OK stored {n} clips")
            ok += 1
        except Exception as exc:
            print(f"  FAIL: {exc}")
            if not args.skip_errors:
                sys.exit(1)

    # Invalidate the generation cache so the server picks up new timestamps
    try:
        from app.generate import invalidate_cache
        invalidate_cache()
    except Exception:
        pass

    print(f"\n{'-'*60}")
    print(f"Done: {ok}/{len(sources)} sources re-aligned.")
    if ok < len(sources):
        # Only sources that failed need doing again, and the ones that
        # succeeded are already written -- re-running the whole range would
        # re-transcribe hours of audio for nothing.
        print(f"{len(sources) - ok} failed. Re-run with --source-id to retry "
              f"just those; the rest are already stored.")
        if not args.skip_errors:
            print("(--skip-errors carries on past a failure instead of stopping.)")


if __name__ == "__main__":
    main()
