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
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db, get_db
from scripts.ingest import MODEL_CHOICES, add_transcription_arguments, transcribe


def realign_source(source: dict, model_name: str, device: str | None = None,
                   language: str | None = None, initial_prompt: str | None = None,
                   replace_edited: bool = False) -> int:
    """Replace word_clips for one source. Returns number of new clips stored."""
    from app.database import relativize_path, resolve_path
    # Stored paths are relative to the corpus directory, not to wherever this
    # was run from -- without resolving, every source looks missing as soon as
    # the corpus is not the current directory.
    source_file = resolve_path(source["source_file"])
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Download not found: {source_file}")

    def fingerprint(conn):
        return [tuple(row) for row in conn.execute(
            "SELECT id, word, start_time, end_time, source_file, edited "
            "FROM word_clips WHERE source_id=? ORDER BY id", (source["id"],))]

    with get_db() as conn:
        previous = fingerprint(conn)
    if any(row[-1] for row in previous) and not replace_edited:
        raise ValueError("Source has hand-edited clips; use --replace-edited to replace them")
    version = os.stat(source_file)

    words = transcribe(source_file, model_name, device, language, initial_prompt)
    print(f"  {len(words)} words transcribed")
    if not words:
        raise ValueError("No usable word timestamps; existing clips were kept")
    after = os.stat(source_file)
    if (version.st_mtime_ns, version.st_size) != (after.st_mtime_ns, after.st_size):
        raise ValueError("Source recording changed during transcription; existing clips were kept")

    # Store what came out of the database, not the absolute path we resolved to
    # run ffmpeg. Writing the absolute one back looks harmless on the machine
    # that did the re-align and breaks the corpus everywhere else: a Windows
    # "C:/..." is not absolute to Linux, so resolve_path appends it to the
    # corpus directory and every clip from a re-aligned source fails to open.
    stored = relativize_path(source_file)

    with get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if fingerprint(conn) != previous:
            raise ValueError("Source changed during transcription; existing clips were kept")
        old = {row[0]: row for row in previous}
        for table in ("clip_alignments", "clip_cache", "boundary_reports", "splice_ratings"):
            conn.execute(f"DELETE FROM {table} WHERE clip_id IN "
                         "(SELECT id FROM word_clips WHERE source_id=?)", (source["id"],))
        # Remove old clips for this source
        conn.execute("DELETE FROM word_clips WHERE source_id = ?", (source["id"],))
        # Insert new clips
        conn.executemany(
            "INSERT INTO word_clips (source_id, word, start_time, end_time, source_file) "
            "VALUES (?, ?, ?, ?, ?)",
            [(source["id"], w["word"], w["start"], w["end"], stored) for w in words],
        )
        new = [dict(row) for row in conn.execute(
            "SELECT id,word,start_time,end_time FROM word_clips WHERE source_id=?", (source["id"],))]
        new_by_word: dict[str, list[dict]] = {}
        for row in new:
            new_by_word.setdefault(row["word"], []).append(row)
        remapped = released = 0
        for row in conn.execute("SELECT word,recipe FROM splice_recipes").fetchall():
            try:
                groups = json.loads(row["recipe"])
            except (ValueError, TypeError):
                continue
            if not isinstance(groups, list):
                continue
            changed = False
            for group in groups:
                if (not isinstance(group, dict) or not isinstance(group.get("clip_id"), int)
                        or group["clip_id"] not in old):
                    continue
                before = old[group["clip_id"]]
                candidates = new_by_word.get(before[1], [])
                if group.get("start") is not None and group.get("end") is not None:
                    if not isinstance(group["start"], (float, int)) or not isinstance(group["end"], (float, int)):
                        continue
                    candidates = [c for c in candidates
                                  if c["start_time"] - .001 <= group["start"] < group["end"]
                                  <= c["end_time"] + .001]
                else:
                    candidates = [c for c in candidates
                                  if min(c["end_time"], before[3]) > max(c["start_time"], before[2])]
                if len(candidates) == 1:
                    group["clip_id"] = candidates[0]["id"]
                    remapped += 1
                else:
                    for field in ("clip_id", "start", "end"):
                        group.pop(field, None)
                    released += 1
                changed = True
            if changed:
                conn.execute("UPDATE splice_recipes SET recipe=? WHERE word=?",
                             (json.dumps(groups), row["word"]))

    # The persistent metadata was removed in the same transaction. Runtime
    # invalidation scans each cache once, instead of once per replaced clip.
    from app import clip_cache, forced_align
    from app.generate import invalidate_cache
    clip_cache.invalidate_many(old, persisted=False)
    forced_align.invalidate_many(old, persisted=False)
    invalidate_cache(alignments=False)
    if remapped or released:
        print(f"  Splice pins: {remapped} remapped, {released} released to automatic selection")

    return len(words)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-align word timestamps using stable-whisper."
    )
    parser.add_argument("--model", default="base",
                        choices=MODEL_CHOICES,
                        help="Whisper model size (default: base)")
    parser.add_argument("--source-id", type=int, default=None,
                        help="Only re-align this DB source id")
    parser.add_argument("--start-from-id", type=int, default=None,
                        metavar="N", help="Skip sources with id < N")
    parser.add_argument("--end-at-id", type=int, default=None,
                        metavar="N", help="Stop after source id N (inclusive)")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Continue if one source fails")
    parser.add_argument("--replace-edited", action="store_true",
                        help="Explicitly replace hand-edited clips during retranscription")
    from app.device import add_argument as _device_arg
    _device_arg(parser)
    add_transcription_arguments(parser)
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
            n = realign_source(s, args.model, args.device, args.language, args.initial_prompt,
                               replace_edited=args.replace_edited)
            print(f"  OK stored {n} clips")
            ok += 1
        except Exception as exc:
            print(f"  FAIL: {exc}")
            if not args.skip_errors:
                sys.exit(1)

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
