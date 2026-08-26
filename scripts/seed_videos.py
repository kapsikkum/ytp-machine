#!/usr/bin/env python3
"""
Seed the database with a curated list of Michael Rosen YouTube videos.

Run:
    python scripts/seed_videos.py
    python scripts/seed_videos.py --model small   # better accuracy
    python scripts/seed_videos.py --skip-errors   # continue on failure
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest import ingest

# ── Curated Michael Rosen videos ──────────────────────────────────────────────
# All from his official channel (MichaelRosenOfficial) or well-known uploads.
# Verify URLs are still live before a first run.
SEED_VIDEOS: list[dict] = [
    {
        "url": "https://www.youtube.com/watch?v=z1cfVQyrQ3Q",
        "note": "User-added video",
    },
    {
        "url": "https://www.youtube.com/watch?v=wCOzQpRSFgg",
        "note": "Chocolate Cake — arguably his most iconic poem",
    },
    {
        "url": "https://www.youtube.com/watch?v=HVTnqIzO6Sc",
        "note": "No Breathing in Class",
    },
    {
        "url": "https://www.youtube.com/watch?v=5GZdQvbg4A8",
        "note": "The Bakerloo Line / Michael Rosen on the tube",
    },
    {
        "url": "https://www.youtube.com/watch?v=rJKMzLPnUvM",
        "note": "Fluff — classic everyday-life poem",
    },
    {
        "url": "https://www.youtube.com/watch?v=L9TtSqG0U4g",
        "note": "Michael Rosen performs at the Edinburgh Book Festival",
    },
    {
        "url": "https://www.youtube.com/watch?v=FgMdyou9aKA",
        "note": "Quick, Let's Get Out of Here",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the Michael Rosen Says database with curated videos."
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue seeding if one video fails",
    )
    args = parser.parse_args()

    n = len(SEED_VIDEOS)
    print(f"Seeding {n} Michael Rosen videos  (Whisper model: {args.model})\n")

    ok = 0
    for i, entry in enumerate(SEED_VIDEOS, 1):
        url = entry["url"]
        note = entry["note"]
        print(f"[{i}/{n}] {note}")
        print(f"      {url}")
        try:
            ingest(url, model_name=args.model)
            ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if not args.skip_errors:
                sys.exit(1)
        print()

    print(f"Seeding complete: {ok}/{n} videos ingested successfully.")


if __name__ == "__main__":
    main()
