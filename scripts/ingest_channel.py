#!/usr/bin/env python3
"""
Batch-ingest all videos from a YouTube channel, optionally filtered by year.

Usage:
    python scripts/ingest_channel.py <channel_url> [--year YYYY] [--model MODEL] [--skip-errors]

Examples:
    python scripts/ingest_channel.py https://www.youtube.com/@MichaelRosenOfficial/videos --year 2008
    python scripts/ingest_channel.py https://www.youtube.com/@MichaelRosenOfficial/videos --year 2008 --model small
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest import ingest


def _fetch_date(vid_id: str, cookies_from_browser: str | None = None) -> dict | None:
    """Fetch upload_date for a single video ID. Returns None if unavailable."""
    import yt_dlp
    from scripts.ingest import _ydl_base_opts
    opts: dict = {
        **_ydl_base_opts(cookies_from_browser),
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={vid_id}", download=False
            )
        if not info:
            return None
        return {
            "id":          vid_id,
            "title":       info.get("title", vid_id),
            "upload_date": info.get("upload_date") or "",
            "url":         f"https://www.youtube.com/watch?v={vid_id}",
        }
    except Exception:
        return None


def list_channel_videos(
    channel_url: str,
    year: int | None = None,
    cookies_from_browser: str | None = None,
) -> list[dict]:
    """
    Return list of {id, title, upload_date, url} dicts, optionally filtered by year.

    YouTube's channel listing doesn't expose upload dates, so we:
      1. Fetch all video IDs via flat-playlist (fast, one request)
      2. Resolve dates sequentially — concurrent requests trigger bot-detection
    """
    import time
    import yt_dlp

    print(f"Fetching channel listing from: {channel_url}")
    from scripts.ingest import _ydl_base_opts
    flat_opts: dict = {
        **_ydl_base_opts(cookies_from_browser),
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(flat_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    if info is None:
        print("ERROR: yt-dlp returned nothing for that URL.")
        return []

    raw = info.get("entries") or []
    if raw and isinstance(raw[0], dict) and "entries" in raw[0]:
        raw = raw[0].get("entries") or []

    all_ids = [e["id"] for e in raw if e and e.get("id")]
    print(f"Found {len(all_ids)} videos in channel listing.")

    if not all_ids:
        return []

    if year is None:
        return [
            {
                "id": vid_id, "title": "", "upload_date": "",
                "url": f"https://www.youtube.com/watch?v={vid_id}",
            }
            for vid_id in all_ids
        ]

    # Need upload dates — sequential requests avoid bot-detection
    print(
        f"Resolving upload dates for {len(all_ids)} videos (year={year}).\n"
        f"Sequential to avoid rate-limiting — ~{len(all_ids) // 2} seconds …"
    )

    results: list[dict] = []
    for i, vid_id in enumerate(all_ids, 1):
        if i % 50 == 0 or i == len(all_ids):
            print(f"  {i}/{len(all_ids)} checked  ({len(results)} matches so far)…",
                  flush=True)
        data = _fetch_date(vid_id, cookies_from_browser)
        if data and data["upload_date"].startswith(str(year)):
            results.append(data)
        time.sleep(0.4)  # polite pause — avoids triggering YouTube's bot-detection

    results.sort(key=lambda x: x["upload_date"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-ingest all (or year-filtered) videos from a YouTube channel."
    )
    parser.add_argument("channel", help="YouTube channel URL (e.g. https://www.youtube.com/@Name/videos)")
    parser.add_argument("--year",  type=int, default=None, help="Only ingest videos uploaded in this year")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument("--skip-errors", action="store_true", help="Continue on per-video failures")
    parser.add_argument("--download-dir", default="downloads", metavar="DIR")
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER",
        help="Pass cookies from this browser to bypass YouTube bot-detection "
             "(e.g. chrome, firefox, edge, chromium). "
             "Required when YouTube demands 'Sign in to confirm you're not a bot'.",
    )
    args = parser.parse_args()

    videos = list_channel_videos(args.channel, args.year, args.cookies_from_browser)

    if not videos:
        sys.exit("No videos found matching the criteria.")

    print(f"\nFound {len(videos)} video(s):\n")
    for v in videos:
        safe_title = v['title'][:60].encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace')
        print(f"  {v['upload_date']}  {safe_title}")
        print(f"  {v['url']}\n")

    print(f"Starting ingestion  (Whisper model: {args.model})\n{'─'*60}")

    ok = 0
    for i, v in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {v['title']}")
        print(f"  {v['url']}")
        try:
            ingest(
                v["url"],
                download_dir=args.download_dir,
                model_name=args.model,
                cookies_from_browser=args.cookies_from_browser,
            )
            ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if not args.skip_errors:
                sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"Done: {ok}/{len(videos)} videos ingested successfully.")


if __name__ == "__main__":
    main()
