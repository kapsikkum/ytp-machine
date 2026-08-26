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

# A redirected stdout on Windows encodes as the locale codepage (cp1252), not
# UTF-8, and this script prints box rules and whatever titles a channel happens
# to use. Piping an ingest to a log file -- the obvious thing to do with a run
# that takes hours -- therefore died on the first character outside Latin-1,
# before a single video had been fetched.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

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

    entries = [e for e in raw if e and e.get("id")]
    all_ids = [e["id"] for e in entries]
    print(f"Found {len(all_ids)} videos in channel listing.")

    if not all_ids:
        return []

    if year is None:
        # The flat listing already carries titles and durations, so use them.
        # Discarding them meant every progress line of a long unfiltered run
        # printed an empty title, which is precisely the run where you need to
        # know which video is being worked on.
        return [
            {
                "id": e["id"],
                "title": e.get("title") or e["id"],
                "upload_date": "",
                "duration": e.get("duration"),
                "url": f"https://www.youtube.com/watch?v={e['id']}",
            }
            for e in entries
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
    parser.add_argument("--download-dir", default=None, metavar="DIR",
                        help="Directory for downloaded videos (default: "
                             "downloads/ inside the active corpus)")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Ingest at most N videos. Vocabulary saturates long before a big "
             "channel runs out, so this is usually what you want.",
    )
    parser.add_argument(
        "--skip", type=int, default=0, metavar="N",
        help="Skip the first N videos of the listing, to sample a different "
             "part of the channel. Not needed to resume -- videos already in "
             "the corpus are skipped anyway.",
    )
    parser.add_argument(
        "--reingest", action="store_true",
        help="Ingest videos even if they are already in this corpus. Adds a "
             "second copy of their clips rather than replacing them.",
    )
    parser.add_argument(
        "--max-height", type=int, default=None, metavar="PX",
        help="Cap the rendition downloaded (e.g. 480).",
    )
    parser.add_argument(
        "--normalise", "--normalize", action="store_true", dest="normalise",
        help="Re-encode each video to the corpus format after download.",
    )
    from app.device import add_argument as _device_arg
    _device_arg(parser)
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

    total_found = len(videos)

    # Drop anything already in this corpus before counting off skip/limit.
    #
    # persist() inserts a source unconditionally, so ingesting a video twice
    # does not update it -- it adds a second copy of every word clip, and the
    # corpus quietly doubles the weight of that speaker's material. Counting
    # positions in the channel listing is no defence either: the listing is
    # newest-first, so a single upload shifts every index by one and "resume
    # from 10" silently re-ingests one and skips another.
    if not args.reingest:
        from app.database import init_db, get_db
        init_db()
        with get_db() as conn:
            known = {r[0] for r in conn.execute("SELECT video_id FROM sources")}
        if known:
            before = len(videos)
            videos = [v for v in videos if v["id"] not in known]
            if before != len(videos):
                print(f"\nSkipping {before - len(videos)} video(s) already in "
                      f"this corpus (pass --reingest to force).")

    if args.skip:
        videos = videos[args.skip:]
    if args.limit is not None:
        videos = videos[:args.limit]
    if not videos:
        sys.exit(f"--skip {args.skip} leaves nothing of {total_found} videos.")
    if len(videos) != total_found:
        print(f"\nSelected {len(videos)} of {total_found} videos "
              f"(skip={args.skip}, limit={args.limit}).")

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
                device=args.device,
                max_height=args.max_height,
                normalise=args.normalise,
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
