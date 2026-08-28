#!/usr/bin/env python3
"""Build a whole corpus from a channel link, in one command.

    python scripts/build_corpus.py https://www.youtube.com/@GarbageTime420 --limit 40

Downloads, transcribes, sharpens the word boundaries, and packs the result into
a bundle ready to drop on a server. It is a wrapper around the steps in the
README -- ingest_channel.py, refine_boundaries.py, corpus.py pack -- and does
nothing they cannot, but running them by hand means naming the corpus and the
data directory correctly in three separate places, and getting one wrong is not
noticed until an ingest has been running for two hours.

The stages are separate processes on purpose. app/database.py reads
$MRS_DATA_DIR and $MRS_CORPUS once, at import, so a single process cannot build
a corpus other than the one it was started against; setting the environment for
each child is the honest way to point the work at a named corpus.

Everything is resumable. Videos already in the corpus are skipped, so a run
interrupted at video 25 of 40 picks up where it stopped, and finished stages
can be left out with --skip-ingest / --skip-refine / --skip-pack.

Not included: correct.py (needs a real transcript to compare against) and
find_noises.py (curated by hand, per speaker). Neither can be driven from a
channel link alone. Nor is realign.py -- the ingest already transcribes with
the model you asked for; realign exists for changing the model afterwards.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A redirected stdout on Windows encodes as the locale codepage, and both the
# rules printed below and any channel's video titles leave Latin-1 immediately.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A channel URL with no tab on the end resolves to the channel home page, which
# yt-dlp reads as a mixture of tabs -- shorts and livestreams included. Neither
# makes good corpus material: shorts are loud and heavily edited, and streams
# are hours of multi-speaker audio. /videos is the tab we actually want.
_TABS = ("videos", "streams", "shorts", "playlists", "featured", "community")

# A single video is a perfectly good corpus -- one long monologue can carry more
# distinct words than a dozen short uploads -- so it is worth taking the same
# route as a channel rather than sending people back to ingest.py.
_VIDEO_ID = r"[A-Za-z0-9_-]{11}"
_VIDEO_RE = re.compile(rf"(?:watch\?v=|youtu\.be/|/shorts/|/live/)({_VIDEO_ID})")


def _video_id(raw: str) -> str | None:
    """The video id, if this names one video rather than a channel."""
    raw = raw.strip()
    m = _VIDEO_RE.search(raw)
    if m:
        return m.group(1)
    # A bare id, pasted on its own. Without this it becomes https://<id>/videos.
    if "/" not in raw and re.fullmatch(_VIDEO_ID, raw):
        return raw
    return None


def _channel_url(raw: str) -> str:
    """Accept @handle, a bare channel URL, or a full one, and aim it at /videos."""
    url = raw.strip()
    if _video_id(url):
        # Already points at one video; adding a tab to it would ask YouTube for
        # a channel that does not exist.
        return url if url.startswith("http") else f"https://www.youtube.com/watch?v={url}"
    if url.startswith("@"):
        url = f"https://www.youtube.com/{url}"
    elif not url.startswith("http"):
        url = f"https://{url}"
    if "list=" in url:
        return url                      # a playlist has no tabs to choose from
    tail = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
    return url if tail in _TABS else url.rstrip("/") + "/videos"


def _slug_from(url: str) -> str | None:
    """A corpus name derived from the channel handle, or None if it is opaque."""
    m = re.search(r"/@([^/?#]+)", url) or re.search(r"/(?:c|user)/([^/?#]+)", url)
    if not m:
        # /channel/UC... is a 24-character random id. Splitting that up gives a
        # corpus called "u-cx-2-lz-a", and nobody wants that on a menu.
        return None
    handle = m.group(1)
    handle = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", handle)   # GarbageTime -> Garbage-Time
    handle = re.sub(r"(?<=[A-Za-z])(?=[0-9])", "-", handle)   # Time420     -> Time-420
    return re.sub(r"[^A-Za-z0-9]+", "-", handle).strip("-").lower() or None


def _preflight(device: str | None) -> None:
    """Fail in the first ten seconds rather than the third hour.

    Every one of these has actually bitten: a missing zstandard turned a
    finished four-hour ingest into an unpackable one, and torch installed
    inside a OneDrive folder imports fine until the sync engine locks a DLL.
    The device check is a real tensor round-trip for the same reason.
    """
    import importlib.util

    missing = [m for m in ("yt_dlp", "whisper", "stable_whisper", "torch",
                           "torchaudio", "num2words", "nltk", "zstandard")
               if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit("Missing Python packages: " + ", ".join(missing) +
                 "\n  pip install -r requirements.txt"
                 "\n  (zstandard is what packs the bundle at the very end.)")

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} is not on PATH. Every clip is cut with it.")

    from app.device import describe, resolve
    print(f"  device:  {describe(resolve(device))}")


def _run(stage: str, argv: list[str], env: dict) -> None:
    """One stage, streaming its output straight through to ours."""
    print(f"\n{'=' * 64}\n{stage}\n{'=' * 64}", flush=True)
    started = time.time()
    proc = subprocess.run([sys.executable, "-u"] + argv, env=env, cwd=PROJECT_ROOT)
    mins = (time.time() - started) / 60
    if proc.returncode != 0:
        sys.exit(f"\n{stage} failed (exit {proc.returncode}) after {mins:.0f} min.\n"
                 f"Nothing already done is lost -- re-run the same command and it "
                 f"resumes, or skip the finished stages with --skip-ingest etc.")
    print(f"\n{stage}: done in {mins:.0f} min.", flush=True)


def _summarise(db_path: str) -> None:
    import sqlite3
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sources = conn.execute("SELECT count(*) FROM sources").fetchone()[0]
        clips = conn.execute("SELECT count(*) FROM word_clips").fetchone()[0]
        words = conn.execute("SELECT count(DISTINCT word) FROM word_clips").fetchone()[0]
    finally:
        conn.close()
    print(f"  {sources} sources, {clips:,} clips, {words:,} distinct words")
    # Distinct words is the number that predicts how it will sound. Anything
    # the corpus has never heard has to be spliced together from phonemes,
    # which is audibly worse than a real recording of the word.
    if words < 2000:
        print("  (under ~2,000 distinct words a lot of lines need splicing; "
              "raise --limit for more material)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Channel link in, packed corpus out.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n"
               "  python scripts/build_corpus.py @GarbageTime420 --limit 40",
    )
    ap.add_argument("channel", help="Channel URL, or just @handle")
    ap.add_argument("--name", metavar="SLUG",
                    help="What to call the corpus (default: from the channel handle)")
    ap.add_argument("--limit", type=int, default=40, metavar="N",
                    help="Maximum videos to ingest (default 40, 0 for the whole "
                         "channel). Vocabulary saturates long before a big "
                         "channel runs out, and every video costs disk forever.")
    ap.add_argument("--model", default="medium",
                    choices=["tiny", "base", "small", "medium", "large"],
                    help="Whisper model (default medium). Every later stage "
                         "inherits these labels, so it is the setting worth "
                         "spending time on.")
    ap.add_argument("--year", type=int, default=None,
                    help="Only ingest videos uploaded in this year")
    ap.add_argument("--data-dir", default=None, metavar="DIR",
                    help="Where corpora/ lives (default $MRS_DATA_DIR, else the "
                         "project root). On Windows, keep this out of OneDrive.")
    ap.add_argument("--out", default=None, metavar="FILE",
                    help="Bundle path (default <name>.tar.zst in the data dir)")
    ap.add_argument("--max-height", type=int, default=480, metavar="PX",
                    help="Cap the rendition downloaded (default 480; the corpus "
                         "format is 480x270, so more is wasted bandwidth)")
    ap.add_argument("--no-normalise", "--no-normalize", dest="normalise",
                    action="store_false", default=True,
                    help="Keep the downloaded encodes as they are. The default "
                         "re-encodes to the corpus format, which is both smaller "
                         "and much cheaper to cut from.")
    ap.add_argument("--cookies-from-browser", default=None, metavar="BROWSER",
                    help="Pass cookies from this browser when YouTube demands "
                         "that you sign in to prove you are not a bot "
                         "(chrome, firefox, edge, …)")
    ap.add_argument("--skip-ingest", action="store_true", help="Already ingested")
    ap.add_argument("--skip-refine", action="store_true", help="Leave boundaries alone")
    ap.add_argument("--skip-pack", action="store_true", help="Stop before the bundle")
    from app.device import add_argument as _device_arg
    _device_arg(ap)
    args = ap.parse_args()

    channel = _channel_url(args.channel)
    one_video = _video_id(channel)
    slug = args.name or _slug_from(channel)
    if not slug:
        sys.exit(f"Cannot derive a corpus name from {channel} -- pass --name."
                 + (" A video URL carries no channel handle to name it after."
                    if one_video else ""))
    slug = re.sub(r"[^a-z0-9_-]+", "-", slug.lower()).strip("-")

    data_dir = os.path.abspath(
        args.data_dir or os.environ.get("MRS_DATA_DIR") or PROJECT_ROOT)
    corpus_dir = os.path.join(data_dir, "corpora", slug)
    db_path = os.path.join(corpus_dir, "corpus.db")
    out = os.path.abspath(args.out or os.path.join(data_dir, f"{slug}.tar.zst"))

    print(f"corpus:  {slug}")
    print(f"{'video:  ' if one_video else 'channel:'} {channel}")
    print(f"into:    {corpus_dir}")
    print(f"bundle:  {out}")
    print(f"model:   {args.model}"
          + ("" if one_video else f"   limit: {args.limit or 'all'}")
          + f"   normalise: {'yes' if args.normalise else 'no'}")
    _preflight(args.device)

    env = dict(os.environ)
    env["MRS_DATA_DIR"] = data_dir
    env["MRS_CORPUS"] = slug
    env["PYTHONIOENCODING"] = "utf-8"
    # An unbuffered child is the difference between a log you can tail and one
    # that sits empty for twenty minutes while a video transcribes.
    env["PYTHONUNBUFFERED"] = "1"
    if args.device:
        env["MRS_DEVICE"] = args.device

    started = time.time()

    if not args.skip_ingest:
        if one_video:
            # ingest.py takes the URL directly. Going through ingest_channel
            # for one video would ask YouTube for a channel listing that does
            # not exist, and --limit and --year have nothing to select from.
            argv = ["scripts/ingest.py", channel, "--model", args.model]
        else:
            argv = ["scripts/ingest_channel.py", channel,
                    "--model", args.model, "--skip-errors"]
            if args.limit:
                argv += ["--limit", str(args.limit)]
            if args.year:
                argv += ["--year", str(args.year)]
        if args.max_height:
            argv += ["--max-height", str(args.max_height)]
        if args.normalise:
            argv += ["--normalise"]
        if args.cookies_from_browser:
            argv += ["--cookies-from-browser", args.cookies_from_browser]
        if args.device:
            argv += ["--device", args.device]
        _run("1/3  Download and transcribe", argv, env)

    if not args.skip_refine:
        # Whisper's boundaries run early and wander by 100-350ms, which is
        # enough for a short clip to open on the tail of the previous word.
        # This is the step that decides whether a splice sounds like speech.
        argv = ["scripts/refine_boundaries.py", "--all", "--apply"]
        if args.device:
            argv += ["--device", args.device]
        _run("2/3  Sharpen word boundaries", argv, env)

    if not args.skip_ingest or not args.skip_refine:
        # Start the corpus's dictionary before packing, so the bundle carries
        # it. A new corpus always contains words no dictionary has -- names,
        # coinages, the speaker's own vocabulary -- and they are unsayable
        # until somebody says how. Left commented out: a wrong pronunciation
        # is worse than none, because the word then gets said confidently
        # instead of reported missing.
        _run("      Note the words it cannot say",
             ["scripts/verify_corpus.py", "--unspliceable", "--write-template",
              "--show", "12"], env)

    if not args.skip_pack:
        _run("3/3  Pack the bundle",
             ["scripts/corpus.py", "pack", "--corpus", slug, "--out", out], env)

    print(f"\n{'=' * 64}\nBuilt '{slug}' in {(time.time() - started) / 60:.0f} min.")
    _summarise(db_path)
    if not args.skip_pack and os.path.exists(out):
        print(f"  bundle: {out}  ({os.path.getsize(out) / 1e6:.0f} MB)")
        csv = os.path.join(corpus_dir, "pronunciations.csv")
        if os.path.exists(csv):
            print(f"  words it cannot say are listed in {csv}\n"
                  f"  (commented out -- uncomment the ones worth teaching it)")
        print("\nInstall it wherever the server runs:")
        print(f"  python scripts/corpus.py install {os.path.basename(out)} --name {slug}")
        print("  curl -X POST localhost:8765/api/reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
