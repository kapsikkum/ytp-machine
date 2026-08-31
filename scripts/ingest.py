#!/usr/bin/env python3
"""
Ingest a YouTube URL or local video file into the Michael Rosen Says database.

Usage:
    python scripts/ingest.py <url_or_file>
    python scripts/ingest.py https://www.youtube.com/watch?v=XXXXXXXXXXX
    python scripts/ingest.py /path/to/video.mp4
    python scripts/ingest.py /path/to/video.mp4 --model medium
"""
import argparse
import os
import re
import subprocess
import sys

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# See the note in ingest_channel.py: a redirected stdout on Windows is cp1252,
# and this prints transcribed words, which are not promised to be Latin-1.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

from app.database import init_db, get_db


# ── Download ──────────────────────────────────────────────────────────────────

def _ydl_base_opts(cookies_from_browser: str | None) -> dict:
    """Return base yt-dlp options including JS runtime and optional cookies."""
    opts: dict = {
        # Node.js must be in PATH; EJS solver is cached after first --remote-components run
        "js_runtimes": {"node": {}},
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


def download_video(
    url: str, download_dir: str, cookies_from_browser: str | None = None,
    max_height: int | None = None,
) -> tuple[str, str, str]:
    """Return (absolute_file_path, video_id, title).

    *max_height* caps the rendition fetched. The corpus is stored at 480x270
    (see normalise_video), so pulling a 1080p master only to throw the pixels
    away costs bandwidth and a slower transcode for nothing.
    """
    import yt_dlp

    if max_height:
        fmt = (f"bestvideo[ext=mp4][height<={max_height}]+bestaudio[ext=m4a]/"
               f"best[ext=mp4][height<={max_height}]/"
               f"best[height<={max_height}]/best[ext=mp4]/best")
    else:
        fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

    ydl_opts = {
        **_ydl_base_opts(cookies_from_browser),
        "outtmpl": os.path.join(download_dir, "%(id)s.%(ext)s"),
        "format": fmt,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id: str = info["id"]
        title: str = info.get("title", video_id)
        candidate = ydl.prepare_filename(info)
        if not os.path.exists(candidate):
            candidate = os.path.splitext(candidate)[0] + ".mp4"
        file_path = os.path.abspath(candidate)

    return file_path, video_id, title


# ── Normalisation ─────────────────────────────────────────────────────────────

# The shape every corpus video is stored in. app/generate.py puts every clip
# through scale=480:270 unconditionally, so this is the format the generator
# wants anyway -- storing it is just doing that work once at ingest instead of
# on every clip of every generation.
CORPUS_WIDTH, CORPUS_HEIGHT, CORPUS_FPS = 480, 270, 25

# A keyframe every second. Cuts are word-length, and -ss seeks to a keyframe
# and decodes forward from there; with the sparse GOPs a 13-minute YouTube
# upload ships with, that forward decode is most of the cost of extracting a
# half-second clip. Denser keyframes cost a little size and make every
# subsequent cut cheap.
CORPUS_GOP = 25

# Quality point for the stored video. The picture is a thumbnail and the audio
# is the entire point of the project, so the video is compressed hard and the
# audio is not.
#
# libx264 rather than the GPU encoder, despite CUDA being right there: measured
# on a 9-minute source, NVENC at comparable quality produced 48 MB against
# libx264's 21 MB. Encoding is not the bottleneck -- transcription is, and it
# has the GPU -- so the smaller file wins, and a corpus that has to be shipped
# to a server is worth less than half the bytes.
CORPUS_CRF = "28"
CORPUS_AUDIO_BITRATE = "96k"


def normalise_video(path: str) -> str:
    """Re-encode in place to the corpus format. Returns the path.

    Two reasons this is worth a pass. Size: a corpus travels as one bundle and
    ships to a server, and 480x270 is roughly a tenth of the 480p rendition
    YouTube serves. Memory: the generator hands ffmpeg one input per clip, up
    to twenty per call, and every one of those is a live decoder -- at 1080p
    that many decoders is what put the container over its memory cap before.
    Storing small keeps the encode budget the one it was tuned against.
    """
    tmp = os.path.splitext(path)[0] + ".norm.mp4"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path,
        "-vf", f"scale={CORPUS_WIDTH}:{CORPUS_HEIGHT}:force_original_aspect_ratio=decrease,"
               f"pad={CORPUS_WIDTH}:{CORPUS_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
               f"fps={CORPUS_FPS},setsar=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", CORPUS_CRF,
        "-g", str(CORPUS_GOP), "-keyint_min", str(CORPUS_GOP),
        "-c:a", "aac", "-b:a", CORPUS_AUDIO_BITRATE, "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart", tmp,
    ]
    before = os.path.getsize(path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(tmp):
        # Not fatal: an un-normalised video still transcribes and still cuts,
        # it is just bigger and slower. Better a working corpus than none.
        print(f"  WARNING: normalise failed, keeping the original\n"
              f"    {(result.stderr or '').strip()[-300:]}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return path

    os.replace(tmp, path)
    after = os.path.getsize(path)
    print(f"  Normalised to {CORPUS_WIDTH}x{CORPUS_HEIGHT} {CORPUS_FPS}fps "
          f"({before / 1e6:.0f} MB -> {after / 1e6:.0f} MB)")
    return path


# ── Transcription ─────────────────────────────────────────────────────────────

# The loaded model, kept between calls. Both batch callers -- ingest_channel
# and realign -- transcribe one source per call, and loading a fresh model each
# time put a second copy of it on the GPU before the first was collected.
# `medium` is about 6 GB against 8 GB of VRAM, so what actually happened was
# that it survived seven sources on leftover fragments and then failed three in
# a row at load time with a bare "CUDA error: out of memory". Holding one model
# also saves the twenty-odd seconds it takes to load, per video.
_model_cache: dict[tuple[str, str], object] = {}


def _load_transcriber(model_name: str, dev: str):
    import stable_whisper
    from app.device import describe

    key = (model_name, dev)
    if key not in _model_cache:
        # Only one model at a time. Asking for a different size mid-run is
        # rare, but keeping both would reintroduce exactly the problem above.
        if _model_cache:
            _model_cache.clear()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"  Loading stable-whisper model '{model_name}' on {describe(dev)} …")
        _model_cache[key] = stable_whisper.load_model(model_name, device=dev)
    return _model_cache[key]


def transcribe(video_path: str, model_name: str = "base",
               device: str | None = None) -> list[dict]:
    """
    Transcribe with stable-whisper for accurate word-level timestamps.

    stable-ts refines Whisper's output using audio-energy curves and modified
    attention patterns, giving ~50ms accuracy vs ~200ms for plain Whisper.
    """
    from app.device import get as get_device

    dev = get_device(device)
    model = _load_transcriber(model_name, dev)

    print(f"  Transcribing {os.path.basename(video_path)} …")
    result = model.transcribe(video_path, word_timestamps=True)

    words: list[dict] = []
    for segment in result.segments:
        for wi in segment.words:
            clean = re.sub(r"[^\w]", "", wi.word).lower().strip()
            if clean:
                words.append({
                    "word":  clean,
                    "start": float(wi.start),
                    "end":   float(wi.end),
                })

    return words


def _transcribe_plain_whisper(video_path: str, model_name: str = "base") -> list[dict]:
    """Fallback using plain openai-whisper (kept for reference)."""
    import whisper

    print(f"  Loading Whisper model '{model_name}' …")
    model = whisper.load_model(model_name)

    print(f"  Transcribing {os.path.basename(video_path)} …")
    result = model.transcribe(video_path, word_timestamps=True)

    words: list[dict] = []
    for segment in result.get("segments", []):
        for wi in segment.get("words", []):
            clean = re.sub(r"[^\w]", "", wi["word"]).lower().strip()
            if clean:
                words.append({
                    "word": clean,
                    "start": float(wi["start"]),
                    "end": float(wi["end"]),
                })

    return words


# ── Database persistence ──────────────────────────────────────────────────────

def persist(
    video_id: str,
    source_file: str,
    title: str,
    url: str | None,
    words: list[dict],
) -> int:
    """Insert source + word clips into DB; return number of words stored."""
    from app.database import relativize_path
    source_file = relativize_path(source_file)   # store portable relative paths
    # relativize_path assumes the file is in the corpus's downloads/, because
    # that is where every path it is given is supposed to end up. Say so if it
    # is not: a corpus that cannot find its own video is worth hearing about
    # now, not on the first generation that tries to use it.
    from app.database import resolve_path
    if not os.path.exists(resolve_path(source_file)):
        print(f"  WARNING: stored as {source_file}, which does not exist. "
              f"The corpus will not be able to cut clips from it.")
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO sources (video_id, source_file, title, url) VALUES (?, ?, ?, ?)",
            (video_id, source_file, title, url),
        )
        source_id = cur.lastrowid

        conn.executemany(
            "INSERT INTO word_clips (source_id, word, start_time, end_time, source_file) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (source_id, w["word"], w["start"], w["end"], source_file)
                for w in words
            ],
        )

    return len(words)


# ── Main entry ────────────────────────────────────────────────────────────────

def _default_download_dir() -> str:
    """downloads/ inside whichever corpus is active.

    Not "downloads" relative to the working directory: with more than one
    corpus installed that path belongs to none of them in particular, and an
    ingest run from the wrong directory would file its videos where the
    database it just wrote could never find them.
    """
    from app.database import active
    return os.path.join(active()["dir"], "downloads")


def ingest(
    input_path: str,
    download_dir: str | None = None,
    model_name: str = "base",
    cookies_from_browser: str | None = None,
    device: str | None = None,
    max_height: int | None = None,
    normalise: bool = False,
    align: bool = True,
) -> None:
    init_db()
    download_dir = download_dir or _default_download_dir()
    os.makedirs(download_dir, exist_ok=True)

    is_url = input_path.startswith("http://") or input_path.startswith("https://")

    if is_url:
        print(f"Downloading: {input_path}")
        source_file, video_id, title = download_video(
            input_path, download_dir, cookies_from_browser, max_height
        )
        url: str | None = input_path
        if normalise:
            source_file = normalise_video(source_file)
    else:
        source_file = os.path.abspath(input_path)
        if not os.path.exists(source_file):
            sys.exit(f"File not found: {source_file}")
        video_id = os.path.splitext(os.path.basename(source_file))[0]
        title = video_id
        url = None
        # Into the corpus, always -- not only when normalising.
        #
        # persist() stores every path as downloads/<name> relative to the
        # corpus, so a file left where the user pointed at it is recorded at a
        # path nothing will ever find. Nothing notices at ingest time either:
        # transcription reads the original, and the corpus only turns out to
        # be broken later, when something tries to cut a clip out of it.
        #
        # (Copying is also what keeps normalise_video from rewriting the
        # user's own file in place.)
        import shutil
        copy = os.path.join(download_dir, f"{video_id}.mp4")
        if os.path.abspath(copy) != source_file:
            shutil.copy2(source_file, copy)
        source_file = os.path.abspath(copy)
        if normalise:
            source_file = normalise_video(source_file)

    print(f"Source file : {source_file}")

    words = transcribe(source_file, model_name, device)
    print(f"  Found {len(words)} word timestamps")

    count = persist(video_id, source_file, title, url, words)
    try:
        from app.generate import invalidate_cache
        invalidate_cache()
    except Exception:
        pass

    # Measure where each phoneme of each new clip falls.
    #
    # Here rather than only in build_corpus.py, because adding a video to an
    # existing corpus is the ordinary way this project grows, and a corpus
    # where some clips know where their sounds are and others do not is worse
    # than either: the ones that do not fall back to inferring cuts from the
    # spelling, which is wrong for a third of words and says nothing about it.
    #
    # Skipped where there is no aligner (the server installs torch from the
    # CPU wheel index and no transformers at all), and cheap when the clips
    # are already done -- it only looks at the ones with no times.
    if align:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from align_phones import align_corpus
        with get_db() as conn:
            row = conn.execute("SELECT id FROM sources WHERE video_id=?",
                               (video_id,)).fetchone()
        if row:
            print("  Placing phonemes …")
            align_corpus(source_id=row["id"], device=device)

    unique = sorted({w["word"] for w in words})
    sample = ", ".join(unique[:25]) + ("…" if len(unique) > 25 else "")
    print(f"  Stored {count} clips  ({len(unique)} unique words)")
    print(f"  Sample: {sample}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a YouTube URL or local video file into the Michael Rosen Says database."
    )
    parser.add_argument("input", help="YouTube URL or path to a local video file")
    parser.add_argument(
        "--download-dir",
        default=None,
        metavar="DIR",
        help="Directory for downloaded videos (default: downloads/ inside the "
             "active corpus)",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER",
        help="Pass cookies from this browser to bypass YouTube bot-detection "
             "(e.g. chrome, firefox, edge, chromium). "
             "The browser must be installed and logged in to YouTube.",
    )
    parser.add_argument(
        "--max-height", type=int, default=None, metavar="PX",
        help="Cap the rendition downloaded (e.g. 480). Pointless above the "
             "corpus format unless you are keeping the originals.",
    )
    parser.add_argument(
        "--no-align", dest="align", action="store_false",
        help="Skip measuring where each phoneme falls. Sub-word splices then "
             "infer it from the spelling, which disagrees with the phoneme "
             "count for a third of words.",
    )
    parser.add_argument(
        "--normalise", "--normalize", action="store_true", dest="normalise",
        help=f"Re-encode to the corpus format ({CORPUS_WIDTH}x{CORPUS_HEIGHT} "
             f"{CORPUS_FPS}fps) after download. Much smaller bundles and "
             f"cheaper generation; recommended for any new corpus.",
    )
    from app.device import add_argument as _device_arg
    _device_arg(parser)
    args = parser.parse_args()

    ingest(args.input, args.download_dir, args.model, args.cookies_from_browser,
           args.device, args.max_height, args.normalise, args.align)
    print("Done.")


if __name__ == "__main__":
    main()
