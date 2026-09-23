"""Re-transcode corpus source videos with a different GOP structure.

Usage:
    python scripts/retranscode.py [--corpus SLUG] [--gop N]

Re-encodes every .mp4 in the corpus's downloads/ directory with the specified
keyframe interval (GOP). Denser keyframes make FFmpeg seeking faster at the
cost of slightly larger files.

  --gop 25   one keyframe per second (default, current corpus format)
  --gop 12   one keyframe per ~0.48s (~1.3x file size, ~2x faster seeking)
  --gop 1    all-intra (~2.5x file size, instant seeking)
"""
import argparse
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.database import active, set_setting, list_corpora, init_db
from scripts.ingest import CORPUS_WIDTH, CORPUS_HEIGHT, CORPUS_FPS, CORPUS_CRF, CORPUS_AUDIO_BITRATE


def retranscode(corpus_dir: str, gop: int) -> dict:
    downloads = os.path.join(corpus_dir, "downloads")
    if not os.path.isdir(downloads):
        print(f"No downloads/ directory in {corpus_dir}")
        return {"processed": 0}
    
    files = sorted(glob.glob(os.path.join(downloads, "*.mp4")))
    total_before = 0
    total_after = 0
    processed = 0
    
    for i, path in enumerate(files, 1):
        before = os.path.getsize(path)
        total_before += before
        tmp = path + ".retranscode.tmp.mp4"
        
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path,
            "-vf", f"scale={CORPUS_WIDTH}:{CORPUS_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={CORPUS_WIDTH}:{CORPUS_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
                   f"fps={CORPUS_FPS},setsar=1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", CORPUS_CRF,
            "-g", str(gop), "-keyint_min", str(gop),
            "-c:a", "aac", "-b:a", CORPUS_AUDIO_BITRATE, "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart", tmp,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(tmp):
            print(f"  [{i}/{len(files)}] FAILED {os.path.basename(path)}: {result.stderr.strip()[-200:]}")
            if os.path.exists(tmp):
                os.remove(tmp)
            continue
        
        os.replace(tmp, path)
        after = os.path.getsize(path)
        total_after += after
        processed += 1
        ratio = after / before if before else 1.0
        print(f"  [{i}/{len(files)}] {os.path.basename(path)}  "
              f"{before/1e6:.1f} MB -> {after/1e6:.1f} MB  ({ratio:.2f}x)")
    
    print(f"\nDone: {processed}/{len(files)} files re-transcoded")
    if total_before:
        print(f"Total: {total_before/1e6:.0f} MB -> {total_after/1e6:.0f} MB  "
              f"({total_after/total_before:.2f}x)")
    
    return {"processed": processed, "total": len(files),
            "before_mb": round(total_before / 1e6, 1),
            "after_mb": round(total_after / 1e6, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", help="Corpus slug (default: active corpus)")
    parser.add_argument("--gop", type=int, default=12,
                        help="Keyframe interval in frames (default: 12, all-intra: 1)")
    args = parser.parse_args()
    
    if args.corpus:
        os.environ["MRS_CORPUS"] = args.corpus
    
    corpus = active()
    init_db()
    print(f"Re-transcoding {corpus['slug']} with GOP={args.gop}")
    print(f"  Directory: {corpus['dir']}")
    
    retranscode(corpus["dir"], args.gop)
    set_setting("corpus_gop", str(args.gop))
    print(f"\nStored corpus_gop={args.gop} in settings.")


if __name__ == "__main__":
    main()
