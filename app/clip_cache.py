"""Pre-rendered word clip cache.

Frequently extracted word clips are cached as short MPEG-TS segments so the
generator can skip FFmpeg decode+filter for repeated words with no dynamic
modifiers (pitch, speed, glitch, reverse, stretch).

The cache is stored on disk under the corpus's cache/clips/ directory and
tracked in the clip_cache table so invalidation on clip edits is reliable.
"""

import logging
import os
import subprocess
import threading

from app.database import active, get_db, DATA_DIR

log = logging.getLogger(__name__)

MAX_CACHE_MB = int(os.environ.get("MRS_CLIP_CACHE_MB", "500"))
_lock = threading.Lock()


def _cache_dir() -> str:
    d = os.path.join(active()["dir"], "cache", "clips")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(clip_id: int, pad_end: float) -> str:
    # Quantize pad_end to 3 decimal places for key stability
    return os.path.join(_cache_dir(), f"{clip_id}_{pad_end:.3f}.ts")


def get_cached(clip_id: int | None, pad_end: float) -> str | None:
    """Return path to a cached clip segment, or None if not cached."""
    if clip_id is None:
        return None
    path = _cache_path(clip_id, pad_end)
    if os.path.exists(path):
        return path
    return None


def cache_clip(clip_id: int, pad_end: float, source_file: str,
               start: float, duration: float) -> str | None:
    """Extract and cache a clip segment. Returns path or None on failure."""
    if clip_id is None:
        return None
    
    path = _cache_path(clip_id, pad_end)
    if os.path.exists(path):
        return path
    
    with _lock:
        # Double-check after acquiring lock
        if os.path.exists(path):
            return path
        
        tmp = path + ".tmp"
        try:
            from app.database import resolve_path
            src = source_file if os.path.isabs(source_file) else resolve_path(source_file)
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.4f}", "-t", f"{duration:.4f}",
                "-i", src,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "44100", "-ac", "2",
                "-f", "mpegts", tmp,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                log.warning("clip cache extract failed for clip %d: %s",
                            clip_id, result.stderr[-200:])
                if os.path.exists(tmp):
                    os.remove(tmp)
                return None
            
            os.replace(tmp, path)
            
            # Track in database
            try:
                with get_db() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO clip_cache "
                        "(clip_id, pad_end, cache_file) VALUES (?, ?, ?)",
                        (clip_id, round(pad_end, 3), os.path.basename(path))
                    )
            except Exception:
                pass
            
            log.debug("Cached clip %d (pad=%.3f) -> %s", clip_id, pad_end, path)
            return path
            
        except Exception as exc:
            log.warning("clip cache failed for clip %d: %s", clip_id, exc)
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return None


def invalidate(clip_id: int | None = None) -> None:
    """Drop cached segments for one clip (after edit) or all clips."""
    try:
        with get_db() as conn:
            if clip_id is None:
                rows = conn.execute("SELECT cache_file FROM clip_cache").fetchall()
                conn.execute("DELETE FROM clip_cache")
            else:
                rows = conn.execute(
                    "SELECT cache_file FROM clip_cache WHERE clip_id=?",
                    (clip_id,)
                ).fetchall()
                conn.execute("DELETE FROM clip_cache WHERE clip_id=?", (clip_id,))
        
        cache_dir = _cache_dir()
        for row in rows:
            fname = row["cache_file"] if isinstance(row, dict) else row[0]
            path = os.path.join(cache_dir, fname)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
    except Exception:
        pass


def evict_lru(max_mb: int | None = None) -> int:
    """Delete oldest cached clips until total size < max_mb. Returns bytes freed."""
    if max_mb is None:
        max_mb = MAX_CACHE_MB
    max_bytes = max_mb * 1024 * 1024
    cache_dir = _cache_dir()
    
    try:
        files = []
        for name in os.listdir(cache_dir):
            if name.endswith(".ts"):
                path = os.path.join(cache_dir, name)
                try:
                    stat = os.stat(path)
                    files.append((path, stat.st_mtime, stat.st_size))
                except OSError:
                    pass
        
        total = sum(s for _, _, s in files)
        if total <= max_bytes:
            return 0
        
        # Sort oldest first
        files.sort(key=lambda x: x[1])
        freed = 0
        for path, _mtime, size in files:
            if total - freed <= max_bytes:
                break
            try:
                os.remove(path)
                freed += size
                # Extract clip_id from filename for DB cleanup
                basename = os.path.basename(path)
                parts = basename.rsplit(".", 1)[0].split("_", 1)
                if len(parts) == 2:
                    clip_id = int(parts[0])
                    pad_end = float(parts[1])
                    with get_db() as conn:
                        conn.execute(
                            "DELETE FROM clip_cache WHERE clip_id=? AND pad_end=?",
                            (clip_id, pad_end)
                        )
            except (OSError, ValueError, Exception):
                pass
        
        return freed
    except Exception:
        return 0


def cache_size_mb() -> float:
    """Current total size of the clip cache in MB."""
    cache_dir = _cache_dir()
    total = 0
    try:
        for name in os.listdir(cache_dir):
            if name.endswith(".ts"):
                try:
                    total += os.path.getsize(os.path.join(cache_dir, name))
                except OSError:
                    pass
    except OSError:
        pass
    return total / (1024 * 1024)
