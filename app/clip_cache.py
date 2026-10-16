"""Pre-rendered word clip cache.

Frequently extracted word clips are cached as short MPEG-TS segments so the
generator can skip FFmpeg decode+filter for repeated words with no dynamic
modifiers (pitch, speed, glitch, reverse, stretch).

The cache is stored on disk under the corpus's cache/clips/ directory and
tracked in the clip_cache table so invalidation on clip edits is reliable.
"""

import logging
import hashlib
import json
import math
import os
import re
import subprocess
import threading

from app.database import active, get_db, resolve_path

log = logging.getLogger(__name__)

MAX_CACHE_MB = int(os.environ.get("MRS_CLIP_CACHE_MB", "500"))
_lock = threading.Lock()


def _cache_dir(create: bool = True) -> str:
    d = os.path.join(active()["dir"], "cache", "clips")
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def _fingerprint(source_file: str, start: float, duration: float) -> tuple[str, str]:
    """The actual source and FFmpeg window, including source replacements."""
    if not math.isfinite(start) or start < 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid cache extraction window")
    source = source_file if os.path.isabs(source_file) else resolve_path(source_file)
    source = os.path.normcase(os.path.realpath(source))
    stat = os.stat(source)
    body = json.dumps([source, stat.st_mtime_ns, stat.st_size,
                       f"{start:.4f}", f"{duration:.4f}"], ensure_ascii=True)
    return source, hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _cache_path(clip_id: int, pad_end: float, fingerprint: str, *, create: bool = True) -> str:
    # Id and pad remain readable for invalidation/eviction; the fingerprint
    # prevents a whole word, another voice or old source from supplying audio
    # for a different extraction with the same id.
    return os.path.join(_cache_dir(create), f"{clip_id}_{pad_end:.3f}_{fingerprint}.ts")


_CACHE_NAME = re.compile(r"(\d+)_(-?\d+(?:\.\d+)?)(?:_[0-9a-f]{24})\.ts(?:\.tmp)?$")
_LEGACY_NAME = re.compile(r"(\d+)_(-?\d+(?:\.\d+)?)\.ts(?:\.tmp)?$")


def _cache_key(name: str) -> tuple[int, float] | None:
    match = _CACHE_NAME.fullmatch(name) or _LEGACY_NAME.fullmatch(name)
    return (int(match[1]), float(match[2])) if match else None


def _remove(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def get_cached(clip_id: int | None, pad_end: float, source_file: str,
               start: float, duration: float) -> str | None:
    """Return only a cache matching this source and exact extraction window."""
    if clip_id is None:
        return None
    try:
        _source, fingerprint = _fingerprint(source_file, start, duration)
        path = _cache_path(clip_id, pad_end, fingerprint, create=False)
        if os.path.isfile(path):
            return path
    except (OSError, ValueError, TypeError):
        pass
    return None


def cache_clip(clip_id: int, pad_end: float, source_file: str,
               start: float, duration: float) -> str | None:
    """Extract and cache a clip segment. Returns path or None on failure."""
    if clip_id is None:
        return None
    try:
        source, fingerprint = _fingerprint(source_file, start, duration)
        path = _cache_path(clip_id, pad_end, fingerprint)
    except (OSError, ValueError, TypeError) as exc:
        log.warning("clip cache source unavailable for clip %d: %s", clip_id, exc)
        return None

    if os.path.isfile(path):
        return path

    with _lock:
        # Double-check after acquiring lock
        if os.path.isfile(path):
            return path

        tmp = path + ".tmp"
        try:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.4f}", "-t", f"{duration:.4f}",
                "-i", source,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "44100", "-ac", "2",
                "-f", "mpegts", tmp,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                log.warning("clip cache extract failed for clip %d: %s",
                            clip_id, result.stderr[-200:])
                _remove(tmp)
                return None

            # A retranscode/replace during extraction must not install audio
            # under the fingerprint of the recording that was there before.
            if _fingerprint(source_file, start, duration)[1] != fingerprint:
                _remove(tmp)
                return None
            os.replace(tmp, path)

            # Track in database
            try:
                with get_db() as conn:
                    old = conn.execute(
                        "SELECT cache_file FROM clip_cache WHERE clip_id=? AND pad_end=?",
                        (clip_id, round(pad_end, 3))).fetchone()
                    conn.execute(
                        "INSERT OR REPLACE INTO clip_cache "
                        "(clip_id, pad_end, cache_file) VALUES (?, ?, ?)",
                        (clip_id, round(pad_end, 3), os.path.basename(path))
                    )
                # The table keeps one window per id/pad. Remove the previous
                # tracked file so a changed fingerprint does not orphan it.
                if old and old[0] != os.path.basename(path) and _cache_key(old[0]):
                    _remove(os.path.join(os.path.dirname(path), old[0]))
            except Exception:
                pass

            log.debug("Cached clip %d (pad=%.3f) -> %s", clip_id, pad_end, path)
            return path

        except Exception as exc:
            log.warning("clip cache failed for clip %d: %s", clip_id, exc)
            _remove(tmp)
            return None


def invalidate(clip_id: int | None = None) -> None:
    """Drop cached segments for one clip (after edit) or all clips."""
    _invalidate_ids(None if clip_id is None else {clip_id})


def invalidate_many(clip_ids, *, persisted: bool = True) -> None:
    """Invalidate a source's clips with one directory scan and transaction."""
    ids = set(clip_ids)
    if ids:
        _invalidate_ids(ids, persisted=persisted)


def _invalidate_ids(ids: set[int] | None, *, persisted: bool = True) -> None:
    with _lock:
        try:
            if persisted:
                with get_db() as conn:
                    if ids is None:
                        conn.execute("DELETE FROM clip_cache")
                    else:
                        conn.executemany("DELETE FROM clip_cache WHERE clip_id=?", [(cid,) for cid in ids])
        except Exception:
            pass
        # Also clean legacy/untracked fingerprints and failed temporary files.
        # Only filenames owned by this cache are eligible, never DB paths.
        cache_dir = _cache_dir(create=False)
        try:
            for name in os.listdir(cache_dir):
                key = _cache_key(name)
                if key and (ids is None or key[0] in ids):
                    _remove(os.path.join(cache_dir, name))
        except OSError:
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
            if name.endswith(".ts") and _cache_key(name):
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
                key = _cache_key(basename)
                if key:
                    clip_id, pad_end = key
                    with get_db() as conn:
                        conn.execute(
                            "DELETE FROM clip_cache WHERE clip_id=? AND pad_end=? AND cache_file=?",
                            (clip_id, pad_end, basename)
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
