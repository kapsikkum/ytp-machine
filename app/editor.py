"""Endpoints for the corpus editor — reading and correcting clips by hand.

Everything else in this project treats the corpus as read-only and derived: the
ingest writes it, the app reads it. That works until a boundary is wrong, and
boundaries are wrong often enough to matter. Nearly every generation fault
found so far -- "mrs rosen" losing its N, "time" taking its T from the word
after it, a word with 100ms of the next word's silence on the end -- was an
aligner putting an edge where it stopped being confident rather than where the
sound was.

Those are trivial to fix if you can see and hear them, and impossible
otherwise. So this exposes the corpus for editing: the source video, the words
in spoken order, and the timings, with an audio envelope so an edge can be
placed by eye rather than guessed at.

Deliberately separate from the generation API. This is the only thing in the
project that writes to word_clips outside an ingest, and it is worth being able
to see all of it in one file.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.database import active, get_db, init_db

router = APIRouter()
log = logging.getLogger(__name__)


class ClipEdit(BaseModel):
    word: str | None = None
    start_time: float | None = None
    end_time: float | None = None


class NewClip(BaseModel):
    word: str
    start_time: float
    end_time: float


# Noises live in their own table with the same shape as word_clips. They are
# kept apart in the schema because they must not join word runs or idle
# detection -- a click is not a word -- but for editing they are the same
# thing: a label and two times against a source.
_TABLES = {"word": "word_clips", "noise": "noise_clips"}


def _table(kind: str) -> str:
    if kind not in _TABLES:
        raise HTTPException(status_code=400,
                            detail=f"kind must be word or noise, not {kind!r}")
    return _TABLES[kind]


def _invalidate() -> None:
    """Drop the caches every edit invalidates."""
    try:
        from app.generate import invalidate_cache
        invalidate_cache()
    except Exception:
        pass


def _source_path(source_id: int) -> str:
    from app.database import resolve_path
    with get_db() as conn:
        row = conn.execute("SELECT source_file FROM sources WHERE id=?",
                           (source_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"no source {source_id}")
    path = resolve_path(row["source_file"])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"file missing: {path}")
    return path


@router.get("/sources")
def sources():
    """Every source in the active corpus, with enough to choose between them."""
    init_db()
    with get_db() as conn:
        rows = conn.execute("""
            SELECT s.id, s.video_id, s.title,
                   count(w.id)      AS clips,
                   min(w.start_time) AS first,
                   max(w.end_time)   AS last
            FROM sources s LEFT JOIN word_clips w ON w.source_id = s.id
            GROUP BY s.id ORDER BY s.id
        """).fetchall()
    return {"corpus": active()["slug"],
            "sources": [dict(r) for r in rows]}


@router.get("/source/{source_id}")
def source_clips(source_id: int):
    """Every clip of one source, in spoken order, with what looks suspect.

    The flags are hints for where to look first, not judgements. A very short
    clip, one that overlaps its neighbour, or one whose gap to the next word is
    negative are all shapes that produced audible faults before.
    """
    init_db()
    with get_db() as conn:
        src = conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        if not src:
            raise HTTPException(status_code=404, detail=f"no source {source_id}")
        rows = [dict(r, kind="word") for r in conn.execute(
            "SELECT id, word, start_time, end_time FROM word_clips "
            "WHERE source_id=? ORDER BY start_time", (source_id,))]
        try:
            rows += [dict(r, kind="noise") for r in conn.execute(
                "SELECT id, word, start_time, end_time FROM noise_clips "
                "WHERE source_id=? ORDER BY start_time", (source_id,))]
            rows.sort(key=lambda c: c["start_time"])
        except Exception:
            pass                      # a corpus packed before noise_clips existed

    words = [c for c in rows if c["kind"] == "word"]
    for i, c in enumerate(rows):
        dur = c["end_time"] - c["start_time"]
        # Measured against the next *word*: a noise sitting inside a gap is
        # not an overlap, it is the point of it.
        later = [w for w in words if w["start_time"] > c["start_time"]]
        nxt = later[0]["start_time"] if later else None
        c["gap_after"] = None if nxt is None else round(nxt - c["end_time"], 4)
        flags = []
        if dur < 0.06:
            flags.append("very short")
        if c["gap_after"] is not None and c["gap_after"] < 0:
            flags.append("overlaps next")
        if dur > 1.5:
            flags.append("very long")
        c["flags"] = flags
    return {"source": dict(src), "clips": rows}


@router.get("/source/{source_id}/video")
def source_video(source_id: int):
    """The video itself, so the browser can seek around in it."""
    return FileResponse(_source_path(source_id), media_type="video/mp4")


@router.get("/source/{source_id}/envelope")
def envelope(source_id: int, start: float = 0.0, end: float = 0.0,
             buckets: int = 900):
    """Loudness across a span, for drawing a waveform.

    An edge is nearly impossible to place from numbers alone and obvious from
    a picture -- every boundary fault found by hand this week was visible the
    moment the audio was plotted.
    """
    import array
    import math
    import subprocess
    import tempfile
    import wave

    if end <= start:
        raise HTTPException(status_code=400, detail="end must be after start")
    span = min(end - start, 60.0)          # a minute is plenty to look at
    path = _source_path(source_id)
    tmp = os.path.join(tempfile.gettempdir(), f"_env_{os.getpid()}.wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.4f}",
                        "-t", f"{span:.4f}", "-i", path,
                        "-ac", "1", "-ar", "16000", tmp], capture_output=True)
        with wave.open(tmp, "rb") as w:
            raw = w.readframes(w.getnframes())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"could not read audio: {exc}")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    samples = array.array("h")
    samples.frombytes(raw[: len(raw) // 2 * 2])
    if not samples:
        return {"start": start, "end": start + span, "peak": 0, "values": []}

    per = max(1, len(samples) // max(1, buckets))
    values = [int(math.sqrt(sum(v * v for v in samples[k:k + per]) / per))
              for k in range(0, len(samples) - per, per)]
    return {"start": start, "end": start + span,
            "peak": max(values) if values else 0, "values": values}


@router.patch("/clip/{clip_id}")
def edit_clip(clip_id: int, edit: ClipEdit, kind: str = "word"):
    """Change a clip's word or its timings."""
    init_db()
    sets, params = [], []
    if edit.word is not None:
        # Stored exactly as the ingest stores them, or the corpus ends up with
        # two spellings of the same word and lookups find only one.
        import re
        word = re.sub(r"[^\w]", "", edit.word).lower()
        if not word:
            raise HTTPException(status_code=400, detail="word cannot be empty")
        sets.append("word=?"); params.append(word)
    if edit.start_time is not None:
        sets.append("start_time=?"); params.append(round(edit.start_time, 4))
    if edit.end_time is not None:
        sets.append("end_time=?"); params.append(round(edit.end_time, 4))
    if not sets:
        raise HTTPException(status_code=400, detail="nothing to change")

    table = _table(kind)
    with get_db() as conn:
        row = conn.execute(f"SELECT * FROM {table} WHERE id=?", (clip_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"no {kind} clip {clip_id}")
        new_start = edit.start_time if edit.start_time is not None else row["start_time"]
        new_end = edit.end_time if edit.end_time is not None else row["end_time"]
        if new_end - new_start < 0.02:
            raise HTTPException(status_code=400,
                                detail="a clip must be at least 20ms long")
        conn.execute(f"UPDATE {table} SET {', '.join(sets)} WHERE id=?",
                     (*params, clip_id))
        out = dict(conn.execute(f"SELECT id, word, start_time, end_time "
                                f"FROM {table} WHERE id=?", (clip_id,)).fetchone(),
                   kind=kind)
    _invalidate()
    log.info("EDIT    clip %s -> %s %.3f-%.3f", clip_id, out["word"],
             out["start_time"], out["end_time"])
    return out


@router.delete("/clip/{clip_id}")
def delete_clip(clip_id: int, kind: str = "word"):
    """Remove a clip. For a word that was never said, or is unusable."""
    init_db()
    table = _table(kind)
    with get_db() as conn:
        row = conn.execute(f"SELECT word FROM {table} WHERE id=?", (clip_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"no {kind} clip {clip_id}")
        conn.execute(f"DELETE FROM {table} WHERE id=?", (clip_id,))
        # Ratings key on clip id, and ids are reused by AUTOINCREMENT only
        # after a vacuum -- but a stale rating on a deleted clip is dead weight
        # either way.
        try:
            conn.execute("DELETE FROM splice_ratings WHERE clip_id=?", (clip_id,))
        except Exception:
            pass
    _invalidate()
    log.info("DELETE  clip %s (%s)", clip_id, row["word"])
    return {"deleted": clip_id, "word": row["word"]}


@router.post("/source/{source_id}/clip")
def add_clip(source_id: int, clip: NewClip, kind: str = "word"):
    """Add a word the transcription missed, or a noise it could not hear.

    Whisper only emits words, so every click, spew and hum in a recording is
    invisible to it. find_noises.py hunts them by energy, which works but is
    blind to what they are -- marking one by hand against the waveform is the
    only way to say "that one is a spew" rather than "that one is loud".
    """
    init_db()
    table = _table(kind)
    import re
    word = re.sub(r"[^\w]", "", clip.word).lower()
    if not word:
        raise HTTPException(status_code=400, detail="word cannot be empty")
    if clip.end_time - clip.start_time < 0.02:
        raise HTTPException(status_code=400, detail="a clip must be at least 20ms long")
    with get_db() as conn:
        src = conn.execute("SELECT source_file FROM sources WHERE id=?",
                           (source_id,)).fetchone()
        if not src:
            raise HTTPException(status_code=404, detail=f"no source {source_id}")
        cur = conn.execute(
            f"INSERT INTO {table} (source_id, word, start_time, end_time, source_file) "
            f"VALUES (?,?,?,?,?)",
            (source_id, word, round(clip.start_time, 4), round(clip.end_time, 4),
             src["source_file"]))
        new_id = cur.lastrowid
    _invalidate()
    log.info("ADD     clip %s = %s %.3f-%.3f", new_id, word,
             clip.start_time, clip.end_time)
    return {"id": new_id, "word": word, "kind": kind,
            "start_time": clip.start_time, "end_time": clip.end_time}
