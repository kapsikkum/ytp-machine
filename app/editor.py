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

import json
import logging
import os
import re
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.database import active, get_db, init_db

router = APIRouter()
log = logging.getLogger(__name__)


def _phones_moved_off(raw, duration: float) -> bool:
    from app.phone_align import outside
    return outside(raw, duration)


class ClipEdit(BaseModel):
    word: str | None = None
    start_time: float | None = None
    end_time: float | None = None


class Rating(BaseModel):
    word: str
    score: int


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


def _invalidate(clip_id: int | None = None) -> None:
    """Drop the caches this edit invalidates.

    When the edit is to one clip, its forced alignment goes with it -- the
    cached character times were measured from the boundaries that just moved,
    and are wrong the moment they do. Only that clip's, though: re-aligning
    costs a model pass each, so dropping every alignment on every 10ms nudge
    would leave the next sentence to pay for it.
    """
    try:
        from app.generate import invalidate_cache
        invalidate_cache(alignments=clip_id is None)
    except Exception:
        pass
    if clip_id is not None:
        try:
            from app.forced_align import invalidate as invalidate_alignment
            invalidate_alignment(clip_id)
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
        # `aligned` rather than the times themselves: a corpus of 9,000
        # clips would otherwise send a phoneme table per clip to the browser
        # to answer a yes/no question.
        rows = [dict(r, kind="word") for r in conn.execute(
            "SELECT id, word, start_time, end_time, "
            "       (phones IS NOT NULL AND phones != '') AS aligned "
            "FROM word_clips WHERE source_id=? ORDER BY start_time", (source_id,))]
        try:
            rows += [dict(r, kind="noise", aligned=1) for r in conn.execute(
                "SELECT id, word, start_time, end_time FROM noise_clips "
                "WHERE source_id=? ORDER BY start_time", (source_id,))]
            rows.sort(key=lambda c: c["start_time"])
        except Exception:
            pass                      # a corpus packed before noise_clips existed

        # A rating is keyed on (target word, clip). The target is the word
        # being *built*, which is often not this clip's own word: a downvote
        # says "this clip sounded wrong used for that", which is why one clip
        # can carry several. Fetched in one query rather than per clip -- a
        # source has thousands.
        ratings: dict[int, dict[str, int]] = {}
        try:
            for r in conn.execute(
                    "SELECT word, clip_id, score FROM splice_ratings"):
                ratings.setdefault(r["clip_id"], {})[r["word"]] = r["score"]
        except Exception:
            pass                      # a corpus packed before splice_ratings
        reports: dict[int, list[str]] = {}
        try:
            for r in conn.execute("SELECT clip_id, kind, count FROM boundary_reports"):
                reports.setdefault(r["clip_id"], []).append(
                    f"reported: {r['kind']}" + (f" ×{r['count']}" if r["count"] > 1 else ""))
        except Exception:
            pass                      # a corpus from before boundary reports

    for c in rows:
        c["ratings"] = ratings.get(c["id"], {}) if c["kind"] == "word" else {}

    corpus_aligned = any(c.get("aligned") for c in rows if c["kind"] == "word")
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
        if any(v < 0 for v in c["ratings"].values()):
            flags.append("downvoted")
        # Only worth saying in a corpus that has been aligned at all --
        # otherwise it is true of every clip and says nothing.
        if corpus_aligned and not c.get("aligned") and c["kind"] == "word":
            # Which of the two it is decides what to do about it: run the
            # alignment, or teach the corpus how the word is said.
            from app.phonemes import word_to_phonemes
            flags.append("not aligned" if word_to_phonemes(c["word"])
                         else "no pronunciation")
        elif _phones_moved_off(c.get("phones"), c["end_time"] - c["start_time"]):
            flags.append("alignment moved off")
        if c["kind"] == "word":
            flags += reports.get(c["id"], [])
        c["flags"] = flags
    return {"source": dict(src), "clips": rows}


# One alignment run at a time, and what it is doing, so the editor can show
# progress rather than a button that goes quiet for a minute.
#
# A dict rather than a job queue: there is one of these, it is per source, and
# app/jobs.py exists for generation, which has to survive a queue of requests
# from a browser. This does not.
_align_state: dict[str, Any] = {"running": False, "source_id": None,
                                "done": 0, "total": 0, "result": None,
                                "error": None}
_align_lock = threading.Lock()


def _align_ready() -> tuple[bool, str]:
    """Can this process align, and if not, why not?

    The server cannot: the model needs about 1.8GB resident and the container
    is capped at 3.2GB with the app already holding 1.3GB of it. Measured
    rather than assumed, and the same regardless of clip length -- torch keeps
    its allocator arena, so a 300ms clip costs what a 1.5s one does.
    """
    try:
        from app import phone_align
    except Exception as exc:
        return False, f"the aligner is not installed here ({exc})"
    if not phone_align.available():
        return False, ("this machine has no aligner -- it needs torch, "
                       "torchaudio and transformers. Align on the machine "
                       "that builds corpora and put the bundle back.")
    return True, ""


@router.get("/source/{source_id}/align")
def align_status(source_id: int):
    """What alignment would do here, and what it is doing now.

    Two different reasons a clip has no phoneme times, and they want
    different things done about them:

      - nobody has run the alignment over it yet. The button fixes that.
      - the dictionary has no pronunciation for the word, so there is no
        sequence to place and never will be until somebody supplies one.
        Michael Rosen says "scheddle", "nibblin" and "dyou", and CMU has
        heard of none of them.

    Counting the second kind as work to be done gives a button that says
    "align 24 clips", changes nothing when pressed, and still says 24.
    """
    init_db()
    from app.phonemes import word_to_phonemes
    ready, why = _align_ready()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT word, count(*) AS n FROM word_clips WHERE source_id=? "
            "AND (phones IS NULL OR phones = '') GROUP BY word", (source_id,))
        pending = 0
        unknown: list[str] = []
        unknown_clips = 0
        for r in rows:
            if word_to_phonemes(r["word"]):
                pending += r["n"]
            else:
                unknown_clips += r["n"]
                unknown.append(r["word"])
        # Times that exist but no longer describe this clip are work to do as
        # much as times never measured -- more so, because they are believed.
        from app.phone_align import outside
        stale = 0
        for r in conn.execute(
                "SELECT word, phones, end_time - start_time AS dur FROM word_clips "
                "WHERE source_id=? AND phones IS NOT NULL AND phones != ''", (source_id,)):
            if outside(r["phones"], r["dur"]):
                if word_to_phonemes(r["word"]):
                    stale += 1
                    pending += 1
                else:
                    unknown_clips += 1
    st = dict(_align_state)
    return {"available": ready, "reason": why, "pending": pending, "stale": stale,
            "no_pronunciation": unknown_clips,
            "unknown_words": sorted(unknown)[:20],
            "running": st["running"] and st["source_id"] == source_id,
            "done": st["done"], "total": st["total"],
            "result": st["result"], "error": st["error"]}


@router.post("/source/{source_id}/align")
def align_source(source_id: int, redo: bool = False):
    """Measure where every phoneme falls, for this source's clips.

    Wanted because a word added by hand has no phoneme times, and until it
    does its sub-word cuts are inferred from the spelling -- which is what the
    alignment was built to stop. Adding three words to a video should not mean
    packing the corpus, moving it to another machine and moving it back.
    """
    init_db()
    ready, why = _align_ready()
    if not ready:
        raise HTTPException(status_code=503, detail=why)

    with _align_lock:
        if _align_state["running"]:
            raise HTTPException(
                status_code=409,
                detail=f"already aligning source {_align_state['source_id']}")
        _align_state.update(running=True, source_id=source_id, done=0,
                            total=0, result=None, error=None)

    # The pass only visits clips with no times, so stale ones lose theirs
    # first -- otherwise the button offers work it then quietly skips.
    from app.phone_align import outside
    with get_db() as conn:
        bad = [r["id"] for r in conn.execute(
            "SELECT id, phones, end_time - start_time AS dur FROM word_clips "
            "WHERE source_id=? AND phones IS NOT NULL AND phones != ''", (source_id,))
            if outside(r["phones"], r["dur"])]
        for cid in bad:
            conn.execute("UPDATE word_clips SET phones=NULL WHERE id=?", (cid,))
    if bad:
        log.info("ALIGN   source %s: dropped %d alignment(s) that had moved off "
                 "their clip", source_id, len(bad))

    def work() -> None:
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "scripts"))
            from align_phones import align_corpus
            done, skipped, failed = align_corpus(source_id=source_id, redo=redo)
            _align_state["result"] = {"aligned": done, "no_pronunciation": skipped,
                                      "failed": failed}
            log.info("ALIGN   source %s: %d aligned, %d without a pronunciation, "
                     "%d failed", source_id, done, skipped, failed)
        except Exception as exc:                       # noqa: BLE001
            _align_state["error"] = str(exc)
            log.exception("alignment of source %s failed", source_id)
        finally:
            _align_state["running"] = False

    threading.Thread(target=work, daemon=True, name="align-source").start()
    return {"started": True, "source_id": source_id}


@router.get("/source/{source_id}/video")
def source_video(source_id: int, c: str = ""):
    """The video itself, so the browser can seek around in it.

    `c` is the corpus slug. Source ids start again at 1 in each corpus, so
    it decides which video this id means when it names another voice than
    the active one (a word said in another voice, auditioned from the word
    menu) -- and it keeps the URL different between corpora, since a cached
    /source/1/video from another corpus is the wrong video entirely.

    no-cache rather than no-store: revalidating on the ETag is enough to
    notice the file changed, and still lets seeking reuse what it has.
    """
    path = None
    if c and c != active()["slug"]:
        from app.generate import voice_source
        path = voice_source(c, source_id)
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"no source {source_id} in {c}")
    return FileResponse(path or _source_path(source_id), media_type="video/mp4",
                        headers={"Cache-Control": "no-cache"})


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
        word = re.sub(r"[^\w]", "", edit.word).lower()
        if not word:
            raise HTTPException(status_code=400, detail="word cannot be empty")
        sets.append("word=?"); params.append(word)
    moved = edit.start_time is not None or edit.end_time is not None
    if edit.start_time is not None:
        sets.append("start_time=?"); params.append(round(edit.start_time, 4))
    if edit.end_time is not None:
        sets.append("end_time=?"); params.append(round(edit.end_time, 4))
    if moved and kind == "word":
        # Marked as chosen rather than found, so the encoder stops nursing the
        # tail outwards past it. Only word clips: noises are already extracted
        # exactly as stored.
        sets.append("edited=1")
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
        if moved and kind == "word":
            # Moved by hand, against the waveform: whatever was reported
            # about where it was cut is answered.
            try:
                conn.execute("DELETE FROM boundary_reports WHERE clip_id=?", (clip_id,))
            except Exception:
                pass

        # Aligned phoneme times are stored relative to the clip's own
        # start_time, so an edit invalidates them -- and moving the start by
        # 20ms would otherwise leave every phoneme 20ms out, which is the size
        # of the errors this whole mechanism exists to remove.
        #
        # They are moved rather than dropped. Nothing about the recording
        # changed: the sounds are still exactly where they were, and only the
        # point they are measured from has moved. Realigning would need the
        # model, which is far larger than this server.
        #
        # A different word is not the same case. The times describe a sequence
        # of phonemes that is no longer this word's, so they go, and the clip
        # falls back to inferring cuts from the spelling until the next
        # alignment pass -- which is what it did before any of this existed.
        if kind == "word" and row["phones"]:
            renamed = edit.word is not None and word != row["word"]
            shift = row["start_time"] - new_start
            if renamed:
                conn.execute("UPDATE word_clips SET phones=NULL WHERE id=?",
                             (clip_id,))
            elif abs(shift) > 1e-9:
                from app.phone_align import outside, shift_stored
                moved_phones = shift_stored(row["phones"], shift)
                # Moving them is right until the boundary moves past them. A
                # clip retimed onto the word beside it kept phonemes sitting
                # entirely before its own start, and nothing noticed, because
                # "aligned" only ever asked whether times existed.
                if outside(moved_phones, new_end - new_start):
                    moved_phones = None
                conn.execute("UPDATE word_clips SET phones=? WHERE id=?",
                             (moved_phones, clip_id))
            else:
                from app.phone_align import outside
                if outside(row["phones"], new_end - new_start):
                    conn.execute("UPDATE word_clips SET phones=NULL WHERE id=?", (clip_id,))

        out = dict(conn.execute(f"SELECT id, word, start_time, end_time "
                                f"FROM {table} WHERE id=?", (clip_id,)).fetchone(),
                   kind=kind)
    _invalidate(clip_id)
    log.info("EDIT    clip %s -> %s %.3f-%.3f", clip_id, out["word"],
             out["start_time"], out["end_time"])
    return out


@router.put("/clip/{clip_id}/rating")
def set_rating(clip_id: int, rating: Rating):
    """Set a clip's score for one target word.

    Ratings are normally cast from the generator -- you hear a bad splice and
    downvote it, which is one vote at a time and blind to what is already
    there. Here the whole picture is visible, so the score is set outright
    rather than nudged: a clip that has collected -3 from three bad sentences
    can be forgiven in one move, or condemned without having to generate the
    same sentence three times.

    Only word clips have them. Noises are never spliced into a word.
    """
    init_db()
    word = re.sub(r"[^\w]", "", rating.word).lower()
    if not word:
        raise HTTPException(status_code=400, detail="a rating needs a word")
    with get_db() as conn:
        if not conn.execute("SELECT 1 FROM word_clips WHERE id=?",
                            (clip_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"no word clip {clip_id}")
        if rating.score == 0:
            # Neutral is the absence of a rating, so it is stored as one. A
            # row of zeroes would read as "judged and found average", which is
            # not a thing the splicer or anybody else means.
            conn.execute("DELETE FROM splice_ratings WHERE word=? AND clip_id=?",
                         (word, clip_id))
        else:
            conn.execute(
                "INSERT INTO splice_ratings (word, clip_id, score) VALUES (?,?,?) "
                "ON CONFLICT(word, clip_id) DO UPDATE SET score=?",
                (word, clip_id, rating.score, rating.score))
        out = {r["word"]: r["score"] for r in conn.execute(
            "SELECT word, score FROM splice_ratings WHERE clip_id=?", (clip_id,))}
    # A rating changes which clip gets chosen, not where anything is cut, so
    # the alignments stay: they cost a model pass each to rebuild.
    _invalidate(clip_id)
    log.info("RATE    clip %s for %r = %s", clip_id, word, rating.score)
    return {"id": clip_id, "ratings": out}


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
    _invalidate(clip_id)
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
        # A clip added here was dragged out against the waveform, so its edges
        # are chosen and the encoder should leave them alone -- same as an
        # edited one. Only word_clips has the column; noises are extracted
        # exactly as stored anyway.
        extra = ", edited" if kind == "word" else ""
        vals = ",1" if kind == "word" else ""
        cur = conn.execute(
            f"INSERT INTO {table} (source_id, word, start_time, end_time, source_file{extra}) "
            f"VALUES (?,?,?,?,?{vals})",
            (source_id, word, round(clip.start_time, 4), round(clip.end_time, 4),
             src["source_file"]))
        new_id = cur.lastrowid
    _invalidate()
    log.info("ADD     clip %s = %s %.3f-%.3f", new_id, word,
             clip.start_time, clip.end_time)
    return {"id": new_id, "word": word, "kind": kind,
            "start_time": clip.start_time, "end_time": clip.end_time}


# ── Splice editor ──────────────────────────────────────────────────────────
#
# A word the splicer builds wrongly is hard to fix from the generator: you
# hear the fault, and the only lever is downvoting clips until the search
# lands somewhere else. This is the search made visible -- the sounds being
# aimed at, what could supply each of them, and what any given combination
# sounds like -- and a way to write the answer down.


class SpliceGroup(BaseModel):
    phones: list[str]
    source: str | None = None          # which word supplies them
    clip_id: int | None = None         # ... and optionally which clip of it
    start: float | None = None         # ... and where in that clip, set by hand
    end: float | None = None


def _where(s: dict) -> dict:
    """Where a piece was cut, and the clip it was cut from: what the waveform
    in the splice editor draws, the clip's own bounds being as far as a grip
    may be dragged."""
    return {"source_id": s.get("source_id"), "clip_id": s.get("id"),
            "start": round(s["start_time"], 4), "end": round(s["end_time"], 4),
            "src_start": round(s.get("_src_start", s["start_time"]), 4),
            "src_end": round(s.get("_src_end", s["end_time"]), 4)}


class SplicePlan(BaseModel):
    groups: list[SpliceGroup]
    mode: str = "strict"


def _cbw() -> dict:
    """The generator's own clip index, built if it isn't already."""
    import app.generate as g
    g._ensure_cache()
    return g._clips_by_word_cache or {}


def _fell_back(seg: dict, group: list[str], source: str) -> bool:
    """Did this piece come out as the whole word instead of the cut asked for?

    _realise falls back to a whole clip when alignment cannot cut one, which
    is the right thing to do mid-sentence and a lie by omission here: you
    asked to hear CH and you are hearing "catch", which looks exactly like the
    pinned clip being ignored again.

    Read from `subword` rather than `_cut`, which is a working flag _realise
    pops before returning -- reading it gave False for every piece, cut or
    not, and would have cried wolf on every audition.
    """
    from app import phonemes as ph
    cph = ph.phones_of(source or "") or []
    pos = ph._find_sub(cph, group)
    if pos is None:
        return False
    needed = pos > 0 or pos + len(group) < len(cph)
    return needed and not seg.get("subword")


def _clip_brief(c: dict, ratings: dict) -> dict:
    return {"id": c.get("id"), "source_id": c.get("source_id"),
            "start_time": round(c["start_time"], 3),
            "end_time": round(c["end_time"], 3),
            "duration": round(c["end_time"] - c["start_time"], 3),
            "ratings": ratings.get(c.get("id"), {})}


def _all_ratings() -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    try:
        with get_db() as conn:
            for r in conn.execute("SELECT word, clip_id, score FROM splice_ratings"):
                out.setdefault(r["clip_id"], {})[r["word"]] = r["score"]
    except Exception:
        pass
    return out


@router.get("/splice/{word}")
def splice_word(word: str, mode: str = "strict"):
    """What the splicer would do with *word*, and what it had to choose from."""
    init_db()
    import app.generate as g
    from app import phonemes as ph

    word = re.sub(r"[^\w]", "", word).lower()
    if not word:
        raise HTTPException(status_code=400, detail="need a word")

    cbw = _cbw()
    ratings = _all_ratings()
    phones = ph.canonical_phones(word, mode)
    exact = [_clip_brief(c, ratings) for c in cbw.get(word, [])[:40]]

    plan = None
    if phones:
        from app.database import max_units
        segs = ph.find_phoneme_splice(word, cbw, g._penalty_for(word), mode,
                                      max_units())
        if segs:
            plan = [dict(s.get("unit") or {},
                         start_time=round(s["start_time"], 3),
                         end_time=round(s["end_time"], 3),
                         subword=bool(s.get("subword")))
                    for s in segs]

    saved = ph.user_recipe(word)
    entry = ph.user_entry(word)
    return {
        "word": word,
        "phones": phones,
        "known": ph.word_to_phonemes(word) is not None,
        "pron": {"scope": entry[0], "value": entry[1]} if entry else None,
        "exact": exact,
        "exact_total": len(cbw.get(word, [])),
        "plan": plan,
        "recipe": ([{"phones": list(grp), "from": list(pref), **cut}
                    for grp, pref, cut in saved] if saved else None),
        "mode": mode,
    }


# How a word is pronounced, from the splice editor. The splicer can only build
# a word out of the sounds it is told the word has, so this is the other half
# of fixing one: the pieces are chosen against these sounds.
class Pronunciation(BaseModel):
    value: str
    scope: str = "all"                 # "all": every voice; "voice": this one only


def _pron_word(word: str) -> str:
    word = re.sub(r"[^\w']", "", word).lower()
    if not word:
        raise HTTPException(status_code=400, detail="need a word")
    return word


def _pron_changed() -> None:
    """Everything worked out from the old pronunciations is out of date."""
    import app.generate as g
    g.invalidate_cache(alignments=False)
    g._known_spelling.cache_clear()


@router.post("/pronunciation/{word}/try")
def try_pronunciation(word: str, p: Pronunciation):
    """The sounds a pronunciation would give, without saving it."""
    from app import phonemes as ph
    word = _pron_word(word)
    try:
        phones, stored, guessed = ph.try_pronunciation(word, p.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"word": word, "phones": phones, "stored": stored, "guessed": guessed}


@router.put("/pronunciation/{word}")
def save_pronunciation(word: str, p: Pronunciation):
    """Say *word* this way from now on, in this voice or every voice."""
    from app import phonemes as ph
    word = _pron_word(word)
    if p.scope not in ("all", "voice"):
        raise HTTPException(status_code=400, detail="scope is all or voice")
    try:
        phones, stored, guessed = ph.try_pronunciation(word, p.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ph.set_pronunciation(word, stored, p.scope)
    _pron_changed()
    log.info("PRONOUNCE  %s = %s (%s)", word, stored, p.scope)
    return {"word": word, "phones": phones, "stored": stored, "scope": p.scope}


@router.delete("/pronunciation/{word}")
def delete_pronunciation(word: str, scope: str = "all"):
    """Back to the dictionary for *word*."""
    from app import phonemes as ph
    word = _pron_word(word)
    ph.set_pronunciation(word, None, "voice" if scope == "voice" else "all")
    _pron_changed()
    log.info("PRONOUNCE  %s cleared (%s)", word, scope)
    return {"word": word}


@router.get("/splice-sources")
def splice_sources(phones: str, limit: int = 30):
    """Which words could supply this run of sounds, best first.

    Takes the phonemes rather than a word and a span so the caller can ask
    about a grouping that does not exist yet -- which is the whole business of
    the editor: trying a different split.
    """
    init_db()
    from app import phonemes as ph
    # Stress digits stripped: CMU writes IH1, the corpus index does not, and a
    # phoneme typed straight off a dictionary would otherwise match nothing at
    # all while looking perfectly correct.
    group = [re.sub(r"\d+$", "", p)
             for p in re.split(r"[\s,]+", phones.upper()) if p]
    if not group:
        raise HTTPException(status_code=400, detail="need phonemes")
    cbw = _cbw()
    ratings = _all_ratings()
    out = []
    for cand in ph.group_sources(group, cbw, limit):
        pool = sorted(cbw.get(cand["word"], []),
                      key=lambda c: -(c["end_time"] - c["start_time"]))
        out.append(dict(cand, clip_list=[_clip_brief(c, ratings) for c in pool[:12]]))
    return {"phones": group, "sources": out}


@router.post("/splice/{word}/preview")
def splice_preview(word: str, plan: SplicePlan):
    """Render one chosen grouping, so it can be judged by ear."""
    init_db()
    import app.generate as g
    from app import phonemes as ph

    word = re.sub(r"[^\w]", "", word).lower()
    groups = [{"phones": [p.upper() for p in gr.phones], "from": gr.source,
               "clip_id": gr.clip_id, "start": gr.start, "end": gr.end} for gr in plan.groups]
    segs = ph.realise_groups(word, groups, _cbw(), g._penalty_for(word), plan.mode)
    if not segs:
        raise HTTPException(
            status_code=400,
            detail="that combination cannot be cut from this corpus -- check "
                   "each group's source actually contains those sounds")

    import uuid
    os.makedirs("output", exist_ok=True)
    name = f"splice_{uuid.uuid4().hex[:10]}.mp4"
    try:
        g._build_video(segs, os.path.join("output", name))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"could not render: {exc}")
    log.info("SPLICE  preview %r from %s", word,
             " + ".join(s.get("spliced_from", "?") for s in segs))
    return {"url": f"/output/{name}",
            "units": [{"from": s.get("spliced_from"), "clip_id": s.get("id"),
                       "phones": (s.get("unit") or {}).get("phones", []),
                       "fell_back": _fell_back(s, (s.get("unit") or {}).get("phones", []),
                                               s.get("spliced_from")),
                       "duration": round(s["end_time"] - s["start_time"], 3),
                       **_where(s)}
                      for s in segs]}


@router.put("/splice/{word}/recipe")
def save_recipe(word: str, plan: SplicePlan):
    """Write a grouping down, so every future generation uses it."""
    init_db()
    import json

    from app import phonemes as ph
    word = re.sub(r"[^\w]", "", word).lower()
    phones = ph.canonical_phones(word, plan.mode)
    flat = [p.upper() for gr in plan.groups for p in gr.phones]
    if not phones:
        raise HTTPException(status_code=400,
                            detail=f"no pronunciation known for {word!r}")
    if flat != phones:
        # Saving a recipe that does not spell the word would leave a rule that
        # can never fire -- _chosen_from_recipe checks the same thing before
        # using one -- and it would fail silently, at generation time.
        raise HTTPException(
            status_code=400,
            detail=f"the groups spell {' '.join(flat)}, but {word} is "
                   f"{' '.join(phones)}")
    if any(not gr.source for gr in plan.groups):
        raise HTTPException(status_code=400,
                            detail="every group needs a source word")

    # The clip and its times go in only as a pair, and only both ends: a
    # recipe cut by hand is played exactly as it was heard in the editor.
    body = json.dumps([{"phones": [p.upper() for p in gr.phones], "from": [gr.source],
                        **({"clip_id": gr.clip_id, "start": gr.start, "end": gr.end}
                           if gr.clip_id and gr.start is not None and gr.end is not None
                           else {})}
                       for gr in plan.groups])
    with get_db() as conn:
        conn.execute("INSERT INTO splice_recipes (word, recipe) VALUES (?,?) "
                     "ON CONFLICT(word) DO UPDATE SET recipe=?",
                     (word, body, body))
    ph.invalidate_recipes()
    _invalidate()
    log.info("RECIPE  %s = %s", word,
             " | ".join(f"{'-'.join(gr.phones)}<{gr.source}" for gr in plan.groups))
    return {"word": word, "saved": True}


@router.delete("/splice/{word}/recipe")
def delete_recipe(word: str):
    """Back to whatever the search decides."""
    init_db()
    from app import phonemes as ph
    word = re.sub(r"[^\w]", "", word).lower()
    with get_db() as conn:
        conn.execute("DELETE FROM splice_recipes WHERE word=?", (word,))
    ph.invalidate_recipes()
    _invalidate()
    log.info("RECIPE  %s cleared", word)
    return {"word": word, "saved": False}


@router.get("/recipes")
def recipes():
    """Everything written down for this corpus."""
    init_db()
    import json
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT word, recipe FROM splice_recipes ORDER BY word").fetchall()
    except Exception:
        return {"recipes": []}
    out = []
    for r in rows:
        try:
            out.append({"word": r["word"], "groups": json.loads(r["recipe"])})
        except Exception:
            pass
    return {"recipes": out}

class Piece(BaseModel):
    phones: list[str]
    source: str
    clip_id: int | None = None
    start: float | None = None
    end: float | None = None


@router.post("/splice-piece")
def splice_piece(piece: Piece, render: bool = True):
    """Render one piece alone, for auditioning a clip.

    Choosing between twelve clips of "catch" by rendering the whole word each
    time is not choosing, it is guessing with an extra step.

    render=0 skips the video and only says where the cut falls: the editor
    draws that on the clip's waveform and plays it from the source itself.
    """
    init_db()
    import app.generate as g
    from app import phonemes as ph

    segs = ph.realise_piece([p.upper() for p in piece.phones], piece.source,
                            _cbw(), piece.clip_id, start=piece.start, end=piece.end)
    if not segs:
        raise HTTPException(status_code=400,
                            detail=f"{piece.source!r} cannot supply those sounds")
    if not render:
        s = segs[0]
        return {"from": s.get("spliced_from"),
                "fell_back": _fell_back(s, [p.upper() for p in piece.phones], piece.source),
                **_where(s)}

    import uuid
    os.makedirs("output", exist_ok=True)
    name = f"piece_{uuid.uuid4().hex[:10]}.mp4"
    try:
        g._build_video(segs, os.path.join("output", name))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"could not render: {exc}")
    s = segs[0]
    return {"url": f"/output/{name}", "from": s.get("spliced_from"),
            "clip_id": s.get("id"),
            "fell_back": _fell_back(s, [p.upper() for p in piece.phones],
                                    piece.source),
            "duration": round(s["end_time"] - s["start_time"], 3), **_where(s)}

@router.get("/clip/{clip_id}/locate")
def locate_clip(clip_id: int):
    """Which source a clip belongs to, and where in it.

    The generator knows a word came from clip 13478 and nothing else -- it
    never had to care which video that is. Following the link the other way
    needs exactly this one lookup.
    """
    init_db()
    with get_db() as conn:
        for kind, table in _TABLES.items():
            row = conn.execute(
                f"SELECT id, source_id, word, start_time, end_time "
                f"FROM {table} WHERE id=?", (clip_id,)).fetchone()
            if row:
                return dict(row, kind=kind)
    raise HTTPException(status_code=404, detail=f"no clip {clip_id}")
