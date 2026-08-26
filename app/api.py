import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.database import get_db
from app.generate import generate_video

router = APIRouter()
log = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    text: str


class RateRequest(BaseModel):
    word: str
    clips: list[int]
    rating: int = -1   # < 0 down-vote, > 0 up-vote (undo)


@router.post("/generate")
def generate(req: GenerateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    log.info("GENERATE  %r", text)
    t0 = time.perf_counter()

    result = generate_video(text)

    elapsed = time.perf_counter() - t0
    log.info(
        "DONE  %.2fs  found=%d  spliced=%d  missing=%d  url=%s",
        elapsed,
        len(result["found"]),
        len(result["spliced"]),
        len(result["missing"]),
        result.get("video_url"),
    )

    if not result["found"] and not result["spliced"]:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "No clips found or spliced for any word in the input.",
                "missing": result["missing"],
            },
        )

    return result


@router.post("/rate")
def rate(req: RateRequest):
    """Record feedback on a phoneme splice.  A down-vote makes the splicer avoid
    the clips that built this splice for this word next time (unless it has no
    other option)."""
    word = req.word.strip().lower()
    if not word or not req.clips:
        raise HTTPException(status_code=400, detail="word and clips are required")
    from app.generate import rate_splice
    delta = 1 if req.rating > 0 else -1
    scores = rate_splice(word, req.clips, delta)
    log.info("RATE  %s  clips=%s  delta=%+d  -> %s", word, req.clips, delta, scores)
    return {"status": "ok", "word": word, "scores": scores}


@router.post("/reload")
def reload():
    """Invalidate the in-memory clip cache so DB edits (corrections, re-aligns,
    new ingests) take effect without restarting the server."""
    from app.generate import invalidate_cache
    invalidate_cache()
    log.info("cache invalidated via /api/reload")
    return {"status": "reloaded"}


@router.get("/words")
def words():
    """Full corpus vocabulary with clip counts — fetched once by the frontend
    for instant client-side word checking while typing."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT word, COUNT(*) FROM word_clips GROUP BY word"
        ).fetchall()
    return {"words": {r[0]: r[1] for r in rows}}


@router.get("/suggest")
def suggest(context: str = "", prefix: str = "", limit: int = 10):
    """Autocomplete: phrase continuations from real spoken runs, plus
    vocabulary prefix matches."""
    from app.generate import suggest_next
    return suggest_next(context, prefix, max(1, min(limit, 25)))


@router.get("/stats")
def stats():
    with get_db() as conn:
        total_clips = conn.execute("SELECT COUNT(*) FROM word_clips").fetchone()[0]
        unique_words = conn.execute(
            "SELECT COUNT(DISTINCT word) FROM word_clips"
        ).fetchone()[0]
        sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        sample = conn.execute(
            "SELECT DISTINCT word FROM word_clips ORDER BY RANDOM() LIMIT 30"
        ).fetchall()
    return {
        "total_clips": total_clips,
        "unique_words": unique_words,
        "sources": sources,
        "sample_words": [r[0] for r in sample],
    }
