import logging
import os
import tempfile
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from app import jobs
from app.database import (max_units as _max_units_setting,
                          PROJECT_ROOT, SPLICE_MODES, active, get_db, init_db,
                          list_corpora, set_active, set_setting, splice_mode)
from app.generate import generate_video

router = APIRouter()
log = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    text: str
    # Burned into the picture rather than written to a sidecar file: the
    # output of this thing is shared as a video, and a .vtt nobody carries
    # with it is a caption nobody sees.
    subtitles: bool = False


class SpliceModeRequest(BaseModel):
    mode: str


class MaxUnitsRequest(BaseModel):
    units: int


class CorpusRequest(BaseModel):
    slug: str


class RateRequest(BaseModel):
    word: str
    clips: list[int]
    rating: int = -1   # < 0 down-vote, > 0 up-vote, 0 clears the vote


@router.post("/generate")
def generate(req: GenerateRequest, wait: bool = False):
    """Queue a video and return its id.

    Asynchronous by default. Generation can take minutes on a long sentence,
    and holding the request open for that meant the proxy gave up first and
    answered 504 -- the work finished, but nobody could collect it. Poll
    /api/jobs/{id} instead.

    `?wait=1` keeps the old blocking behaviour for scripts and curl, where
    there is no proxy in the way and a single answer is easier to consume.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    if wait:
        log.info("GENERATE (sync)  %r", text)
        t0 = time.perf_counter()
        try:
            result = generate_video(text, subtitles=req.subtitles)
        except RuntimeError as exc:
            detail = str(exc)
            crowded = "temporarily unavailable" in detail or "Too many open files" in detail
            log.error("GENERATE failed  %r  %s", text, detail[-500:])
            raise HTTPException(
                status_code=503 if crowded else 500,
                detail={
                    "message": (
                        "Too much at once — that resolved to more clips than the "
                        "video encoder can open in one go. Try a shorter sentence."
                        if crowded else
                        "The video encoder failed on this input."
                    ),
                },
            ) from exc
        elapsed = time.perf_counter() - t0
        log.info("DONE  %.2fs  found=%d  spliced=%d  missing=%d  url=%s",
                 elapsed, len(result["found"]), len(result["spliced"]),
                 len(result["missing"]), result.get("video_url"))
        if not result["found"] and not result["spliced"]:
            raise HTTPException(status_code=404, detail={
                "message": "No clips found or spliced for any word in the input.",
                "missing": result["missing"],
            })
        return result

    job = jobs.submit(text, subtitles=req.subtitles)
    log.info("QUEUE  %s  %r", job.id, text[:80])
    body = job.as_dict()
    body["position"] = jobs.position(job.id)
    return JSONResponse(status_code=202, content=body)


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    """How a queued generation is going, and its result once it is done."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={
            "message": "No such job. It may have finished long enough ago to be forgotten.",
        })
    body = job.as_dict()
    if job.status == "queued":
        body["position"] = jobs.position(job_id)

    # A job that produced nothing usable is reported the same way the old
    # synchronous endpoint did, so the page can keep one code path for it.
    r = job.result
    if job.status == "done" and r is not None and not r["found"] and not r["spliced"]:
        body["status"] = "error"
        body["error"] = "No clips found or spliced for any word in the input."
        body["error_kind"] = "nothing_found"
        body["missing"] = r["missing"]
    return body


@router.get("/queue")
def queue_stats():
    return jobs.stats()


@router.post("/rate")
def rate(req: RateRequest):
    """Record feedback on the clips behind a word, however it was made.

    Applies to found clips and runs as well as phoneme splices: a vote shifts
    how likely those clips are to be chosen for this word next time, rather
    than ruling them in or out. rating 0 clears an earlier vote.
    """
    word = req.word.strip().lower()
    if not word or not req.clips:
        raise HTTPException(status_code=400, detail="word and clips are required")
    from app.generate import rate_splice, _vote_weight
    delta = 0 if req.rating == 0 else (1 if req.rating > 0 else -1)
    scores = rate_splice(word, req.clips, delta)
    log.info("RATE  %s  clips=%s  delta=%+d  -> %s", word, req.clips, delta, scores)
    return {
        "status": "ok",
        "word": word,
        "scores": scores,
        # What the vote actually did, so the UI can say so rather than implying
        # a downvote banned the clip.
        "weights": {str(cid): round(_vote_weight(s), 3) for cid, s in scores.items()},
    }


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


def _corpus_totals() -> dict:
    with get_db() as conn:
        return {
            "clips": conn.execute("SELECT COUNT(*) FROM word_clips").fetchone()[0],
            "words": conn.execute("SELECT COUNT(DISTINCT word) FROM word_clips").fetchone()[0],
            "sources": conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
            "splice_mode": splice_mode(),
            "max_units": _max_units_setting(),
        }


@router.get("/corpora")
def corpora():
    """Every installed corpus, with the size of each.

    Counting means opening each database in turn, which is cheap at this scale
    and saves the frontend from having to switch corpus just to find out how
    big one is.
    """
    current = active()
    out = []
    for c in list_corpora():
        entry = {"slug": c["slug"], "name": c["name"], "active": c["slug"] == current["slug"]}
        try:
            set_active(c["slug"])
            entry.update(_corpus_totals())
        except Exception as exc:
            entry["error"] = str(exc)
        out.append(entry)
    # Always put the selection back, including when a corpus above failed to
    # open -- otherwise merely listing them would change which one is live.
    set_active(current["slug"])
    return {"corpora": out, "active": current["slug"]}


def _corpus_pack():
    """scripts/corpus.py, reached the way editor.py reaches align_phones --
    it lives outside app/ because it is a standalone CLI too."""
    import sys
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import corpus as corpus_pack
    return corpus_pack


@router.get("/corpus/export")
def export_corpus(slug: str | None = None):
    """Download a bundle of a corpus: database, videos, transcripts,
    pronunciations. The same tarball `scripts/corpus.py pack` builds --
    streamed back and deleted after, so exporting from the browser leaves
    nothing extra sitting on the server.
    """
    target = slug or active()["slug"]
    if not any(c["slug"] == target for c in list_corpora()):
        raise HTTPException(status_code=404, detail=f"No corpus called {target!r}")

    fd, tmp = tempfile.mkstemp(suffix=".tar.zst", prefix="export-")
    os.close(fd)
    os.remove(tmp)          # pack_bundle writes the file itself; this only reserved a name
    try:
        out = _corpus_pack().pack_bundle(target, tmp)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (Exception, SystemExit) as exc:                      # noqa: BLE001
        # SystemExit too: _corpus_root() raises it for a corpus that vanished
        # between the check above and here, and that must not kill the worker.
        raise HTTPException(status_code=500, detail=f"pack failed: {exc}") from exc

    log.info("EXPORT  %s -> %s (%s)", target, out,
             f"{os.path.getsize(out) / 1e6:.1f} MB")
    ext = ".tar.zst" if out.endswith(".zst") else ".tar.gz"
    return FileResponse(out, filename=f"{target}-{time.strftime('%Y-%m-%d')}{ext}",
                        media_type="application/octet-stream",
                        background=BackgroundTask(os.remove, out))


@router.post("/corpus/import")
async def import_corpus(name: str = Form(...), bundle: UploadFile = File(...),
                        force: bool = Form(False)):
    """Install an uploaded bundle as a new corpus, alongside any already
    installed. What `scripts/corpus.py install` does from a terminal, from
    the browser instead -- the same round trip this project already leans on
    to move a corpus between machines without losing hand edits.
    """
    slug_name = name.strip()
    if not slug_name:
        raise HTTPException(status_code=400, detail="name must not be empty")

    fname = bundle.filename or ""
    suffix = ".tar.zst" if fname.endswith(".zst") else \
             ".tar.gz" if fname.endswith((".gz", ".tgz")) else ".tar"
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="import-")
    size = 0
    try:
        with os.fdopen(fd, "wb") as fh:
            while chunk := await bundle.read(1 << 20):
                fh.write(chunk)
                size += len(chunk)

        try:
            manifest = _corpus_pack().install_bundle(tmp, slug_name, force)
        except FileExistsError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"a corpus already lives at {exc} (tick overwrite to replace it)"
            ) from exc
        except Exception as exc:                                # noqa: BLE001
            # Anything else here means the upload wasn't a readable bundle --
            # not a tarfile, not zstd/gzip, or missing what pack always writes.
            raise HTTPException(
                status_code=400, detail=f"could not read that bundle: {exc}") from exc
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    log.info("IMPORT  %s -> %s (%s uploaded)", fname, manifest["slug"],
             f"{size / 1e6:.1f} MB")
    return {"status": "ok", **manifest}


@router.get("/splice-mode")
def get_splice_mode():
    """How hard the splicer may push for the active corpus."""
    return {"mode": splice_mode(), "modes": list(SPLICE_MODES)}


@router.post("/splice-mode")
def put_splice_mode(req: SpliceModeRequest):
    """Set it. Stored in the corpus database, so it travels with the corpus
    and survives a restart without anything being configured on the server."""
    mode = req.mode.lower().strip()
    if mode not in SPLICE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of {', '.join(SPLICE_MODES)}, got {req.mode!r}")
    set_setting("splice_mode", mode)
    # Only the plan changes, not the clips, so the cache stays valid.
    log.info("SPLICE  mode set to %s for %s", mode, active()["slug"])
    return {"mode": mode, "corpus": active()["slug"]}


@router.get("/max-units")
def get_max_units():
    """How many pieces a spliced word may be built from."""
    from app.database import (DEFAULT_MAX_UNITS, MAX_MAX_UNITS, MIN_MAX_UNITS,
                              max_units)
    return {"units": max_units(), "default": DEFAULT_MAX_UNITS,
            "min": MIN_MAX_UNITS, "max": MAX_MAX_UNITS}


@router.post("/max-units")
def put_max_units(req: MaxUnitsRequest):
    """Set it, for the active corpus.

    Raising it lets a small corpus build words it otherwise reports missing --
    a thirty-word corpus has to take a phoneme at a time -- at the cost of a
    join per piece, which is where a splice sounds like one. Stored beside the
    splice mode, in the corpus, so it travels with it.
    """
    from app.database import MAX_MAX_UNITS, MIN_MAX_UNITS
    if not MIN_MAX_UNITS <= req.units <= MAX_MAX_UNITS:
        raise HTTPException(
            status_code=400,
            detail=f"units must be between {MIN_MAX_UNITS} and {MAX_MAX_UNITS}, "
                   f"got {req.units}")
    set_setting("max_units", str(req.units))
    # The plan changes, not the clips, so the clip cache stays valid -- but the
    # generator caches nothing about splices, so there is nothing else to drop.
    log.info("SPLICE  max units set to %s for %s", req.units, active()["slug"])
    return {"units": req.units, "corpus": active()["slug"]}


@router.post("/corpus")
def switch_corpus(req: CorpusRequest):
    """Select a corpus. Everything cached from the old one is dropped."""
    try:
        chosen = set_active(req.slug)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No corpus called {req.slug!r}")

    # The clip cache, the word positions and the splice scores are all built
    # from the previous database. Left in place they would splice the new
    # corpus's words out of the old corpus's video files.
    from app.generate import invalidate_cache
    invalidate_cache()
    init_db()

    log.info("CORPUS  switched to %s", chosen["slug"])
    return {"status": "ok", "active": chosen["slug"], "name": chosen["name"], **_corpus_totals()}


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
    current = active()
    return {
        "total_clips": total_clips,
        "unique_words": unique_words,
        "sources": sources,
        "sample_words": [r[0] for r in sample],
        "corpus": current["slug"],
        "corpus_name": current["name"],
    }
