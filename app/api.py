import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from app import jobs
from app.database import (max_units as _max_units_setting,
                          SPLICE_MODES, active, get_db, init_db,
                          list_corpora, set_active, set_setting, splice_mode)

router = APIRouter()
log = logging.getLogger(__name__)

# Longest sentence accepted. Far past anything anyone types -- a whole page of
# text is a quarter of this -- and short of a paste that ties the one worker
# up for hours.
MAX_TEXT = 4000

# Largest corpus bundle accepted. The shipped corpus packs to ~110 MB.
MAX_BUNDLE_BYTES = 4 << 30
_webhook_slots = threading.BoundedSemaphore(32)
_active_mutations = 0
_request_mutation = ContextVar("request_mutation", default=False)


@contextmanager
def request_mutation():
    """Reserve admitted writes until they finish, including asynchronous I/O."""
    global _active_mutations
    with jobs.corpus_guard():
        if jobs.maintenance_in_progress():
            raise HTTPException(status_code=409, detail="a corpus import or export is in progress")
        _active_mutations += 1
    token = _request_mutation.set(True)
    try:
        yield
    finally:
        _request_mutation.reset(token)
        with jobs.corpus_guard():
            _active_mutations -= 1


def _queued(submit):
    """Put a job in the queue, or say plainly that the queue is full."""
    try:
        return submit()
    except jobs.CorpusBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except jobs.QueueFull:
        raise HTTPException(status_code=503, detail={
            "message": f"{jobs.MAX_QUEUED} videos are already waiting. Try again in a few minutes."})


class GenerateRequest(BaseModel):
    text: str
    # Burned into the picture rather than written to a sidecar file: the
    # output of this thing is shared as a video, and a .vtt nobody carries
    # with it is a caption nobody sees.
    #
    # On by default. A spliced sentence is not always intelligible first time
    # -- that is half the joke -- and the words on screen are what let someone
    # hear what it was going for. Pass false to turn them off.
    subtitles: bool = True
    # How it is said: speed, pitch, gaps, how long a stretched letter is held,
    # and whether to prefer clean takes and real phrases. See
    # app.generate.OPTIONS; anything left out takes its default.
    options: dict = {}


class BatchGenerateRequest(BaseModel):
    sentences: list[str] = Field(default_factory=list, max_length=jobs.MAX_QUEUED)
    texts: list[str] = Field(default_factory=list, max_length=jobs.MAX_QUEUED)
    subtitles: bool = False
    options: dict = {}
    webhook_url: str | None = None


class SpliceModeRequest(BaseModel):
    mode: str


class MaxUnitsRequest(BaseModel):
    units: int


class CorpusRequest(BaseModel):
    slug: str


class RateRequest(BaseModel):
    word: str
    clips: list[int] = Field(max_length=256)
    rating: int = -1   # < 0 down-vote, > 0 up-vote, 0 clears the vote


class BoundaryReport(BaseModel):
    clips: list[int] = Field(max_length=256)
    kind: str


@router.post("/boundary")
def boundary(req: BoundaryReport):
    """Report the clips behind a word as cut in the wrong place."""
    if not req.clips:
        raise HTTPException(status_code=400, detail="clips are required")
    from app.generate import report_boundary
    try:
        totals = report_boundary(req.clips, req.kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log.info("BOUNDARY  clips=%s  %s", req.clips, req.kind)
    return {"status": "ok", "reports": totals}


@router.get("/clips/{word}")
def clips_for(word: str, voice: str = ""):
    """The recordings a word can be said with, and where each came from."""
    from app.database import active
    from app.generate import clip_choices
    return {"corpus": voice or active()["slug"], "clips": clip_choices(word, voice=voice or None)}


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
    if len(text) > MAX_TEXT:
        raise HTTPException(status_code=400, detail={
            "message": f"That is {len(text):,} characters; {MAX_TEXT:,} is the most one video takes."})

    job = _queued(lambda: jobs.submit(text, subtitles=req.subtitles, options=req.options))
    log.info("QUEUE  %s  %r%s", job.id, text[:80], "  (waiting)" if wait else "")

    if wait:
        # Through the queue like everything else. Generating right here ran a
        # second video beside the worker's -- the one-at-a-time rule the queue
        # exists for, and the seed and voice both being process-wide.
        job.ended.wait()
        if job.status == "cancelled":
            raise HTTPException(status_code=409, detail={"message": "that job was cancelled"})
        if job.status == "error":
            raise HTTPException(status_code=503 if job.error_kind == "crowded" else 500,
                                detail={"message": job.error})
        result = job.result
        if not result["found"] and not result["spliced"]:
            raise HTTPException(status_code=404, detail={
                "message": "No clips found or spliced for any word in the input.",
                "missing": result["missing"],
            })
        return result

    body = job.as_dict()
    body["position"] = jobs.position(job.id)
    return JSONResponse(status_code=202, content=body)


@router.post("/generate/batch")
def generate_batch(req: BatchGenerateRequest):
    """Queue multiple sentences as batch generation jobs with optional webhook notification."""
    sentences = req.sentences or req.texts
    if not sentences:
        raise HTTPException(status_code=400, detail="sentences must not be empty")

    cleaned = []
    for s in sentences:
        if not isinstance(s, str) or not s.strip():
            raise HTTPException(status_code=400, detail="Each sentence must be a non-empty string")
        t = s.strip()
        if len(t) > MAX_TEXT:
            raise HTTPException(status_code=400, detail=f"Sentence exceeds maximum length of {MAX_TEXT:,} characters")
        cleaned.append(t)

    webhook_host = None
    if req.webhook_url:
        from app import webhooks
        try:
            webhook_host = webhooks.validate(req.webhook_url).host
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not _webhook_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="too many webhook notifications are waiting")

    queued_jobs = []
    try:
        queued_jobs = jobs.submit_batch(cleaned, subtitles=req.subtitles, options=req.options)
    except jobs.CorpusBusy as exc:
        if req.webhook_url:
            _webhook_slots.release()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except jobs.QueueFull:
        if req.webhook_url:
            _webhook_slots.release()
        raise HTTPException(status_code=503, detail={"message": f"{jobs.MAX_QUEUED} videos are already waiting. Try again in a few minutes."})
    except BaseException:
        if req.webhook_url:
            _webhook_slots.release()
        raise

    if req.webhook_url:
        def _notify_webhook(url: str, batch: list[jobs.Job]):
            try:
                for j in batch:
                    j.ended.wait()
                webhooks.deliver(url, {"event": "batch_completed", "count": len(batch),
                                      "jobs": [j.as_dict() for j in batch]})
                log.info("WEBHOOK delivered to %s for %d jobs", webhook_host, len(batch))
            except Exception:
                # Callback query strings can contain credentials. Keep them
                # and caller-controlled exception text out of the log.
                log.warning("WEBHOOK delivery failed to %s", webhook_host)
            finally:
                _webhook_slots.release()

        t = threading.Thread(target=_notify_webhook, args=(req.webhook_url, list(queued_jobs)), daemon=True)
        try:
            t.start()
        except BaseException:
            _webhook_slots.release()
            raise

    return JSONResponse(status_code=202, content={
        "status": "queued",
        "count": len(queued_jobs),
        "job_ids": [j.id for j in queued_jobs],
        "jobs": [j.as_dict() for j in queued_jobs],
    })


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
    if (job.kind == "say" and job.status == "done" and r is not None
            and not r["found"] and not r["spliced"]):
        body["status"] = "error"
        body["error"] = "No clips found or spliced for any word in the input."
        body["error_kind"] = "nothing_found"
        body["missing"] = r["missing"]
    return body


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    """Cancel a queued or running job."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={
            "message": "No such job. It may have finished long enough ago to be forgotten.",
        })
    success = jobs.cancel(job_id)
    return {"id": job_id, "status": job.status, "cancelled": success}


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
    from app.ytpmv import samples
    samples.clear_cache()       # built from the old timings
    _sample_words_cache.clear()
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


@router.get("/noises")
def noises():
    """The non-verbal clips this voice has, and how many of each.

    The tokens are typed as *spew* / *click*, and nothing anywhere said which
    ones existed -- the markup was documented and its vocabulary was not, so
    the only way to find a noise was to guess a word and see.
    """
    from app.ytpmv import samples
    return {"noises": samples.noises()}


@router.get("/suggest")
def suggest(context: str = "", prefix: str = "", limit: int = 10):
    """Autocomplete: phrase continuations from real spoken runs, plus
    vocabulary prefix matches."""
    from app.generate import suggest_next
    return suggest_next(context, prefix, max(1, min(limit, 25)))


def _corpus_totals() -> dict:
    return {**_totals_of(active()), "splice_mode": splice_mode(),
            "max_units": _max_units_setting()}


def _totals_of(corpus: dict) -> dict:
    """Count one corpus without making it the active one.

    Counting used to mean switching to each corpus in turn and switching back.
    That is a global change for a read: a request arriving mid-count built its
    clip cache against the wrong corpus, and every path in it then pointed at
    another voice's directory -- ffmpeg looking for a Michael Rosen video
    inside james-channel. Read-only, and the live corpus never moves.
    """
    import sqlite3
    db = corpus.get("db")
    if not db or not os.path.exists(db):
        return {"clips": 0, "words": 0, "sources": 0}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        out = {
            "clips": conn.execute("SELECT COUNT(*) FROM word_clips").fetchone()[0],
            "words": conn.execute("SELECT COUNT(DISTINCT word) FROM word_clips").fetchone()[0],
            "sources": conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
        }
        for key, name in (("splice_mode", "splice_mode"), ("max_units", "max_units"), ("schema_version", "schema_version")):
            try:
                row = conn.execute("SELECT value FROM settings WHERE key=?", (name,)).fetchone()
            except sqlite3.Error:
                row = None
            if row:
                out[key] = int(row[0]) if key in ("max_units", "schema_version") else row[0]
            elif key == "schema_version":
                out[key] = 0
        return out
    finally:
        conn.close()


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
            entry.update(_totals_of(c))
        except Exception as exc:                                # noqa: BLE001
            entry["error"] = str(exc)
        out.append(entry)
    return {"corpora": out, "active": current["slug"]}


def _corpus_pack():
    """scripts/corpus.py -- it lives outside app/ because it is a standalone CLI too."""
    from scripts import corpus as corpus_pack
    return corpus_pack


def _remove_files(*paths):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _require_corpus_idle():
    """Called under jobs.corpus_guard, shared with job/alignment reservation."""
    from app.editor import alignment_running
    if jobs.has_live_jobs() or alignment_running():
        raise HTTPException(status_code=409,
                            detail="wait for generation and alignment to finish before changing a corpus")


def _begin_maintenance(require_idle=False):
    with jobs.corpus_guard():
        if _active_mutations > int(_request_mutation.get()):
            raise HTTPException(status_code=409, detail="wait for other corpus changes to finish")
        if require_idle:
            _require_corpus_idle()
        try:
            jobs.begin_corpus_maintenance()
        except jobs.CorpusBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


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
    _begin_maintenance()
    try:
        out = _corpus_pack().pack_bundle(target, tmp)
    except FileNotFoundError as exc:
        _remove_files(tmp, tmp.removesuffix(".tar.zst") + ".tar.gz")
        raise HTTPException(status_code=404, detail="that corpus is no longer available") from exc
    except (Exception, SystemExit) as exc:                      # noqa: BLE001
        # SystemExit too: _corpus_root() raises it for a corpus that vanished
        # between the check above and here, and that must not kill the worker.
        log.exception("EXPORT  %s failed", target)
        _remove_files(tmp, tmp.removesuffix(".tar.zst") + ".tar.gz")
        raise HTTPException(status_code=500, detail="could not pack that corpus") from exc
    finally:
        jobs.end_corpus_maintenance()

    log.info("EXPORT  %s -> %s (%s)", target, out,
             f"{os.path.getsize(out) / 1e6:.1f} MB")
    ext = ".tar.zst" if out.endswith(".zst") else ".tar.gz"
    return FileResponse(out, filename=f"{target}-{time.strftime('%Y-%m-%d')}{ext}",
                        media_type="application/octet-stream",
                        background=BackgroundTask(_remove_files, out))


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
        await bundle.close()
        raise HTTPException(status_code=400, detail="name must not be empty")

    fname = bundle.filename or ""
    suffix = ".tar.zst" if fname.endswith(".zst") else \
             ".tar.gz" if fname.endswith((".gz", ".tgz")) else ".tar"
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="import-")
    def install_upload():
        size = 0
        with os.fdopen(fd, "wb") as fh:
            while chunk := bundle.file.read(1 << 20):
                size += len(chunk)
                if size > MAX_BUNDLE_BYTES:
                    raise HTTPException(status_code=413, detail=(
                        f"that bundle is over {MAX_BUNDLE_BYTES >> 30} GB"))
                fh.write(chunk)

        try:
            _begin_maintenance(require_idle=force)
            try:
                manifest = _corpus_pack().install_bundle(tmp, slug_name, force)
                # Reserve maintenance until old-file caches have been dropped.
                from app.database import forget_ready
                from app.generate import invalidate_cache
                forget_ready()
                invalidate_cache()
                _sample_words_cache.clear()
            finally:
                jobs.end_corpus_maintenance()
            return manifest, size
        except FileExistsError as exc:
            raise HTTPException(
                status_code=409,
                detail="a corpus with that name already exists (tick overwrite to replace it)"
            ) from exc
        except HTTPException:
            raise
        except (Exception, SystemExit) as exc:                  # noqa: BLE001
            # Anything else here means the upload wasn't a readable bundle --
            # not a tarfile, not zstd/gzip, or missing what pack always writes.
            log.warning("IMPORT could not read bundle (%s)", type(exc).__name__)
            raise HTTPException(status_code=400, detail="could not read that bundle") from exc

    try:
        # Multipart parsing already spooled the upload. Disk copies, extraction
        # and validation can take seconds; keep the event loop free to respond.
        manifest, size = await run_in_threadpool(install_upload)
    finally:
        _remove_files(tmp)
        await bundle.close()
    log.info("IMPORT  %s -> %s (%s uploaded)", fname, manifest["slug"],
             f"{size / 1e6:.1f} MB")
    return {"status": "ok", **{k: v for k, v in manifest.items() if k not in ("target", "backup")},
            "backup_saved": bool(manifest.get("backup"))}


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
    with jobs.corpus_guard():
        if jobs.maintenance_in_progress():
            raise HTTPException(status_code=409, detail="a corpus import or export is in progress")
        _require_corpus_idle()
        try:
            chosen = set_active(req.slug)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"No corpus called {req.slug!r}")

        from app.database import using_corpus
        from app.generate import invalidate_cache
        with using_corpus(chosen):
            invalidate_cache()
            _sample_words_cache.clear()
            init_db()
            totals = _corpus_totals()
        log.info("CORPUS  switched to %s", chosen["slug"])
        return {"status": "ok", "active": chosen["slug"], "name": chosen["name"], **totals}


_sample_words_cache: dict[str, list[str]] = {}


@router.get("/stats")
def stats():
    current = active()
    totals = _totals_of(current)
    slug = current["slug"]
    if slug not in _sample_words_cache:
        with get_db() as conn:
            sample = conn.execute(
                "SELECT DISTINCT word FROM word_clips ORDER BY RANDOM() LIMIT 30"
            ).fetchall()
            _sample_words_cache[slug] = [r[0] for r in sample]
    return {
        "total_clips": totals["clips"],
        "unique_words": totals["words"],
        "sources": totals["sources"],
        "sample_words": _sample_words_cache[slug],
        "corpus": current["slug"],
        "corpus_name": current["name"],
        "schema_version": totals.get("schema_version", 0),
        "needs_upgrade": totals.get("schema_version", 0) < 1,
    }


@router.post("/corpus/upgrade")
def upgrade_corpus():
    """Bring the active corpus database to the current schema version."""
    from app.database import forget_ready, init_db, corpus_meta
    forget_ready()
    init_db()
    return corpus_meta()


@router.get("/settings/server")
def get_server_settings():
    from app import clip_cache
    import app.generate as gen
    return {
        "encode_workers": gen._encode_workers(),
        "encoder": os.environ.get("MRS_ENCODER", "libx264"),
        "available_encoders": gen._available_encoders(),
        "clip_cache_mb": round(clip_cache.cache_size_mb(), 1),
        "clip_cache_max_mb": clip_cache.MAX_CACHE_MB,
    }


@router.post("/settings/server")
def set_server_settings(req: dict):
    if "encode_workers" in req:
        os.environ["MRS_ENCODE_WORKERS"] = str(max(1, min(8, int(req["encode_workers"]))))
    if "encoder" in req:
        os.environ["MRS_ENCODER"] = str(req["encoder"]).lower().strip()
    return get_server_settings()


@router.get("/corpus/diagnostics")
def api_corpus_diagnostics():
    from app.editor import corpus_diagnostics
    return corpus_diagnostics()


# ── YTPMV ─────────────────────────────────────────────────────────────────────
# A MIDI file played by the corpus. Upload one to have it read and a sound
# suggested for every part; audition parts; then render through the same
# queue as sentences.

class YtpmvSampleRequest(BaseModel):
    text: str
    take: int = 0
    hit: str | None = None      # a drum to cut it into: kick, snare, hats...
    corpus: str | None = None   # installed voice for this sample; omitted uses the active voice


class YtpmvPreviewRequest(BaseModel):
    midi_id: str
    part: dict
    seconds: float = 6.0
    chip: str | None = None     # the song's chip switch, so a part previews on it
    synth: bool = False         # no voice: the part is the chip and nothing else


class YtpmvRenderRequest(BaseModel):
    midi_id: str
    parts: list[dict] = []
    options: dict = {}


class YtpmvMixPreviewRequest(YtpmvRenderRequest):
    start: float = 0
    seconds: float | None = 10  # null plays through to the song end
    part_id: str | None = None
    note: int | None = None      # instrument audition: one MIDI note, not the song pattern
    notes: list[int] | None = None  # prepare the visible keyboard as one audio bank


def _ytpmv_call(fn, *args, **kwargs):
    from app.ytpmv.render import YtpmvError
    from app.ytpmv.samples import SampleError
    try:
        return fn(*args, **kwargs)
    except (YtpmvError, SampleError) as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc


@router.post("/ytpmv/midi")
async def ytpmv_upload(midi: UploadFile = File(...)):
    """Store a MIDI file and describe it: key, tempo, parts, suggested sounds.

    The first song on a voice takes a few seconds longer while the suggested
    words are measured; after that the measurements are cached with the voice.
    """
    from starlette.concurrency import run_in_threadpool
    from app.ytpmv import render
    data = await midi.read(render.MAX_MIDI_BYTES + 1)
    midi_id = _ytpmv_call(render.save_midi, data, midi.filename or "")
    log.info("YTPMV  upload %s -> %s", midi.filename, midi_id)
    return await run_in_threadpool(_ytpmv_call, render.analyse, midi_id)


@router.get("/ytpmv/library")
def ytpmv_library():
    """Every MIDI file kept, newest first, with what is worth knowing about each."""
    from app.ytpmv import render
    return render.library()


@router.get("/ytpmv/midi/{midi_id}")
def ytpmv_describe(midi_id: str):
    """The same description for a file already uploaded."""
    from app.ytpmv import render
    return _ytpmv_call(render.analyse, midi_id)


class YtpmvRenameRequest(BaseModel):
    title: str


@router.patch("/ytpmv/midi/{midi_id}")
def ytpmv_rename(midi_id: str, req: YtpmvRenameRequest):
    from app.ytpmv import render
    return _ytpmv_call(render.rename_midi, midi_id, req.title)


@router.delete("/ytpmv/midi/{midi_id}")
def ytpmv_delete(midi_id: str):
    from app.ytpmv import render
    with jobs.corpus_guard():
        if jobs.midi_in_use(midi_id):
            raise HTTPException(409, "This MIDI is in use. Stop its playback preparation or cancel its render before deleting it.")
        return _ytpmv_call(render.delete_midi, midi_id)


@router.post("/ytpmv/sample")
def ytpmv_sample(req: YtpmvSampleRequest):
    """What a word sounds like as an instrument: the clip, and the note it is on."""
    from app.ytpmv import samples
    return _ytpmv_call(lambda: samples.build(req.text, req.take, req.hit or None,
                                           corpus=req.corpus).describe())


@router.get("/ytpmv/midi/{midi_id}/part/{part_id}")
def ytpmv_part_voice(midi_id: str, part_id: str, corpus: str | None = None):
    """Refresh one part's recommendations and sounds for the chosen voice."""
    from app.ytpmv import render
    return _ytpmv_call(render.recommend_part, midi_id, part_id, corpus)


@router.get("/ytpmv/midi/{midi_id}/notes")
def ytpmv_timeline_notes(midi_id: str, start: float = 0, end: float | None = None, parts: str = ""):
    """Bounded, accurate note detail for the visible timeline tracks and range."""
    from app.ytpmv import render
    if len(parts) > 4096:
        raise HTTPException(400, "Too many timeline parts.")
    return _ytpmv_call(render.timeline_notes, midi_id, start, end, parts.split(",") if parts else [])


@router.post("/ytpmv/preview")
def ytpmv_preview(req: YtpmvPreviewRequest):
    """A few seconds of one part, sung as it will be in the render (audio only)."""
    from app.ytpmv import render
    seconds = max(1.0, min(float(req.seconds), 15.0))
    return _ytpmv_call(render.preview_part, req.midi_id, req.part, seconds, req.chip, req.synth)


@router.post("/ytpmv/render")
def ytpmv_render(req: YtpmvRenderRequest):
    """Queue a render; poll /api/jobs/{id} as for a sentence."""
    from app.ytpmv import render
    song = _ytpmv_call(render.load_song, req.midi_id)
    parts = _ytpmv_call(render.prepare_parts, req.parts, bool(req.options.get("synth", False)))
    job = _queued(lambda: jobs.submit_ytpmv({"midi_id": req.midi_id, "parts": parts,
                             "options": req.options},
                            label=song.title or f"midi {req.midi_id}"))
    log.info("QUEUE  %s  ytpmv %s", job.id, req.midi_id)
    body = job.as_dict()
    body["position"] = jobs.position(job.id)
    return JSONResponse(status_code=202, content=body)


@router.post("/ytpmv/preview-mix")
def ytpmv_preview_mix(req: YtpmvMixPreviewRequest):
    """Queue audio playback of a selection, or the remaining song when seconds is null."""
    from app.ytpmv import render
    song = _ytpmv_call(render.load_song, req.midi_id)
    parts = _ytpmv_call(render.prepare_parts, req.parts, bool(req.options.get("synth", False)))
    start = render._num(req.start, 0, 0, render.MAX_SECONDS)
    if req.note is not None and (not req.part_id or not 0 <= req.note <= 127):
        raise HTTPException(400, "Choose a track and a MIDI note from 0 to 127.")
    if req.notes is not None and (not req.part_id or req.note is not None or not 1 <= len(req.notes) <= 24
                                 or len(set(req.notes)) != len(req.notes)
                                 or any(not 0 <= note <= 127 for note in req.notes)):
        raise HTTPException(400, "Choose a track and 1–24 distinct keyboard notes from 0 to 127.")
    if req.part_id:
        part = song.part(req.part_id)
        if part is None:
            raise HTTPException(status_code=400, detail="No such part.")
        speed = render._num(req.options.get("speed", 1), 1, .25, 4)
        start = max(0, (part.notes[0].start / speed if part.notes else 0) - .05)
        if req.note is not None or req.notes is not None:
            start = 0
        given = {p.get("id"): p for p in parts}
        parts = [{**given.get(p.id, {"id": p.id}), "mute": p.id != req.part_id,
                  "visible": False} for p in song.parts]
        parts = _ytpmv_call(render.prepare_parts, parts, bool(req.options.get("synth", False)))
    seconds = (render._num(req.seconds, 10, 1, 15) if req.part_id else
               None if req.seconds is None else render._num(req.seconds, 10, 1, render.MAX_SECONDS))
    job = _queued(lambda: jobs.submit_ytpmv({"midi_id": req.midi_id, "parts": parts,
        "options": req.options, "audio_preview": True, "start": start, "seconds": seconds,
        "part_id": req.part_id, "note": req.note, "keyboard_notes": req.notes},
        label="Preview: " + (song.title or req.midi_id)))
    return JSONResponse(status_code=202, content={"id": job.id, "status": job.status,
                                                "position": jobs.position(job.id)})
