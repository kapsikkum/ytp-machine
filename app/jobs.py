"""A one-at-a-time job queue for video generation.

Generation used to run inside the request. A long sentence takes minutes --
one ffmpeg input per clip, a forced-alignment pass per spliced word -- and
nginx gives up well before that, so the browser got a 504 and the work carried
on invisibly to nobody's benefit. Requests are now short: submit returns a job
id straight away and the page asks how it is going.

Deliberately a single worker. Generation is ffmpeg-bound and this box shares
its CPU with several other services; two of these at once was what exhausted
the container's pid ceiling in the first place. Serialising is a feature, not a
limitation to be tuned away later -- a queue that runs two jobs concurrently
would reintroduce exactly the failure it was built to stop.

In-process and in-memory on purpose: one uvicorn worker, jobs that mean nothing
after a restart, and no appetite for a broker to run a toy. If this ever needs
to survive a restart or scale past one process, that is the point to reach for
something real, not before.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# How many finished jobs to remember. Enough that a page which polls slowly, or
# a tab reopened a few minutes later, still finds its result; small enough that
# nothing accumulates.
_KEEP = 50

# A finished job is forgotten after this long even if the cap is not reached,
# so a quiet day does not leave stale results lying around indefinitely.
_TTL_SECONDS = 30 * 60

# How many may wait at once. One worker at minutes a job: past this the last
# in line would wait an hour, and a script posting in a loop fills memory.
MAX_QUEUED = 20


class QueueFull(Exception):
    """Too many jobs waiting already."""


class CorpusBusy(Exception):
    """Corpus files are temporarily being maintained."""


@dataclass
class Job:
    id: str
    text: str                       # the sentence, or a label for other kinds
    subtitles: bool = False         # burn the words onto the picture
    kind: str = "say"               # say | ytpmv
    corpus: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"          # queued | running | done | error | cancelled
    stage: str = "waiting"          # loading | resolving | encoding | joining | cancelled
    done: int = 0
    total: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    error_kind: str | None = None   # "crowded" when ffmpeg ran out of room
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    ended: threading.Event = field(default_factory=threading.Event, repr=False)
    abort_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "kind": self.kind,
            "corpus": self.corpus,
            "status": self.status,
            "stage": self.stage,
            "done": self.done,
            "total": self.total,
            "queued_for": round((self.started or time.time()) - self.created, 1),
        }
        if self.status == "running":
            out["running_for"] = round(time.time() - (self.started or time.time()), 1)
        if self.finished:
            out["took"] = round(self.finished - (self.started or self.created), 1)
        if self.result is not None:
            out["result"] = self.result
        if self.error is not None:
            out["error"] = self.error
            out["error_kind"] = self.error_kind
        return out


_jobs: dict[str, Job] = {}
_order: list[str] = []
_queue: deque[str] = deque()
_lock = threading.Lock()
_ready = threading.Condition(_lock)
_worker_lock = threading.Lock()
_corpus_lock = threading.RLock()
_maintenance = False
_worker: threading.Thread | None = None


@contextmanager
def corpus_guard():
    """Make corpus maintenance checks atomic with new work reservations."""
    with _corpus_lock:
        yield


def has_live_jobs() -> bool:
    """Call inside corpus_guard when deciding whether maintenance can start."""
    with _lock:
        return any(job.status in ("queued", "running") for job in _jobs.values())


def midi_in_use(midi_id: str) -> bool:
    """Call inside corpus_guard before deleting a queued/running job's source."""
    with _lock:
        return any(job.kind == "ytpmv" and job.params.get("midi_id") == midi_id
                   and job.status in ("queued", "running") for job in _jobs.values())


def maintenance_in_progress() -> bool:
    with corpus_guard():
        return _maintenance


def begin_corpus_maintenance() -> None:
    """Reserve maintenance under corpus_guard, then release it during I/O."""
    global _maintenance
    with corpus_guard():
        if _maintenance:
            raise CorpusBusy
        _maintenance = True


def end_corpus_maintenance() -> None:
    global _maintenance
    with corpus_guard():
        _maintenance = False


def _prune() -> None:
    """Drop finished jobs that are old or surplus. Never drops live ones."""
    now = time.time()
    with _lock:
        stale = [
            jid for jid in _order
            if (j := _jobs.get(jid))
            and j.status in ("done", "error", "cancelled")
            and j.finished
            and now - j.finished > _TTL_SECONDS
        ]
        for jid in stale:
            _jobs.pop(jid, None)
            _order.remove(jid)

        finished = [jid for jid in _order
                    if _jobs.get(jid) and _jobs[jid].status in ("done", "error", "cancelled")]
        while len(finished) > _KEEP:
            jid = finished.pop(0)
            _jobs.pop(jid, None)
            _order.remove(jid)


def _run(job: Job) -> None:
    if not job.corpus or job.abort_event.is_set():
        _run_in_corpus(job)
        return
    from app.database import using_corpus
    try:
        with using_corpus(job.corpus):
            _run_in_corpus(job)
    except Exception as exc:
        job.status = "error"
        job.error = f"The requested corpus '{job.corpus}' is unavailable."
        job.error_kind = "request"
        job.finished = time.time()
        job.ended.set()
        log.warning("Failed to activate corpus %s for job %s: %s", job.corpus, job.id, exc)


def _run_in_corpus(job: Job) -> None:
    with _lock:
        if job.abort_event.is_set():
            job.status = "cancelled"
            job.stage = "cancelled"
            job.error = "Job cancelled"
            job.error_kind = "cancelled"
            job.finished = time.time()
            job.ended.set()
            return
        job.status = "running"
        job.started = time.time()

    def progress(stage: str, done: int, total: int) -> None:
        if job.abort_event.is_set():
            raise RuntimeError("Job cancelled")
        job.stage, job.done, job.total = stage, done, total

    try:
        if job.kind == "ytpmv":
            # Through the same single worker as sentences, on purpose: a song
            # is the heaviest thing this box renders, and running one beside a
            # sentence is exactly the contention the queue exists to prevent.
            from app.ytpmv.render import YtpmvError, render_ytpmv, preview_mix
            try:
                perform = preview_mix if job.params.get("audio_preview") else render_ytpmv
                job.result = perform(job.params, progress=progress)
            except YtpmvError as exc:
                job.error = str(exc)
                job.error_kind = "request"
                job.status = "error"
                return
        else:
            from app.generate import generate_video
            job.result = generate_video(job.text, progress=progress,
                                        subtitles=job.subtitles, options=job.params)
        if job.abort_event.is_set():
            job.status = "cancelled"
            job.stage = "cancelled"
            job.error = "Job cancelled"
            job.error_kind = "cancelled"
        else:
            job.status = "done"
            job.stage = "finished"
    except RuntimeError as exc:
        if job.abort_event.is_set():
            job.status = "cancelled"
            job.stage = "cancelled"
            job.error = "Job cancelled"
            job.error_kind = "cancelled"
            log.info("JOB %s cancelled", job.id)
            return
        detail = str(exc)
        crowded = "temporarily unavailable" in detail or "Too many open files" in detail
        # An OOM kill leaves nothing behind: ffmpeg dies on a signal without
        # printing, so the complaint is empty. Reporting "the encoder failed"
        # for that was true and useless -- an empty error is itself the clue.
        killed = not detail.split("FFmpeg failed:")[-1].strip()
        if crowded or killed:
            job.error_kind = "crowded"
            job.error = (
                "Ran out of room part-way through. Try a shorter stretch of "
                "the song, or fewer tiles."
                if job.kind == "ytpmv" else
                "Ran out of room part-way through. That usually means the "
                "sentence produced more clips than one encode can hold — "
                "try a shorter one."
            )
        else:
            job.error_kind = "encoder"
            job.error = "The video encoder failed on this input."
        job.status = "error"
        log.error("JOB %s failed: %s", job.id, detail[-500:])
    except Exception as exc:  # noqa: BLE001 -- a worker thread must not die
        if job.abort_event.is_set():
            job.status = "cancelled"
            job.stage = "cancelled"
            job.error = "Job cancelled"
            job.error_kind = "cancelled"
            log.info("JOB %s cancelled", job.id)
            return
        job.error = f"{type(exc).__name__}: {exc}"
        job.error_kind = "internal"
        job.status = "error"
        log.exception("JOB %s crashed", job.id)
    finally:
        job.finished = time.time()
        job.ended.set()


def _loop() -> None:
    while True:
        with _ready:
            while not _queue:
                _ready.wait()
            jid = _queue.popleft()
            job = _jobs.get(jid)
        if job is None:          # pruned before it ran; nothing to do
            continue
        if job.abort_event.is_set() or job.status == "cancelled":
            job.status = "cancelled"
            job.stage = "cancelled"
            job.finished = time.time()
            job.ended.set()
            _prune()
            continue
        log.info("JOB %s start  %r", job.id, job.text[:80])
        _run(job)
        log.info("JOB %s %s in %.1fs", job.id, job.status,
                 (job.finished or 0) - (job.started or 0))
        _prune()


def _ensure_worker() -> None:
    """Start the worker on first use rather than at import.

    Daemon, so it never keeps the process alive on shutdown -- a half-finished
    video is worth less than a container that stops when told to.
    """
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, name="ytp-worker", daemon=True)
            _worker.start()


def _enqueue(job: Job) -> Job:
    return _enqueue_many([job])[0]


def _enqueue_many(batch: list[Job]) -> list[Job]:
    with corpus_guard():
        if _maintenance:
            raise CorpusBusy
        _prune()
        with _ready:
            if sum(j.status == "queued" for j in _jobs.values()) + len(batch) > MAX_QUEUED:
                raise QueueFull
            for job in batch:
                _jobs[job.id] = job
                _order.append(job.id)
                _queue.append(job.id)
            _ready.notify()
        try:
            _ensure_worker()
        except BaseException:
            with _ready:
                for job in batch:
                    _jobs.pop(job.id, None)
                    _order.remove(job.id)
                    try:
                        _queue.remove(job.id)
                    except ValueError:
                        pass
            raise
    return batch


def submit(text: str, subtitles: bool = False, options: dict[str, Any] | None = None,
           corpus: str | None = None) -> Job:
    with corpus_guard():
        if corpus is None:
            try:
                from app.database import active
                corpus = active()["slug"]
            except Exception:
                corpus = ""
        return _enqueue(Job(id=uuid.uuid4().hex[:12], text=text, subtitles=subtitles,
                            corpus=corpus, params=deepcopy(options or {})))


def submit_ytpmv(params: dict[str, Any], label: str = "ytpmv",
                 corpus: str | None = None) -> Job:
    with corpus_guard():
        if corpus is None:
            try:
                from app.database import active
                corpus = active()["slug"]
            except Exception:
                corpus = ""
        return _enqueue(Job(id=uuid.uuid4().hex[:12], text=label, kind="ytpmv",
                            corpus=corpus, params=deepcopy(params)))


def submit_batch(texts: list[str], subtitles: bool = False,
                 options: dict[str, Any] | None = None) -> list[Job]:
    """Reserve an entire batch before any of its jobs can start."""
    with corpus_guard():
        try:
            from app.database import active
            corpus = active()["slug"]
        except Exception:
            corpus = ""
        batch = [Job(id=uuid.uuid4().hex[:12], text=text, subtitles=subtitles,
                     corpus=corpus, params=deepcopy(options or {})) for text in texts]
        return _enqueue_many(batch)


def cancel(job_id: str) -> bool:
    """Abort a queued or running job."""
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return False
        if job.status in ("done", "error", "cancelled"):
            return False
        job.abort_event.set()
        if job.status == "queued":
            # Removing a cancelled reservation keeps the physical backlog
            # bounded too, while the single worker is busy with a long render.
            try:
                _queue.remove(job_id)
            except ValueError:
                pass                            # the worker already took it
            job.status = "cancelled"
            job.stage = "cancelled"
            job.error = "Job cancelled"
            job.error_kind = "cancelled"
            job.finished = time.time()
            job.ended.set()
    _prune()
    return True


def get(job_id: str) -> Job | None:
    _prune()
    with _lock:
        return _jobs.get(job_id)


def position(job_id: str) -> int:
    """How many jobs are ahead of this one, 0 meaning next or already running."""
    with _lock:
        waiting = [jid for jid in _order
                   if _jobs.get(jid) and _jobs[jid].status == "queued"]
    return waiting.index(job_id) if job_id in waiting else 0


def stats() -> dict[str, int]:
    with _lock:
        statuses = [j.status for j in _jobs.values()]
    return {
        "queued": statuses.count("queued"),
        "running": statuses.count("running"),
        "cancelled": statuses.count("cancelled"),
        "remembered": len(statuses),
    }
