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
import queue
import threading
import time
import uuid
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


@dataclass
class Job:
    id: str
    text: str
    status: str = "queued"          # queued | running | done | error
    stage: str = "waiting"          # loading | resolving | encoding | joining
    done: int = 0
    total: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    error_kind: str | None = None   # "crowded" when ffmpeg ran out of room
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
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
_queue: "queue.Queue[str]" = queue.Queue()
_lock = threading.Lock()
_worker: threading.Thread | None = None


def _prune() -> None:
    """Drop finished jobs that are old or surplus. Never drops live ones."""
    now = time.time()
    with _lock:
        stale = [
            jid for jid in _order
            if (j := _jobs.get(jid))
            and j.status in ("done", "error")
            and j.finished
            and now - j.finished > _TTL_SECONDS
        ]
        for jid in stale:
            _jobs.pop(jid, None)
            _order.remove(jid)

        finished = [jid for jid in _order
                    if _jobs.get(jid) and _jobs[jid].status in ("done", "error")]
        while len(finished) > _KEEP:
            jid = finished.pop(0)
            _jobs.pop(jid, None)
            _order.remove(jid)


def _run(job: Job) -> None:
    from app.generate import generate_video

    def progress(stage: str, done: int, total: int) -> None:
        job.stage, job.done, job.total = stage, done, total

    job.status = "running"
    job.started = time.time()
    try:
        job.result = generate_video(job.text, progress=progress)
        job.status = "done"
        job.stage = "finished"
    except RuntimeError as exc:
        detail = str(exc)
        crowded = "temporarily unavailable" in detail or "Too many open files" in detail
        # An OOM kill leaves nothing behind: ffmpeg dies on a signal without
        # printing, so the complaint is empty. Reporting "the encoder failed"
        # for that was true and useless -- an empty error is itself the clue.
        killed = not detail.split("FFmpeg failed:")[-1].strip()
        if crowded or killed:
            job.error_kind = "crowded"
            job.error = (
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
        job.error = f"{type(exc).__name__}: {exc}"
        job.error_kind = "internal"
        job.status = "error"
        log.exception("JOB %s crashed", job.id)
    finally:
        job.finished = time.time()


def _loop() -> None:
    while True:
        jid = _queue.get()
        job = _jobs.get(jid)
        if job is None:          # pruned before it ran; nothing to do
            _queue.task_done()
            continue
        log.info("JOB %s start  %r", job.id, job.text[:80])
        _run(job)
        log.info("JOB %s %s in %.1fs", job.id, job.status,
                 (job.finished or 0) - (job.started or 0))
        _queue.task_done()
        _prune()


def _ensure_worker() -> None:
    """Start the worker on first use rather than at import.

    Daemon, so it never keeps the process alive on shutdown -- a half-finished
    video is worth less than a container that stops when told to.
    """
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_loop, name="ytp-worker", daemon=True)
        _worker.start()


def submit(text: str) -> Job:
    _ensure_worker()
    job = Job(id=uuid.uuid4().hex[:12], text=text)
    with _lock:
        _jobs[job.id] = job
        _order.append(job.id)
    _queue.put(job.id)
    return job


def get(job_id: str) -> Job | None:
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
        "remembered": len(statuses),
    }
