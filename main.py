import logging
import os
import stat
import threading
import time
from urllib.parse import urlsplit
from contextlib import nullcontext

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.database import active, init_db, list_corpora, using_corpus
from app import jobs
from app.api import MAX_BUNDLE_BYTES, request_mutation, router

# ── Logging ───────────────────────────────────────────────────────────────────
# Use a file handler on the "app" logger with propagate=False so our messages
# go to app.log regardless of how uvicorn configures the root logger.
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")
_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", "%H:%M:%S")

_app_logger = logging.getLogger("app")
_app_logger.setLevel(logging.INFO)
_app_logger.propagate = False

# Avoid duplicate handlers on hot-reload
if not _app_logger.handlers:
    _fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    _fh.setFormatter(_fmt)
    _app_logger.addHandler(_fh)

    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    _app_logger.addHandler(_sh)

# ── App ───────────────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
os.makedirs("downloads", exist_ok=True)

init_db()

_MAX_BODY = MAX_BUNDLE_BYTES + (1 << 20)     # a bundle, and the form around it

app = FastAPI(title="YTP Machine", version="1.0.0")

# No CORS. The pages are served from here, so nothing needed it, and
# allow_origins=["*"] let any site you happened to visit delete clips or
# replace a corpus on this one.
#
# That still leaves the requests a browser sends cross-site without asking
# first -- a form post, a no-cors fetch of an upload -- so a change made from
# another origin is refused outright. Sec-Fetch-Site is set by the browser and
# cannot be forged by a page; the bot and curl send none and are unaffected.
#
# And the size of a request body is held here, before anything reads it:
# Starlette writes a whole multipart upload to a temporary file before the
# endpoint runs, so a limit inside /corpus/import came after the disk had
# already filled.
class _Guard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        writing = scope["method"] not in ("GET", "HEAD", "OPTIONS")
        if writing:
            denied = headers.get(b"sec-fetch-site") in (b"same-site", b"cross-site")
            if b"origin" in headers:
                try:
                    origin = urlsplit(headers[b"origin"].decode("ascii"))
                    here = urlsplit(scope["scheme"] + "://" + headers.get(b"host", b"").decode("ascii"))
                    origin_port = origin.port or (443 if origin.scheme == "https" else 80)
                    here_port = here.port or (443 if here.scheme == "https" else 80)
                    denied |= (origin.scheme not in ("http", "https") or origin.scheme != here.scheme
                               or origin.hostname != here.hostname or origin_port != here_port
                               or bool(origin.path or origin.query or origin.fragment or origin.username))
                except (ValueError, UnicodeError):
                    denied = True
            if denied:
                return await JSONResponse({"detail": "not from another origin"}, 403)(scope, receive, send)
        # MIDI and JSON requests cannot use the multi-gigabyte bundle budget.
        # Enforce before multipart parsing spools the entire file to disk.
        path = scope.get("path", "")
        limit = min(_MAX_BODY, MAX_BUNDLE_BYTES + (1 << 20) if path == "/api/corpus/import"
                    else 3 << 20 if path == "/api/ytpmv/midi" else 1 << 20)
        length = headers.get(b"content-length", b"")
        if length.isdigit() and int(length) > limit:
            return await JSONResponse({"detail": "that is too big to send"}, 413)(scope, receive, send)
        if writing and path in ("/api/corpus/import", "/api/ytpmv/midi"):
            content_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
            if content_type != b"multipart/form-data":
                # These endpoints require a file. Never give URL-encoded form
                # parsing the bundle's multi-gigabyte size allowance.
                return await JSONResponse({"detail": "file uploads require multipart/form-data"}, 415)(
                    scope, receive, send)

        seen = 0

        async def counted():
            # No length given (chunked): count it as it arrives instead.
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > limit:
                raise ValueError("request body over the limit")
            return message

        # ContextVars propagate through FastAPI's threadpool. A request keeps
        # using the corpus it began with even if another client switches later.
        if not path.startswith("/api/"):
            return await self.app(scope, counted, send)
        mismatch = False
        maintaining = False
        with jobs.corpus_guard():
            maintaining = writing and jobs.maintenance_in_progress()
            current = active()
            requested = headers.get(b"x-corpus", b"").decode("utf-8", errors="replace")
            if requested and requested != current["slug"]:
                # YTPMV operations create previews/renders, not corpus edits.
                # Their page-local default may differ from the server's voice.
                voice_operation = path in {"/api/ytpmv/midi", "/api/ytpmv/sample",
                                           "/api/ytpmv/preview", "/api/ytpmv/render",
                                           "/api/ytpmv/preview-mix"} or (
                    scope["method"] == "DELETE" and path.startswith("/api/jobs/"))
                if writing and not voice_operation:
                    mismatch = True
                else:
                    current = next((c for c in list_corpora() if c["slug"] == requested), None)
                    mismatch = current is None
        if maintaining:
            return await JSONResponse({"detail": "a corpus import or export is in progress"}, 409)(
                scope, receive, send)
        if mismatch:
            return await JSONResponse({"detail": "the active corpus changed; reload before editing"}, 409)(
                scope, receive, send)
        with using_corpus(current):
            try:
                with request_mutation() if writing else nullcontext():
                    await self.app(scope, counted, send)
            except HTTPException as exc:
                return await JSONResponse({"detail": exc.detail}, exc.status_code)(scope, receive, send)


app.add_middleware(_Guard)

app.include_router(router, prefix="/api")

# The corpus editor, on its own prefix. It is the only thing here that writes
# to word_clips outside an ingest, so it is kept apart from the generation API
# rather than folded into it.
from app.editor import router as editor_router  # noqa: E402
app.include_router(editor_router, prefix="/api/edit")

# ── Housekeeping ──────────────────────────────────────────────────────────────
# Every generation and every preview leaves an mp4 in output/, and nothing ever
# read them twice. Left alone that is a disk filling up with videos nobody will
# watch again -- the editor makes it faster, because auditioning twelve clips
# of one word writes twelve files in a minute.
#
# Previews go sooner than generations: a preview is listened to once, while a
# generated video is a link somebody may still have open.
_KEEP_HOURS = {"splice_": 24, "piece_": 24, "ytpmv_sample_": 24}
_KEEP_DEFAULT_HOURS = 24 * 7


def _sweep_output() -> None:
    now = time.time()
    removed = freed = 0
    try:
        names = os.listdir("output")
    except OSError:
        return
    for name in names:
        path = os.path.join("output", name)
        keep = next((h for p, h in _KEEP_HOURS.items() if name.startswith(p)),
                    _KEEP_DEFAULT_HOURS)
        try:
            st = os.stat(path)
            if not stat.S_ISREG(st.st_mode) or now - st.st_mtime < keep * 3600:
                continue
            os.remove(path)
        except OSError:
            continue           # in use, or gone already: it will come round again
        removed += 1
        freed += st.st_size
    _app_logger.info("SWEEP   %d file(s) kept, %d removed (%.1f MB)",
                     len(names) - removed, removed, freed / 1e6)


def _sweep_loop() -> None:
    """Once at startup, then daily.

    A plain daemon thread rather than a startup event: it needs no event loop,
    it cannot hold a shutdown open, and there is nothing here worth the
    ceremony of a lifespan handler.
    """
    while True:
        try:
            _sweep_output()
        except Exception:
            _app_logger.exception("output sweep failed")
        time.sleep(24 * 3600)


threading.Thread(target=_sweep_loop, daemon=True, name="output-sweep").start()


# Serve generated videos
app.mount("/output", StaticFiles(directory="output"), name="output")

# Serve frontend last so it catches "/"
# Compressed, but only the pages: index.html alone is 80 KB of inline script.
# Not app-wide -- that would gzip the videos too, and a compressed body has no
# byte ranges, so the player could no longer seek.
app.mount("/", GZipMiddleware(StaticFiles(directory="frontend", html=True), minimum_size=1024),
          name="frontend")
