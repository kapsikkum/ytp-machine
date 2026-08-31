import logging
import os
import stat
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import init_db
from app.api import router

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

app = FastAPI(title="YTP Machine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
_KEEP_HOURS = {"splice_": 24, "piece_": 24}
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
    if removed:
        _app_logger.info("SWEEP   %d old output file(s), %.1f MB",
                         removed, freed / 1e6)


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
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
