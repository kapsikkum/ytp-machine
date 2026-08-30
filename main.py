import os
import logging

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

# Serve generated videos
app.mount("/output", StaticFiles(directory="output"), name="output")

# Serve frontend last so it catches "/"
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
