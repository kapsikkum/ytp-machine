import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.environ.get("MRS_DB_PATH", "michael_rosen.db")

# Project root (parent of app/) — source files are stored RELATIVE to this so
# the whole project can be moved without breaking clip paths.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalise_sep(p: str) -> str:
    r"""Backslashes to forward slashes.

    Stored paths are written by whichever machine ran the ingest, and
    os.path.join on Windows produces "downloads\clip.mp4". On Linux that is
    not a directory and a file -- it is one filename containing a backslash --
    so every clip lookup misses and the app generates nothing at all. Forward
    slashes resolve on both platforms, so they are the stored form.
    """
    return p.replace("\\", "/")


def resolve_path(p: str) -> str:
    """Turn a stored (relative) source path into an absolute one."""
    if not p:
        return p
    p = normalise_sep(p)
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def relativize_path(p: str) -> str:
    """Store paths relative to the project root (e.g. downloads/x.mp4)."""
    if not p:
        return p
    return "downloads/" + os.path.basename(normalise_sep(p))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id    TEXT NOT NULL,
                source_file TEXT NOT NULL,
                title       TEXT,
                url         TEXT,
                ingested_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS word_clips (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id   INTEGER NOT NULL REFERENCES sources(id),
                word        TEXT NOT NULL,
                start_time  REAL NOT NULL,
                end_time    REAL NOT NULL,
                source_file TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_wc_word ON word_clips(word);

            -- Non-verbal vocalisations (clicks, spews, pops, blows) that occur
            -- in the gaps Whisper skipped.  Kept separate so they don't break
            -- word runs / idle detection.
            CREATE TABLE IF NOT EXISTS noise_clips (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id   INTEGER NOT NULL REFERENCES sources(id),
                word        TEXT NOT NULL,
                start_time  REAL NOT NULL,
                end_time    REAL NOT NULL,
                source_file TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_nc_word ON noise_clips(word);

            -- User feedback on phoneme splices.  A negative score means "this
            -- clip sounded bad when used to splice this target word", so the
            -- splicer should avoid it for that word unless it has no choice.
            CREATE TABLE IF NOT EXISTS splice_ratings (
                word     TEXT NOT NULL,
                clip_id  INTEGER NOT NULL,
                score    INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (word, clip_id)
            );
        """)

        # Normalise any Windows separators left by an ingest run on Windows.
        # resolve_path() copes with them on the way out, but a database full of
        # backslashes is only portable because of that fallback -- fixing the
        # stored form makes the corpus itself portable, which is the point of
        # bundling it. Idempotent, and a no-op once done.
        for table in ("sources", "word_clips", "noise_clips"):
            conn.execute(
                f"UPDATE {table} SET source_file = replace(source_file, char(92), '/') "
                f"WHERE source_file LIKE '%' || char(92) || '%'"
            )
