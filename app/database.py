import os
import sqlite3
from contextlib import contextmanager

# Project root (parent of app/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where corpora live. Each is a directory holding one .db and a downloads/ of
# the videos that database indexes:
#
#     $MRS_DATA_DIR/corpora/michael-rosen/{corpus.db,downloads/}
#     $MRS_DATA_DIR/corpora/attenborough/{corpus.db,downloads/}
#
# Source paths are stored relative to the corpus directory rather than to the
# project, which is what lets several sit side by side: two corpora can both
# have a downloads/clip.mp4 without colliding, and a corpus stays portable
# because nothing in it refers to where it happens to be installed.
DATA_DIR = os.environ.get("MRS_DATA_DIR", PROJECT_ROOT)
CORPORA_DIR = os.path.join(DATA_DIR, "corpora")

# The legacy single-corpus layout: a database sitting directly in the data
# directory with downloads/ beside it. Still honoured so an existing install
# keeps working untouched; it appears in the list as one corpus like any other.
LEGACY_DB = os.environ.get("MRS_DB_PATH", os.path.join(DATA_DIR, "michael_rosen.db"))


def _slugify(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in name.lower()]
    return "".join(keep).strip("-") or "corpus"


def _db_in(directory: str) -> str | None:
    """The database inside a corpus directory, whatever it is called."""
    if not os.path.isdir(directory):
        return None
    preferred = os.path.join(directory, "corpus.db")
    if os.path.exists(preferred):
        return preferred
    dbs = sorted(
        f for f in os.listdir(directory)
        if f.endswith(".db") and os.path.isfile(os.path.join(directory, f))
    )
    return os.path.join(directory, dbs[0]) if dbs else None


def list_corpora() -> list[dict]:
    """Every installed corpus, newest layout first, legacy last."""
    found: list[dict] = []
    if os.path.isdir(CORPORA_DIR):
        for slug in sorted(os.listdir(CORPORA_DIR)):
            directory = os.path.join(CORPORA_DIR, slug)
            db = _db_in(directory)
            if db:
                found.append({"slug": slug, "name": slug.replace("-", " ").title(),
                              "dir": directory, "db": db})
    if os.path.exists(LEGACY_DB) and not any(c["db"] == LEGACY_DB for c in found):
        found.append({
            "slug": "default",
            "name": os.path.splitext(os.path.basename(LEGACY_DB))[0].replace("_", " ").title(),
            "dir": os.path.dirname(LEGACY_DB) or ".",
            "db": LEGACY_DB,
        })
    return found


# The corpus every query and every clip path currently resolves against.
_active: dict | None = None


def active() -> dict:
    """The selected corpus, picking one on first use if nothing is set."""
    global _active
    if _active is None:
        available = list_corpora()
        if not available:
            # Nothing installed. Point at the legacy path so init_db() can
            # create an empty database rather than raising on import.
            _active = {"slug": "default", "name": "Default",
                       "dir": os.path.dirname(LEGACY_DB) or ".", "db": LEGACY_DB}
        else:
            # Order of preference: what MRS_CORPUS asks for, then the legacy
            # "default" corpus, then whatever sorts first. Without the middle
            # step a restart silently switched voices, because the listing is
            # alphabetical and an installed pack can sort ahead of the corpus
            # that was there all along.
            preferred = os.environ.get("MRS_CORPUS")
            _active = (
                next((c for c in available if c["slug"] == preferred), None)
                or next((c for c in available if c["slug"] == "default"), None)
                or available[0]
            )
    return _active


def set_active(slug: str) -> dict:
    """Switch corpus. Raises KeyError if there is no such corpus."""
    global _active
    for c in list_corpora():
        if c["slug"] == slug:
            _active = c
            return c
    raise KeyError(slug)


def install_dir(name: str) -> str:
    """Path a new corpus called `name` should be unpacked into."""
    return os.path.join(CORPORA_DIR, _slugify(name))


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
    """Turn a stored (relative) source path into an absolute one.

    Relative to the *active corpus*, not the project: that is what keeps two
    corpora from tripping over each other when both contain downloads/ files
    of the same name.
    """
    if not p:
        return p
    p = normalise_sep(p)
    return p if os.path.isabs(p) else os.path.join(active()["dir"], p)


def relativize_path(p: str) -> str:
    """Store paths relative to the corpus directory (e.g. downloads/x.mp4)."""
    if not p:
        return p
    return "downloads/" + os.path.basename(normalise_sep(p))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(active()["db"])
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
