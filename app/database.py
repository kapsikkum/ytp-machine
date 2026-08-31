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


def _corpus_entry(slug: str) -> dict:
    """Where a corpus called *slug* lives, installed or not yet.

    In the modern one-directory-per-corpus form: pointing a fresh install at
    the legacy loose path meant the very first ingest recreated the layout
    everything else is migrating away from, and it would need migrating again
    immediately.
    """
    directory = os.path.join(CORPORA_DIR, slug)
    return {"slug": slug, "name": slug.replace("-", " ").title(),
            "dir": directory, "db": os.path.join(directory, "corpus.db")}


def active() -> dict:
    """The selected corpus, picking one on first use if nothing is set."""
    global _active
    if _active is None:
        available = list_corpora()
        preferred = os.environ.get("MRS_CORPUS")

        if preferred:
            # An explicit name always wins, and names the corpus even when it
            # does not exist yet -- that is precisely the case where a new one
            # is about to be built.
            #
            # list_corpora() only reports directories that already hold a .db,
            # so a corpus being created for the first time matched nothing here
            # and fell through to "whatever sorts first". Asking to build
            # "james-channel" in a data directory that already held
            # "michael-rosen" therefore downloaded into michael-rosen's
            # downloads/ -- the shipped corpus, in git -- while the header
            # printed the name it was supposed to be using. Every build that
            # ever worked did so only because it pointed --data-dir at an empty
            # directory, where the no-corpora branch below got it right by
            # accident.
            slug = _slugify(preferred)
            _active = (next((c for c in available if c["slug"] == slug), None)
                       or _corpus_entry(slug))
        elif available:
            # Nothing asked for: the legacy "default" corpus, else whatever
            # sorts first. Without the middle step a restart silently switched
            # voices, because the listing is alphabetical and an installed pack
            # can sort ahead of the corpus that was there all along.
            _active = (next((c for c in available if c["slug"] == "default"), None)
                       or available[0])
        else:
            _active = _corpus_entry("default")
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
    target = active()["db"]
    # sqlite will not create a missing parent, and reports it as the same
    # "unable to open database file" it gives for a permissions problem.
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    conn = sqlite3.connect(target)
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

            -- How to build one word out of others, saved from the splice
            -- editor. The same shape as the hand-tuned recipes in phonemes.py
            -- and consulted before them: those were tuned against one speaker,
            -- and this one belongs to the corpus it sits in. JSON, because it
            -- is a list of (phonemes, preferred sources) and reading it back
            -- as rows would mean three tables to express one sentence.
            CREATE TABLE IF NOT EXISTS splice_recipes (
                word   TEXT PRIMARY KEY,
                recipe TEXT NOT NULL
            );

            -- Per-corpus settings. In the database rather than the environment
            -- because they belong to the corpus, not to the server: how hard
            -- to push a splice depends on how much material there is, and a
            -- 30-word corpus and a 7,000-word one want opposite answers while
            -- being served by the same process. Travels inside the bundle, so
            -- a corpus arrives already knowing how it wants to be spliced.
            CREATE TABLE IF NOT EXISTS settings (
                key    TEXT PRIMARY KEY,
                value  TEXT NOT NULL
            );
        """)

        # Added after the fact, so every corpus built before it gets it here
        # rather than needing a rebuild. `edited` marks a clip whose boundaries
        # were set by hand: the encoder nurses a word's tail outwards to catch
        # the decay of a sonorant, which is right for a machine-placed edge and
        # wrong for one somebody chose while looking at the waveform.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(word_clips)")}
        if "edited" not in cols:
            conn.execute("ALTER TABLE word_clips ADD COLUMN edited INTEGER DEFAULT 0")
        # Where each phoneme of this clip actually falls, as JSON
        # [[phone, start, end], ...] in seconds from the clip's start_time.
        #
        # Written at corpus build time by scripts/align_phones.py, because the
        # model that produces it is larger than the whole server container.
        # Empty for a corpus built before it existed, and the splicer falls
        # back to guessing from the spelling, which is what it always did.
        if "phones" not in cols:
            conn.execute("ALTER TABLE word_clips ADD COLUMN phones TEXT")

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


# ── Per-corpus settings ───────────────────────────────────────────────────────

# How hard the splicer is allowed to push when the corpus cannot say a word
# outright. Ordered from most to least faithful:
#
#   strict      only real recordings and clean phoneme splices. A word that
#               cannot be built from what is actually there is reported
#               missing. This is what every corpus did before the setting
#               existed, and it stays the default.
#   loose       substitute a near-enough phoneme when the exact one has no
#               coverage, and guess a pronunciation for words the dictionary
#               has never heard of.
#   desperate   as loose, and also drop phonemes nothing can cover rather than
#               give up on the word. Always produces something. Whether it
#               sounds like the word you asked for is another matter.
SPLICE_MODES = ("strict", "loose", "desperate")
DEFAULT_SPLICE_MODE = "strict"


def get_setting(key: str, default: str | None = None) -> str | None:
    """One setting from the active corpus, or *default*."""
    try:
        with get_db() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    except sqlite3.Error:
        return default          # a corpus packed before the table existed
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def splice_mode() -> str:
    """The active corpus's splice mode, validated."""
    mode = (get_setting("splice_mode") or DEFAULT_SPLICE_MODE).lower()
    return mode if mode in SPLICE_MODES else DEFAULT_SPLICE_MODE
