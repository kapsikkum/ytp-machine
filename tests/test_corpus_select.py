#!/usr/bin/env python3
"""Which corpus a run resolves to, as plain asserts.

    python tests/test_corpus_select.py

This decides where every downloaded video and every clip row is written, so a
mistake here does not fail -- it succeeds against the wrong corpus. That is
what happened: building a new corpus in a data directory that already held one
resolved to the existing corpus and downloaded 227 MB into it, while the header
printed the name it was supposed to be using.

app.database reads MRS_DATA_DIR at import, so each case reloads the module.
"""
import importlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

failures: list[str] = []


def resolve(data_dir: str, corpus: str | None) -> dict:
    """active() as a fresh process would see it."""
    os.environ["MRS_DATA_DIR"] = data_dir
    os.environ.pop("MRS_DB_PATH", None)
    if corpus is None:
        os.environ.pop("MRS_CORPUS", None)
    else:
        os.environ["MRS_CORPUS"] = corpus
    import app.database as db
    importlib.reload(db)
    return db.active()


def install(data_dir: str, *slugs: str) -> None:
    """Make each slug look like an installed corpus (a directory with a .db)."""
    for slug in slugs:
        d = os.path.join(data_dir, "corpora", slug)
        os.makedirs(os.path.join(d, "downloads"), exist_ok=True)
        open(os.path.join(d, "corpus.db"), "a").close()


def check(label: str, got: str, want: str) -> None:
    if got != want:
        failures.append(f"  {label}: expected {want!r}, got {got!r}")


# An empty data directory: whatever was asked for, or "default".
empty = tempfile.mkdtemp(prefix="ytp-sel-empty-")
check("empty dir, no name", resolve(empty, None)["slug"], "default")
check("empty dir, named", resolve(empty, "brand-new")["slug"], "brand-new")

# A data directory that already holds a corpus -- the case that was wrong.
used = tempfile.mkdtemp(prefix="ytp-sel-used-")
install(used, "michael-rosen", "zebra")

check("named, already installed", resolve(used, "michael-rosen")["slug"], "michael-rosen")
check("named, NOT installed yet", resolve(used, "james-channel")["slug"], "james-channel")

# and it must point *inside* the named corpus, not merely report the name: the
# name was right all along, the paths were not.
entry = resolve(used, "james-channel")
if "james-channel" not in entry["dir"] or "michael-rosen" in entry["dir"]:
    failures.append(f"  a new corpus resolved to another corpus's files: {entry['dir']}")
if "james-channel" not in entry["db"]:
    failures.append(f"  new corpus database in the wrong place: {entry['db']}")

# Nothing asked for: alphabetical, which is the long-standing behaviour.
check("nothing asked for", resolve(used, None)["slug"], "michael-rosen")

# A legacy "default" corpus beats alphabetical order, so installing a pack that
# sorts earlier does not silently switch voices on the next restart.
legacy = tempfile.mkdtemp(prefix="ytp-sel-legacy-")
install(legacy, "aaa-pack", "default")
check("default beats alphabetical", resolve(legacy, None)["slug"], "default")
check("explicit still wins over default", resolve(legacy, "aaa-pack")["slug"], "aaa-pack")

# A name needing slugifying still lands somewhere sane rather than creating a
# directory with a space in it.
check("name is slugified", resolve(used, "James Channel")["slug"], "james-channel")

if failures:
    print(f"FAILED ({len(failures)}):")
    print("\n".join(failures))
    sys.exit(1)
print("ok: 9 corpus-selection cases")
