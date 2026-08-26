#!/usr/bin/env python3
"""Pack and unpack the corpus: the database and the source videos it points at.

The corpus is deliberately not in git. It is 48 mp4 files and a SQLite
database -- around 110 MB of binary that never delta-compresses, so committing
it would put the whole lot in every clone's history forever and grow with each
re-ingest. It is also not source: it is derived from YouTube by scripts/ingest,
and it is the one part of this project that is expensive to rebuild but cheap
to copy.

So it travels as a single bundle instead. `pack` makes one, `unpack` restores
it, and the container entrypoint unpacks it automatically the first time it
finds an empty data directory. Attach a bundle to a GitHub release, drop it on
a host, or keep one as a backup -- it is just a tarball with a checksum.

    python scripts/corpus.py pack                    # -> corpus-YYYY-MM-DD.tar.zst
    python scripts/corpus.py pack --out /tmp/c.tar.zst
    python scripts/corpus.py unpack corpus-*.tar.zst
    python scripts/corpus.py unpack https://example/corpus.tar.zst
    python scripts/corpus.py info corpus-*.tar.zst

The database stores source paths relative to the project root (see
app.database.relativize_path), which is what makes the pair portable: unpack
anywhere and the clip paths still resolve.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import urllib.request
from datetime import date

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# What goes in a bundle. `transcripts` is small and in git already, but it is
# included so an unpacked bundle is enough to re-run the ingest scripts without
# a checkout.
#
# Every corpus names its database corpus.db, whoever it is of. The old name is
# still read so a bundle packed before the layout changed still unpacks, but
# nothing writes it any more -- a per-corpus filename was the reason the packer
# had one speaker's name baked into it while claiming to be general.
DB_NAME = "corpus.db"
LEGACY_DB_NAME = "michael_rosen.db"
MEMBERS = [DB_NAME, "downloads", "transcripts"]
READABLE_MEMBERS = MEMBERS + [LEGACY_DB_NAME]


def _db_in(directory: str) -> str | None:
    """The database inside a corpus directory, new name or old."""
    for name in (DB_NAME, LEGACY_DB_NAME):
        p = os.path.join(directory, name)
        if os.path.exists(p):
            return p
    return None

# Generated videos are excluded on purpose: ~1 GB, and every one of them can be
# regenerated from the corpus in seconds.
EXCLUDE_DIRS = {"output", "__pycache__", ".venv", "venv"}


def _data_root() -> str:
    """Where corpora live -- the volume in a container, else the repo."""
    return os.environ.get("MRS_DATA_DIR", PROJECT_ROOT)


def _corpus_root(slug: str | None = None) -> str:
    """The directory of the corpus to act on.

    Named corpus if given, otherwise whichever one the app would serve. Falls
    back to the data root so a legacy install -- database loose in the data
    directory with downloads/ beside it -- still packs without being migrated
    first.
    """
    import sys as _sys
    _sys.path.insert(0, PROJECT_ROOT)
    from app.database import list_corpora, active  # noqa: E402

    if slug:
        for c in list_corpora():
            if c["slug"] == slug:
                return c["dir"]
        raise SystemExit(f"no corpus called {slug!r}. Try: corpus.py list")
    try:
        return active()["dir"]
    except Exception:
        return _data_root()


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_write(path: str):
    """Prefer zstd, fall back to gzip. Both are readable by `unpack`."""
    if path.endswith(".zst"):
        try:
            import zstandard  # noqa: F401
        except ImportError:
            alt = path[: -len(".zst")] + ".gz"
            print(f"  zstandard not installed, writing gzip instead: {alt}")
            return tarfile.open(alt, "w:gz"), alt
        import zstandard

        cctx = zstandard.ZstdCompressor(level=10)
        raw = open(path, "wb")
        stream = cctx.stream_writer(raw)
        return tarfile.open(fileobj=stream, mode="w|"), path
    return tarfile.open(path, "w:gz"), path


def _open_read(path: str):
    if path.endswith(".zst"):
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        raw = open(path, "rb")
        return tarfile.open(fileobj=dctx.stream_reader(raw), mode="r|")
    return tarfile.open(path, "r:*")


def _checkpoint(db: str) -> None:
    """Fold the WAL into the database file so the copy in the bundle is whole.

    A live SQLite database with an unmerged WAL is not self-contained; tarring
    it alone can capture a torn state that is missing recent writes.
    """
    if not os.path.exists(db):
        return
    try:
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
    except sqlite3.Error as exc:
        print(f"  warning: could not checkpoint the database: {exc}")


def cmd_pack(args: argparse.Namespace) -> int:
    root = _corpus_root(args.corpus)
    out = args.out or os.path.join(
        PROJECT_ROOT, f"corpus-{date.today().isoformat()}.tar.zst"
    )
    print(f"packing {root}")

    present = [m for m in READABLE_MEMBERS if os.path.exists(os.path.join(root, m))]
    if not present:
        print(f"nothing to pack: none of {MEMBERS} found under {root}", file=sys.stderr)
        return 1
    missing = [m for m in MEMBERS if m not in present]
    if missing:
        print(f"  note: not present, skipping: {', '.join(missing)}")

    db = _db_in(root)
    if db:
        _checkpoint(db)

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        base = os.path.basename(info.name)
        if base in EXCLUDE_DIRS or base.endswith((".db-wal", ".db-shm")):
            return None
        # Reproducible-ish: drop owner and mtime noise so re-packing an
        # unchanged corpus gives a similar bundle.
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    tar, out = _open_write(out)
    count = 0
    try:
        for m in present:
            src = os.path.join(root, m)
            for_added = 0
            # A bundle always carries corpus.db, even when packed from a legacy
            # install that still calls it something else, so what comes out is
            # in the modern layout no matter what went in.
            tar.add(src, arcname=(DB_NAME if m == LEGACY_DB_NAME else m),
                    filter=_filter)
            if os.path.isdir(src):
                for_added = sum(len(f) for _, _, f in os.walk(src))
            count += for_added or 1
    finally:
        tar.close()

    size = os.path.getsize(out)
    print(f"packed {count} files -> {out}")
    print(f"  size    {_human(size)}")
    print(f"  sha256  {_sha256(out)}")
    return 0


def _fetch(url: str) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".tar.zst")
    os.close(fd)
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    print(f"  {_human(os.path.getsize(tmp))}")
    return tmp


def cmd_unpack(args: argparse.Namespace) -> int:
    root = args.into or _data_root()
    src = args.bundle
    tmp = None
    if src.startswith(("http://", "https://")):
        tmp = src = _fetch(src)

    if _db_in(root) and not args.force:
        print(
            f"refusing to overwrite an existing corpus at {root}\n"
            f"  (pass --force if that is what you want)",
            file=sys.stderr,
        )
        return 1

    os.makedirs(root, exist_ok=True)
    try:
        tar = _open_read(src)
        try:
            # Streaming tars cannot be rewound, so extract in one pass.
            if hasattr(tarfile, "data_filter"):
                tar.extractall(root, filter="data")
            else:  # Python < 3.12
                tar.extractall(root)  # noqa: S202
        finally:
            tar.close()
    finally:
        if tmp:
            os.unlink(tmp)

    print(f"unpacked into {root}")
    for m in MEMBERS:
        p = os.path.join(root, m)
        if os.path.isdir(p):
            n = sum(len(f) for _, _, f in os.walk(p))
            print(f"  {m:18} {n} files")
        elif os.path.exists(p):
            print(f"  {m:18} {_human(os.path.getsize(p))}")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    """Put a bundle in as a named corpus, alongside any already installed.

    This is what makes a second voice possible: each corpus is a directory of
    its own holding one database and the videos it indexes, so two of them
    never argue over a downloads/ filename.
    """
    import sys as _sys
    _sys.path.insert(0, PROJECT_ROOT)
    from app.database import install_dir  # noqa: E402

    target = install_dir(args.name)
    if os.path.exists(target) and os.listdir(target) and not args.force:
        print(f"{target} already exists and is not empty (pass --force)", file=sys.stderr)
        return 1

    src = args.bundle
    tmp = None
    if src.startswith(("http://", "https://")):
        tmp = src = _fetch(src)

    os.makedirs(target, exist_ok=True)
    try:
        tar = _open_read(src)
        try:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(target, filter="data")
            else:  # Python < 3.12
                tar.extractall(target)  # noqa: S202
        finally:
            tar.close()
    finally:
        if tmp:
            os.unlink(tmp)

    print(f"installed as {os.path.basename(target)} in {target}")
    for m in MEMBERS:
        pth = os.path.join(target, m)
        if os.path.isdir(pth):
            print(f"  {m:18} {sum(len(f) for _, _, f in os.walk(pth))} files")
        elif os.path.exists(pth):
            print(f"  {m:18} {_human(os.path.getsize(pth))}")
    print("")
    print("Restart the app, or POST /api/corpus to switch to it.")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """Move a legacy loose install into corpora/<name>/, in place.

    The original layout put the database straight in the data directory with
    downloads/ next to it. That works, and is still read, but it is the one
    corpus that cannot sit beside another -- there is only one downloads/ to
    go round -- and it is why "default" existed as a name for "the old one".
    Moving it makes every corpus the same shape.

    Idempotent: with nothing loose to move it reports that and changes nothing,
    so it is safe to run on every container start.
    """
    root = _data_root()
    legacy = os.path.join(root, LEGACY_DB_NAME)
    if not os.path.exists(legacy):
        if not args.quiet:
            print(f"nothing to migrate: no {LEGACY_DB_NAME} in {root}")
        return 0

    target = os.path.join(root, "corpora", _slug(args.name))
    if os.path.exists(os.path.join(target, DB_NAME)) and not args.force:
        print(f"{target} already holds a corpus; not overwriting (use --force)",
              file=sys.stderr)
        return 1

    print(f"migrating loose corpus in {root}\n  -> {target}")
    if args.dry_run:
        print("  (dry run: pass --apply to move anything)")
        return 0

    os.makedirs(target, exist_ok=True)

    # The database first, and by rename rather than copy: on the same
    # filesystem it is atomic and instant, and it means a 100 MB downloads/
    # is never duplicated on a server that may not have room for two copies.
    _checkpoint(legacy)
    for suffix in ("-wal", "-shm"):
        stray = legacy + suffix
        if os.path.exists(stray):
            os.remove(stray)          # folded into the db by the checkpoint
    os.replace(legacy, os.path.join(target, DB_NAME))
    print(f"  {LEGACY_DB_NAME} -> {DB_NAME}")

    for member in ("downloads", "transcripts"):
        src = os.path.join(root, member)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(target, member)
        if os.path.exists(dst):
            # Merge rather than clobber: a half-finished earlier attempt should
            # be completed, not have its contents replaced.
            os.makedirs(dst, exist_ok=True)
            for entry in os.listdir(src):
                s, d = os.path.join(src, entry), os.path.join(dst, entry)
                if not os.path.exists(d):
                    os.replace(s, d)
            if not os.listdir(src):
                os.rmdir(src)
        else:
            os.replace(src, dst)
        n = sum(len(f) for _, _, f in os.walk(dst))
        print(f"  {member}/ -> {n} files")

    print("\nDone. The corpus is now "
          f"{os.path.join('corpora', os.path.basename(target))}.")
    print(f"Set MRS_CORPUS={os.path.basename(target)} if it was pinned to 'default'.")
    return 0


def _slug(name: str) -> str:
    import sys as _sys
    _sys.path.insert(0, PROJECT_ROOT)
    from app.database import _slugify  # noqa: E402
    return _slugify(name)


def cmd_list(args: argparse.Namespace) -> int:
    import sys as _sys
    _sys.path.insert(0, PROJECT_ROOT)
    from app.database import list_corpora, active  # noqa: E402

    corpora = list_corpora()
    if not corpora:
        print("no corpora installed")
        return 0
    current = active()["slug"]
    for c in corpora:
        mark = "*" if c["slug"] == current else " "
        print(f" {mark} {c['slug']:24} {c['dir']}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    src = args.bundle
    print(f"{src}")
    print(f"  size    {_human(os.path.getsize(src))}")
    print(f"  sha256  {_sha256(src)}")
    tar = _open_read(src)
    try:
        sizes: dict[str, list[int]] = {}
        for info in tar:
            if not info.isfile():
                continue
            top = info.name.split("/")[0]
            sizes.setdefault(top, []).append(info.size)
    finally:
        tar.close()
    for top, entries in sorted(sizes.items()):
        print(f"  {top:18} {len(entries):4} files  {_human(sum(entries))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="build a corpus bundle")
    p.add_argument("--out", help="output path (default corpus-<date>.tar.zst)")
    p.add_argument("--corpus", metavar="SLUG",
                   help="which installed corpus to pack (default: the active one)")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("migrate",
                       help="move a legacy loose install into corpora/<name>/")
    p.add_argument("--name", default="michael-rosen",
                   help="what to call the migrated corpus (default michael-rosen)")
    p.add_argument("--apply", dest="dry_run", action="store_false", default=True,
                   help="actually move the files (without this it only reports)")
    p.add_argument("--force", action="store_true",
                   help="migrate even if the target already holds a corpus")
    p.add_argument("--quiet", action="store_true",
                   help="say nothing when there is nothing to migrate")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("unpack", help="restore a corpus bundle (path or URL)")
    p.add_argument("bundle")
    p.add_argument("--into", help="target directory (default the data dir)")
    p.add_argument("--force", action="store_true", help="overwrite an existing corpus")
    p.set_defaults(func=cmd_unpack)

    p = sub.add_parser("install", help="install a bundle as a named corpus")
    p.add_argument("bundle")
    p.add_argument("--name", required=True, help="what to call it, e.g. attenborough")
    p.add_argument("--force", action="store_true", help="overwrite an existing corpus")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("list", help="list installed corpora")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="describe a bundle without unpacking it")
    p.add_argument("bundle")
    p.set_defaults(func=cmd_info)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
