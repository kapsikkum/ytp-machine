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
MEMBERS = ["michael_rosen.db", "downloads", "transcripts"]

# Generated videos are excluded on purpose: ~1 GB, and every one of them can be
# regenerated from the corpus in seconds.
EXCLUDE_DIRS = {"output", "__pycache__", ".venv", "venv"}


def _data_root() -> str:
    """Where the live corpus lives -- the volume in a container, else the repo."""
    return os.environ.get("MRS_DATA_DIR", PROJECT_ROOT)


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
    root = _data_root()
    out = args.out or os.path.join(
        PROJECT_ROOT, f"corpus-{date.today().isoformat()}.tar.zst"
    )

    present = [m for m in MEMBERS if os.path.exists(os.path.join(root, m))]
    if not present:
        print(f"nothing to pack: none of {MEMBERS} found under {root}", file=sys.stderr)
        return 1
    missing = [m for m in MEMBERS if m not in present]
    if missing:
        print(f"  note: not present, skipping: {', '.join(missing)}")

    _checkpoint(os.path.join(root, "michael_rosen.db"))

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
            tar.add(src, arcname=m, filter=_filter)
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

    db = os.path.join(root, "michael_rosen.db")
    if os.path.exists(db) and not args.force:
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
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("unpack", help="restore a corpus bundle (path or URL)")
    p.add_argument("bundle")
    p.add_argument("--into", help="target directory (default the data dir)")
    p.add_argument("--force", action="store_true", help="overwrite an existing corpus")
    p.set_defaults(func=cmd_unpack)

    p = sub.add_parser("info", help="describe a bundle without unpacking it")
    p.add_argument("bundle")
    p.set_defaults(func=cmd_info)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
