#!/usr/bin/env python3
"""Pack and unpack the corpus: the database and the source videos it points at.

The repository includes a demo corpus. Additional channel corpora can travel
as bundles without committing their databases and recordings to git. Each
bundle includes the database, source media, transcripts and pronunciations;
these are expensive to rebuild but straightforward to copy consistently.

`pack` makes a bundle, `unpack` restores
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
import codecs
import hashlib
import io
import ntpath
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import urllib.request
from urllib.parse import urlsplit
from datetime import date
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:      # for app.database, when run as a script
    sys.path.insert(0, PROJECT_ROOT)

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
# pronunciations.csv travels with the corpus: the words a speaker invents
# belong to that speaker, and a corpus that arrives without them cannot say
# them.
MEMBERS = [DB_NAME, "downloads", "transcripts", "pronunciations.csv"]
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
    """Prefer zstd, fall back to gzip. Returns (tar, path, finish).

    Call finish(), never tar.close(). With zstd there are three things stacked
    up -- the tarfile writes into a compressor which writes into a file -- and
    closing only the top of that stack leaves the archive unfinished.
    """
    if path.endswith(".zst"):
        try:
            import zstandard  # noqa: F401
        except ImportError:
            alt = path[: -len(".zst")] + ".gz"
            print(f"  zstandard not installed, writing gzip instead: {alt}")
            tar = tarfile.open(alt, "w:gz")
            return tar, alt, tar.close
        import zstandard

        cctx = zstandard.ZstdCompressor(level=10)
        raw = open(path, "wb")
        stream = cctx.stream_writer(raw)
        tar = tarfile.open(fileobj=stream, mode="w|")

        def finish() -> None:
            # Every step is load-bearing and the order is the point: the tar
            # writes its trailer into the compressor, the compressor flushes
            # its final frame into the file, and only then is the archive
            # complete. Closing the tarfile alone -- which is what this did --
            # produced a .tar.zst that looked entirely normal, had a stable
            # checksum, copied faithfully to another machine, and then failed
            # to extract there with "unexpected end of data".
            tar.close()
            stream.close()
            raw.close()

        return tar, path, finish

    # gzip needs no help: tarfile owns the file and closing it closes both.
    tar = tarfile.open(path, "w:gz")
    return tar, path, tar.close


def _open_read(path: str):
    if path.endswith(".zst"):
        try:
            import zstandard
        except ImportError:
            # _open_write quietly falls back to gzip when zstandard is absent,
            # so it is entirely possible to build a bundle on one machine and
            # be unable to open it on another. Say which, rather than letting a
            # bare ModuleNotFoundError surface from inside a container start.
            raise SystemExit(
                f"cannot read {os.path.basename(path)}: zstandard is not "
                f"installed here.\n"
                f"  pip install zstandard\n"
                f"  (or re-pack as .tar.gz, which needs nothing extra)"
            ) from None

        dctx = zstandard.ZstdDecompressor()
        raw = open(path, "rb")
        try:
            return tarfile.open(fileobj=dctx.stream_reader(raw), mode="r|")
        except BaseException:
            # tarfile.open reads the first block before returning, so a
            # corrupt or truncated frame raises here rather than handing back
            # an object whose .close() would tidy up -- close it ourselves, or
            # `raw` stays open. Harmless on Linux; on Windows it holds the file
            # locked, so a caller's own cleanup (os.remove on a temp upload)
            # fails right behind it.
            raw.close()
            raise
    return tarfile.open(path, "r:*")


def _checkpoint(db: str) -> None:
    """Require a complete checkpoint before an offline legacy migration."""
    if not os.path.exists(db):
        return
    with closing(sqlite3.connect(db, timeout=1)) as conn:
        busy, remaining, checkpointed = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy or (remaining >= 0 and remaining != checkpointed):
            raise RuntimeError("database is busy; stop the app before migrating the corpus")


def _snapshot_database(db: str, target: str) -> None:
    """Copy a consistent committed SQLite view, including live WAL writes.

    Checkpointing the source then copying its main file leaves a window where
    another writer can change it. The backup API holds a consistent view and
    makes a self-contained copy without checkpointing the live database.
    """
    source = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        snapshot = sqlite3.connect(target)
        try:
            source.backup(snapshot)
            # The copied header may carry WAL mode. Only the temporary copy
            # changes: the archive must not need its own -wal or -shm files.
            snapshot.execute("PRAGMA journal_mode=DELETE")
        finally:
            snapshot.close()
    finally:
        source.close()


def _effective_pronunciations(root: str) -> bytes | None:
    """The global dictionary followed by this voice's overrides, as one CSV.

    Retain comments, row order, quoting and line endings: later rows can name
    earlier pronunciations, so a parsed/sorted dictionary changes its meaning.
    Strip only a later file's BOM and separate files whose last row has no
    newline. Neither source file is rewritten.
    """
    from app.database import DATA_DIR

    paths: list[str] = []
    chunks: list[bytes] = []
    for path in (os.path.join(DATA_DIR, "pronunciations.csv"),
                 os.path.join(root, "pronunciations.csv")):
        if not os.path.isfile(path):
            continue
        if any(os.path.samefile(path, previous) for previous in paths):
            continue                    # legacy corpus and global file coincide
        paths.append(path)
        with open(path, "rb") as fh:
            body = fh.read()
        if chunks:
            body = body.removeprefix(codecs.BOM_UTF8)
            if body and chunks[-1] and not chunks[-1].endswith((b"\n", b"\r")):
                chunks.append(b"\n")
        chunks.append(body)
    return b"".join(chunks) if paths else None


def _check_paths(db: str) -> None:
    """Warn if any clip points somewhere a different machine cannot follow.

    A stored path has to be relative to the corpus directory, because that is
    the only thing that is the same on the machine that built the corpus and
    the one that serves it. An absolute path survives packing, unpacking and a
    checksum without complaint and then fails at the only moment that matters:
    ffmpeg opening the file. Worse, a Windows "C:/..." is not absolute to
    Linux, so it gets quietly appended to the corpus directory and the error
    names a path that never existed anywhere.

    This is a warning rather than a refusal -- the bundle is still the best
    copy of a corpus that took hours to build, and the paths can be repaired
    in place afterwards.
    """
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return
    try:
        bad = conn.execute(
            "SELECT count(*), count(DISTINCT source_file) FROM word_clips "
            "WHERE source_file NOT LIKE 'downloads/%'"
        ).fetchone()
    except sqlite3.Error:
        return                      # an older or partial schema: nothing to say
    finally:
        conn.close()

    if bad and bad[0]:
        print(f"  WARNING: {bad[0]} clips across {bad[1]} file(s) have paths "
              f"that are not relative to the corpus.\n"
              f"           They will not resolve anywhere but this machine.")


def pack_bundle(slug: str | None = None, out: str | None = None) -> str:
    """Build a bundle for corpus *slug* (default: whichever the app would
    serve). Returns the path actually written -- which can differ from *out*
    when zstandard is not installed and it fell back to gzip.

    This is what both the CLI (`corpus.py pack`) and the editor's export
    button call; kept as one function so a bug fixed for one is fixed for
    both.
    """
    root = _corpus_root(slug)
    out = out or os.path.join(
        PROJECT_ROOT, f"corpus-{date.today().isoformat()}.tar.zst"
    )
    print(f"packing {root}")

    db = _db_in(root)
    pronunciations = _effective_pronunciations(root)
    present = [m for m in MEMBERS
               if m not in (DB_NAME, "pronunciations.csv")
               and os.path.exists(os.path.join(root, m))]
    if db:
        present.insert(0, DB_NAME)
    if pronunciations is not None:
        present.append("pronunciations.csv")
    if not present:
        raise FileNotFoundError(
            f"nothing to pack: none of {MEMBERS} found under {root}")
    missing = [m for m in MEMBERS if m not in present]
    if missing:
        print(f"  note: not present, skipping: {', '.join(missing)}")

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        base = os.path.basename(info.name)
        if base in EXCLUDE_DIRS or base.endswith((".db-wal", ".db-shm")):
            return None
        # Reproducible-ish: drop owner and mtime noise so re-packing an
        # unchanged corpus gives a similar bundle.
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    count = 0
    with tempfile.TemporaryDirectory(prefix="mrs-pack-") as staging:
        snapshot = os.path.join(staging, DB_NAME)
        if db:
            _snapshot_database(db, snapshot)
            _check_paths(snapshot)
            conn = sqlite3.connect(snapshot)
            try:
                try:
                    row = conn.execute(
                        "SELECT value FROM settings WHERE key='schema_version'").fetchone()
                    print(f"  schema version: {row[0] if row else 0}")
                except sqlite3.Error:
                    pass
            finally:
                conn.close()

        tar, out, finish = _open_write(out)
        try:
            for m in present:
                if m == "pronunciations.csv":
                    info = tarfile.TarInfo(m)
                    info.size = len(pronunciations)
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(pronunciations))
                    count += 1
                    continue
                src = snapshot if m == DB_NAME else os.path.join(root, m)
                # Legacy and modern databases both travel as one corpus.db.
                tar.add(src, arcname=m, filter=_filter)
                for_added = (sum(len(f) for _, _, f in os.walk(src))
                             if os.path.isdir(src) else 0)
                count += for_added or 1
        finally:
            finish()

    print(f"packed {count} files -> {out}")
    return out


def cmd_pack(args: argparse.Namespace) -> int:
    try:
        out = pack_bundle(args.corpus, args.out)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"  size    {_human(os.path.getsize(out))}")
    print(f"  sha256  {_sha256(out)}")
    return 0


def _fetch(url: str) -> str:
    path = urlsplit(url).path.lower()
    suffix = ".tar.zst" if path.endswith(".zst") else ".tar.gz" if path.endswith((".gz", ".tgz")) else ".tar"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    print(f"downloading {url}")
    try:
        with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(url, timeout=30) as resp:
            size = 0
            while chunk := resp.read(1 << 20):
                size += len(chunk)
                if size > MAX_UNPACKED_BYTES:
                    raise ValueError("download exceeds the maximum corpus bundle size")
                fh.write(chunk)
    except BaseException:
        os.unlink(tmp)
        raise
    print(f"  {_human(os.path.getsize(tmp))}")
    return tmp


MAX_UNPACKED_BYTES = 8 << 30
MAX_MEMBER_FILES = 100_000


def _safe_extract(tar: tarfile.TarFile, root: str) -> None:
    """Extract portable regular corpus files, with bounded archive contents."""
    real_root = os.path.realpath(root)
    total_bytes = 0
    total_files = 0
    seen: set[str] = set()
    for member in tar:
        total_files += 1
        if total_files > MAX_MEMBER_FILES:
            raise ValueError(f"Tar bomb protection: exceeded maximum member files limit ({MAX_MEMBER_FILES:,})")

        name = member.name.rstrip("/")
        parts = name.split("/")
        if (not name or "\\" in name or ntpath.splitdrive(name)[0]
                or any(part in ("", ".", "..") or ":" in part
                       or part.endswith((".", " ")) for part in parts)):
            raise ValueError("bundle contains a nonportable or unsafe path")
        if parts[0] not in READABLE_MEMBERS:
            raise ValueError(f"unexpected bundle member: {parts[0]!r}")
        if parts[0] not in ("downloads", "transcripts") and len(parts) != 1:
            raise ValueError("bundle files must have the expected corpus layout")
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}
        if any(part.split(".", 1)[0].upper() in reserved for part in parts):
            raise ValueError("bundle contains a reserved device filename")
        if not (member.isfile() or member.isdir()):
            raise ValueError("bundle links and special files are not supported")
        if parts[0] not in ("downloads", "transcripts") and not member.isfile():
            raise ValueError("expected a regular corpus file")
        # Check on every platform: an archive that is ambiguous on Windows
        # should not quietly produce a different corpus when installed there.
        key = name.casefold()
        if key in seen:
            raise ValueError("bundle contains a duplicate member path")
        seen.add(key)
        target_path = os.path.realpath(os.path.join(real_root, *parts))
        if os.path.commonpath([target_path, real_root]) != real_root:
            raise ValueError(f"Zip slip protection: member {member.name!r} resolves outside target directory")

        if member.isfile():
            if member.size < 0:
                raise ValueError("bundle member has a negative size")
            total_bytes += member.size
            if total_bytes > MAX_UNPACKED_BYTES:
                raise ValueError(
                    f"Tar bomb protection: exceeded maximum uncompressed size ({MAX_UNPACKED_BYTES:,} bytes)"
                )

        if member.isdir():
            os.makedirs(target_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("bundle member could not be read")
            # Never restore archive ownership, special permission bits or links.
            with source, open(target_path, "xb") as output:
                shutil.copyfileobj(source, output, length=1 << 20)


def _validate_install(directory: str) -> None:
    database = _db_in(directory)
    if not database:
        raise ValueError("bundle must contain a corpus database")
    try:
        with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            if conn.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("bundle database failed its integrity check")
            for table, required in (("sources", {"id", "video_id", "source_file"}),
                                    ("word_clips", {"id", "source_id", "word", "start_time",
                                                    "end_time", "source_file"})):
                if conn.execute("SELECT type FROM sqlite_master WHERE name=?", (table,)).fetchone() != ("table",):
                    raise ValueError(f"bundle database is missing required {table} table")
                columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                if not required <= columns:
                    raise ValueError(f"bundle database is missing required {table} columns")
    except sqlite3.Error as exc:
        raise ValueError("bundle does not contain a readable corpus database") from exc


def cmd_unpack(args: argparse.Namespace) -> int:
    root = args.into or _data_root()
    src = args.bundle
    tmp = None

    if _db_in(root) and not args.force:
        print(
            f"refusing to overwrite an existing corpus at {root}\n"
            f"  (pass --force if that is what you want)",
            file=sys.stderr,
        )
        return 1

    if src.startswith(("http://", "https://")):
        tmp = src = _fetch(src)

    os.makedirs(root, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".unpack-", dir=root)
    backup = None
    moved_old: list[str] = []
    moved_new: list[str] = []
    try:
        tar = _open_read(src)
        try:
            # Streaming tars cannot be rewound, so extract in one pass.
            _safe_extract(tar, staging)
        finally:
            tar.close()
        _validate_install(staging)
        existing = [m for m in READABLE_MEMBERS if os.path.lexists(os.path.join(root, m))]
        if existing and not args.force:
            raise FileExistsError("corpus files already exist; pass --force to replace them")
        if existing:
            backup_root = os.path.join(root, ".corpus-backups")
            os.makedirs(backup_root, exist_ok=True)
            backup = tempfile.mkdtemp(prefix="unpack-", dir=backup_root)
        try:
            for member in existing:
                os.replace(os.path.join(root, member), os.path.join(backup, member))
                moved_old.append(member)
            for member in READABLE_MEMBERS:
                source = os.path.join(staging, member)
                if os.path.exists(source):
                    os.replace(source, os.path.join(root, member))
                    moved_new.append(member)
        except BaseException:
            for member in reversed(moved_new):
                os.replace(os.path.join(root, member), os.path.join(staging, member))
            for member in reversed(moved_old):
                os.replace(os.path.join(backup, member), os.path.join(root, member))
            raise
    finally:
        shutil.rmtree(staging)
        if tmp:
            os.unlink(tmp)

    print(f"unpacked into {root}")
    if backup:
        print(f"  previous corpus kept in {backup}")
    for m in MEMBERS:
        p = os.path.join(root, m)
        if os.path.isdir(p):
            n = sum(len(f) for _, _, f in os.walk(p))
            print(f"  {m:18} {n} files")
        elif os.path.exists(p):
            print(f"  {m:18} {_human(os.path.getsize(p))}")
    return 0


def _activate_install(staging: str, target: str, force: bool = False) -> str | None:
    """Swap a validated staging directory, restoring the old one on failure."""
    backup = None
    try:
        if os.path.exists(target):
            if os.listdir(target) and not force:
                raise FileExistsError(target)
            backup_root = os.path.join(os.path.dirname(target), ".backups")
            os.makedirs(backup_root, exist_ok=True)
            backup = tempfile.mkdtemp(prefix=os.path.basename(target) + "-", dir=backup_root)
            os.rmdir(backup)
            os.replace(target, backup)
        os.replace(staging, target)
    except BaseException:
        if backup and not os.path.exists(target):
            os.replace(backup, target)
        raise
    return backup


def _install_manifest(target: str, backup: str | None = None) -> dict:
    from app.database import forget_ready
    forget_ready()
    manifest: dict = {"target": target, "slug": os.path.basename(target), "members": {}}
    if backup:
        manifest["backup"] = backup
    for member in MEMBERS:
        path = os.path.join(target, member)
        if os.path.isdir(path):
            files = [os.path.join(dp, f) for dp, _, fs in os.walk(path) for f in fs]
            manifest["members"][member] = {"files": len(files),
                                          "bytes": sum(os.path.getsize(f) for f in files)}
        elif os.path.exists(path):
            manifest["members"][member] = {"bytes": os.path.getsize(path)}
    return manifest


def seed_directory(source: str, name: str) -> dict:
    """Seed a mounted loose corpus with a WAL-safe snapshot and atomic install."""
    from app.database import install_dir
    source_db = _db_in(source)
    if not source_db:
        raise ValueError("source directory does not contain a corpus database")
    target = install_dir(name)
    if os.path.exists(target) and os.listdir(target):
        raise FileExistsError(target)
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".install-", dir=parent)
    try:
        for member in ("downloads", "transcripts"):
            path = os.path.join(source, member)
            if os.path.isdir(path):
                shutil.copytree(path, os.path.join(staging, member))
        pronunciations = _effective_pronunciations(source)
        if pronunciations is not None:
            Path(staging, "pronunciations.csv").write_bytes(pronunciations)
        _snapshot_database(source_db, os.path.join(staging, DB_NAME))
        _validate_install(staging)
        backup = _activate_install(staging, target)
    finally:
        if os.path.isdir(staging):
            shutil.rmtree(staging)
    return _install_manifest(target, backup)


def cmd_seed_directory(args: argparse.Namespace) -> int:
    manifest = seed_directory(args.source, args.name)
    print(f"seeded {manifest['slug']} in {manifest['target']}")
    return 0


def install_bundle(bundle: str, name: str, force: bool = False) -> dict:
    """Extract *bundle* (a local file) as a new corpus called *name*,
    alongside any already installed. Returns a manifest of what was written.

    This is what makes a second voice possible: each corpus is a directory of
    its own holding one database and the videos it indexes, so two of them
    never argue over a downloads/ filename.

    Raises FileExistsError if the target already holds something and *force*
    is not set, so a caller (CLI or API) can turn that into its own kind of
    "already there" answer instead of this deciding what that looks like.
    """
    from app.database import install_dir  # noqa: E402

    target = install_dir(name)
    if os.path.exists(target) and os.listdir(target) and not force:
        raise FileExistsError(target)

    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".install-", dir=parent)
    backup = None
    try:
        tar = _open_read(bundle)
        try:
            _safe_extract(tar, staging)
        finally:
            tar.close()
        _validate_install(staging)
        # Verify again just before the swap, including a corpus created while
        # extraction ran. Failed input never touches an existing installation.
        backup = _activate_install(staging, target, force)
    finally:
        if os.path.isdir(staging):
            shutil.rmtree(staging)

    return _install_manifest(target, backup)


def cmd_install(args: argparse.Namespace) -> int:
    """Put a bundle in as a named corpus, alongside any already installed."""
    src = args.bundle
    tmp = None
    if src.startswith(("http://", "https://")):
        tmp = src = _fetch(src)

    try:
        manifest = install_bundle(src, args.name, args.force)
    except FileExistsError as exc:
        print(f"{exc} already exists and is not empty (pass --force)", file=sys.stderr)
        return 1
    finally:
        if tmp:
            os.unlink(tmp)

    target = manifest["target"]
    print(f"installed as {manifest['slug']} in {target}")
    for m, info in manifest["members"].items():
        if "files" in info:
            print(f"  {m:18} {info['files']} files")
        else:
            print(f"  {m:18} {_human(info['bytes'])}")
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

    # Abort before moving any media if a live reader prevents WAL recovery.
    # This command runs while the app is stopped (including container startup).
    _checkpoint(legacy)
    os.makedirs(target, exist_ok=True)

    # Directories first, database last. This whole operation is not atomic, and
    # whether it can be retried depends entirely on the order: the one thing
    # that marks an install as "still legacy" is the loose database, so moving
    # it first and then failing on downloads/ leaves a corpus split across two
    # places that a re-run reports as nothing to migrate. Moving it last means
    # any failure leaves a legacy install that is still legacy, and running it
    # again simply carries on. (Directory renames do fail in practice -- a
    # stale handle on the directory node is enough, even with no file inside it
    # open -- so this is a real case, not a theoretical one.)
    for member in ("downloads", "transcripts"):
        src = os.path.join(root, member)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(target, member)
        _move_dir(src, dst)
        n = sum(len(f) for _, _, f in os.walk(dst))
        print(f"  {member}/ -> {n} files")

    # The database last, so that until this succeeds the install still reads as
    # a legacy one and re-running resumes. Renamed rather than copied: on one
    # filesystem that is atomic and instant, and it never needs room for two
    # copies of a corpus on a server that may not have it.
    dictionary = os.path.join(root, "pronunciations.csv")
    dictionary_target = os.path.join(target, "pronunciations.csv")
    if os.path.isfile(dictionary) and not os.path.exists(dictionary_target):
        shutil.copy2(dictionary, dictionary_target)
    for suffix in ("-wal", "-shm"):
        stray = legacy + suffix
        if os.path.exists(stray):
            os.remove(stray)          # folded into the db by the checkpoint
    os.replace(legacy, os.path.join(target, DB_NAME))
    print(f"  {LEGACY_DB_NAME} -> {DB_NAME}")

    print("\nDone. The corpus is now "
          f"{os.path.join('corpora', os.path.basename(target))}.")
    print(f"Set MRS_CORPUS={os.path.basename(target)} if it was pinned to 'default'.")
    return 0


def _move_dir(src: str, dst: str) -> None:
    """Move a directory, falling back to moving its contents one by one.

    Renaming the directory is one instant operation and is tried first. It can
    fail on Windows with "the process cannot access the file" even when nothing
    inside is open -- a stale handle on the directory node itself is enough,
    and a just-killed process leaves exactly that. The files themselves move
    perfectly well in that state, so the fallback is not a lesser path, just a
    slower one. Existing entries are left alone rather than overwritten, so an
    interrupted run finishes instead of clobbering what it already moved.
    """
    if not os.path.exists(dst):
        try:
            os.replace(src, dst)
            return
        except OSError:
            pass                      # fall through to per-entry

    os.makedirs(dst, exist_ok=True)
    for entry in os.listdir(src):
        s, d = os.path.join(src, entry), os.path.join(dst, entry)
        if not os.path.exists(d):
            os.replace(s, d)
    try:
        if not os.listdir(src):
            os.rmdir(src)
    except OSError:
        pass                          # empty but still held; harmless


def _slug(name: str) -> str:
    from app.database import _slugify  # noqa: E402
    return _slugify(name)


def cmd_list(args: argparse.Namespace) -> int:
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

    p = sub.add_parser("seed-directory", help="snapshot a mounted corpus into a named install")
    p.add_argument("source")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_seed_directory)

    p = sub.add_parser("list", help="list installed corpora")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="describe a bundle without unpacking it")
    p.add_argument("bundle")
    p.set_defaults(func=cmd_info)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
