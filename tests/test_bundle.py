#!/usr/bin/env python3
"""Unit tests for the bundle stream helpers.

    python tests/test_bundle.py

These exist because they did not, and every .tar.zst the packer produced was
truncated. _open_write stacks three things -- a tarfile writing into a zstd
compressor writing into a file -- and only the tarfile was being closed, so the
compressor never flushed its final frame. Nothing noticed: pack reported
success, the checksum matched (computed over the same truncated bytes), and
even listing the archive worked, because a tar reader returns the entries it
can reach and then stops. It failed days later, on a server, as "unexpected end
of data".

The property that was violated is the only one worth asserting: what goes in
comes back out, byte for byte, through the real writer and the real reader.
"""
import os
import shutil
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.corpus import _open_read, _open_write  # noqa: E402

failures: list[str] = []

# Large and incompressible. The bug lost only the unflushed tail, so a handful
# of tiny files fits in the compressor's buffer and round-trips even when
# broken -- which is exactly why small-scale checking would have missed it.
MEMBERS = {
    "corpus.db":              os.urandom(300_000),
    "downloads/aaa111.mp4":   os.urandom(900_000),
    "downloads/bbb222.mp4":   os.urandom(900_000),
    "downloads/ccc333.mp4":   b"\x00\xff" * 500_000,
}


def write_bundle(path: str, members: dict[str, bytes]) -> str:
    """Build a bundle the way cmd_pack does, and finish it the way it must be."""
    staging = tempfile.mkdtemp(prefix="ytp-stage-")
    try:
        for name, blob in members.items():
            full = os.path.join(staging, name.replace("/", os.sep))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(blob)
        tar, out, finish = _open_write(path)
        try:
            for name in members:
                tar.add(os.path.join(staging, name.replace("/", os.sep)), arcname=name)
        finally:
            finish()          # never tar.close() -- that is the whole bug
        return out
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def read_bundle(path: str) -> dict[str, bytes]:
    got: dict[str, bytes] = {}
    tar = _open_read(path)
    try:
        # A streaming reader must be walked in order; it cannot seek back.
        for info in tar:
            if info.isfile():
                f = tar.extractfile(info)
                got[info.name.replace(os.sep, "/")] = f.read() if f else b""
    finally:
        tar.close()
    return got


try:
    import zstandard  # noqa: F401
    HAVE_ZSTD = True
except ImportError:
    HAVE_ZSTD = False

# _open_write falls back to gzip when zstandard is missing, which is right for
# a user and useless for a test: this whole file exists for a bug in the zstd
# path, and without the check it would pack gzip twice and report success.
FORMATS = [".tar.zst", ".tar.gz"] if HAVE_ZSTD else [".tar.gz"]

work = tempfile.mkdtemp(prefix="ytp-bundle-")
try:
    for ext in FORMATS:
        path = os.path.join(work, "subject" + ext)
        try:
            actual = write_bundle(path, MEMBERS)
        except Exception as exc:
            failures.append(f"  {ext}: packing raised {type(exc).__name__}: {exc}")
            continue

        if not os.path.exists(actual):
            failures.append(f"  {ext}: nothing was written")
            continue
        if not actual.endswith(ext):
            failures.append(f"  {ext}: silently written as {os.path.basename(actual)} "
                            f"instead -- this format was never actually tested")
            continue

        try:
            got = read_bundle(actual)
        except Exception as exc:
            # This is the shape the original failure took: writing "succeeded",
            # reading blew up at the cut.
            failures.append(f"  {ext}: reading it back raised "
                            f"{type(exc).__name__}: {exc}")
            continue

        missing = set(MEMBERS) - set(got)
        if missing:
            failures.append(f"  {ext}: {sorted(missing)} never came back -- "
                            f"the stream ended early")
        for name, blob in MEMBERS.items():
            if name in got and got[name] != blob:
                failures.append(
                    f"  {ext}: {name} came back {len(got[name])} bytes, "
                    f"sent {len(blob)}")

    # The reader must reject a truncated bundle rather than quietly returning
    # a short member list, which is what let the original bug travel so far.
    good = os.path.join(work, "subject" + FORMATS[0])
    if os.path.exists(good):
        # Same extension as the bundle it was cut from: _open_read dispatches on
        # it, so a gzip bundle named .zst fails as "no zstandard" rather than as
        # the truncation this is checking for.
        cut = os.path.join(work, "cut" + FORMATS[0])
        with open(good, "rb") as f:
            whole = f.read()
        with open(cut, "wb") as f:
            f.write(whole[: len(whole) * 2 // 3])
        try:
            recovered = read_bundle(cut)
            if len(recovered) == len(MEMBERS):
                failures.append("  a truncated bundle read back as complete")
        except Exception:
            pass                      # raising is the correct outcome
finally:
    shutil.rmtree(work, ignore_errors=True)

if failures:
    print(f"FAILED ({len(failures)}):")
    print("\n".join(failures))
    sys.exit(1)
print(f"ok: {len(MEMBERS)} members round-trip byte-for-byte through "
      f"{' and '.join(f.lstrip('.') for f in FORMATS)}, and a truncated bundle "
      f"is not mistaken for a whole one")
if not HAVE_ZSTD:
    print("   NOTE: zstandard is not installed here, so the zstd path -- the one "
          "that broke -- was not tested. pip install zstandard")
