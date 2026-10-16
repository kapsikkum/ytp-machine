#!/usr/bin/env python3
"""Untrusted bundles cannot damage an installed corpus or escape staging."""
from contextlib import ExitStack, redirect_stdout
import io
import os
from pathlib import Path
import sqlite3
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import database as db
from scripts import corpus


class CorpusInstallTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="mrs-install-")))
        self.stack.enter_context(patch.object(db, "CORPORA_DIR", str(self.root / "corpora")))
        self.stack.enter_context(patch.object(db, "LEGACY_DB", str(self.root / "absent.db")))
        self.target = self.root / "corpora" / "voice"
        self.bundle = self.root / "new.tar"
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def make_database(self, target, label="new"):
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(target) as conn:
            conn.executescript("CREATE TABLE sources (id INTEGER PRIMARY KEY, video_id TEXT, source_file TEXT);"
                               "CREATE TABLE word_clips (id INTEGER PRIMARY KEY, source_id INTEGER, word TEXT,"
                               "start_time REAL, end_time REAL, source_file TEXT);")
            conn.execute("INSERT INTO sources VALUES (1, ?, 'downloads/video.mp4')", (label,))
        conn.close()

    def archive(self, extra=(), database=True):
        staging = self.root / "fixture"
        staging.mkdir(exist_ok=True)
        if database:
            self.make_database(staging / "corpus.db")
        with tarfile.open(self.bundle, "w") as tar:
            if database:
                tar.add(staging / "corpus.db", arcname="corpus.db")
            info = tarfile.TarInfo("downloads/video.mp4")
            info.size = 3
            tar.addfile(info, io.BytesIO(b"new"))
            for name, kind, content in extra:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.linkname = content.decode() if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
                info.size = len(content) if kind == tarfile.REGTYPE else 0
                tar.addfile(info, io.BytesIO(content) if kind == tarfile.REGTYPE else None)

    def existing(self):
        self.make_database(self.target / "corpus.db", "old")
        (self.target / "downloads").mkdir()
        (self.target / "downloads" / "old.mp4").write_bytes(b"old recording")
        (self.target / "private-note.txt").write_text("keep this")

    def test_bad_forced_import_keeps_database_and_all_existing_files(self):
        self.existing()
        original = (self.target / "corpus.db").read_bytes()
        self.archive(extra=[("../outside.txt", tarfile.REGTYPE, b"bad")])
        with self.assertRaises(ValueError):
            corpus.install_bundle(str(self.bundle), "voice", force=True)
        self.assertEqual((self.target / "corpus.db").read_bytes(), original)
        self.assertEqual((self.target / "downloads" / "old.mp4").read_bytes(), b"old recording")
        self.assertFalse((self.root / "corpora" / "outside.txt").exists())
        self.assertEqual([c["slug"] for c in db.list_corpora()], ["voice"])
        self.assertFalse(list((self.root / "corpora").glob(".install-*")))

    def test_forced_import_keeps_recoverable_backup_and_hides_staging(self):
        self.existing()
        original = (self.target / "corpus.db").read_bytes()
        self.archive()
        result = corpus.install_bundle(str(self.bundle), "voice", force=True)
        backup = Path(result["backup"])
        self.assertEqual((backup / "corpus.db").read_bytes(), original)
        self.assertEqual((backup / "private-note.txt").read_text(), "keep this")
        self.assertEqual((self.target / "downloads" / "video.mp4").read_bytes(), b"new")
        self.assertEqual([c["slug"] for c in db.list_corpora()], ["voice"])

    def test_failed_final_swap_restores_previous_corpus(self):
        self.existing()
        self.archive()
        replace = os.replace

        def fail_swap(source, destination):
            if Path(source).name.startswith(".install-") and Path(destination) == self.target:
                raise PermissionError("fixture swap failed")
            return replace(source, destination)

        with patch.object(corpus.os, "replace", side_effect=fail_swap), self.assertRaises(PermissionError):
            corpus.install_bundle(str(self.bundle), "voice", force=True)
        self.assertEqual((self.target / "downloads" / "old.mp4").read_bytes(), b"old recording")
        self.assertTrue((self.target / "private-note.txt").exists())

    def test_missing_or_invalid_database_never_becomes_installed(self):
        for invalid in (False, True):
            self.archive(database=False, extra=[("corpus.db", tarfile.REGTYPE, b"not SQLite")] if invalid else [])
            with self.assertRaises(ValueError):
                corpus.install_bundle(str(self.bundle), "voice")
            self.assertFalse(self.target.exists())

    def test_links_devices_unknown_and_nonportable_members_are_rejected(self):
        cases = [("downloads/link", tarfile.SYMTYPE, b"../../secret"),
                 ("downloads/link", tarfile.LNKTYPE, b"corpus.db"),
                 ("downloads/fifo", tarfile.FIFOTYPE, b""),
                 ("main.py", tarfile.REGTYPE, b"bad"),
                 ("C:/downloads/x", tarfile.REGTYPE, b"bad"),
                 ("downloads/../x", tarfile.REGTYPE, b"bad"),
                 ("downloads\\x", tarfile.REGTYPE, b"bad"),
                 ("downloads/x:stream", tarfile.REGTYPE, b"bad"),
                 ("downloads/CON.mp4", tarfile.REGTYPE, b"bad"),
                 ("downloads/x. ", tarfile.REGTYPE, b"bad"),
                 ("downloads/VIDEO.mp4", tarfile.REGTYPE, b"bad")]
        for member in cases:
            with self.subTest(member=member[0]):
                self.archive(extra=[member], database=False)
                with self.assertRaises(ValueError):
                    corpus.install_bundle(str(self.bundle), "voice")
                self.assertFalse(self.target.exists())

    def test_archive_limits_are_applied_before_install(self):
        self.archive()
        for limit in (patch.object(corpus, "MAX_UNPACKED_BYTES", 1),
                      patch.object(corpus, "MAX_MEMBER_FILES", 1)):
            with limit, self.assertRaises(ValueError):
                corpus.install_bundle(str(self.bundle), "voice")
            self.assertFalse(self.target.exists())

    def test_unpack_preserves_unrelated_files_and_backs_up_replaced_members(self):
        self.existing()
        self.archive()
        args = SimpleNamespace(into=str(self.target), bundle=str(self.bundle), force=True)
        self.assertEqual(corpus.cmd_unpack(args), 0)
        self.assertEqual((self.target / "private-note.txt").read_text(), "keep this")
        backup = next((self.target / ".corpus-backups").iterdir())
        self.assertEqual((backup / "downloads" / "old.mp4").read_bytes(), b"old recording")

    def test_download_format_and_failure_cleanup(self):
        made = []
        mkstemp = tempfile.mkstemp

        def make_temp(**kwargs):
            fd, path = mkstemp(dir=self.root, **kwargs)
            made.append(path)
            return fd, path

        response = io.BytesIO(b"gzip bytes")
        with patch.object(corpus.tempfile, "mkstemp", side_effect=make_temp), \
                patch.object(corpus.urllib.request, "urlopen", return_value=response) as fetch:
            result = corpus._fetch("https://example.invalid/archive.tar.gz?token=unused")
        self.assertTrue(result.endswith(".tar.gz"))
        fetch.assert_called_once_with("https://example.invalid/archive.tar.gz?token=unused", timeout=30)
        with patch.object(corpus.tempfile, "mkstemp", side_effect=make_temp), \
                patch.object(corpus.urllib.request, "urlopen", side_effect=OSError("fixture offline")), \
                self.assertRaises(OSError):
            corpus._fetch("https://example.invalid/archive.tar.zst")
        self.assertFalse(Path(made[-1]).exists())

    def test_directory_seed_includes_live_wal_and_pronunciations(self):
        source = self.root / "mounted"
        self.make_database(source / "corpus.db", "first")
        (source / "downloads").mkdir()
        (source / "downloads" / "video.mp4").write_bytes(b"recording")
        (source / "pronunciations.csv").write_bytes(b"zoop,Z UW P\n")
        with self.stack.enter_context(closing_connection(source / "corpus.db")) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            reader = sqlite3.connect(source / "corpus.db")
            try:
                reader.execute("BEGIN")
                reader.execute("SELECT * FROM sources").fetchall()
                writer.execute("INSERT INTO sources VALUES (2, 'committed-wal', 'downloads/video.mp4')")
                writer.commit()
                self.assertGreater(Path(str(source / "corpus.db") + "-wal").stat().st_size, 0)
                result = corpus.seed_directory(str(source), "voice")
            finally:
                reader.close()
        with sqlite3.connect(self.target / "corpus.db") as snapshot:
            self.assertEqual(snapshot.execute("SELECT video_id FROM sources WHERE id=2").fetchone()[0],
                             "committed-wal")
            self.assertEqual(snapshot.execute("PRAGMA journal_mode").fetchone()[0], "delete")
        snapshot.close()
        self.assertEqual(result["slug"], "voice")
        self.assertIn(b"zoop,Z UW P", (self.target / "pronunciations.csv").read_bytes())
        self.assertEqual((self.target / "downloads" / "video.mp4").read_bytes(), b"recording")

    def test_failed_directory_seed_is_retryable_and_keeps_source(self):
        source = self.root / "mounted"
        self.make_database(source / "corpus.db")
        (source / "downloads").mkdir()
        with patch.object(corpus.shutil, "copytree", side_effect=OSError("fixture copy failure")), \
                self.assertRaises(OSError):
            corpus.seed_directory(str(source), "voice")
        self.assertFalse(self.target.exists())
        self.assertTrue((source / "corpus.db").exists())
        self.assertFalse(list((self.root / "corpora").glob(".install-*")))
        corpus.seed_directory(str(source), "voice")
        with self.assertRaises(FileExistsError):
            corpus.seed_directory(str(source), "voice")

    def test_busy_wal_migration_aborts_before_moving_files(self):
        legacy = self.root / "michael_rosen.db"
        self.make_database(legacy)
        (self.root / "downloads").mkdir()
        (self.root / "downloads" / "video.mp4").write_bytes(b"recording")
        with closing_connection(legacy) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            reader = sqlite3.connect(legacy)
            try:
                reader.execute("BEGIN")
                reader.execute("SELECT * FROM sources").fetchall()
                writer.execute("INSERT INTO sources VALUES (2, 'wal', 'downloads/video.mp4')")
                writer.commit()
                args = SimpleNamespace(name="voice", dry_run=False, force=False, quiet=True)
                with patch.object(corpus, "_data_root", return_value=str(self.root)), \
                        self.assertRaisesRegex(RuntimeError, "stop the app"):
                    corpus.cmd_migrate(args)
                self.assertTrue(legacy.exists())
                self.assertTrue((self.root / "downloads" / "video.mp4").exists())
                self.assertFalse(self.target.exists())
                self.assertGreater(Path(str(legacy) + "-wal").stat().st_size, 0)
            finally:
                reader.close()


def closing_connection(path):
    from contextlib import closing
    return closing(sqlite3.connect(path))


if __name__ == "__main__":
    unittest.main()
