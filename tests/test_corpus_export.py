#!/usr/bin/env python3
"""Portable exports of live SQLite data and effective pronunciations.

    python tests/test_corpus_export.py

All databases, dictionaries and archives belong to temporary fixtures. No
models, downloads or live corpus modifications are involved.
"""
from __future__ import annotations

import codecs
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sqlite3
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import database as db, phonemes
from scripts import corpus


class CorpusExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mrs-export-test-")
        self.data = Path(self.tmp.name)
        self.voice = self.data / "corpora" / "voice"
        (self.voice / "downloads").mkdir(parents=True)
        (self.voice / "downloads" / "video.mp4").write_bytes(b"test source")
        self.database = self.voice / "corpus.db"
        self.conn = sqlite3.connect(self.database)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA wal_autocheckpoint=0")
        self.conn.executescript(
            "CREATE TABLE sources (id INTEGER PRIMARY KEY, video_id TEXT);"
            "CREATE TABLE word_clips (id INTEGER PRIMARY KEY, source_id INTEGER, "
            "word TEXT, start_time REAL, end_time REAL, source_file TEXT);"
            "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);"
            "INSERT INTO settings VALUES ('schema_version', '1');")
        self.conn.commit()
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_source(self, name: str):
        cur = self.conn.execute("INSERT INTO sources (video_id) VALUES (?)", (name,))
        self.conn.execute(
            "INSERT INTO word_clips (source_id, word, start_time, end_time, source_file) "
            "VALUES (?, 'hello', 0, 0.4, 'downloads/video.mp4')", (cur.lastrowid,))
        self.conn.commit()

    def pack(self, root=None, before_open=None) -> Path:
        output = self.data / "export.tar.gz"
        original_open = corpus._open_write

        def open_after_snapshot(path):
            if before_open:
                before_open()
            return original_open(path)

        with patch.object(db, "DATA_DIR", str(self.data)), \
             patch.object(corpus, "_corpus_root", return_value=str(root or self.voice)), \
             patch.object(corpus, "_open_write", side_effect=open_after_snapshot), \
             redirect_stdout(io.StringIO()):
            result = corpus.pack_bundle(out=str(output))
        return Path(result)

    def unpack(self, bundle: Path) -> Path:
        target = self.data / "relocated" / "corpora" / "copied"
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle, "r:gz") as archive:
            self.assertTrue(all(not Path(m.name).is_absolute() and
                                str(self.data) not in m.name for m in archive))
            corpus._safe_extract(archive, str(target))
        return target

    def read_sources(self, database: Path) -> list[str]:
        connection = sqlite3.connect(database)
        try:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            return [r[0] for r in connection.execute("SELECT video_id FROM sources ORDER BY id")]
        finally:
            connection.close()

    def dictionary(self, data: Path, voice: Path) -> dict:
        entry = {"slug": "voice", "dir": str(voice), "db": str(voice / "corpus.db")}
        with patch.object(db, "DATA_DIR", str(data)), patch.object(db, "_active", entry), \
             patch.object(phonemes, "_user_dict", None), \
             patch.object(phonemes, "_user_entries", None), patch.object(phonemes, "_DICT", {}):
            return dict(phonemes._load_user_dict())

    def entry(self, data: Path, voice: Path, word: str):
        active = {"slug": "voice", "dir": str(voice), "db": str(voice / "corpus.db")}
        with patch.object(db, "DATA_DIR", str(data)), patch.object(db, "_active", active), \
             patch.object(phonemes, "_user_dict", None), \
             patch.object(phonemes, "_user_entries", None), patch.object(phonemes, "_DICT", {}), \
             patch.object(phonemes.log, "warning"):
            return phonemes.user_entry(word)

    def test_live_uncheckpointed_wal_is_exported_without_changing_source(self):
        # A reader pinned before the write makes a live checkpoint incomplete.
        reader = sqlite3.connect(self.database)
        try:
            reader.execute("BEGIN")
            self.assertEqual(reader.execute("SELECT count(*) FROM sources").fetchone()[0], 0)
            self.add_source("committed-in-wal")
            wal = Path(str(self.database) + "-wal")
            self.assertGreater(wal.stat().st_size, 0)
            database_before, wal_before = self.database.read_bytes(), wal.read_bytes()
            bundle = self.pack()
            self.assertEqual(self.database.read_bytes(), database_before)
            self.assertEqual(wal.read_bytes(), wal_before)
            with tarfile.open(bundle, "r:gz") as archive:
                names = archive.getnames()
                self.assertEqual(names.count("corpus.db"), 1)
                self.assertFalse(any(n.endswith(("-wal", "-shm")) for n in names))
            extracted = self.unpack(bundle)
            self.assertEqual(self.read_sources(extracted / "corpus.db"), ["committed-in-wal"])
            self.assertEqual((extracted / "downloads" / "video.mp4").read_bytes(), b"test source")
        finally:
            reader.close()

    def test_archive_uses_fixed_snapshot_while_live_writes_continue(self):
        self.add_source("before-snapshot")
        bundle = self.pack(before_open=lambda: self.add_source("after-snapshot"))
        extracted = self.unpack(bundle)
        self.assertEqual(self.read_sources(extracted / "corpus.db"), ["before-snapshot"])
        self.assertEqual(self.read_sources(self.database), ["before-snapshot", "after-snapshot"])

    def test_effective_dictionary_survives_relocation_with_voice_overrides(self):
        shared = (codecs.BOM_UTF8 + b"# Shared comments\r\n"
                  b"customseed,K AH S T AH M\r\n"
                  b"globalalias,customseed\r\n"
                  b"suppressed,S AH P R EH S T")
        voice = (codecs.BOM_UTF8 + b"# Voice comments\n"
                 b"customseed,V OY S\n"
                 b"voicealias,customseed\n"
                 b"suppressed,=skip\n")
        global_path = self.data / "pronunciations.csv"
        voice_path = self.voice / "pronunciations.csv"
        global_path.write_bytes(shared)
        voice_path.write_bytes(voice)
        effective_before = self.dictionary(self.data, self.voice)
        bundle = self.pack()
        self.assertEqual(global_path.read_bytes(), shared)
        self.assertEqual(voice_path.read_bytes(), voice)
        with tarfile.open(bundle, "r:gz") as archive:
            self.assertEqual(archive.getnames().count("pronunciations.csv"), 1)
            exported = archive.extractfile("pronunciations.csv").read()
        self.assertEqual(exported, shared + b"\n" + voice.removeprefix(codecs.BOM_UTF8))
        self.assertNotIn("\ufeff", exported.decode("utf-8-sig"))
        extracted = self.unpack(bundle)
        effective_after = self.dictionary(self.data / "relocated", extracted)
        self.assertEqual(effective_after, effective_before)
        self.assertEqual(effective_after["customseed"], ["V", "OY", "S"])
        self.assertEqual(effective_after["globalalias"], ["K", "AH", "S", "T", "AH", "M"])
        self.assertEqual(effective_after["voicealias"], ["V", "OY", "S"])
        self.assertIsNone(effective_after["suppressed"])
        self.assertEqual(self.entry(self.data / "relocated", extracted, "customseed"),
                         ("voice", "V OY S"))

    def test_editor_uses_last_valid_row_and_ignores_rejected_trailing_row(self):
        (self.data / "pronunciations.csv").write_text("customword,K AH S T AH M\n",
                                                     encoding="utf-8")
        (self.voice / "pronunciations.csv").write_text(
            "customword,V OY S\ncustomword,!!!\n", encoding="utf-8")
        self.assertEqual(self.entry(self.data, self.voice, "customword"), ("voice", "V OY S"))

    def test_editor_rejects_a_forward_reference_just_as_runtime_did(self):
        (self.data / "pronunciations.csv").write_text("customword,K AH S T AH M\n",
                                                     encoding="utf-8")
        (self.voice / "pronunciations.csv").write_text(
            "customword,definedlater\ndefinedlater,V OY S\n", encoding="utf-8")
        self.assertEqual(self.entry(self.data, self.voice, "customword"),
                         ("all", "K AH S T AH M"))

    def test_global_only_dictionary_is_included(self):
        shared = b"# Global only\ncustomword,K AH S T AH M\n"
        (self.data / "pronunciations.csv").write_bytes(shared)
        extracted = self.unpack(self.pack())
        self.assertEqual((extracted / "pronunciations.csv").read_bytes(), shared)
        self.assertEqual(self.dictionary(self.data / "relocated", extracted)["customword"],
                         ["K", "AH", "S", "T", "AH", "M"])

    def test_empty_voice_dictionary_does_not_add_a_newline(self):
        shared = b"customword,K AH S T AH M"
        (self.data / "pronunciations.csv").write_bytes(shared)
        (self.voice / "pronunciations.csv").write_bytes(codecs.BOM_UTF8)
        with tarfile.open(self.pack(), "r:gz") as archive:
            self.assertEqual(archive.extractfile("pronunciations.csv").read(), shared)

    def test_legacy_global_and_voice_dictionary_is_not_duplicated(self):
        shared = codecs.BOM_UTF8 + b"# Legacy dictionary\r\ncustomword,K AH S T AH M"
        path = self.data / "pronunciations.csv"
        path.write_bytes(shared)
        with tarfile.open(self.pack(root=self.data), "r:gz") as archive:
            self.assertEqual(archive.getnames().count("pronunciations.csv"), 1)
            self.assertEqual(archive.extractfile("pronunciations.csv").read(), shared)
        self.assertEqual(path.read_bytes(), shared)


if __name__ == "__main__":
    unittest.main()
