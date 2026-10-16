#!/usr/bin/env python3
"""Retranscription preserves edits and reconciles only replaced clip data."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import database as db, forced_align as fa
from scripts import realign


class RealignLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="mrs-realign-")))
        self.stack.enter_context(patch.object(db, "_active", {
            "slug": "test", "dir": str(self.root), "db": str(self.root / "corpus.db")}))
        self.stack.enter_context(patch.object(db, "_ready", set()))
        self.stack.enter_context(patch.object(fa, "_char_cache", {}))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        (self.root / "downloads").mkdir()
        self.recording = self.root / "downloads" / "video.mp4"
        self.recording.write_bytes(b"fixture recording")
        db.init_db()
        self.cache = self.root / "cache" / "clips"
        self.cache.mkdir(parents=True)
        with db.get_db() as conn:
            for source in (1, 2):
                conn.execute("INSERT INTO sources (id,video_id,source_file) VALUES (?,?,?)",
                             (source, str(source), "downloads/video.mp4"))
            for cid, source, word, start, end in ((1, 1, "cat", 0., 1.), (2, 1, "dog", 1., 2.), (3, 2, "bird", 0., 1.)):
                conn.execute("INSERT INTO word_clips (id,source_id,word,start_time,end_time,source_file) VALUES (?,?,?,?,?,?)",
                             (cid, source, word, start, end, "downloads/video.mp4"))
                conn.execute("INSERT INTO splice_ratings VALUES ('target',?, -2)", (cid,))
                conn.execute("INSERT INTO boundary_reports VALUES (?, 'ends late', 1)", (cid,))
                conn.execute("INSERT INTO clip_alignments (clip_id,alignment) VALUES (?, '[]')", (cid,))
                name = f"{cid}_0.180.ts"
                conn.execute("INSERT INTO clip_cache (clip_id,pad_end,cache_file) VALUES (?,0.18,?)", (cid, name))
                (self.cache / name).write_bytes(b"cached recording")
            self.source = dict(conn.execute("SELECT * FROM sources WHERE id=1").fetchone())
            conn.execute("INSERT INTO splice_recipes VALUES ('target', ?)", (json.dumps([
                {"phones": ["K"], "from": ["cat"], "clip_id": 1, "start": .1, "end": .2},
                {"phones": ["D"], "from": ["dog"], "clip_id": 2, "start": 1.1, "end": 1.2}]),))

    def words(self):
        return [{"word": "cat", "start": 0., "end": 1.}, {"word": "fox", "start": 1., "end": 2.}]

    def metadata_ids(self, table):
        with db.get_db() as conn:
            return [r[0] for r in conn.execute(f"SELECT clip_id FROM {table} ORDER BY clip_id")]

    def test_replacement_clears_only_old_metadata_and_remaps_valid_pins(self):
        with patch.object(realign, "transcribe", return_value=self.words()):
            self.assertEqual(realign.realign_source(self.source, "tiny"), 2)
        for table in ("clip_cache", "clip_alignments", "splice_ratings", "boundary_reports"):
            self.assertEqual(self.metadata_ids(table), [3])
        self.assertFalse((self.cache / "1_0.180.ts").exists())
        self.assertTrue((self.cache / "3_0.180.ts").exists())
        with db.get_db() as conn:
            groups = json.loads(conn.execute("SELECT recipe FROM splice_recipes").fetchone()[0])
            cat_id = conn.execute("SELECT id FROM word_clips WHERE source_id=1 AND word='cat'").fetchone()[0]
        self.assertEqual(groups[0]["clip_id"], cat_id)
        self.assertEqual((groups[0]["start"], groups[0]["end"]), (.1, .2))
        self.assertEqual(groups[1], {"phones": ["D"], "from": ["dog"]})

    def test_hand_edits_are_protected_before_loading_transcriber(self):
        with db.get_db() as conn:
            conn.execute("UPDATE word_clips SET edited=1 WHERE id=1")
        with patch.object(realign, "transcribe") as transcribe, self.assertRaisesRegex(ValueError, "--replace-edited"):
            realign.realign_source(self.source, "tiny")
        transcribe.assert_not_called()
        with patch.object(realign, "transcribe", return_value=self.words()):
            self.assertEqual(realign.realign_source(self.source, "tiny", replace_edited=True), 2)

    def test_concurrent_edit_aborts_instead_of_overwriting_it(self):
        def edit(*args):
            with db.get_db() as conn:
                conn.execute("UPDATE word_clips SET word='hand edit' WHERE id=1")
            return self.words()

        with patch.object(realign, "transcribe", side_effect=edit), self.assertRaisesRegex(ValueError, "changed during"):
            realign.realign_source(self.source, "tiny")
        self.assertEqual(self.metadata_ids("clip_alignments"), [1, 2, 3])
        with db.get_db() as conn:
            self.assertEqual(conn.execute("SELECT word FROM word_clips WHERE id=1").fetchone()[0], "hand edit")

    def test_changed_recording_aborts_before_replacing_clips(self):
        def replace(*args):
            self.recording.write_bytes(b"different recording now")
            return self.words()

        with patch.object(realign, "transcribe", side_effect=replace), self.assertRaisesRegex(ValueError, "recording changed"):
            realign.realign_source(self.source, "tiny")
        self.assertEqual(self.metadata_ids("clip_alignments"), [1, 2, 3])

    def test_insert_failure_rolls_back_clips_and_dependent_metadata(self):
        with db.get_db() as conn:
            conn.executescript("CREATE TRIGGER fail_insert BEFORE INSERT ON word_clips "
                               "BEGIN SELECT RAISE(ABORT,'fixture failure'); END;")
        with patch.object(realign, "transcribe", return_value=self.words()), self.assertRaises(sqlite3.IntegrityError):
            realign.realign_source(self.source, "tiny")
        for table in ("clip_cache", "clip_alignments", "splice_ratings", "boundary_reports"):
            self.assertEqual(self.metadata_ids(table), [1, 2, 3])
        self.assertTrue((self.cache / "1_0.180.ts").exists())


if __name__ == "__main__":
    unittest.main()
