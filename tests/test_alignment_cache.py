#!/usr/bin/env python3
"""Character alignment caches belong to a clip, including its corpus.

    python tests/test_alignment_cache.py

Use two real temporary databases with overlapping clip ids and source names.
Stub only model inference; no model download, GPU or corpus mutation is needed.
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import database as db, forced_align as fa


class AlignmentCacheTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="mrs-alignment-cache-")
        self.addCleanup(self.scratch.cleanup)
        self.corpora = {}
        self.alignments = {"alpha": [("C", 0.01, 0.09)], "beta": [("C", 0.21, 0.29)]}
        for slug in ("alpha", "beta"):
            directory = os.path.join(self.scratch.name, slug)
            os.makedirs(os.path.join(directory, "downloads"))
            source = os.path.join(directory, "downloads", "same.mp4")
            with open(source, "wb") as f:
                f.write(b"test source -- inference is stubbed")
            target = os.path.join(directory, "corpus.db")
            self.corpora[slug] = {"slug": slug, "dir": directory, "db": target}
            with contextlib.closing(sqlite3.connect(target)) as conn, conn:
                conn.executescript("""
                    CREATE TABLE word_clips (id INTEGER PRIMARY KEY, word TEXT,
                        source_file TEXT, start_time REAL, end_time REAL);
                    CREATE TABLE clip_alignments (clip_id INTEGER PRIMARY KEY, alignment TEXT);
                    INSERT INTO word_clips VALUES (1, 'cat', 'downloads/same.mp4', 1.0, 1.5);
                """)
                conn.execute("INSERT INTO clip_alignments VALUES (1, ?)",
                             (json.dumps(self.alignments[slug]),))
        for patcher in (patch.object(db, "_active", self.corpora["alpha"]),
                        patch.object(fa, "_char_cache", {})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def clip(self, slug="alpha", **changes):
        clip = {"id": 1, "word": "cat", "start_time": 1.0, "end_time": 1.5,
                "source_file": os.path.join(self.corpora[slug]["dir"], "downloads", "same.mp4")}
        clip.update(changes)
        return clip

    def stored(self, slug):
        with contextlib.closing(sqlite3.connect(self.corpora[slug]["db"])) as conn:
            row = conn.execute("SELECT alignment FROM clip_alignments WHERE clip_id=1").fetchone()
        return [(ch, a, b) for ch, a, b in json.loads(row[0])] if row else None

    def clear_stored(self, slug):
        with contextlib.closing(sqlite3.connect(self.corpora[slug]["db"])) as conn, conn:
            conn.execute("DELETE FROM clip_alignments")

    def test_matching_absolute_local_clip_reads_persisted_alignment(self):
        with patch.object(fa, "_align") as align:
            self.assertEqual(fa.char_times(self.clip()), self.alignments["alpha"])
            align.assert_not_called()

    def test_relative_and_absolute_local_source_share_memory_entry(self):
        with patch.object(fa, "_align") as align:
            first = fa.char_times(self.clip())
            with patch.object(fa, "_load_alignment") as load:
                self.assertEqual(fa.char_times(self.clip(source_file="downloads/same.mp4")), first)
                load.assert_not_called()
            align.assert_not_called()

    def test_matching_local_clip_saves_new_alignment(self):
        self.clear_stored("alpha")
        got = [("C", 0.03123456, 0.09876543)]
        with patch.object(fa, "_align", return_value=got):
            self.assertEqual(fa.char_times(self.clip()), got)
        self.assertEqual(self.stored("alpha"), [("C", 0.031235, 0.098765)])
        self.assertEqual(self.stored("beta"), self.alignments["beta"])

    def test_borrowed_same_id_label_and_times_does_not_read_or_overwrite_active(self):
        got = [("C", 0.31, 0.39)]
        with patch.object(fa, "_align", return_value=got) as align:
            self.assertEqual(fa.char_times(self.clip("beta")), got)
            self.assertEqual(fa.char_times(self.clip("beta")), got)
            align.assert_called_once()
            self.assertEqual(fa.char_times(self.clip()), self.alignments["alpha"])
            align.assert_called_once()
        self.assertEqual(self.stored("alpha"), self.alignments["alpha"])
        self.assertEqual(self.stored("beta"), self.alignments["beta"])

    def test_borrowed_clip_does_not_create_active_persistent_entry(self):
        self.clear_stored("alpha")
        with patch.object(fa, "_align", return_value=[("C", 0.31, 0.39)]):
            fa.char_times(self.clip("beta"))
        self.assertIsNone(self.stored("alpha"))
        self.assertEqual(self.stored("beta"), self.alignments["beta"])

    def test_changed_word_or_bounds_realign_without_touching_stored_local_clip(self):
        with patch.object(fa, "_align", return_value=[("D", 0.1, 0.2)]) as align:
            fa.char_times(self.clip())
            self.assertEqual(align.call_count, 0)
            fa.char_times(self.clip(word="dog"))
            fa.char_times(self.clip(start_time=1.1))
            fa.char_times(self.clip(end_time=1.6))
            self.assertEqual(align.call_count, 3)
        self.assertEqual(self.stored("alpha"), self.alignments["alpha"])

    def test_changed_borrowed_fingerprint_does_not_reuse_memory_result(self):
        outputs = [[("C", 0.1, 0.2)], [("D", 0.2, 0.3)], [("D", 0.3, 0.4)]]
        with patch.object(fa, "_align", side_effect=outputs) as align:
            self.assertEqual(fa.char_times(self.clip("beta")), outputs[0])
            self.assertEqual(fa.char_times(self.clip("beta", word="dog")), outputs[1])
            self.assertEqual(fa.char_times(self.clip("beta", word="dog", start_time=1.1)), outputs[2])
            self.assertEqual(align.call_count, 3)

    def test_relative_cache_keys_resolve_against_each_active_corpus(self):
        clip = self.clip(source_file="downloads/same.mp4")
        with patch.object(fa, "_align") as align:
            self.assertEqual(fa.char_times(clip), self.alignments["alpha"])
            with patch.object(db, "_active", self.corpora["beta"]):
                self.assertEqual(fa.char_times(clip), self.alignments["beta"])
            align.assert_not_called()

    def test_relative_source_outside_corpus_declines_alignment(self):
        with patch.object(fa, "_align") as align, self.assertLogs(fa.log, level="WARNING"):
            self.assertIsNone(fa.char_times(self.clip(source_file="../beta/downloads/same.mp4")))
            align.assert_not_called()
        self.assertEqual(self.stored("alpha"), self.alignments["alpha"])
        self.assertEqual(self.stored("beta"), self.alignments["beta"])

    def test_save_rechecks_local_fingerprint_after_inference(self):
        self.clear_stored("alpha")

        def edit_during_alignment(_clip):
            with db.get_db() as conn:
                conn.execute("UPDATE word_clips SET start_time=1.1 WHERE id=1")
            return [("C", 0.1, 0.2)]

        with patch.object(fa, "_align", side_effect=edit_during_alignment):
            fa.char_times(self.clip())
        self.assertIsNone(self.stored("alpha"))

    def test_invalidate_removes_all_matching_runtime_ids_and_only_active_storage(self):
        with patch.object(fa, "_align", return_value=[("C", 0.3, 0.4)]):
            fa.char_times(self.clip())
            fa.char_times(self.clip("beta"))
            fa.char_times(self.clip("beta", word="dog"))
        self.assertEqual(len(fa._char_cache), 3)
        fa.invalidate(1)
        self.assertEqual(fa._char_cache, {})
        self.assertIsNone(self.stored("alpha"))
        self.assertEqual(self.stored("beta"), self.alignments["beta"])


class ModelModeTest(unittest.TestCase):
    def test_model_load_uses_evaluation_mode(self):
        model = Mock()
        model.eval.return_value = model
        model.to.return_value = model
        bundle = types.SimpleNamespace(get_model=lambda: model, get_labels=lambda: ("-", "A", "|"))
        torch = types.ModuleType("torch")
        torchaudio = types.ModuleType("torchaudio")
        torchaudio.pipelines = types.SimpleNamespace(WAV2VEC2_ASR_BASE_960H=bundle)
        with patch.dict(sys.modules, {"torch": torch, "torchaudio": torchaudio}), \
                patch.object(fa, "_model", None), patch.object(fa, "_labels", None), \
                patch.object(fa, "_dict", None), patch.object(fa, "_device", "cpu"), \
                patch("app.device.get", return_value="cpu"), \
                patch("app.device.describe", return_value="CPU"):
            self.assertTrue(fa._ensure_model())
        model.eval.assert_called_once_with()
        model.to.assert_called_once_with("cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
