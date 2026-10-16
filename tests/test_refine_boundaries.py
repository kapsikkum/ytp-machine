#!/usr/bin/env python3
"""Boundary refinement regressions without torch, models or source downloads.

    python tests/test_refine_boundaries.py

The transcript encoding, window coordinates and writes decide which word is
heard. Exercise those with a deterministic alignment and an isolated SQLite
database; acoustic accuracy still needs a separately annotated audio sample.
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

from scripts import refine_boundaries as rb


def word(cid, start=0.0, end=0.2, edited=0, text="a"):
    return {"id": cid, "word": text, "start_time": start,
            "end_time": end, "edited": edited}


class Audio:
    def __init__(self, samples):
        self.samples = samples

    def size(self, dim):
        return self.samples if dim == 1 else 1

    def __getitem__(self, item):
        selected = item[1]
        return Audio(selected.stop - selected.start)

    def to(self, _device):
        return self


class RefineTests(unittest.TestCase):
    def setUp(self):
        self.vocab = {ch: i + 1 for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ'|")}
        self.vocab_patch = patch.object(rb, "_DICT", self.vocab)
        self.vocab_patch.start()
        self.addCleanup(self.vocab_patch.stop)

    def test_targets_keep_word_delimiters_and_letter_ranges(self):
        targets, ranges = rb._ctc_targets([word(1, text="a"), word(2, text="nice")])
        self.assertEqual(targets, [self.vocab[c] for c in "A|NICE"])
        self.assertEqual(ranges, [(0, 1), (2, 6)])
        other, _ = rb._ctc_targets([word(1, text="an"), word(2, text="ice")])
        self.assertNotEqual(targets, other)

    def test_unencodable_word_keeps_its_place_in_ranges(self):
        targets, ranges = rb._ctc_targets([word(1), word(2, text="ps5"), word(3, text="b")])
        self.assertEqual(targets, [self.vocab[c] for c in "A|B"])
        self.assertEqual(ranges, [(0, 1), None, (2, 3)])

    def test_dense_words_do_not_override_neighbor_limit(self):
        updates, _ = rb._boundary_updates(
            [word(1, 1.0, 1.4), word(2, 1.4, 1.7)],
            [(1.0, 1.4), (1.4, 1.7)], 2.0)
        self.assertEqual(updates[0], (1.0, 1.4, 1))

    def test_tails_stop_at_eof_and_keep_roomy_gap(self):
        updates, _ = rb._boundary_updates([word(1)], [(0.0, 0.2)], 0.22)
        self.assertEqual(updates, [(0.0, 0.22, 1)])
        updates, _ = rb._boundary_updates([word(1)], [(0.0, 0.2)], 1.0)
        self.assertEqual(updates, [(0.0, 0.5, 1)])

    def test_edited_and_unaligned_neighbors_constrain_the_previous_word(self):
        words = [word(1), word(2, 0.3, 0.5, edited=1)]
        updates, _ = rb._boundary_updates(words, [(0.0, 0.2), (0.6, 0.8)], 1.0)
        self.assertEqual(updates, [(0.0, 0.28, 1)])
        words[1]["edited"] = 0
        updates, _ = rb._boundary_updates(words, [(0.0, 0.2), None], 1.0)
        self.assertEqual(updates, [(0.0, 0.28, 1)])

    def test_degenerate_and_nonfinite_alignment_is_not_written(self):
        for aligned in [(float("nan"), 0.2), (0.1, float("inf")), (0.2, 0.1)]:
            self.assertEqual(rb._boundary_updates([word(1)], [aligned], 1.0), ([], []))

    def test_ctc_uses_log_probabilities_and_repeat_frame_requirement(self):
        torch = types.ModuleType("torch")
        torch.inference_mode = contextlib.nullcontext
        torch.int32 = "int32"
        torch.tensor = lambda values, **_kw: values
        functional = types.ModuleType("torchaudio.functional")
        torchaudio = types.ModuleType("torchaudio")
        torchaudio.functional = functional
        logp = object()
        emission = Mock(device="cpu")
        emission.size.return_value = 3
        emission.log_softmax.return_value = logp
        cpu = Mock()
        cpu.cpu.return_value = cpu
        functional.forced_align = Mock(return_value=([cpu], [cpu]))
        functional.merge_tokens = Mock(return_value=[
            types.SimpleNamespace(start=0, end=1), types.SimpleNamespace(start=2, end=3)])
        with patch.dict(sys.modules, {"torch": torch, "torchaudio": torchaudio,
                                      "torchaudio.functional": functional}), \
                patch.object(rb, "_model", Mock(return_value=(emission, None))):
            got = rb._align_window(Audio(960), [word(1, text="aa")])
            self.assertEqual(got, [(0.0, 0.06)])
            emission.log_softmax.assert_called_once_with(dim=-1)
            self.assertIs(functional.forced_align.call_args.args[0], logp)
            functional.forced_align.reset_mock()
            emission.size.return_value = 2  # two A's require a blank too
            self.assertEqual(rb._align_window(Audio(640), [word(1, text="aa")]), [None])
            functional.forced_align.assert_not_called()

    def test_load_model_uses_evaluation_mode(self):
        model = Mock()
        model.eval.return_value = model
        model.to.return_value = model
        bundle = types.SimpleNamespace(get_model=lambda: model, get_labels=lambda: ("-", "A", "|"))
        torchaudio = types.ModuleType("torchaudio")
        torchaudio.pipelines = types.SimpleNamespace(WAV2VEC2_ASR_BASE_960H=bundle)
        with patch.dict(sys.modules, {"torchaudio": torchaudio}), \
                patch.object(rb, "_model", None), patch.object(rb, "_labels", None), \
                patch.object(rb, "_device", "cpu"), \
                patch("app.device.get", return_value="cpu"), \
                patch("app.device.describe", return_value="CPU"):
            rb._load_model("cpu")
        model.eval.assert_called_once_with()
        model.to.assert_called_once_with("cpu")


class DatabaseRefineTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="mrs-refine-test-")
        self.addCleanup(self.scratch.cleanup)
        self.db = os.path.join(self.scratch.name, "test.db")
        with contextlib.closing(sqlite3.connect(self.db)) as conn, conn:
            conn.executescript("""
                CREATE TABLE sources (id INTEGER PRIMARY KEY, source_file TEXT);
                CREATE TABLE word_clips (id INTEGER PRIMARY KEY, source_id INTEGER,
                    word TEXT, start_time REAL, end_time REAL, edited INTEGER, phones TEXT);
                CREATE TABLE clip_alignments (clip_id INTEGER PRIMARY KEY, alignment TEXT);
                INSERT INTO sources VALUES (1, 'downloads/test.mp4');
            """)
            conn.executemany("INSERT INTO word_clips VALUES (?, 1, ?, ?, ?, ?, ?)", [
                (1, "cat", 0.1, 0.35, 0, json.dumps([["K", 0.05, 0.1]])),
                (2, "dog", 0.4, 0.6, 1, json.dumps([["D", 0.0, 0.1]])),
                (3, "end", 0.7, 1.0, 0, json.dumps([["EH", 0.0, 0.8]])),
            ])
            conn.executemany("INSERT INTO clip_alignments VALUES (?, 'cached')", [(1,), (2,), (3,)])

        @contextlib.contextmanager
        def get_db():
            conn = sqlite3.connect(self.db)
            conn.row_factory = sqlite3.Row
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

        for patcher in (patch.object(rb, "get_db", get_db),
                        patch("app.database.resolve_path", lambda _path: "test.mp4")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_apply_preserves_edits_and_updates_phone_origin(self):
        with patch.object(rb.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)), \
                patch.object(rb, "_load_wav", return_value=Audio(rb._SR)), \
                patch.object(rb, "_align_window", return_value=[(0.15, 0.3), (0.5, 0.65), (0.75, 0.95)]) as align:
            updated, total, _median = rb.refine_source(1, apply=True)
        self.assertEqual((updated, total), (2, 3))
        self.assertEqual([w["word"] for w in align.call_args.args[1]], ["cat", "dog", "end"])
        with contextlib.closing(sqlite3.connect(self.db)) as conn:
            rows = conn.execute("SELECT id, start_time, end_time, phones FROM word_clips ORDER BY id").fetchall()
            cached = conn.execute("SELECT clip_id FROM clip_alignments").fetchall()
        self.assertEqual(rows[0][:3], (1, 0.15, 0.38))
        self.assertEqual(json.loads(rows[0][3]), [["K", 0.0, 0.05]])
        self.assertEqual(rows[1][:3], (2, 0.4, 0.6))
        self.assertEqual(rows[2][:3], (3, 0.75, 1.0))
        self.assertIsNone(rows[2][3])  # old phone extends beyond the shortened clip
        self.assertEqual(cached, [(2,)])

    def test_decode_error_cleans_temporary_wav_and_does_not_write(self):
        paths = []

        def failed_decode(command, **_kw):
            paths.append(command[-1])
            self.assertTrue(os.path.exists(command[-1]))
            return types.SimpleNamespace(returncode=1, stderr="invalid media")

        with patch.object(rb.subprocess, "run", side_effect=failed_decode), \
                patch.object(rb, "_load_wav") as loader:
            with self.assertRaisesRegex(RuntimeError, "Could not decode source 1: invalid media"):
                rb.refine_source(1, apply=True)
            loader.assert_not_called()
        self.assertEqual(len(paths), 1)
        self.assertFalse(os.path.exists(paths[0]))
        with contextlib.closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT start_time FROM word_clips WHERE id=1").fetchone()[0], 0.1)

    def test_load_error_also_cleans_temporary_wav(self):
        paths = []

        def decoded(command, **_kw):
            paths.append(command[-1])
            return types.SimpleNamespace(returncode=0)

        with patch.object(rb.subprocess, "run", side_effect=decoded), \
                patch.object(rb, "_load_wav", side_effect=ValueError("broken wav")):
            with self.assertRaisesRegex(ValueError, "broken wav"):
                rb.refine_source(1, apply=False)
        self.assertFalse(os.path.exists(paths[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
