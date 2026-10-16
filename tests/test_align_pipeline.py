#!/usr/bin/env python3
"""Alignment checkpoints and resumability without models or media downloads.

    python tests/test_align_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import database, phone_align
from scripts import align_phones


class AlignmentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="mrs-align-"))
        self.stack.enter_context(patch.object(database, "_active", {
            "slug": "test", "dir": directory, "db": os.path.join(directory, "corpus.db"),
        }))
        self.stack.enter_context(patch.object(database, "_ready", set()))
        database.init_db()

        self.batches = []
        self.fail_write = False
        owner = self

        class Connection:
            def __init__(self, conn):
                self.conn = conn
                self.batch_size = None

            def execute(self, *args):
                return self.conn.execute(*args)

            def executemany(self, sql, rows):
                rows = list(rows)
                if owner.fail_write:
                    raise RuntimeError("checkpoint failed")
                result = self.conn.executemany(sql, rows)
                self.batch_size = len(rows)
                return result

        @contextmanager
        def tracked_db():
            with database.get_db() as conn:
                tracked = Connection(conn)
                yield tracked
            if tracked.batch_size is not None:
                self.batches.append(tracked.batch_size)

        self.stack.enter_context(patch.object(align_phones, "get_db", tracked_db))
        self.stack.enter_context(patch.object(align_phones, "word_to_phonemes", return_value=["K"]))
        self.stack.enter_context(patch.object(phone_align, "available", return_value=True))
        self.aligner = self.stack.enter_context(patch.object(
            phone_align, "align", return_value=[("K", 0.06, 0.16)]))
        self.decoder = self.stack.enter_context(patch.object(
            align_phones, "_decode", side_effect=lambda path: object()))
        self.slicer = self.stack.enter_context(patch.object(
            align_phones, "_slice", return_value=object()))
        self.invalidate = Mock()
        self.stack.enter_context(patch.dict(sys.modules, {
            "app.generate": SimpleNamespace(invalidate_cache=self.invalidate),
        }))

    def seed(self, *counts):
        with database.get_db() as conn:
            for source, count in enumerate(counts, start=1):
                path = f"downloads/source-{source}.mp4"
                conn.execute("INSERT INTO sources (id, video_id, source_file) VALUES (?, ?, ?)",
                             (source, str(source), path))
                conn.executemany(
                    "INSERT INTO word_clips (source_id, word, start_time, end_time, source_file) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [(source, "clip", i + 0.1, i + 0.3, path) for i in range(count)])

    def stored(self):
        with database.get_db() as conn:
            return [r[0] for r in conn.execute("SELECT phones FROM word_clips ORDER BY id")]

    def test_batches_and_final_partial_checkpoint(self):
        self.seed(56, 56)
        self.assertEqual(align_phones.align_corpus(device="cpu"), (112, 0, 0))
        self.assertEqual(self.batches, [50, 50, 12])
        self.assertEqual(self.decoder.call_count, 2)
        self.assertEqual(self.slicer.call_count, 112)
        self.assertEqual(self.aligner.call_args.args[2], "cpu")
        self.assertTrue(all(json.loads(body) == [["K", 0.0, 0.1]] for body in self.stored()))
        self.invalidate.assert_called_once_with(alignments=False)

    def test_interruption_preserves_partial_checkpoint_and_resumes(self):
        self.seed(65)
        interrupted = RuntimeError("model stopped")
        self.aligner.side_effect = [[("K", 0.06, 0.16)]] * 57 + [interrupted]
        with self.assertRaises(RuntimeError) as caught:
            align_phones.align_corpus()
        self.assertIs(caught.exception, interrupted)
        self.assertEqual(self.batches, [50, 7])
        self.assertEqual(sum(body is not None for body in self.stored()), 57)
        self.invalidate.assert_called_once_with(alignments=False)

        self.aligner.side_effect = None
        self.aligner.reset_mock()
        self.invalidate.reset_mock()
        self.assertEqual(align_phones.align_corpus(), (8, 0, 0))
        self.assertEqual(self.aligner.call_count, 8)
        self.assertEqual(self.batches, [50, 7, 8])
        self.assertTrue(all(self.stored()))
        self.invalidate.assert_called_once_with(alignments=False)

    def test_failed_emergency_checkpoint_preserves_original_error(self):
        self.seed(2)
        interrupted = RuntimeError("original alignment error")
        self.aligner.side_effect = [[("K", 0.06, 0.16)], interrupted]
        self.fail_write = True
        with self.assertRaises(RuntimeError) as caught:
            align_phones.align_corpus()
        self.assertIs(caught.exception, interrupted)
        self.assertEqual(self.stored(), [None, None])
        self.invalidate.assert_not_called()

    def test_keyboard_interrupt_preserves_completed_work(self):
        self.seed(3)
        self.aligner.side_effect = [[("K", 0.06, 0.16)], KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            align_phones.align_corpus()
        self.assertEqual(self.batches, [1])
        self.assertEqual(sum(body is not None for body in self.stored()), 1)
        self.invalidate.assert_called_once_with(alignments=False)

    def test_source_decode_exception_flushes_previous_source(self):
        self.seed(3, 2)
        self.decoder.side_effect = [object(), RuntimeError("decode stopped")]
        with self.assertRaisesRegex(RuntimeError, "decode stopped"):
            align_phones.align_corpus()
        self.assertEqual(self.batches, [3])
        self.assertEqual(sum(body is not None for body in self.stored()), 3)
        self.invalidate.assert_called_once_with(alignments=False)

    def test_no_successful_alignment_does_not_invalidate(self):
        self.seed(4)
        self.aligner.return_value = None
        self.assertEqual(align_phones.align_corpus(), (0, 0, 4))
        self.assertEqual(self.batches, [])
        self.invalidate.assert_not_called()

    def test_edit_during_inference_cannot_reinstall_outdated_phones(self):
        self.seed(1)
        for field, value in (("word", "edited"), ("start_time", 0.15),
                             ("end_time", 0.4), ("source_file", "downloads/replaced.mp4")):
            with self.subTest(field=field):
                with database.get_db() as conn:
                    conn.execute("UPDATE word_clips SET word='clip', start_time=0.1, end_time=0.3, "
                                 "source_file='downloads/source-1.mp4', phones=NULL WHERE id=1")

                def edit_during_inference(wav, phones, device):
                    with database.get_db() as conn:
                        conn.execute(f"UPDATE word_clips SET {field}=? WHERE id=1", (value,))
                    return [("K", 0.06, 0.16)]

                self.aligner.side_effect = edit_during_inference
                self.assertEqual(align_phones.align_corpus(), (0, 1, 0))
                self.assertEqual(self.stored(), [None])
                with database.get_db() as conn:
                    self.assertEqual(conn.execute(f"SELECT {field} FROM word_clips WHERE id=1").fetchone()[0], value)
                self.invalidate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
