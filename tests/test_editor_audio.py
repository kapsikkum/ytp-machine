#!/usr/bin/env python3
"""Bounded measured spectrograms and recoverable alignment reservations."""
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import database as db, editor, jobs


class EditorAudioTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="mrs-editor-audio-")))
        self.corpus = {"slug": "audio-test", "dir": str(self.directory),
                       "db": str(self.directory / "corpus.db")}
        self.stack.enter_context(patch.object(db, "_active", self.corpus))
        self.stack.enter_context(patch.object(db, "_ready", set()))
        self.stack.enter_context(patch.object(editor, "_envelopes", editor.OrderedDict()))
        self.stack.enter_context(patch.object(editor, "_align_state", {
            "running": False, "source_id": None, "corpus": None, "done": 0,
            "total": 0, "result": None, "error": None}))
        self.stack.enter_context(patch.object(editor, "_align_ready", return_value=(True, "")))
        self.stack.enter_context(patch.object(jobs, "maintenance_in_progress", return_value=False))
        db.init_db()
        self.source = self.directory / "downloads" / "tone.wav"
        self.source.parent.mkdir()
        self.source.write_bytes(b"source fixture")
        with db.get_db() as conn:
            conn.execute("INSERT INTO sources (id,video_id,source_file,title) VALUES (1,'test',?,'Tone')",
                         ("downloads/tone.wav",))

    def test_measured_frequency_and_silence(self):
        t = np.arange(16000) / 16000
        tone = editor._spectral_bins((0.8 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
        matrix = np.asarray(tone["values"])
        self.assertEqual(matrix.shape[1], 64)
        dominant = matrix[2:-2].mean(axis=0).argmax() * tone["max_hz"] / tone["bins"]
        self.assertLessEqual(abs(dominant - 1000), 125)
        self.assertGreater(matrix.max(), 200)
        self.assertEqual(np.asarray(editor._spectral_bins(np.zeros(16000))["values"]).max(), 0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_real_ffmpeg_decode_of_tone_and_silence(self):
        t = np.arange(16000) / 16000
        samples = np.concatenate([0.8 * np.sin(2 * np.pi * 2000 * t), np.zeros(16000)])
        with wave.open(str(self.source), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes((samples * 32767).astype("<i2").tobytes())
        measured = editor.envelope(1, 0, 1, spectrogram=True)
        dominant = np.asarray(measured["spectrogram"]["values"])[2:-2].mean(axis=0).argmax() * 125
        self.assertLessEqual(abs(dominant - 2000), 125)
        silent = editor.envelope(1, 1, 2, spectrogram=True)
        self.assertEqual(silent["peak"], 0)
        self.assertEqual(np.asarray(silent["spectrogram"]["values"]).max(), 0)

    def test_window_payload_cache_and_source_replacement(self):
        samples = np.zeros(960000, dtype=np.float32)
        with patch("app.ytpmv.pitch.decode_audio", return_value=samples) as decode:
            plain = editor.envelope(1, 0, 3600, buckets=100000)
            self.assertNotIn("spectrogram", plain)
            self.assertLessEqual(len(plain["values"]), 4000)
            self.assertEqual(plain["end"], 60)
            self.assertTrue(os.path.samefile(decode.call_args.args[0], self.source))
            self.assertEqual(decode.call_args.args[1:], (0, 60.0, 16000, "s16le"))
            measured = editor.envelope(1, 0, 3600, buckets=100000, spectrogram=True)
            self.assertEqual(np.asarray(measured["spectrogram"]["values"]).shape, (300, 64))
            self.assertIs(editor.envelope(1, 0, 3600, buckets=100000, spectrogram=True), measured)
            self.assertEqual(decode.call_count, 2)
            self.source.write_bytes(b"replacement with different file size")
            editor.envelope(1, 0, 3600, buckets=100000, spectrogram=True)
            self.assertEqual(decode.call_count, 3)
        with patch("app.ytpmv.pitch.decode_audio", return_value=np.zeros(100)):
            for start in range(30):
                editor.envelope(1, start, start + 1)
        self.assertEqual(len(editor._envelopes), 24)

    def test_invalid_window_does_not_decode(self):
        for a, b in [(-1, 1), (1, 1), (2, 1), (float("nan"), 1), (0, float("inf"))]:
            with self.subTest(a=a, b=b), patch("app.ytpmv.pitch.decode_audio") as decode:
                with self.assertRaises(HTTPException) as err:
                    editor.envelope(1, a, b)
                self.assertEqual(err.exception.status_code, 400)
                decode.assert_not_called()

    def test_missing_source_and_maintenance_never_reserve(self):
        with self.assertRaises(HTTPException) as err:
            editor.align_source(999)
        self.assertEqual(err.exception.status_code, 404)
        self.assertFalse(editor.alignment_running())
        with patch.object(jobs, "maintenance_in_progress", return_value=True):
            with self.assertRaises(HTTPException) as err:
                editor.align_source(1)
        self.assertEqual(err.exception.status_code, 409)
        self.assertFalse(editor.alignment_running())

    def test_start_failure_and_setup_failure_release_reservation(self):
        with patch.object(editor.threading, "Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("cannot start")
            with self.assertRaisesRegex(RuntimeError, "cannot start"):
                editor.align_source(1)
        self.assertFalse(editor.alignment_running())
        with patch.object(editor, "get_db", side_effect=RuntimeError("cannot query")), \
                patch.object(editor, "_source_path", return_value=str(self.source)):
            with self.assertRaisesRegex(RuntimeError, "cannot query"):
                editor.align_source(1)
        self.assertFalse(editor.alignment_running())

    def test_worker_pins_corpus_and_status_is_scoped(self):
        captured = []
        thread = Mock()
        def create_thread(**kwargs):
            captured.append(kwargs["target"])
            return thread
        with patch.object(editor.threading, "Thread", side_effect=create_thread):
            started = editor.align_source(1)
        self.assertEqual(started["corpus"], "audio-test")
        self.assertTrue(editor.alignment_running())
        other = {**self.corpus, "slug": "other"}
        seen = []
        def align(**kwargs):
            seen.append(db.active()["slug"])
            return 3, 1, 0
        with patch.object(db, "_active", other), \
                patch.dict(sys.modules, {"scripts.align_phones": SimpleNamespace(align_corpus=align)}):
            st = editor.align_status(1)
            self.assertFalse(st["running"])
            self.assertIsNone(st["result"])
            captured[0]()
            self.assertEqual(db.active()["slug"], "other")
        self.assertEqual(seen, ["audio-test"])
        self.assertFalse(editor.alignment_running())
        self.assertEqual(editor.align_status(1)["result"]["aligned"], 3)


if __name__ == "__main__":
    unittest.main()
