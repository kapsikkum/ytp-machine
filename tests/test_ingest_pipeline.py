#!/usr/bin/env python3
"""Transcription validation and build resumption without a model or network."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import database as db
from scripts import build_corpus as build, ingest as pipeline, realign


class IngestPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mrs-ingest-test-")
        self.corpus = Path(self.tmp.name) / "corpora" / "test"
        self.corpus.mkdir(parents=True)
        (self.corpus / "downloads").mkdir()
        entry = {"slug": "test", "name": "Test", "dir": str(self.corpus),
                 "db": str(self.corpus / "corpus.db")}
        self.active = patch.object(db, "_active", entry)
        self.ready = patch.object(db, "_ready", set())
        self.active.start()
        self.ready.start()
        db.init_db()
        self.source = self.corpus / "downloads" / "video.mp4"
        self.source.touch()

    def tearDown(self):
        self.ready.stop()
        self.active.stop()
        self.tmp.cleanup()

    def words(self):
        return [{"word": "hello", "start": 0.0, "end": 0.4}]

    def transcribe_fake(self, timestamps, **kwargs):
        model = SimpleNamespace(transcribe=lambda *a, **k: SimpleNamespace(
            segments=[SimpleNamespace(words=[SimpleNamespace(word=w, start=s, end=e)
                                             for w, s, e in timestamps])]))
        with patch.object(pipeline, "_load_transcriber", return_value=model), \
             patch("app.device.get", return_value="cpu"), redirect_stdout(io.StringIO()):
            return pipeline.transcribe("video.mp4", **kwargs)

    def test_only_positive_finite_timestamps_are_stored(self):
        words = self.transcribe_fake([
            ("Hello!", 0, 0.4), ("zero", 1, 1), ("back", 2, 1),
            ("negative", -1, 0), ("nan", float("nan"), 1),
            ("infinity", 0, float("inf")), ("missing", None, 1), ("...", 3, 4)])
        self.assertEqual(words, self.words())

    def test_language_and_vocabulary_reach_the_model(self):
        from unittest.mock import MagicMock
        model = MagicMock()
        model.transcribe.return_value.segments = []
        with patch.object(pipeline, "_load_transcriber", return_value=model), \
             patch("app.device.get", return_value="cpu"), redirect_stdout(io.StringIO()):
            pipeline.transcribe("video.mp4", "turbo", language="en", initial_prompt="Morshu")
        model.transcribe.assert_called_once_with(
            "video.mp4", word_timestamps=True, language="en", initial_prompt="Morshu")

    def test_empty_transcription_does_not_mark_a_source_as_ingested(self):
        with self.assertRaisesRegex(ValueError, "No usable word"):
            pipeline.persist("video", str(self.source), "Video", None, [])
        with db.get_db() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM sources").fetchone()[0], 0)

    def test_unportable_source_is_rejected_even_if_names_collide(self):
        outside = Path(self.tmp.name) / "video.mp4"
        outside.touch()
        with self.assertRaisesRegex(ValueError, "downloads/"):
            pipeline.persist("video", str(outside), "Video", None, self.words())

    def test_outside_download_directory_fails_before_expensive_work(self):
        with patch.object(pipeline, "download_video") as download, \
             patch.object(pipeline, "transcribe") as transcribe, \
             self.assertRaisesRegex(ValueError, "MRS_DATA_DIR"):
            pipeline.ingest("https://www.youtube.com/watch?v=Akwm2UZJ34o",
                            download_dir=str(Path(self.tmp.name) / "elsewhere"))
        download.assert_not_called()
        transcribe.assert_not_called()

    def test_local_filename_collision_cannot_overwrite_recorded_audio(self):
        self.source.write_bytes(b"existing recording")
        original = Path(self.tmp.name) / "video.mp4"
        original.write_bytes(b"different recording")
        pipeline.persist("video", str(self.source), "Video", None, self.words())
        with patch.object(pipeline, "transcribe") as transcribe, \
             self.assertRaisesRegex(FileExistsError, "not replaced"):
            pipeline.ingest(str(original), align=False)
        transcribe.assert_not_called()
        self.assertEqual(self.source.read_bytes(), b"existing recording")
        self.assertEqual(original.read_bytes(), b"different recording")

    def test_reingest_aligns_the_new_source(self):
        pipeline.persist("video", str(self.source), "Video", None, self.words())
        with patch.object(pipeline, "transcribe", return_value=self.words()), \
             patch("scripts.align_phones.align_corpus") as align, \
             redirect_stdout(io.StringIO()):
            # ingest currently imports the script by its top-level name.
            with patch.dict(sys.modules, {"align_phones": sys.modules["scripts.align_phones"]}):
                pipeline.ingest(str(self.source))
        self.assertEqual(align.call_args.kwargs["source_id"], 2)
        with db.get_db() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM sources").fetchone()[0], 2)

    def test_empty_retranscription_preserves_existing_clips(self):
        pipeline.persist("video", str(self.source), "Video", None, self.words())
        with db.get_db() as conn:
            source = dict(conn.execute("SELECT * FROM sources").fetchone())
        with patch.object(realign, "transcribe", return_value=[]), \
             redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            realign.realign_source(source, "medium")
        with db.get_db() as conn:
            self.assertEqual(conn.execute("SELECT word FROM word_clips").fetchone()[0], "hello")

    def run_build(self, extra=(), channel="@Test"):
        argv = ["build_corpus.py", channel, "--name", "test", "--data-dir", self.tmp.name,
                "--skip-refine", "--skip-align", "--skip-pack", *extra]
        with patch.object(sys, "argv", argv), patch.object(build, "_preflight"), \
             patch.object(build, "_run") as run, redirect_stdout(io.StringIO()):
            self.assertEqual(build.main(), 0)
        return run

    def test_resuming_build_uses_remaining_total_budget(self):
        pipeline.persist("video", str(self.source), "Video", None, self.words())
        run = self.run_build(("--limit", "3", "--language", "en", "--initial-prompt", "Morshu"))
        stage = run.call_args_list[0].args[1]
        self.assertEqual(stage[stage.index("--limit") + 1], "2")
        self.assertIn("--no-align", stage)
        self.assertEqual(stage[stage.index("--language") + 1], "en")
        self.assertEqual(stage[stage.index("--initial-prompt") + 1], "Morshu")

    def test_completed_build_skips_ingestion(self):
        pipeline.persist("video", str(self.source), "Video", None, self.words())
        run = self.run_build(("--limit", "1"))
        self.assertFalse(any("Download and transcribe" in call.args[0]
                             for call in run.call_args_list))

    def test_legacy_empty_source_does_not_consume_build_budget(self):
        with db.get_db() as conn:
            conn.execute("INSERT INTO sources (video_id, source_file) "
                         "VALUES ('empty', 'downloads/empty.mp4')")
        self.assertEqual(build._known_video_ids(str(self.corpus / "corpus.db")), set())
        run = self.run_build(("--limit", "1"))
        stage = run.call_args_list[0].args[1]
        self.assertEqual(stage[stage.index("--limit") + 1], "1")

    def test_single_video_build_resumes_without_duplicate_clips(self):
        pipeline.persist("Akwm2UZJ34o", str(self.source), "Video", None, self.words())
        run = self.run_build(channel="youtu.be/Akwm2UZJ34o")
        self.assertFalse(any("Download and transcribe" in call.args[0]
                             for call in run.call_args_list))

    def test_unlimited_build_still_checks_for_new_videos(self):
        pipeline.persist("video", str(self.source), "Video", None, self.words())
        run = self.run_build(("--limit", "0"))
        stage = run.call_args_list[0].args[1]
        self.assertIn("scripts/ingest_channel.py", stage)
        self.assertNotIn("--limit", stage)

    def test_bare_short_url_resolves_to_a_single_video(self):
        self.assertEqual(build._channel_url("youtu.be/Akwm2UZJ34o"),
                         "https://www.youtube.com/watch?v=Akwm2UZJ34o")

    def test_channel_share_query_stays_after_the_videos_path(self):
        self.assertEqual(build._channel_url("@Test?si=abc#featured"),
                         "https://www.youtube.com/@Test/videos?si=abc#featured")
        self.assertEqual(build._channel_url("https://youtube.com/@Test/videos?si=abc"),
                         "https://youtube.com/@Test/videos?si=abc")

    def test_single_video_query_can_put_id_after_other_parameters(self):
        self.assertEqual(build._video_id("https://youtube.com/watch?si=abc&v=Akwm2UZJ34o"),
                         "Akwm2UZJ34o")

    def test_pronunciation_template_is_written_for_the_bundled_voice(self):
        run = self.run_build(("--limit", "0"))
        template = next(call.args[1] for call in run.call_args_list
                        if "scripts/verify_corpus.py" in call.args[1])
        self.assertIn("--per-corpus", template)

    def test_preflight_skipped_stages_need_no_models_or_gpu(self):
        with patch("importlib.util.find_spec") as find_spec, \
             patch.object(build.shutil, "which") as which, \
             patch("app.device.resolve") as device:
            build._preflight("cuda", ingest=False, refine=False, align=False,
                             pack=False, verify=False)
        find_spec.assert_not_called()
        which.assert_not_called()
        device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
