#!/usr/bin/env python3
"""Channel resume and failure handling, without downloads or model loading.

    python tests/test_ingest_channel.py

The database belongs to a temporary corpus. yt-dlp listings and transcription
are mocked, so these checks also run on the bare interpreter used by CI.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import database as db
from scripts import ingest_channel as channel


def video(vid: str) -> dict:
    return {"id": vid, "title": f"Video {vid}", "upload_date": "",
            "url": f"https://www.youtube.com/watch?v={vid}"}


class ChannelIngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mrs-channel-test-")
        entry = {"slug": "test", "name": "Test", "dir": self.tmp.name,
                 "db": os.path.join(self.tmp.name, "corpus.db")}
        self.active_patch = patch.object(db, "_active", entry)
        self.ready_patch = patch.object(db, "_ready", set())
        self.active_patch.start()
        self.ready_patch.start()
        self.aligner = MagicMock(return_value=(1, 0, 0))
        fake_align = types.ModuleType("scripts.align_phones")
        fake_align.align_corpus = self.aligner
        self.align_patch = patch.dict(sys.modules, {"scripts.align_phones": fake_align})
        self.align_patch.start()
        db.init_db()

    def tearDown(self):
        self.align_patch.stop()
        self.ready_patch.stop()
        self.active_patch.stop()
        self.tmp.cleanup()

    def remember(self, vid: str, *, with_clip=True, phones="[]") -> int:
        with db.get_db() as conn:
            cur = conn.execute(
                "INSERT INTO sources (video_id, source_file, title) VALUES (?, ?, ?)",
                (vid, f"downloads/{vid}.mp4", f"Video {vid}"))
            source_id = cur.lastrowid
            if with_clip:
                conn.execute(
                    "INSERT INTO word_clips "
                    "(source_id, word, start_time, end_time, source_file, phones) "
                    "VALUES (?, 'hello', 0, 0.4, ?, ?)",
                    (source_id, f"downloads/{vid}.mp4", phones))
        return source_id

    def run_channel(self, videos: list[dict], args=(), side_effect=None):
        output = io.StringIO()
        argv = ["ingest_channel.py", "https://www.youtube.com/@Test/videos", *args]
        with patch.object(channel, "list_channel_videos", return_value=videos), \
             patch.object(channel, "ingest", side_effect=side_effect) as ingest, \
             patch.object(sys, "argv", argv), redirect_stdout(output):
            channel.main()
        return ingest, output.getvalue()

    def test_alignment_can_be_deferred(self):
        for args, expected in (((), True), (("--no-align",), False)):
            with self.subTest(args=args):
                ingest, _ = self.run_channel([video("new")], args)
                self.assertEqual(ingest.call_args.kwargs["align"], expected)

    def test_transcription_options_reach_each_video(self):
        ingest, _ = self.run_channel(
            [video("first"), video("second")],
            ("--model", "medium.en", "--language", "en", "--initial-prompt", "Morshu"))
        self.assertEqual(ingest.call_count, 2)
        for call in ingest.call_args_list:
            self.assertEqual(call.kwargs["model_name"], "medium.en")
            self.assertEqual(call.kwargs["language"], "en")
            self.assertEqual(call.kwargs["initial_prompt"], "Morshu")

    def test_finished_channel_is_a_successful_noop(self):
        self.remember("old")
        ingest, output = self.run_channel([video("old")], ("--limit", "3"))
        ingest.assert_not_called()
        self.assertIn("All matching videos are already", output)
        self.aligner.assert_not_called()

    def test_legacy_empty_source_is_retried(self):
        self.remember("empty", with_clip=False)
        ingest, _ = self.run_channel([video("empty")])
        self.assertEqual(ingest.call_count, 1)
        self.aligner.assert_not_called()

    def test_legacy_invalid_source_is_retried(self):
        source_id = self.remember("invalid")
        with db.get_db() as conn:
            conn.execute("UPDATE word_clips SET end_time=? WHERE source_id=?",
                         (float("inf"), source_id))
        ingest, _ = self.run_channel([video("invalid")])
        self.assertEqual(ingest.call_count, 1)
        self.aligner.assert_not_called()

    def test_committed_partial_alignment_is_resumed_without_transcription(self):
        source_id = self.remember("partial", phones=None)
        ingest, _ = self.run_channel([video("partial")])
        ingest.assert_not_called()
        self.aligner.assert_called_once_with(source_id=source_id, device=None)

    def test_no_align_preserves_partial_alignment_for_later(self):
        self.remember("partial", phones=None)
        ingest, _ = self.run_channel([video("partial")], ("--no-align",))
        ingest.assert_not_called()
        self.aligner.assert_not_called()

    def test_resume_alignment_error_fails_the_run(self):
        self.remember("partial", phones=None)
        self.aligner.side_effect = RuntimeError("Model unavailable")
        with self.assertRaises(SystemExit) as raised:
            self.run_channel([video("partial")], ("--skip-errors",))
        self.assertEqual(raised.exception.code, 1)

    def test_resume_alignment_failed_clips_fail_the_run(self):
        self.remember("partial", phones=None)
        self.aligner.return_value = (0, 0, 1)
        with self.assertRaises(SystemExit) as raised:
            self.run_channel([video("partial")], ("--skip-errors",))
        self.assertEqual(raised.exception.code, 1)

    def test_resume_error_allows_new_ingests_but_reports_failure(self):
        self.remember("partial", phones=None)
        self.aligner.side_effect = RuntimeError("Model unavailable")
        attempted = []
        with self.assertRaises(SystemExit) as raised:
            self.run_channel([video("partial"), video("new")], ("--skip-errors",),
                             side_effect=lambda *a, **kw: attempted.append(a[0]))
        self.assertEqual(attempted, [video("new")["url"]])
        self.assertEqual(raised.exception.code, 1)

    def test_existing_sources_do_not_consume_limit(self):
        self.remember("old")
        ingest, _ = self.run_channel(
            [video("old"), video("new"), video("reserve")], ("--limit", "1"))
        self.assertEqual(ingest.call_count, 1)
        self.assertEqual(ingest.call_args.args[0], video("new")["url"])

    def test_failed_candidates_do_not_consume_limit(self):
        ingest, output = self.run_channel(
            [video("private"), video("good"), video("reserve")],
            ("--limit", "1", "--skip-errors"),
            side_effect=[RuntimeError("Members only"), None])
        self.assertEqual(ingest.call_count, 2)
        self.assertEqual(ingest.call_args.args[0], video("good")["url"])
        self.assertIn("Done: 1/1", output)

    def test_all_failed_ingests_fail_the_stage(self):
        # A bare RuntimeError has no text, and must still be reportable.
        with self.assertRaises(SystemExit) as raised:
            self.run_channel([video("private")], ("--skip-errors",),
                             side_effect=RuntimeError())
        self.assertEqual(raised.exception.code, 1)

    def test_reingest_includes_existing_sources(self):
        self.remember("old")
        ingest, _ = self.run_channel([video("old")], ("--reingest", "--no-align"))
        self.assertEqual(ingest.call_count, 1)
        self.assertFalse(ingest.call_args.kwargs["align"])

    def listing(self, entries: list[dict], year=None):
        fake = types.ModuleType("yt_dlp")
        fake.YoutubeDL = MagicMock()
        fake.YoutubeDL.return_value.__enter__.return_value.extract_info.return_value = {
            "entries": entries}
        with patch.dict(sys.modules, {"yt_dlp": fake}), redirect_stdout(io.StringIO()):
            return channel.list_channel_videos("https://www.youtube.com/@Test/videos", year)

    def test_duplicate_listing_entries_are_ingested_once(self):
        result = self.listing([
            {"id": "first", "title": "First", "duration": 42},
            {"id": "first", "title": "Duplicate"},
            {"id": "second", "title": "Second"},
        ])
        self.assertEqual([v["id"] for v in result], ["first", "second"])
        self.assertEqual(result[0]["title"], "First")
        self.assertEqual(result[0]["duration"], 42)

    def test_duplicate_listing_dates_are_resolved_once(self):
        def dated(vid, cookies):
            return {**video(vid), "upload_date": "20260101"}

        with patch.object(channel, "_fetch_date", side_effect=dated) as fetch, \
             patch("time.sleep"):
            result = self.listing([{"id": "first"}, {"id": "first"}], year=2026)
        self.assertEqual(len(result), 1)
        fetch.assert_called_once_with("first", None)


if __name__ == "__main__":
    unittest.main()
