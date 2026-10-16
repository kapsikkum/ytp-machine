#!/usr/bin/env python3
"""Caption parsing and reference filenames remain accurate and confined."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import database as db
from scripts import import_transcripts as transcripts, verify_corpus as verify


class CorpusToolsTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="mrs-tools-")))
        self.stack.enter_context(patch.object(db, "_active", {
            "slug": "test", "dir": str(self.root), "db": str(self.root / "corpus.db")}))
        self.stack.enter_context(patch.object(db, "_ready", set()))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        db.init_db()

    def test_source_ids_cannot_escape_reference_or_caption_directories(self):
        for identifier in ("../../outside", "/absolute", "C:\\secret", "a:b", "CON", "..", "x " * 200):
            with self.subTest(identifier=identifier):
                name = db.artifact_filename(identifier, ".txt")
                self.assertEqual(Path(name).name, name)
                self.assertNotIn("\\", name)
                self.assertEqual(Path(verify._cache_path(identifier)).parent, self.root / "captions")
        self.assertEqual(db.artifact_filename("Akwm2UZJ34o", ".txt"), "Akwm2UZJ34o.txt")

    def test_transcript_import_of_malicious_id_writes_only_inside_output(self):
        with db.get_db() as conn:
            conn.execute("INSERT INTO sources(video_id,source_file,title) VALUES ('../../outside','downloads/a.mp4','Hot Food')")
        dump = self.root / "wiki.json"
        dump.write_text(json.dumps({"parse": {"wikitext": "==Hot Food==\nreference"}}), encoding="utf-8")
        output = self.root / "transcripts"
        with patch.object(sys, "argv", ["import", "--json", str(dump), "--write"]):
            transcripts.main()
        files = list(output.iterdir())
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_text(), "reference")
        self.assertTrue(files[0].name.startswith("id-"))
        self.assertFalse((self.root.parent / "outside.txt").exists())

    def test_caption_segments_keep_separate_words_and_shared_coarse_time(self):
        words = verify._caption_words({"events": [
            {"tStartMs": 1000, "segs": [{"utf8": "It's hot food!", "tOffsetMs": 200}]},
            {"tStartMs": "bad", "segs": [{"utf8": "ignore"}]},
            {"tStartMs": float("inf"), "segs": [{"utf8": "ignore"}]},
            {"tStartMs": -1, "segs": [{"utf8": "ignore"}]},
            {"tStartMs": 0, "segs": [{"utf8": "..."}, {"utf8": None}]}]})
        self.assertEqual(words, [("its", 1.2), ("hot", 1.2), ("food", 1.2)])
        report = verify.compare(words, [("its", 1.2, 1.4), ("hot", 1.4, 1.6), ("food", 1.6, 1.8)])
        self.assertEqual(report["agreed"], 3)

    def test_broken_caption_cache_is_refetched_and_saved_atomically(self):
        cache = Path(verify._cache_path("video"))
        cache.write_text("{interrupted")
        extractor = Mock()
        extractor.__enter__ = Mock(return_value=extractor)
        extractor.__exit__ = Mock(return_value=False)
        extractor.extract_info.return_value = {"subtitles": {"en": [{"ext": "json3", "url": "https://example.invalid/captions"}]}}
        response = io.BytesIO(json.dumps({"events": [{"tStartMs": 0, "segs": [{"utf8": "one two"}]}]}).encode())
        with patch.dict(sys.modules, {"yt_dlp": SimpleNamespace(YoutubeDL=lambda opts: extractor)}), \
                patch.object(verify.urllib.request, "urlopen", return_value=response):
            self.assertEqual(verify.fetch_captions("video"), [("one", 0.), ("two", 0.)])
        self.assertEqual(json.loads(cache.read_text()), [["one", 0.], ["two", 0.]])
        self.assertFalse(list(cache.parent.glob(".captions-*")))
        with patch.dict(sys.modules, {"yt_dlp": SimpleNamespace(YoutubeDL=Mock(side_effect=AssertionError("no download")))}):
            self.assertEqual(verify.fetch_captions("video"), [("one", 0.), ("two", 0.)])

    def test_empty_network_exception_has_a_useful_failure(self):
        extractor = Mock()
        extractor.__enter__ = Mock(return_value=extractor)
        extractor.__exit__ = Mock(return_value=False)
        extractor.extract_info.side_effect = RuntimeError()
        with patch.dict(sys.modules, {"yt_dlp": SimpleNamespace(YoutubeDL=lambda opts: extractor)}):
            self.assertIsNone(verify.fetch_captions("video"))


if __name__ == "__main__":
    unittest.main()
