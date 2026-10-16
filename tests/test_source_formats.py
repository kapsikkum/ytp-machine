#!/usr/bin/env python3
"""Mixed original/normalized sources render correctly; probes are cached."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import database as db, generate as g


class SourceFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mrs-formats-")
        self.root = Path(self.tmp.name)
        (self.root / "downloads").mkdir()
        self.path = self.root / "downloads" / "source.mp4"
        self.path.touch()
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(db, "_active", {
            "dir": str(self.root), "db": str(self.root / "corpus.db"), "slug": "test"}))
        g._native_video_format.cache_clear()

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()
        g._native_video_format.cache_clear()

    def probe(self, width=480, height=270, rate="25/1"):
        return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [{
            "width": width, "height": height, "r_frame_rate": rate, "avg_frame_rate": rate}]}))

    def test_location_does_not_make_an_original_video_native(self):
        with patch.object(g.subprocess, "run", return_value=self.probe(1920, 1080, "30/1")):
            self.assertFalse(g._is_corpus_native({"source_file": "downloads/source.mp4"}))

    def test_source_versions_are_probed_once_and_changes_reprobe(self):
        with patch.object(g.subprocess, "run", side_effect=[self.probe(), self.probe(640, 480)]) as run:
            self.assertTrue(g._is_corpus_native({"source_file": str(self.path)}))
            self.assertTrue(g._is_corpus_native({"source_file": "downloads/source.mp4"}))
            self.assertEqual(run.call_count, 1)
            self.path.write_bytes(b"new source version")
            self.assertFalse(g._is_corpus_native({"source_file": str(self.path)}))
            self.assertEqual(run.call_count, 2)

    def test_unknown_or_missing_sources_use_normalization(self):
        for response in (SimpleNamespace(returncode=1, stdout=""),
                         SimpleNamespace(returncode=0, stdout="not json"), self.probe(rate="0/0")):
            g._native_video_format.cache_clear()
            with patch.object(g.subprocess, "run", return_value=response):
                self.assertFalse(g._is_corpus_native({"source_file": str(self.path)}))
        self.assertFalse(g._is_corpus_native({"source_file": "downloads/missing.mp4"}))

    def test_display_rotation_requires_normalization(self):
        for metadata in ({"side_data_list": [{"rotation": 90}]},
                         {"tags": {"rotate": "-90"}}):
            response = self.probe()
            data = json.loads(response.stdout)
            data["streams"][0].update(metadata)
            response.stdout = json.dumps(data)
            g._native_video_format.cache_clear()
            with patch.object(g.subprocess, "run", return_value=response):
                self.assertFalse(g._is_corpus_native({"source_file": str(self.path)}))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_real_mixed_source_render(self):
        native = self.root / "downloads" / "native.mp4"
        for path, shape, rate, colour in ((self.path, "320x240", "30", "red"),
                                          (native, "480x270", "25", "blue")):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                            f"color=c={colour}:s={shape}:r={rate}:d=1", "-f", "lavfi", "-i",
                            "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-shortest", str(path)], check=True, capture_output=True)
        self.assertFalse(g._is_corpus_native({"source_file": str(self.path)}))
        self.assertTrue(g._is_corpus_native({"source_file": str(native)}))
        segments = [{"source_file": str(path), "word": "test", "start_time": .1,
                     "end_time": .5, "edited": True} for path in (self.path, native)]
        output = self.root / "mixed.mp4"
        g._encode_chunk(segments, str(output), final_tail=True)
        info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                               "-show_entries", "stream=width,height,r_frame_rate",
                               "-of", "json", str(output)], check=True, capture_output=True, text=True)
        stream = json.loads(info.stdout)["streams"][0]
        self.assertEqual((stream["width"], stream["height"], stream["r_frame_rate"]), (480, 270, "25/1"))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_real_rotated_source_render(self):
        native = self.root / "downloads" / "native.mp4"
        rotated = self.root / "downloads" / "rotated.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=blue:s=480x270:r=25:d=1", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(native)], check=True, capture_output=True)
        rotation = subprocess.run(["ffmpeg", "-y", "-v", "error", "-display_rotation", "90",
                                   "-i", str(native), "-c", "copy", str(rotated)], capture_output=True)
        if rotation.returncode:
            # Older ffmpeg writes the same display matrix via a stream tag.
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(native), "-c", "copy",
                            "-metadata:s:v:0", "rotate=90", str(rotated)], check=True, capture_output=True)
        self.assertFalse(g._is_corpus_native({"source_file": str(rotated)}))
        self.assertTrue(g._is_corpus_native({"source_file": str(native)}))
        segments = [{"source_file": str(path), "word": "test", "start_time": .1,
                     "end_time": .5, "edited": True} for path in (rotated, native)]
        output = self.root / "rotated-mixed.mp4"
        g._encode_chunk(segments, str(output), final_tail=True)
        info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                               "-show_entries", "stream=width,height,r_frame_rate",
                               "-of", "json", str(output)], check=True, capture_output=True, text=True)
        stream = json.loads(info.stdout)["streams"][0]
        self.assertEqual((stream["width"], stream["height"], stream["r_frame_rate"]), (480, 270, "25/1"))


if __name__ == "__main__":
    unittest.main()
