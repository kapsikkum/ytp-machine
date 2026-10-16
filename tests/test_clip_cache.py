#!/usr/bin/env python3
"""Rendered caches must match the recording and exact extraction window.

    python tests/test_clip_cache.py
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import clip_cache as cache, database as db


class ClipCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="mrs-render-cache-"))
        self.corpora = {}
        self.sources = {}
        for slug in ("alpha", "beta"):
            root = Path(directory) / slug
            source = root / "downloads" / "same.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(slug.encode())
            self.sources[slug] = str(source)
            self.corpora[slug] = {"slug": slug, "dir": str(root), "db": str(root / "corpus.db")}
            with patch.object(db, "_active", self.corpora[slug]), patch.object(db, "_ready", set()):
                db.init_db()
        self.stack.enter_context(patch.object(db, "_active", self.corpora["alpha"]))
        self.commands = []

    def fake_ffmpeg(self, command, **kwargs):
        self.commands.append(command)
        Path(command[-1]).write_bytes(b"rendered audio/video")
        return SimpleNamespace(returncode=0, stderr="")

    def render(self, source=None, *, clip_id=1, start=1.0, duration=0.5, pad=0.18):
        with patch.object(cache.subprocess, "run", side_effect=self.fake_ffmpeg):
            return cache.cache_clip(clip_id, pad, source or self.sources["alpha"], start, duration)

    def stored(self):
        with db.get_db() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM clip_cache")]

    def test_local_relative_and_absolute_source_share_exact_window(self):
        path = self.render()
        self.assertIsNotNone(path)
        self.assertEqual(cache.get_cached(1, 0.18, "downloads/same.mp4", 1.0, 0.5), path)
        self.assertEqual(cache.get_cached(1, 0.18, self.sources["alpha"], 1.0000001, 0.5000001), path)
        self.assertIsNone(cache.get_cached(1, 0.18, self.sources["alpha"], 1.1, 0.5))
        self.assertIsNone(cache.get_cached(1, 0.18, self.sources["alpha"], 1.0, 0.7))
        command = self.commands[0]
        self.assertEqual(command[command.index("-ss") + 1], "1.0000")
        self.assertEqual(command[command.index("-t") + 1], "0.5000")
        self.assertEqual(len(self.stored()), 1)

    def test_borrowed_voice_with_same_id_cannot_use_active_voice_audio(self):
        local = self.render()
        self.assertIsNone(cache.get_cached(1, 0.18, self.sources["beta"], 1.0, 0.5))
        borrowed = self.render(self.sources["beta"])
        self.assertNotEqual(local, borrowed)
        self.assertEqual(cache.get_cached(1, 0.18, self.sources["beta"], 1.0, 0.5), borrowed)
        self.assertIsNone(cache.get_cached(1, 0.18, self.sources["alpha"], 1.0, 0.5))
        self.assertFalse(Path(local).exists())

    def test_changed_window_replaces_tracked_file_without_orphaning(self):
        previous = self.render()
        current = self.render(start=1.2, duration=0.2)
        self.assertNotEqual(previous, current)
        self.assertFalse(Path(previous).exists())
        self.assertEqual(self.stored()[0]["cache_file"], Path(current).name)
        self.assertEqual(len(list(Path(current).parent.glob("*.ts"))), 1)

    def test_source_size_and_mtime_replacements_miss_old_cache(self):
        previous = self.render()
        source = Path(self.sources["alpha"])
        source.write_bytes(b"new source, different size")
        self.assertIsNone(cache.get_cached(1, 0.18, str(source), 1.0, 0.5))
        current = self.render()
        self.assertNotEqual(previous, current)
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertIsNone(cache.get_cached(1, 0.18, str(source), 1.0, 0.5))

    def test_legacy_id_only_cache_is_never_reused(self):
        legacy = Path(cache._cache_dir()) / "1_0.180.ts"
        legacy.write_bytes(b"old unrelated audio")
        self.assertIsNone(cache.get_cached(1, 0.18, self.sources["alpha"], 1.0, 0.5))

    def test_lookup_does_not_create_cache_directory(self):
        self.assertIsNone(cache.get_cached(1, 0.18, self.sources["alpha"], 1.0, 0.5))
        self.assertFalse(Path(cache._cache_dir(create=False)).exists())
        self.assertIsNone(cache.get_cached(1, 0.18, "../outside.mp4", 1.0, 0.5))

    def test_source_replaced_during_render_does_not_install_stale_audio(self):
        def replace_during_render(command, **kwargs):
            self.fake_ffmpeg(command, **kwargs)
            Path(self.sources["alpha"]).write_bytes(b"replacement source")
            return SimpleNamespace(returncode=0, stderr="")

        with patch.object(cache.subprocess, "run", side_effect=replace_during_render):
            self.assertIsNone(cache.cache_clip(1, 0.18, self.sources["alpha"], 1.0, 0.5))
        self.assertEqual(list(Path(cache._cache_dir()).iterdir()), [])
        self.assertEqual(self.stored(), [])

    def test_failed_render_removes_temporary_file(self):
        def fail(command, **kwargs):
            Path(command[-1]).write_bytes(b"partial")
            return SimpleNamespace(returncode=1, stderr="fake ffmpeg failed")

        with patch.object(cache.subprocess, "run", side_effect=fail), self.assertLogs(cache.log, level="WARNING"):
            self.assertIsNone(cache.cache_clip(1, 0.18, self.sources["alpha"], 1.0, 0.5))
        self.assertEqual(list(Path(cache._cache_dir()).iterdir()), [])
        self.assertEqual(self.stored(), [])

    def test_invalidation_removes_untracked_legacy_and_temporary_variants(self):
        first = self.render()
        second = self.render(clip_id=2)
        directory = Path(cache._cache_dir())
        legacy = directory / "1_0.450.ts"
        orphan = directory / ("1_0.180_" + "a" * 24 + ".ts")
        temporary = Path(str(orphan) + ".tmp")
        unrelated = directory / "keep.txt"
        for path in (legacy, orphan, temporary, unrelated):
            path.write_bytes(b"keep only the unrelated file")
        cache.invalidate(1)
        self.assertFalse(any(path.exists() for path in (Path(first), legacy, orphan, temporary)))
        self.assertTrue(Path(second).exists())
        self.assertTrue(unrelated.exists())
        self.assertEqual([row["clip_id"] for row in self.stored()], [2])
        cache.invalidate()
        self.assertFalse(Path(second).exists())
        self.assertTrue(unrelated.exists())
        self.assertEqual(self.stored(), [])

    def test_evicting_old_fingerprint_preserves_new_tracked_variant(self):
        current = Path(self.render())
        current.write_bytes(b"x" * (512 * 1024))
        stale = current.parent / ("1_0.180_" + "b" * 24 + ".ts")
        stale.write_bytes(b"x" * (2 * 1024 * 1024))
        os.utime(stale, (1, 1))
        self.assertEqual(cache.evict_lru(1), 2 * 1024 * 1024)
        self.assertFalse(stale.exists())
        self.assertTrue(current.exists())
        self.assertEqual(self.stored()[0]["cache_file"], current.name)
        self.assertEqual(cache.evict_lru(0), 512 * 1024)
        self.assertEqual(self.stored(), [])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not installed")
    def test_real_ffmpeg_cache_is_readable_and_has_requested_audio(self):
        source = self.sources["alpha"]
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=size=480x270:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "0.8",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", source,
        ], check=True, capture_output=True)
        path = cache.cache_clip(1, 0.18, source, 0.1, 0.2)
        self.assertIsNotNone(path)
        self.assertEqual(cache.get_cached(1, 0.18, source, 0.1, 0.2), path)
        audio = subprocess.run([
            "ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
            "-f", "s16le", "-",
        ], check=True, capture_output=True).stdout
        self.assertGreater(len(audio) / 2 / 16000, 0.15)
        self.assertLess(len(audio) / 2 / 16000, 0.35)


if __name__ == "__main__":
    unittest.main(verbosity=2)
