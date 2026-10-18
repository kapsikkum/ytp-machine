"""Media cancellation, cleanup and cache budgets without corpus/model writes."""

import os
import shutil
import contextlib
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import generate as g, media_process
from app.ytpmv import render, samples
from app.ytpmv.music import Note, Part, Song
from app.ytpmv import pitch
from app.ytpmv import recommend


class MediaResourcesTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required for real framing checks')
    def test_fit_preserves_whole_picture_cover_crops_and_cache_keeps_them_separate(self):
        samples.clear_cache()
        self.addCleanup(samples.clear_cache)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory,'edges.mp4')
            subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i',
                'color=red:s=160x90:r=25:d=0.3,drawbox=x=0:y=0:w=8:h=ih:color=blue:t=fill,drawbox=x=iw-8:y=0:w=8:h=ih:color=green:t=fill',
                '-c:v','libx264','-pix_fmt','yuv420p',str(source)],check=True,capture_output=True)
            sample=samples.Sample('edges',0,1,None,str(source),0,SimpleNamespace(duration=.2))
            fitted=sample.frames(90,90)
            cropped=sample.frames(90,90,fit='cover')
            self.assertLess(int(fitted[0,5].max()),5)
            self.assertGreater(int(fitted[0,45,1,2]),150,'left blue edge is retained')
            self.assertGreater(int(fitted[0,45,88,1]),70,'right green edge is retained')
            self.assertGreater(int(cropped[0,5,45,0]),200)
            self.assertLess(int(cropped[0,:,:,2].max()),30,'cover removes the blue edge')
            self.assertIs(sample.frames(90,90),fitted)
            self.assertIs(sample.frames(90,90,fit='cover'),cropped)
            retro=sample.frames(90,90,pixel_aspect=2)
            self.assertLess(int(retro[0,:,1].max()),5,'fit accounts for the final canvas upscale')
            self.assertGreater(int(retro[0,5,45,0]),200)
            self.assertGreater(int(retro[0,45,6,2]),150)
            for width,height in [(181,41),(91,93),(1,1)]:
                frames=sample.frames(width,height)
                self.assertEqual(frames.shape[1:],(height,width,3),'odd tile sizes must remain exact')

    def test_process_output_is_byte_identical_with_cancellation_checks(self):
        cmd = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes(range(256))*1024); sys.stderr.buffer.write(b'error'*4096)"]
        before = subprocess.run(cmd, capture_output=True, timeout=10)
        after = media_process.run(cmd, capture_output=True, check_cancel=lambda: None, timeout=10)
        self.assertEqual((after.returncode, after.stdout, after.stderr),
                         (before.returncode, before.stdout, before.stderr))

    def test_cancel_kills_and_reaps_child_promptly(self):
        spawned = []
        real_popen = subprocess.Popen

        def popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            spawned.append(proc)
            return proc

        start = time.monotonic()

        def cancelled():
            if time.monotonic() - start > 0.15:
                raise RuntimeError("cancelled")

        with patch.object(media_process.subprocess, "Popen", side_effect=popen):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                media_process.run([sys.executable, "-c", "import time; time.sleep(30)"],
                                  capture_output=True, check_cancel=cancelled, timeout=10)
        self.assertLess(time.monotonic() - start, 2)
        self.assertIsNotNone(spawned[0].poll())
        self.assertTrue(spawned[0].stdout.closed)

    def test_timeout_reaps_child(self):
        start = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            media_process.run([sys.executable, "-c", "import time; time.sleep(30)"],
                              capture_output=True, check_cancel=lambda: None, timeout=0.15)
        self.assertLess(time.monotonic() - start, 2)

    def test_cancel_unblocks_a_stalled_raw_video_pipe(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.PIPE)
        start = time.monotonic()

        def cancel():
            if time.monotonic() - start > 0.15:
                raise RuntimeError("cancelled during pipe write")

        try:
            with self.assertRaisesRegex(RuntimeError, "cancelled during pipe write"):
                with media_process.cancellation_watch(proc, cancel):
                    try:
                        proc.stdin.write(b"frame" * (1024 * 1024))
                    except (BrokenPipeError, OSError):
                        pass
        finally:
            media_process.stop(proc)
        self.assertIsNotNone(proc.poll())
        self.assertLess(time.monotonic() - start, 2)

    def test_failed_chunks_keep_original_error_and_remove_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory, "out.mp4")

            def encode(segments, path, **kwargs):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("original encoder failure")

            with patch.object(g, "_MAX_INPUTS_PER_CALL", 1), patch.object(g, "_encode_chunk", side_effect=encode):
                with self.assertRaisesRegex(RuntimeError, "original encoder failure"):
                    g._build_video([{"source_file": "a"}, {"source_file": "b"}], str(out))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cancel_while_waiting_for_encode_slot(self):
        g._ENCODING.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                g._build_video([], "unused.mp4", progress=lambda *_: (_ for _ in ()).throw(RuntimeError("cancelled")))
        finally:
            g._ENCODING.release()

    def test_frame_cache_uses_lru_byte_budget_across_samples(self):
        samples.clear_cache()
        with patch.object(samples, "_FRAME_CACHE_BYTES", 72):
            voice = SimpleNamespace(duration=0.1)
            a = samples.Sample("a", 0, 1, None, "a.mp4", 0, voice)
            b = samples.Sample("b", 0, 1, None, "b.mp4", 0, voice)
            payload = bytes(range(36))
            with patch.object(media_process, "run", return_value=SimpleNamespace(returncode=0, stdout=payload)) as decode:
                first = a.frames(2, 2)
                self.assertEqual(first.tobytes(), payload)
                a.frames(2, 2)
                self.assertEqual(decode.call_count, 1)
                b.frames(2, 2)
                a.frames(2, 2)  # a is newest
                b.frames(1, 1)  # 36 bytes: evicts oldest, b's 2x2 frames
                self.assertEqual(samples._frame_bytes, 72)
                a.frames(2, 2)
                self.assertEqual(decode.call_count, 3)
                b.frames(2, 2)
                self.assertEqual(decode.call_count, 4)
                # An over-budget item is usable by its render without being retained.
                samples._store_frames(("large",), np.zeros(73, np.uint8))
                self.assertNotIn(("large",), samples._frame_cache)
        samples.clear_cache()
        self.assertEqual(samples._frame_bytes, 0)

    def test_sample_identity_tracks_source_paths_and_replacements(self):
        samples.clear_cache()
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
                a, b = Path(directory, "a.mp4"), Path(directory, "b.mp4")
                a.write_bytes(b"voice a")
                b.write_bytes(b"voice b")
                clip = {"id": 1, "word": "word", "start_time": 0.0, "end_time": 0.05,
                        "edited": True, "source_file": str(a)}

                def encode(segments, path, **kwargs):
                    Path(path).write_bytes(Path(segments[0]["source_file"]).read_bytes())

                def voice(audio):
                    return SimpleNamespace(x=audio, duration=len(audio) / pitch.SR)

                with patch.object(samples, "active", return_value={"slug": "voice"}), \
                        patch.object(g, "_ensure_cache"), patch.object(samples, "takes", return_value=[clip]), \
                        patch.object(g, "_build_video", side_effect=encode) as build, \
                        patch.object(samples, "decode_audio", return_value=np.full(2205, 0.1, np.float32)), \
                        patch.object(samples, "Voice", side_effect=voice):
                    first = samples.build("word")
                    self.assertIs(samples.build("word"), first)
                    self.assertEqual(build.call_count, 1)
                    # Imported rows can reuse the same id, label and times.
                    samples.clear_cache()
                    clip["source_file"] = str(b)
                    second = samples.build("word")
                    self.assertNotEqual(first.mp4, second.mp4)
                    self.assertEqual(Path(second.mp4).read_bytes(), b"voice b")
                    # In-place changed media also invalidates the memory hit.
                    b.write_bytes(b"changed voice b")
                    third = samples.build("word")
                    self.assertNotEqual(second.mp4, third.mp4)
                    self.assertEqual(Path(third.mp4).read_bytes(), b"changed voice b")
                    # A tar import may restore both the original mtime and size.
                    before = samples._source_stamp(str(b))
                    original_stat = b.stat()
                    replacement = Path(directory, "replacement.mp4")
                    replacement.write_bytes(b"z" * original_stat.st_size)
                    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                    os.replace(replacement, b)
                    self.assertNotEqual(samples._source_stamp(str(b)), before)
                    fourth = samples.build("word")
                    self.assertNotEqual(third.mp4, fourth.mp4)
                    self.assertEqual(Path(fourth.mp4).read_bytes(), b"z" * original_stat.st_size)
                    self.assertEqual(build.call_count, 4)
        finally:
            os.chdir(previous)
            samples.clear_cache()

    def test_build_locks_leave_no_permanent_registry_entries(self):
        import gc
        for i in range(100):
            with samples._build_lock(str(i)):
                pass
        gc.collect()
        self.assertEqual(len(samples._building), 0)

    def test_note_cache_budget_preserves_recomputed_audio(self):
        voice = pitch.Voice(np.linspace(-0.3, 0.3, 2205, dtype=np.float32))
        with patch.object(pitch, "_NOTE_CACHE_BYTES", 24000):
            original, warp = voice.render(None, 0.04, "raw", semitones=-2)
            for shift in (-1, 0, 1):
                voice.render(None, 0.04, "raw", semitones=shift)
            self.assertLessEqual(voice._cache_bytes, 24000)
            self.assertEqual(len(voice._cache), 3)
            repeated, repeated_warp = voice.render(None, 0.04, "raw", semitones=-2)
            self.assertIsNot(repeated, original)  # oldest was evicted
            self.assertTrue(np.array_equal(repeated, original))
            self.assertEqual(repeated_warp, warp)
            before = len(voice._cache)
            voice.render(None, 1.0, "raw")
            self.assertEqual(len(voice._cache), before)
            self.assertLessEqual(voice._cache_bytes, 24000)

    def test_concurrent_note_cache_misses_keep_exact_byte_accounting(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        voice = pitch.Voice(np.linspace(-0.3, 0.3, 2205, dtype=np.float32))
        barrier = threading.Barrier(8)
        real_fade = pitch._fade

        def fade(*args):
            result = real_fade(*args)
            barrier.wait(timeout=3)
            return result

        with patch.object(pitch, "_fade", side_effect=fade), ThreadPoolExecutor(max_workers=8) as pool:
            outputs = [future.result()[0] for future in
                       [pool.submit(voice.render, None, 0.08, "raw", 3) for _ in range(8)]]
        self.assertEqual(len(voice._cache), 1)
        self.assertEqual(voice._cache_bytes, outputs[0].nbytes)
        self.assertTrue(all(np.array_equal(outputs[0], out) for out in outputs))

    def test_encode_thread_inherits_operation_corpus(self):
        from app import database
        entries = [{"slug": "a"}, {"slug": "b"}]
        heard = []

        def encode(segments, path, **kwargs):
            heard.append(database.active()["slug"])
            raise RuntimeError("stop before media")

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(database, "list_corpora", return_value=entries), \
                patch.object(database, "_active", entries[1]), \
                patch.object(g, "_MAX_INPUTS_PER_CALL", 1), \
                patch.object(g, "_encode_chunk", side_effect=encode):
            with database.using_corpus("a"):
                with self.assertRaisesRegex(RuntimeError, "stop before media"):
                    g._build_video([{"source_file": "a"}, {"source_file": "b"}], str(Path(directory, "out.mp4")))
            self.assertEqual(database.active()["slug"], "b")
        self.assertTrue(heard)
        self.assertEqual(set(heard), {"a"})

    def test_cancelled_recommendation_does_not_measure_the_whole_backlog(self):
        clips = {str(i): [{"id": i, "word": str(i), "start_time": 0.0,
                          "end_time": 0.1, "source_file": "unused.mp4"}] for i in range(40)}
        started = []
        part = Part(id="lead", name="lead", track=0, channel=0, program=0,
                    notes=[Note(0, 0.1, 60, 100)], role="lead")
        song = Song([part], 0.1, 120, (4, 4), "C", 1)

        def measure(clip, groups=(), check_cancel=None):
            check_cancel()
            started.append(clip["id"])
            for _ in range(10):
                time.sleep(0.005)
                check_cancel()
            return {"v": recommend._MEASURE_VERSION, "word": {}}

        def progress(stage, done, total):
            if stage == "measuring" and done:
                raise RuntimeError("recommendation cancelled")

        with patch.object(recommend, "active", return_value={"slug": "fake"}), \
                patch.object(g, "_ensure_cache"), patch.object(g, "_clips_by_word_cache", clips), \
                patch.object(samples, "noises", return_value={}), patch.object(samples, "units", return_value={}), \
                patch.object(samples, "takes_for", side_effect=lambda word: clips[word]), \
                patch.object(recommend, "_words_for", return_value=list(clips)), \
                patch.object(recommend, "_load", return_value={}), patch.object(recommend, "_save") as save, \
                patch.object(recommend, "_measure_clip", side_effect=measure):
            with self.assertRaisesRegex(RuntimeError, "recommendation cancelled"):
                recommend.recommend(song, progress=progress)
        self.assertLess(len(started), len(clips))
        save.assert_not_called()

    def test_concurrent_measurement_cache_saves_use_atomic_unique_files(self):
        import json
        from concurrent.futures import ThreadPoolExecutor
        real_dump = json.dump

        def dump(*args, **kwargs):
            time.sleep(0.005)
            return real_dump(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(recommend, "_cache_path", return_value=str(Path(directory, "analysis.json"))), \
                patch.object(recommend, "_mem", {"voice": {"take": {"f0": 150}}}), \
                patch.object(recommend.json, "dump", side_effect=dump):
            with ThreadPoolExecutor(max_workers=8) as pool:
                for future in [pool.submit(recommend._save, "voice") for _ in range(8)]:
                    future.result()
            self.assertEqual(json.loads(Path(directory, "analysis.json").read_text()), {"take": {"f0": 150}})
            self.assertEqual([path.name for path in Path(directory).iterdir()], ["analysis.json"])

    def test_failed_note_pass_closes_mappings_before_cleanup(self):
        parts = [Part(id=str(i), name=str(i), track=i, channel=0, program=0,
                      notes=[Note(0, 0.1, 60, 100)], role="lead") for i in range(2)]
        song = Song(parts, 0.1, 120, (4, 4), "C", 1)
        mappings = []
        real_memmap = np.lib.format.open_memmap
        real_tempdir = tempfile.TemporaryDirectory

        def mapping(*args, **kwargs):
            arr = real_memmap(*args, **kwargs)
            mappings.append(arr)
            return arr

        def play(part, settings, takes, mix, *args):
            if part.id == "1":
                raise RuntimeError("note pass cancelled")
            mix[:100] = 0.01
            args[-1][:] = [0, 100]  # span, the final positional argument
            return []

        with real_tempdir() as directory:
            def tempdir(**kwargs):
                return real_tempdir(dir=directory, **kwargs)

            with patch.object(render, "load_song", return_value=song), \
                    patch.object(render, "_play", side_effect=play), \
                    patch.object(render.tempfile, "TemporaryDirectory", side_effect=tempdir), \
                    patch.object(np.lib.format, "open_memmap", side_effect=mapping):
                with self.assertRaisesRegex(RuntimeError, "note pass cancelled"):
                    render.render_ytpmv({"options": {"synth": True, "balance": True}})
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertTrue(all(arr._mmap.closed for arr in mappings))


if __name__ == "__main__":
    unittest.main()
