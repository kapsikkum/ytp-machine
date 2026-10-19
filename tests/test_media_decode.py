"""Real PCM parity and cancellation for the optional PyAV audio backend."""
from contextlib import chdir
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import media_decode
from app.ytpmv.pitch import decode_audio
from app.ytpmv import samples
from types import SimpleNamespace
from app import generate as g


class DispatchTests(unittest.TestCase):
    def test_cancellation_is_never_retried(self):
        for exception in (RuntimeError, ValueError, KeyboardInterrupt):
            def cancel():
                raise exception("cancelled")
            with patch.dict(os.environ, MRS_AUDIO_DECODER="pyav"), \
                    patch.object(media_decode, "_audio_pyav") as native, \
                    patch("app.media_process.run") as cli:
                with self.assertRaisesRegex(exception, "cancelled"):
                    decode_audio("unused", check_cancel=cancel)
                native.assert_not_called()
                cli.assert_not_called()

    def test_cancellation_during_native_decode_is_never_retried(self):
        def native(*args):
            args[-1]()
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("cancelled during decode")
        with patch.dict(os.environ, MRS_AUDIO_DECODER="pyav"), \
                patch.object(media_decode, "_audio_pyav", side_effect=native), \
                patch("app.media_process.run") as cli:
            with self.assertRaisesRegex(ValueError, "during decode"):
                decode_audio("unused", check_cancel=cancel)
            cli.assert_not_called()

    def test_missing_optional_dependency_and_unsupported_media_use_cli(self):
        for error in (ImportError("no av"), media_decode.DecodeError("no timestamps")):
            expected = np.ones(10, np.float32)
            with patch.dict(os.environ, MRS_AUDIO_DECODER="pyav"), \
                    patch.object(media_decode, "_audio_pyav", side_effect=error), \
                    patch("app.ytpmv.pitch._decode_audio_ffmpeg", return_value=expected) as cli:
                self.assertIs(decode_audio("unused"), expected)
                cli.assert_called_once()


@unittest.skipUnless(importlib.util.find_spec("av"), "optional PyAV is not installed")
class PCMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        for name, codec in (("audio.m4a", "aac"), ("audio.wav", "pcm_s16le")):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                "aevalsrc=0.3*sin(2*PI*147*t)|0.2*sin(2*PI*391*t):s=48000:d=1.4",
                "-c:a", codec, str(cls.root / name)], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc2=s=160x90:r=25:d=1.4", "-i", str(cls.root / "audio.wav"),
            "-c:v", "libx264", "-c:a", "aac", str(cls.root / "video.mp4")],
            check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_pcm_windows_rates_and_unbounded_reads(self):
        for name in ("audio.wav", "video.mp4", "audio.m4a"):
            for sr in (8000, 16000, 44100, 48000):
                for pcm in ("f32le", "s16le"):
                    for start, duration in ((None, None), (0, .08), (.123, .35),
                                             (.123, None), (-1, .2), (1.3, .3), (3, .1)):
                        with self.subTest(name=name, sr=sr, pcm=pcm, start=start, duration=duration):
                            kwargs = dict(start=start, duration=duration, sr=sr, pcm=pcm)
                            with patch.dict(os.environ, MRS_AUDIO_DECODER="ffmpeg"):
                                expected = decode_audio(str(self.root / name), **kwargs)
                            with patch.dict(os.environ, MRS_AUDIO_DECODER="pyav"):
                                actual = decode_audio(str(self.root / name), **kwargs)
                            np.testing.assert_array_equal(actual, expected)

    def test_verified_windows_use_native_decoder(self):
        with patch.dict(os.environ, MRS_AUDIO_DECODER="pyav"), \
                patch("app.ytpmv.pitch._decode_audio_ffmpeg", side_effect=AssertionError("unexpected fallback")):
            for name in ("video.mp4", "audio.wav"):
                self.assertGreater(len(decode_audio(str(self.root / name), .123, .35)), 0)

    def test_malformed_input_has_normal_cli_error(self):
        source = self.root / "broken.mp4"
        source.write_bytes(b"broken input")
        with patch.dict(os.environ, MRS_AUDIO_DECODER="pyav"):
            with self.assertRaisesRegex(RuntimeError, "FFmpeg failed"):
                decode_audio(str(source))

    def test_video_fit_fill_seek_odd_cells_and_pixel_aspect(self):
        for start in (0, .123, .4):
            for fit in ("contain", "cover"):
                for w, h, aspect in ((90,90,1), (181,91,1), (128,112,1.5555556)):
                    with self.subTest(start=start, fit=fit, size=(w,h), aspect=aspect):
                        sample = samples.Sample("video", 0, 1, None,
                            str(self.root / "video.mp4"), start, SimpleNamespace(duration=.35))
                        samples.clear_cache()
                        with patch.dict(os.environ, MRS_VIDEO_DECODER="ffmpeg"):
                            expected = sample.frames(w, h, fit=fit, pixel_aspect=aspect)
                        samples.clear_cache()
                        with patch.dict(os.environ, MRS_VIDEO_DECODER="pyav"), \
                                patch.object(sample, "_frames_ffmpeg", wraps=sample._frames_ffmpeg) as cli:
                            actual = sample.frames(w, h, fit=fit, pixel_aspect=aspect)
                            self.assertEqual(cli.call_count, int(start == .123))
                        np.testing.assert_array_equal(actual, expected)
        samples.clear_cache()

    def test_video_fractional_duration_matches_cli_frame_count(self):
        for duration in (.02, .04, .1, .12, .14, .2, .3, .4, .45, .5, 1.2):
            sample = samples.Sample("video", 0, 1, None,
                str(self.root / "video.mp4"), .4, SimpleNamespace(duration=duration))
            samples.clear_cache()
            with patch.dict(os.environ, MRS_VIDEO_DECODER="ffmpeg"):
                expected = sample.frames(90,90)
            samples.clear_cache()
            with patch.dict(os.environ, MRS_VIDEO_DECODER="pyav"):
                actual = sample.frames(90,90)
            with self.subTest(duration=duration):
                np.testing.assert_array_equal(actual, expected)
        samples.clear_cache()

    def test_rotated_video_uses_cli_autorotation(self):
        import av
        rotated = self.root / "rotated.mp4"
        with av.open(str(self.root / "video.mp4")) as source, av.open(str(rotated), "w") as output:
            track = output.add_stream_from_template(source.streams.video[0])
            track.set_display_rotation(90)
            for packet in source.demux(source.streams.video[0]):
                if packet.dts is not None:
                    packet.stream = track
                    output.mux(packet)
        with av.open(str(rotated)) as source:
            self.assertEqual(abs(next(source.decode(video=0)).rotation), 90)
        sample = samples.Sample("video", 0, 1, None, str(rotated),
                                0, SimpleNamespace(duration=.2))
        samples.clear_cache()
        with patch.dict(os.environ, MRS_VIDEO_DECODER="ffmpeg"):
            expected = sample.frames(90,90)
        samples.clear_cache()
        with patch.dict(os.environ, MRS_VIDEO_DECODER="pyav"), \
                patch.object(sample, "_frames_ffmpeg", wraps=sample._frames_ffmpeg) as cli:
            actual = sample.frames(90,90)
            cli.assert_called_once()
        np.testing.assert_array_equal(actual, expected)
        samples.clear_cache()

    def test_repeated_native_decode_releases_threads_and_bounds_memory(self):
        import gc
        try:
            import psutil
        except ImportError:
            self.skipTest("optional psutil required for native resource test")
        process = psutil.Process()
        sample = samples.Sample("video", 0, 1, None, str(self.root / "video.mp4"),
                                .4, SimpleNamespace(duration=.1))
        def cycle():
            samples.clear_cache()
            decode_audio(str(self.root / "video.mp4"), .123, .08)
            sample.frames(90,90)
        with patch.dict(os.environ, MRS_AUDIO_DECODER="pyav", MRS_VIDEO_DECODER="pyav"):
            for _ in range(5):
                cycle()
            gc.collect()
            before_rss, before_threads = process.memory_info().rss, process.num_threads()
            for _ in range(80):
                cycle()
            gc.collect()
            self.assertLessEqual(process.num_threads(), before_threads + 2)
            self.assertLess(process.memory_info().rss - before_rss, 32 * 1024 * 1024)
            # No video cleanup in between: audio resampling itself must release
            # its graph/worker pool instead of relying on another media call.
            before_rss, before_threads = process.memory_info().rss, process.num_threads()
            for _ in range(80):
                decode_audio(str(self.root / "video.mp4"), .123, .08, sr=16000)
            self.assertLessEqual(process.num_threads(), before_threads + 2)
            self.assertLess(process.memory_info().rss - before_rss, 32 * 1024 * 1024)
        samples.clear_cache()

    def test_native_video_cancel_does_not_fall_back(self):
        sample = samples.Sample("video", 0, 1, None, str(self.root / "video.mp4"),
                                0, SimpleNamespace(duration=1))
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls > 8:
                raise RuntimeError("cancelled video")
        samples.clear_cache()
        with patch.dict(os.environ, MRS_VIDEO_DECODER="pyav"), \
                patch.object(sample, "_frames_ffmpeg") as cli:
            with self.assertRaisesRegex(RuntimeError, "cancelled video"):
                sample.frames(90,90,check_cancel=cancel)
            cli.assert_not_called()
        samples.clear_cache()


class LazyVideoTests(unittest.TestCase):
    def test_audio_does_not_encode_and_later_video_reuses_original_voice(self):
        samples.clear_cache()
        self.addCleanup(samples.clear_cache)
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            source = Path(directory, "recording.mp4")
            source.write_bytes(b"original")
            clip = {"id": 1, "word": "word", "start_time": 0., "end_time": .2,
                    "source_file": str(source), "edited": True}
            entry = {"slug": "voice"}
            def encode(segments, path, **kwargs):
                self.assertEqual(segments, [clip])
                Path(path).write_bytes(b"video")
            with patch.object(samples, "active", return_value=entry), \
                    patch.object(samples, "list_corpora", return_value=[entry]), \
                    patch.object(samples, "takes", return_value=[clip]), \
                    patch.object(samples, "decode_audio", return_value=np.full(8820, .1, np.float32)) as decode, \
                    patch.object(g, "_build_video", side_effect=encode) as build:
                sample = samples.build("word", audio_only=True)
                self.assertFalse(Path(sample.mp4).exists())
                build.assert_not_called()
                decode.assert_called_once_with(str(source), 0., .2)
                original = sample.voice.x.copy()
                # A second audio audition uses the in-memory sample even
                # though its picture has not been encoded yet.
                self.assertIs(samples.build("word", audio_only=True), sample)
                self.assertIsNone(sample.describe(include_video=False)["sample_url"])
                build.assert_not_called()
                self.assertEqual(sample.url, "/output/" + Path(sample.mp4).name)
                sample.ensure_video()
                build.assert_called_once()
                np.testing.assert_array_equal(sample.voice.x, original)
                self.assertIs(samples.build("word"), sample)
                # Output cleanup must not evict the prepared audio; requesting
                # the video afterwards regenerates it under the build lock.
                Path(sample.mp4).unlink()
                self.assertIs(samples.build("word", audio_only=True), sample)
                sample.ensure_video()
                self.assertEqual(build.call_count, 2)


if __name__ == "__main__":
    unittest.main()
