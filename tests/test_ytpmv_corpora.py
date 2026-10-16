"""YTPMV parts keep their installed voice through sampling, previews and renders."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, chdir
from pathlib import Path
from types import SimpleNamespace
import os
import asyncio
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import database as db
from app import generate as g
from app.ytpmv import render, samples
from app.ytpmv.music import Note, Part, Song
from app.ytpmv.pitch import SR


class YtpmvCorpusTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="ytpmv-corpora-")))
        self.stack.enter_context(chdir(self.root))
        self.stack.enter_context(patch.object(db, "CORPORA_DIR", str(self.root / "corpora")))
        self.stack.enter_context(patch.object(db, "LEGACY_DB", str(self.root / "missing.db")))
        self.stack.enter_context(patch.object(db, "_ready", set()))
        self.stack.enter_context(patch.object(g, "_POOLS", {}))
        self.stack.enter_context(patch.object(g, "_borrowing", None))
        self.stack.enter_context(patch.object(g, "_cache_corpus", None))
        self.stack.enter_context(patch.object(g, "_clips_by_word_cache", None))
        self.entries = {}
        for slug in ("alpha", "beta"):
            directory = self.root / "corpora" / slug
            (directory / "downloads").mkdir(parents=True)
            (directory / "downloads" / "same.mp4").write_bytes(slug.encode())
            entry = {"slug": slug, "name": slug.title(), "dir": str(directory),
                     "db": str(directory / "corpus.db")}
            self.entries[slug] = entry
            with db.using_corpus(entry):
                db.init_db()
                with db.get_db() as conn:
                    conn.execute("INSERT INTO sources (id, video_id, source_file) VALUES (1, 'same', 'downloads/same.mp4')")
                    conn.execute("INSERT INTO word_clips (id, source_id, word, start_time, end_time, source_file, edited, phones) "
                                 "VALUES (1, 1, 'hello', 0.0, 0.2, 'downloads/same.mp4', 1, ?)",
                                 ('[["AH", 0.02, 0.18]]' if slug == "alpha" else '[["IY", 0.02, 0.18]]',))
                    conn.execute("INSERT INTO noise_clips (source_id, word, start_time, end_time, source_file) "
                                 "VALUES (1, ?, 0.0, 0.1, 'downloads/same.mp4')", (slug,))
        self.stack.enter_context(patch.object(db, "_active", self.entries["alpha"]))
        self.parts = [Part(id=slug, name=slug, track=i, channel=i, program=0,
                           notes=[Note(0, 0.1, 60 + i * 4, 100)], role="lead")
                      for i, slug in enumerate(("alpha", "beta"))]
        self.song = Song(self.parts, 0.15, 120, (4, 4), "C", 1)
        self.stack.enter_context(patch.object(render, "load_song", return_value=self.song))
        samples.clear_cache()
        self.addCleanup(samples.clear_cache)

    def stub_media(self):
        def encode(segments, path, **kwargs):
            Path(path).write_bytes(Path(segments[0]["source_file"]).read_bytes())

        def decode(path, *args, **kwargs):
            freq = 220 if Path(path).read_bytes() == b"alpha" else 440
            return (0.2 * np.sin(2 * np.pi * freq * np.arange(SR // 5) / SR)).astype(np.float32)

        self.stack.enter_context(patch.object(g, "_build_video", side_effect=encode))
        self.stack.enter_context(patch.object(samples, "decode_audio", side_effect=decode))

    def test_parallel_voice_catalogues_use_local_pools(self):
        g._use_voice(None)
        installed = g._installed
        ready = threading.Barrier(2)

        def read(slug):
            with samples.corpus_context(slug):
                ready.wait(timeout=5)
                for _ in range(20):
                    self.assertEqual(samples.noises(), {slug: 1})
                    self.assertEqual(set(samples.units()), {"ah" if slug == "alpha" else "iy"})
                    self.assertIn(str(Path("corpora", slug)), samples.takes("hello")[0]["source_file"])

        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in [pool.submit(read, slug) for slug in self.entries]:
                result.result(timeout=10)
        self.assertIs(g._installed, installed)
        self.assertEqual(db.active()["slug"], "alpha")

    def test_samples_and_cache_distinguish_identical_words_and_take_ids(self):
        self.stub_media()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = [result.result(timeout=10) for result in
                    [pool.submit(samples.build, "hello", corpus=slug) for slug in self.entries]]
        self.assertEqual(a.describe()["corpus"], "alpha")
        self.assertEqual(b.describe()["corpus"], "beta")
        self.assertNotEqual(a.mp4, b.mp4)
        self.assertEqual(Path(a.mp4).read_bytes(), b"alpha")
        self.assertEqual(Path(b.mp4).read_bytes(), b"beta")
        self.assertIs(samples.build("hello", corpus="beta"), b)
        self.assertEqual(db.active()["slug"], "alpha")

    def test_missing_voice_is_rejected_without_changing_selection(self):
        for corpus in ("missing", "../alpha", ""):
            with self.subTest(corpus=corpus), self.assertRaises(samples.SampleError):
                samples.build("hello", corpus=corpus)
        self.assertEqual(db.active()["slug"], "alpha")

    def test_part_recommendations_and_catalogues_are_scoped(self):
        def recommend(song, **kwargs):
            self.assertEqual(db.active()["slug"], "beta")
            self.assertIn(str(Path("corpora", "beta")), kwargs["pool"]["clips_by_word"]["hello"][0]["source_file"])
            return {song.parts[0].id: {"text": "hello", "take": 0}}

        with patch.object(render.recommend, "recommend", side_effect=recommend):
            result = render.recommend_part("midi", "alpha", "beta")
        self.assertEqual(result["corpus"], "beta")
        self.assertEqual(result["settings"]["corpus"], "beta")
        self.assertEqual(result["noises"], {"beta": 1})
        self.assertEqual(result["units"], {"iy": 1})
        self.assertEqual(db.active()["slug"], "alpha")

    def test_part_preparation_pins_defaults_and_rejects_unknown_voices(self):
        parts = render.prepare_parts([{"id": "alpha"}, {"id": "beta", "corpus": "beta"}])
        self.assertEqual([p["corpus"] for p in parts], ["alpha", "beta"])
        with self.assertRaises(samples.SampleError):
            render.prepare_parts([{"id": "alpha", "corpus": "gone"}])
        self.assertEqual(render.prepare_parts([{"id": "alpha", "corpus": "gone"}], synth=True)[0]["corpus"], "gone")

    def test_preview_uses_chosen_voice_and_separates_audio_urls(self):
        self.stub_media()
        settings = {"id": "alpha", "text": "hello"}
        first = render.preview_part("midi", {**settings, "corpus": "alpha"}, seconds=1)
        second = render.preview_part("midi", {**settings, "corpus": "beta"}, seconds=1)
        longer = render.preview_part("midi", {**settings, "corpus": "beta"}, seconds=2)
        self.assertEqual(second["sample"]["corpus"], "beta")
        self.assertNotEqual(first["audio_url"], second["audio_url"])
        self.assertNotEqual(second["audio_url"], longer["audio_url"])
        self.assertEqual(db.active()["slug"], "alpha")

    def test_render_prepares_each_part_with_its_own_sample(self):
        self.stub_media()
        heard = []

        def play(part, settings, choices, *args, **kwargs):
            heard.append((part.id, settings["corpus"], choices[0].corpus))
            if len(heard) == 2:
                raise RuntimeError("samples verified")
            return []

        with patch.object(render, "_play", side_effect=play):
            with self.assertRaisesRegex(RuntimeError, "samples verified"):
                render.render_ytpmv({"midi_id": "midi", "parts": [
                    {"id": "alpha", "text": "hello", "corpus": "alpha"},
                    {"id": "beta", "text": "hello", "corpus": "beta"}],
                    "options": {"balance": False, "max_seconds": 1}})
        self.assertEqual(heard, [("alpha", "alpha", "alpha"), ("beta", "beta", "beta")])

    def test_render_missing_words_are_recommended_in_each_selected_voice(self):
        self.stub_media()
        requested = []

        def recommend(song, **kwargs):
            requested.append((db.active()["slug"], [p.id for p in song.parts]))
            return {p.id: {"text": "hello"} for p in song.parts}

        with patch.object(render.recommend, "recommend", side_effect=recommend), \
                patch.object(render, "_play", side_effect=RuntimeError("samples verified")):
            with self.assertRaisesRegex(RuntimeError, "samples verified"):
                render.render_ytpmv({"midi_id": "midi", "parts": [
                    {"id": "alpha", "corpus": "beta"}, {"id": "beta", "corpus": "alpha"}],
                    "options": {"balance": False}})
        self.assertEqual(requested, [("beta", ["alpha"]), ("alpha", ["beta"])])

    def test_api_validates_uninstalled_voice_before_queueing(self):
        from app import api
        from fastapi import HTTPException
        request = api.YtpmvRenderRequest(midi_id="midi", parts=[{"id": "alpha", "corpus": "missing"}])
        with patch.object(api.jobs, "submit_ytpmv") as submit:
            with self.assertRaises(HTTPException) as error:
                api.ytpmv_render(request)
        self.assertEqual(error.exception.status_code, 400)
        submit.assert_not_called()

    def test_http_guard_allows_page_local_ytpmv_voices_but_not_stale_edits(self):
        Path('frontend').mkdir(exist_ok=True)
        with patch.object(threading.Thread, 'start'):
            import main
        from starlette.responses import JSONResponse
        seen = []

        async def endpoint(scope, receive, send):
            seen.append(db.active()['slug'])
            await JSONResponse({'corpus': db.active()['slug']})(scope, receive, send)

        async def request(path, corpus):
            sent = []
            async def receive():
                return {'type': 'http.request', 'body': b'', 'more_body': False}
            async def send(message):
                sent.append(message)
            await main._Guard(endpoint)({'type': 'http', 'method': 'POST', 'path': path,
                'scheme': 'http', 'headers': [(b'x-corpus', corpus.encode()),
                    (b'content-type', b'multipart/form-data; boundary=test')]}, receive, send)
            return sent[0]['status']

        for path in ('midi', 'sample', 'preview', 'render'):
            self.assertEqual(asyncio.run(request('/api/ytpmv/' + path, 'beta')), 200)
        self.assertEqual(seen, ['beta'] * 4)
        self.assertEqual(asyncio.run(request('/api/edit/clip/1', 'beta')), 409)
        self.assertEqual(asyncio.run(request('/api/ytpmv/sample', 'missing')), 409)
        self.assertEqual(db.active()['slug'], 'alpha')

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for the real render check")
    def test_real_render_contains_two_corpus_videos_and_voices(self):
        for slug, colour, frequency in (("alpha", "red", 220), ("beta", "blue", 440)):
            path = Path(self.entries[slug]["dir"], "downloads", "same.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            f"color=c={colour}:s=64x64:r=25:d=0.6", "-f", "lavfi", "-i",
                            f"sine=frequency={frequency}:sample_rate={SR}:duration=0.6",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                            "-shortest", str(path)], check=True, capture_output=True)
        result = render.render_ytpmv({"midi_id": "midi", "parts": [
            {"id": "alpha", "text": "hello", "corpus": "alpha"},
            {"id": "beta", "text": "hello", "corpus": "beta"}],
            "options": {"balance": False, "max_seconds": 1, "flip": False,
                        "flash": False, "dim": False}})
        self.assertEqual([p["corpus"] for p in result["parts"]], ["alpha", "beta"])
        audio_a, audio_b = [samples.build("hello", corpus=slug) for slug in self.entries]
        self.assertAlmostEqual(audio_a.voice.info.f0, 220, delta=8)
        self.assertAlmostEqual(audio_b.voice.info.f0, 440, delta=8)
        path = str(self.root / result["video_url"].lstrip("/"))
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             check=True, capture_output=True).stdout
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int16)
        # The square sources are padded to the renderer's widescreen tiles.
        self.assertGreater(np.mean((pixels[:, 0] > 150) & (pixels[:, 0] > pixels[:, 2] + 100)), 0.25)
        self.assertGreater(np.mean((pixels[:, 2] > 150) & (pixels[:, 2] > pixels[:, 0] + 100)), 0.25)
        self.assertEqual(db.active()["slug"], "alpha")


if __name__ == "__main__":
    unittest.main()
