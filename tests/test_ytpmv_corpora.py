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
import wave
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

    def test_grid_fills_widescreen_and_balances_incomplete_rows(self):
        self.assertEqual(render.grid_cells(3), [(0, 0, 960, 540), (960, 0, 960, 540), (0, 540, 1920, 540)])
        for width, height in ((1920, 1080), (256, 224), (160, 144)):
            for count in range(1, 33):
                cells = render.grid_cells(count, width, height)
                self.assertEqual(len(cells), count)
                self.assertEqual(sum(w*h for x,y,w,h in cells), width*height)
                last_y = 0
                for y in sorted({c[1] for c in cells}):
                    row = [c for c in cells if c[1] == y]
                    self.assertEqual(y, last_y)
                    last_x = 0
                    for x, _, w, h in row:
                        self.assertEqual(x, last_x)
                        self.assertGreater(min(w,h), 0)
                        last_x += w
                    self.assertEqual(last_x, width)
                    last_y = y + row[0][3]
                self.assertEqual(last_y, height)

    def test_timeline_detail_keeps_sustained_notes_and_bounds_large_windows(self):
        self.parts[0].notes = [Note(0, 4, 60, 100), Note(2, .2, 64, 100), Note(9, .2, 67, 100)]
        got = render.timeline_notes('midi', 1, 3, ['alpha'])
        self.assertEqual(got['parts']['alpha'], [[0,4,60],[2,.2,64]])
        self.assertFalse(got['sampled'])
        for a,b,ids in [(float('nan'),3,['alpha']), (3,2,['alpha']), (0,3,['missing']), (0,3,[])]:
            with self.assertRaises(render.YtpmvError): render.timeline_notes('midi',a,b,ids)
        self.parts[0].notes = [Note(i*.001,.01,60,100) for i in range(50000)]
        got = render.timeline_notes('midi',0,100,['alpha'])
        self.assertLessEqual(len(got['parts']['alpha']),24000)
        self.assertEqual(got['counts']['alpha'],50000)
        self.assertTrue(got['sampled'])

    def test_every_grid_preset_partitions_the_canvas(self):
        for layout in (layout for layout in render.GRID_LAYOUTS if layout != 'classic'):
            for count in range(1, 18):
                with self.subTest(layout=layout, count=count):
                    pixels = np.zeros((90, 160), dtype=np.uint8)
                    cells = render.grid_cells(count, 160, 90, layout)
                    self.assertEqual(len(cells), count)
                    for x,y,w,h in cells:
                        self.assertGreater(min(w,h), 0)
                        pixels[y:y+h,x:x+w] += 1
                    self.assertTrue(np.all(pixels == 1))
        parts = render.ordered_tiles(self.parts, ['missing', 'beta', 'beta', 'alpha'])
        self.assertEqual([p.id for p in parts], ['beta','alpha'])

    def test_classic_grid_keeps_equal_widescreen_tiles_and_centers_incomplete_rows(self):
        self.assertEqual(render.grid_cells(3, layout='classic'),
                         [(0,0,960,540),(960,0,960,540),(480,540,960,540)])
        self.assertEqual(render.grid_cells(3, layout='classic', centered=False)[-1], (0,540,960,540))
        self.assertEqual(render.grid_cells(2, layout='classic')[0][0], 480)
        for width,height in [(1920,1080),(256,224),(160,144)]:
            for count in range(1,33):
                cells = render.grid_cells(count,width,height,'classic')
                pixels = np.zeros((height,width),dtype=np.uint8)
                self.assertEqual(len(cells),count)
                self.assertEqual(len({(w,h) for x,y,w,h in cells}),1)
                for x,y,w,h in cells:
                    self.assertTrue(0 <= x < x+w <= width and 0 <= y < y+h <= height)
                    self.assertLessEqual(abs(w-h*16/9),2)
                    pixels[y:y+h,x:x+w] += 1
                self.assertLessEqual(int(pixels.max()),1)
                for y in {y for x,y,w,h in cells}:
                    row=[c for c in cells if c[1]==y]
                    self.assertLessEqual(abs(row[0][0]-(width-row[-1][0]-row[-1][2])),1)

    def test_instrument_note_audition_is_isolated_and_does_not_mutate_cached_score(self):
        self.song.duration = 30
        self.parts[0].notes = [Note(20, .1, 40, 100)]
        before = list(self.parts[0].notes)
        result = render.preview_mix({'midi_id':'midi', 'part_id':'alpha', 'note':72,
            'parts':[{'id':'alpha'}, {'id':'beta', 'mute':True, 'visible':False}],
            'options':{'synth':True,'chip':'sms','balance':False}, 'seconds':2})
        self.assertEqual([p['id'] for p in result['parts']], ['alpha'])
        self.assertEqual(result['from'], 0)
        self.assertLess(result['duration'], 2)
        self.assertEqual(self.song.duration, 30)
        self.assertEqual(self.parts[0].notes, before)
        with wave.open(str(self.root / result['audio_url'].lstrip('/'))) as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
        self.assertGreater(int(np.abs(pcm.astype(np.int32)).max()), 100)

    def test_note_audition_keeps_auto_instrument_from_the_original_arrangement(self):
        self.parts[0].role = self.parts[1].role = 'rhythm'
        self.parts[0].notes = [Note(0, .1, 40, 100)]
        self.parts[1].notes = [Note(0, .1, 48, 100)]
        with patch.object(render, '_play', wraps=render._play) as play:
            render.preview_mix({'midi_id':'midi','part_id':'alpha','note':72,
                'parts':[{'id':'alpha'},{'id':'beta','mute':True,'visible':False}],
                'options':{'synth':True,'chip':'nes','balance':False},'seconds':2})
        self.assertEqual(play.call_args.args[1]['tone'], render.chip_tone(self.parts[0], 'nes', as_bass=True))

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for layout validation')
    def test_render_uses_featured_preset_and_reordered_tiles(self):
        self.stub_media()
        def frames(sample, w, h, **kwargs):
            colour = (220, 20, 20) if sample.corpus == 'alpha' else (20, 20, 220)
            return np.full((1,h,w,3), colour, dtype=np.uint8)
        with patch.object(samples.Sample, 'frames', frames), \
                patch.object(render, '_OUT_W', 320), patch.object(render, '_OUT_H', 180):
            result = render.render_ytpmv({'midi_id':'midi', 'parts':[
                {'id':'alpha','text':'hello','corpus':'alpha'},
                {'id':'beta','text':'hello','corpus':'beta'}],
                'options':{'balance':False,'flip':False,'flash':False,'dim':False,
                           'layout':'featured','tile_order':['beta','alpha']}})
        raw = subprocess.run(['ffmpeg','-v','error','-i',str(self.root/result['video_url'].lstrip('/')),
                              '-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],
                             check=True,capture_output=True).stdout
        image = np.frombuffer(raw,dtype=np.uint8).reshape(180,320,3)
        self.assertGreater(int(image[90,100,2]), 170)
        self.assertLess(int(image[90,100,0]), 60)
        self.assertGreater(int(image[90,280,0]), 170)
        self.assertEqual(result['options']['tile_order'], ['beta','alpha'])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for layout validation')
    def test_classic_export_centers_last_tile_and_passes_framing_to_decoder(self):
        self.stub_media()
        self.song.parts.append(Part(id='gamma',name='third',track=2,channel=2,program=0,
                                   notes=[Note(0,.1,67,100)],role='chords'))
        fits=[]
        def frames(sample,w,h,**kwargs):
            fits.append(kwargs['fit'])
            return np.full((1,h,w,3),(220,20,20),dtype=np.uint8)
        with patch.object(samples.Sample,'frames',frames),patch.object(render,'_OUT_W',320),patch.object(render,'_OUT_H',180):
            result=render.render_ytpmv({'midi_id':'midi','parts':[
                {'id':p.id,'text':'hello','corpus':'alpha'} for p in self.song.parts],
                'options':{'balance':False,'flip':False,'flash':False,'dim':False,
                           'layout':'classic','media_fit':'cover','center_tiles':True}})
        raw=subprocess.run(['ffmpeg','-v','error','-i',str(self.root/result['video_url'].lstrip('/')),
                            '-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],check=True,capture_output=True).stdout
        image=np.frombuffer(raw,dtype=np.uint8).reshape(180,320,3)
        self.assertLess(int(image[135,20].max()),10)
        self.assertGreater(int(image[135,160,0]),170)
        self.assertLess(int(image[135,300].max()),10)
        self.assertEqual(fits,['cover']*3)
        self.assertEqual(result['options']['media_fit'],'cover')
        self.assertTrue(result['options']['center_tiles'])

    def test_keyboard_bank_renders_all_pitches_together_once(self):
        notes = list(range(60,73))
        before = list(self.parts[0].notes)
        with patch.object(render, '_play', wraps=render._play) as play:
            result = render.preview_mix({'midi_id':'midi','part_id':'alpha','keyboard_notes':notes,
                'parts':[{'id':'alpha'},{'id':'beta','mute':True,'visible':False}],
                'options':{'synth':True,'chip':'sms','balance':False},'seconds':2})
        self.assertEqual(play.call_count,1)
        self.assertEqual([n.pitch for n in play.call_args.args[0].notes],notes)
        self.assertEqual(self.parts[0].notes,before)
        self.assertEqual([s['note'] for s in result['keyboard']],notes)
        with wave.open(str(self.root/result['audio_url'].lstrip('/'))) as wav:
            pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').reshape(-1,2).astype(float)
        self.assertGreater(len(pcm)/SR,25)
        for segment in result['keyboard']:
            start=round((segment['offset']+.1)*SR)
            clip=pcm[start:start+SR//4,0]
            spectrum=np.abs(np.fft.rfft(clip*np.hanning(len(clip))))
            frequency=np.fft.rfftfreq(len(clip),1/SR)[np.argmax(spectrum)]
            expected=440*2**((segment['note']-69)/12)
            self.assertAlmostEqual(frequency,expected,delta=8)
        for notes in [[],[60,60],list(range(25)),[128]]:
            with self.subTest(notes=notes),self.assertRaises(render.YtpmvError):
                render.preview_mix({'midi_id':'midi','part_id':'alpha','keyboard_notes':notes})

    def test_chip_parts_play_without_corpus_samples_in_a_voice_project(self):
        with patch.object(samples, 'build', side_effect=AssertionError('chip requested a corpus sample')), \
                patch.object(render.recommend, 'recommend', side_effect=AssertionError('chip requested a word')):
            result = render.preview_mix({'midi_id': 'midi', 'parts': [
                {'id': 'alpha', 'tone': 'piano'}, {'id': 'beta', 'tone': 'pulse'}],
                'seconds': .4, 'options': {'balance': False}})
        with wave.open(str(self.root / result['audio_url'].lstrip('/'))) as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
        self.assertGreater(int(np.abs(pcm.astype(np.int32)).max()), 1000)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for waveform tiles')
    def test_chip_video_tiles_work_in_mixed_mode_and_retro_draft(self):
        self.song.parts.append(Part(id='gamma', name='third', track=2, channel=2, program=0,
                                   notes=[Note(0, .1, 67, 100)], role='chords'))
        with patch.object(samples, 'build', side_effect=AssertionError('chip requested a corpus sample')), \
                patch.object(render.recommend, 'recommend', side_effect=AssertionError('chip requested a word')), \
                patch.object(render, '_OUT_W', 640), patch.object(render, '_OUT_H', 360):
            result = render.render_ytpmv({'midi_id': 'midi', 'parts': [
                {'id': 'alpha', 'tone': 'piano'}, {'id': 'beta', 'tone': 'pulse'},
                {'id': 'gamma', 'tone': 'snes', 'program': 0}],
                'options': {'chip': 'nes', 'draft': True, 'balance': False,
                            'max_seconds': .2, 'flash': False, 'dim': False}})
        self.assertEqual(result['size'], [640,360])
        raw = subprocess.run(['ffmpeg','-v','error','-i',str(self.root/result['video_url'].lstrip('/')),
                              '-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],
                             check=True,capture_output=True).stdout
        pixels = np.frombuffer(raw,dtype=np.uint8).reshape(360,640,3)
        for x,y,w,h in render.grid_cells(3,640,360):
            self.assertGreater(int(pixels[y:y+h,x:x+w].max()), 30)

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

    def test_mix_preview_uses_render_speed_key_pan_and_bounded_range(self):
        self.parts[0].notes = [Note(1, .6, 60, 100)]
        self.song.duration = 3
        params = {"midi_id": "midi", "parts": [
            {"id": "alpha", "volume": .65, "pan": -1},
            {"id": "beta", "mute": True, "visible": False}],
            "options": {"synth": True, "chip": "sms", "balance": False,
                        "speed": 2, "transpose": 12}, "start": .5, "seconds": 1}
        with patch.object(samples.Sample, "frames", side_effect=AssertionError('audio preview must not draw video')):
            result = render.preview_mix(params)
        self.assertEqual(result['from'], .5)
        self.assertEqual(result['duration'], 1)
        self.assertEqual([p['id'] for p in result['parts']], ['alpha'])
        self.assertEqual(result['parts'][0]['volume'], .65)
        path = Path(self.root, result['audio_url'].lstrip('/'))
        with wave.open(str(path)) as wav:
            self.assertEqual(wav.getnframes(), SR)
            pcm = np.frombuffer(wav.readframes(SR), dtype='<i2').reshape(-1, 2).astype(float)
        self.assertGreater(float(np.abs(pcm[:, 0]).max()), 100)
        self.assertLess(float(np.abs(pcm[:, 1]).max()), 2)
        spectrum = np.abs(np.fft.rfft(pcm[:SR // 5, 0] * np.hanning(SR // 5)))
        peak = np.fft.rfftfreq(SR // 5, 1 / SR)[np.argmax(spectrum)]
        self.assertAlmostEqual(peak, 523.25, delta=8)

    def test_mix_preview_keeps_selected_corpora_and_skips_muted_parts(self):
        self.stub_media()
        result = render.preview_mix({'midi_id': 'midi', 'parts': [
            {'id': 'alpha', 'text': 'hello', 'corpus': 'beta', 'pan': -.3, 'volume': .5},
            {'id': 'beta', 'mute': True, 'visible': False}],
            'options': {'balance': False}, 'seconds': 50})
        self.assertEqual(result['parts'], [{'id': 'alpha', 'corpus': 'beta', 'volume': .5, 'pan': -.3}])
        self.assertLessEqual(result['duration'], 15)

    def test_whole_song_audio_reaches_notes_beyond_short_audition_limit(self):
        self.song.duration = 24
        self.parts[0].notes = [Note(20, .3, 60, 100)]
        result = render.preview_mix({'midi_id': 'midi', 'parts': [
            {'id': 'alpha'}, {'id': 'beta', 'mute': True, 'visible': False}],
            'options': {'synth': True, 'chip': 'sms', 'balance': False}, 'seconds': None})
        self.assertGreater(result['duration'], 24)
        with wave.open(str(Path(self.root, result['audio_url'].lstrip('/')))) as wav:
            self.assertGreater(wav.getnframes(), 24 * SR)
            wav.setpos(20 * SR)
            self.assertGreater(np.abs(np.frombuffer(wav.readframes(SR), dtype='<i2')).max(), 100)

    def test_whole_song_playback_request_preserves_unbounded_selection(self):
        from app import api, jobs
        request = api.YtpmvMixPreviewRequest(midi_id='midi', parts=[],
            options={'synth': True, 'chip': 'sms'}, seconds=None)
        with patch.object(jobs, '_ensure_worker'):
            response = api.ytpmv_preview_mix(request)
        import json
        job = jobs.get(json.loads(response.body)['id'])
        self.addCleanup(lambda: jobs.cancel(job.id))
        self.assertIsNone(job.params['seconds'])

    def test_seek_includes_held_notes_and_matches_full_mix_with_haas_and_variety(self):
        self.stub_media()
        choices = [samples.build('hello', corpus=slug) for slug in self.entries]
        part = self.parts[0]
        part.notes = [Note(0, .8, 60, 100), Note(.9, .2, 62, 100)]
        settings = render._merge(render._default_settings_for(part, {'text': 'hello'}),
                                 {'text': 'hello', 'mode': 'perfect', 'haas': True, 'pan': -.3})
        full = np.zeros((2 * SR, 2), np.float32)
        excerpt = np.zeros((SR, 2), np.float32)
        render._play(part, settings, choices, full, 0, 2, 1, 0, False, jitter=True, vary='rotate')
        render._play(part, settings, choices, excerpt, .25, 1.25, 1, 0, False, jitter=True, vary='rotate')
        self.assertGreater(float(np.abs(excerpt[:SR // 5]).max()), .01)
        np.testing.assert_allclose(excerpt, full[SR // 4:SR // 4 + SR], atol=1e-6)

    def test_queued_part_audition_is_bounded_and_uses_audio_worker(self):
        from app import api, jobs
        request = api.YtpmvMixPreviewRequest(midi_id='midi', part_id='alpha', parts=[],
            options={'synth': True, 'chip': 'sms'}, seconds=999)
        with patch.object(jobs, '_ensure_worker'):
            response = api.ytpmv_preview_mix(request)
        import json
        job = jobs.get(json.loads(response.body)['id'])
        self.addCleanup(lambda: jobs.cancel(job.id))
        self.assertEqual(job.params['seconds'], 15)
        self.assertEqual([(p['id'], p['mute']) for p in job.params['parts']], [('alpha',False), ('beta',True)])
        with patch.object(render, 'preview_mix', return_value={'audio_url':'/output/test.wav'}) as preview, \
                patch.object(render, 'render_ytpmv', side_effect=AssertionError('preview must not encode video')):
            jobs._run(job)
        preview.assert_called_once()
        self.assertEqual(job.status, 'done')
        self.assertEqual(job.result['audio_url'], '/output/test.wav')

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

        async def request(path, corpus, method='POST'):
            sent = []
            async def receive():
                return {'type': 'http.request', 'body': b'', 'more_body': False}
            async def send(message):
                sent.append(message)
            await main._Guard(endpoint)({'type': 'http', 'method': method, 'path': path,
                'scheme': 'http', 'headers': [(b'x-corpus', corpus.encode()),
                    (b'content-type', b'multipart/form-data; boundary=test')]}, receive, send)
            return sent[0]['status']

        for path in ('midi', 'sample', 'preview', 'render', 'preview-mix'):
            self.assertEqual(asyncio.run(request('/api/ytpmv/' + path, 'beta')), 200)
        self.assertEqual(seen, ['beta'] * 5)
        self.assertEqual(asyncio.run(request('/api/jobs/queued-preview', 'beta', 'DELETE')), 200)
        self.assertEqual(seen[-1], 'beta')
        self.assertEqual(asyncio.run(request('/api/edit/clip/1', 'beta')), 409)
        self.assertEqual(asyncio.run(request('/api/ytpmv/sample', 'missing')), 409)
        self.assertEqual(db.active()['slug'], 'alpha')

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for the real render check")
    def test_real_render_contains_two_corpus_videos_and_voices(self):
        for slug, colour, frequency in (("alpha", "red", 220), ("beta", "blue", 440)):
            path = Path(self.entries[slug]["dir"], "downloads", "same.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            f"color=c={colour}:s=160x90:r=25:d=0.6", "-f", "lavfi", "-i",
                            f"sine=frequency={frequency}:sample_rate={SR}:duration=0.6",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                            "-shortest", str(path)], check=True, capture_output=True)
        self.song.parts.append(Part(id='gamma', name='third', track=2, channel=2, program=0,
                                   notes=[Note(0, .1, 67, 100)], role='chords'))
        result = render.render_ytpmv({"midi_id": "midi", "parts": [
            {"id": "alpha", "text": "hello", "corpus": "alpha"},
            {"id": "beta", "text": "hello", "corpus": "beta"},
            {"id": "gamma", "text": "hello", "corpus": "alpha"}],
            "options": {"balance": False, "max_seconds": 1, "flip": False,
                        "flash": False, "dim": False}})
        self.assertEqual([p["corpus"] for p in result["parts"]], ["alpha", "beta", "alpha"])
        self.assertEqual(result['size'], [1920, 1080])
        audio_a, audio_b = [samples.build("hello", corpus=slug) for slug in self.entries]
        self.assertAlmostEqual(audio_a.voice.info.f0, 220, delta=8)
        self.assertAlmostEqual(audio_b.voice.info.f0, 440, delta=8)
        path = str(self.root / result["video_url"].lstrip("/"))
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             check=True, capture_output=True).stdout
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(1080, 1920, 3).astype(np.int16)
        # Fit keeps the complete 16:9 picture centered even in the wide
        # bottom cell, rather than zooming it until its sides disappear.
        for (x,y,w,h), channel in zip(render.grid_cells(3), [0,2,0]):
            tile = pixels[y:y+h, x:x+w]
            scale = min(w/16,h/9)
            expected_area = (16*scale)*(9*scale)/(w*h)
            self.assertAlmostEqual(float(np.mean(tile[:,:,channel] > 150)), expected_area, delta=.015)
            self.assertGreater(int(tile[h//2,w//2,channel]),150)
        self.assertEqual(result['options']['media_fit'],'contain')
        self.assertEqual(db.active()["slug"], "alpha")


if __name__ == "__main__":
    unittest.main()
