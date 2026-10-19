"""Performance shortcuts preserve pronunciation, instrument and voice semantics."""
from contextlib import ExitStack
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import phonemes
from app.ytpmv import render, samples, tone
from app.ytpmv.music import Part, Note, Song


class PerformanceParityTests(unittest.TestCase):
    def test_dictionary_keeps_all_variants_order_and_stress_rules(self):
        entries = [("word", ["W", "ER1", "D"]), ("other", ["AH0", "DH", "ER2"]),
                   ("word", ["W", "AO2", "R", "D"]), ("digits", ["AH12", "X٣"])]
        cmu = SimpleNamespace(entries=Mock(return_value=entries))
        corpus = ModuleType("nltk.corpus")
        corpus.cmudict = cmu
        with patch.dict(sys.modules, {"nltk.corpus": corpus}):
            actual = phonemes._load_cmudict()
        expected = {}
        for word, pronunciation in entries:
            expected.setdefault(word, []).append([re.sub(r"\d+", "", p) for p in pronunciation])
        self.assertEqual(actual, expected)
        self.assertEqual(list(actual), list(expected))
        actual["word"][0].append("S")
        self.assertEqual(entries[0][1], ["W", "ER1", "D"])

    def test_every_chip_note_is_bit_identical_and_cache_results_are_independent(self):
        y = np.full(1103, .375, np.float32)
        cache = tone.SynthCache()
        for name in sorted(tone.SYNTH):
            with self.subTest(tone=name):
                kwargs = dict(hz=247.1234567, level=.63, program=48)
                expected = tone.apply(y, name, **kwargs)
                self.assertEqual(cache.apply(y, name, **kwargs).tobytes(), expected.tobytes())
                changed = cache.apply(y, name, **kwargs)
                changed[:] = 99
                self.assertEqual(cache.apply(y, name, **kwargs).tobytes(), expected.tobytes())
                self.assertLessEqual(cache.bytes, cache.budget)

    def test_exact_key_covers_duration_peak_pitch_velocity_patch_key_rate_and_model(self):
        y = np.full(2205, .5, np.float32)
        cache = tone.SynthCache()
        variants = [(y,"opl3",{}), (y,"opl3",{"hz":220.00000001}),
                    (y,"opl3",{"hz":220.00000002}), (y,"opl3",{"level":.5}),
                    (y,"opl3",{"program":1}), (y,"opl3",{"key":38}),
                    (y,"opl3",{"sr":16000}), (y,"opl3",{"model":"8580"}),
                    (y[:1000],"opl3",{}), (y*.6,"opl3",{})]
        with patch.object(tone,"apply", wraps=tone.apply) as apply:
            for audio, name, settings in variants:
                kwargs = {"hz":220.,"level":.8,"program":0,**settings}
                cache.apply(audio,name,**kwargs)
                cache.apply(audio,name,**kwargs)
            self.assertEqual(apply.call_count, len(variants))

    def test_effects_silence_and_nonfinite_input_do_not_reuse_waveforms(self):
        cache = tone.SynthCache()
        for name in ("clean", "sb16-pcm", "opl3"):
            for audio in (np.zeros(16,np.float32), -np.zeros(16,np.float32)):
                self.assertEqual(cache.apply(audio,name).tobytes(),tone.apply(audio,name).tobytes())
        a = np.linspace(-.5,.5,1103,dtype=np.float32)
        b = a[::-1].copy()
        for audio in (a,b):
            self.assertEqual(cache.apply(audio,"sb16-pcm").tobytes(),tone.apply(audio,"sb16-pcm").tobytes())
        for value in (float("nan"),float("inf")):
            audio = np.full(16,value,np.float32)
            with np.errstate(invalid="ignore"):
                self.assertEqual(cache.apply(audio,"opl3").tobytes(),tone.apply(audio,"opl3").tobytes())

    def test_byte_budget_eviction_and_oversize_notes_preserve_output(self):
        audio = np.full(2205,.5,np.float32)
        cache = tone.SynthCache(budget=1024)
        expected = tone.apply(audio,"opl3",hz=220)
        for _ in range(2):
            self.assertEqual(cache.apply(audio,"opl3",hz=220).tobytes(),expected.tobytes())
        self.assertEqual(cache.bytes,0)
        cache = tone.SynthCache(budget=expected.nbytes+100)
        first = cache.apply(audio,"opl3",hz=220)
        cache.apply(audio,"opl3",hz=330)
        self.assertEqual(len(cache.notes),1)
        self.assertEqual(cache.apply(audio,"opl3",hz=220).tobytes(),first.tobytes())

    def test_program_tables_remain_independent_between_parts(self):
        parts = [Part(str(i),str(i),i,i,0,[Note(0,.1,60,100)],role="rhythm") for i in range(3)]
        song = Song(parts,.2,120,(4,4),"C",1)
        with ExitStack() as stack:
            for module,name,value in ((render,"load_song",song),(render.recommend,"recommend",{}),
                    (samples,"corpus_pool",{}),(samples,"noises",{}),(samples,"units",{}),
                    (render,"active",{"name":"Profile","slug":"profile"})):
                stack.enter_context(patch.object(module,name,return_value=value))
            result = render.analyse("profile")
        a,b = result["parts"][:2]
        self.assertEqual(a["program_tones"],b["program_tones"])
        a["program_tones"]["opl3"][0] = "changed"
        self.assertEqual(b["program_tones"]["opl3"][0],"opl3")

    def test_bulk_voice_lookup_scans_once_and_preserves_default_and_errors(self):
        entry = {"slug":"alpha","name":"Alpha"}
        with patch.object(samples,"active",return_value=entry), \
                patch.object(samples,"list_corpora",return_value=[entry]) as scan:
            payload = [{"id":str(i),"corpus":"alpha"} for i in range(300)]
            self.assertEqual(render.prepare_parts(payload),payload)
            scan.assert_called_once()
            self.assertEqual(render.prepare_parts([{"id":"default"}]),[{"id":"default","corpus":"alpha"}])
            for choice in ("missing", "", 42, []):
                with self.subTest(choice=choice):
                    with self.assertRaises(samples.SampleError):
                        render.prepare_parts([{"id":"bad","corpus":choice}])


if __name__ == "__main__":
    unittest.main()
