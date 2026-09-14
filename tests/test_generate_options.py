#!/usr/bin/env python3
"""The sentence generator's settings, held letters, and timeline. Plain asserts.

    python tests/test_generate_options.py

No corpus and no audio: the corpus lookups are stubbed, and everything here
is the arithmetic that decides what gets cut and how long it plays for.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.generate as g

failures: list[str] = []


def check(label, got, want, tol=None):
    ok = (abs(got - want) <= tol) if tol is not None else got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


# A pretend corpus: which words it has heard, and how often.
HEARD = {"long": 5, "so": 30, "good": 12, "god": 2, "cool": 3, "hmm": 4, "yes": 9,
         "too": 20, "soon": 6, "stop": 3, "no": 40, "way": 8}
g._ensure_cache = lambda: None
g._clips_by_word_cache = {w: [{}] * n for w, n in HEARD.items()}
g._in_dictionary = lambda w: w in HEARD

print("settings")
o = g.generation_options({})
check("every setting has its default", o["speed"], 1.0)
check("including the word gap", o["word_gap"], g._WORD_GAP)
check("and clean takes are on", o["clean_takes"], True)
o = g.generation_options({"speed": 9, "pitch": -40, "stretch": "nonsense", "phrases": 0})
check("too fast is held to the fastest", o["speed"], 2.0)
check("too low is held to the lowest", o["pitch"], -12.0)
check("nonsense is the default", o["stretch"], 0.09)
check("a falsy switch turns it off", o["phrases"], False)
check("NaN is the default", g.generation_options({"speed": float("nan")})["speed"], 1.0)

print("held letters")
check("loooong is long, its o held three letters longer", g.unstretch("loooong"), ("long", [(1, 1, 3)]))
check("punctuation after it is fine", g.unstretch("loooong!")[0], "long")
check("goooood is good, which the corpus says more than god", g.unstretch("goooood"), ("good", [(1, 2, 3)]))
check("hmmmm keeps the two m's hmm is spelt with", g.unstretch("hmmmm"), ("hmm", [(1, 2, 2)]))
check("two of a letter in a real word is spelling, not a stretch", g.unstretch("soon"), None)
check("a held stop still means its word, with nothing held", g.unstretch("stoppp"), ("stop", []))
check("two held runs in one word", g.unstretch("nooo")[0], "no")
check("anything that is not letters is left alone", g.unstretch("4x4"), None)
toks = g.tokenize_full("sooo gooood. too soon")
check("the tokeniser gives the word to look up", [t["word"] for t in toks], ["so", "good", "too", "soon"])
check("and remembers how it was typed, for the caption", toks[1]["shown"], "gooood")
check("a full stop still ends a sentence", toks[1]["ends"], True)
check("unstretched words carry no stretch", toks[2]["stretch"], [])
check("and the old tuples are unchanged", g.tokenize_marked("sooo gooood.")[1], ("good", True, False, False))

print("holding part of a clip")
seg = {"word": "long", "source_file": "x.mp4", "start_time": 10.0, "end_time": 10.4,
       "prev_end": 9.8, "next_start": 10.6, "fade_in": None, "fade_out": None, "pause_after": 0.03}
pieces = g._hold(seg, 10.1, 10.25, 3.0)
check("three pieces: before, held, after", len(pieces), 3)
check("they cover the clip end to end", (pieces[0]["start_time"], pieces[0]["end_time"], pieces[1]["end_time"], pieces[2]["end_time"]),
      (10.0, 10.1, 10.25, 10.4))
check("only the middle is held", [p.get("stretch") for p in pieces], [None, 3.0, None])
check("the joins are butted, with no fades", (pieces[0]["fade_out"], pieces[1]["fade_in"], pieces[1]["fade_out"], pieces[2]["fade_in"]),
      (0.0, 0.0, 0.0, 0.0))
check("the pause stays at the end of the word", (pieces[0]["pause_after"], pieces[2]["pause_after"]), (0.0, 0.03))
check("the word keeps its own lead-in and tail", (pieces[0]["prev_end"], pieces[2]["next_start"]), (9.8, 10.6))
w0, w1, w2 = (g.extract_window(p) for p in pieces)
check("the held piece is cut exactly", w1, (10.1, 10.25))
check("and the pieces meet with no gap", (w0[1], w2[0]), (10.1, 10.25))
check("holding the whole clip leaves one piece", len(g._hold(seg, 10.0, 10.4, 2.0)), 1)
check("holding nothing leaves it alone", g._hold(seg, 10.1, 10.25, 1.0), [seg])

print("where the held letter is")
import app.forced_align as fa
# The aligner's letters for "long": l, o, n, g -- with the o marked in a
# couple of frames, as CTC does, though it sounds until the n.
fa.char_times = lambda c: [("L", 0.00, 0.04), ("O", 0.06, 0.09), ("N", 0.22, 0.25), ("G", 0.30, 0.33)]
check("from the letter's start to the next letter's, not the aligner's end",
      g._letter_span(seg, "long", 1, 1), (10.06, 10.22))
check("the last letter runs to the end of the word", g._letter_span(seg, "long", 3, 1), (10.30, 10.4))
held = g._stretch_word([dict(seg)], "long", [(1, 1, 3)], 0.1)
check("a held word is split around the letter", [round(p["start_time"], 3) for p in held], [10.0, 10.06, 10.22])
mid = held[1]
check("and the letter plays for its length plus 0.1s a letter", (mid["end_time"] - mid["start_time"]) * mid["stretch"], 0.46, tol=1e-6)
tiny = g._stretch_word([dict(seg)], "long", [(1, 1, 40)], 0.1)
check("no piece is held harder than the most a stretch can take", max(p.get("stretch") or 1 for p in tiny) <= g._MAX_STRETCH, True)
added = sum((p["end_time"] - p["start_time"]) * ((p.get("stretch") or 1) - 1) for p in tiny)
check("and no word grows by more than the cap", added <= g._MAX_STRETCH_ADD + 1e-6, True)

print("lengths and the timeline")
opts = g.generation_options({"speed": 2.0})
span, sounding, total = g.segment_durations({"stretch": 3.0, "pause_after": 0.2}, (1.0, 1.2), opts)
check("a held piece sounds for its span times the hold", sounding, 0.6, tol=1e-9)
check("and lasts that plus its pause, at the chosen speed, in whole frames", total, round((0.8 / 2.0) * g._FPS) / g._FPS)
check("atempo is chained past its range", g._atempo_chain(0.25), ["atempo=0.5", "atempo=0.50000"])
check("both ways", g._atempo_chain(3.0), ["atempo=2.0", "atempo=1.50000"])
segs = [dict(seg, _tok=0, pause_after=0.0, next_start=10.4), {"word": "", "source_file": "x.mp4", "start_time": 20.0, "end_time": 20.5,
        "prev_end": 20.0, "next_start": 20.5, "fade_in": 0.0, "fade_out": 0.0},
        dict(seg, start_time=30.0, end_time=30.3, prev_end=29.0, next_start=None, _tok=1)]
g._sound_ends = lambda f, s, e: e
tl = g.timeline(segs, g.generation_options({}))
check("each word gets a place in the timeline", [w["i"] for w in tl], [0, 1])
check("the second starts after the first and the pause between", tl[1]["start"] > tl[0]["end"] + 0.4, True)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
