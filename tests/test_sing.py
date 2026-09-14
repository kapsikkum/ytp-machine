"""app/sing.py: tunes, and a sung clip actually landing on its note."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from app import sing
from app.ytpmv import pitch

fails = []
def check(name, got, want):
    ok = got == want
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok: fails.append(name)

check("arch goes up and back", sing.degrees(9, 0), [0, 1, 2, 3, 4, 3, 2, 1, 0])
check("every shape gives a note a word", [len(sing.degrees(13, s)) for s in range(5)], [13] * 5)
check("wander stays in the octave", all(0 <= d <= 7 for d in sing.degrees(200, 3, seed=5)), True)
n = sing.notes(8, 2, 1, centre=50.0)
check("rising in D starts on a D", n[0] % 12, 2)
check("and the tune straddles the voice", n[0] <= 50 <= n[-1], True)

# A gliding synthetic vowel, sung onto A3.
sr = pitch.SR
t = np.arange(int(0.5 * sr)) / sr
f = 120 + 30 * t
x = (np.sin(2 * np.pi * np.cumsum(f) / sr) + 0.4 * np.sin(4 * np.pi * np.cumsum(f) / sr)).astype(np.float32) * 0.5
y, _ = pitch.Voice(x).render(pitch.midi_to_hz(57), 0.5, "perfect", stretch=False)
got = pitch.detect_pitch(y).f0
check("a sung word lands on its note (within 20 cents)", abs(1200 * np.log2(got / 220.0)) < 20, True)

# Tune strength, vibrato and a raised formant, on the same vowel.
v = pitch.Voice(x)
half = sing.sung(v, 220.0, len(x), amount=0.5)
f_half = pitch.detect_pitch(half).f0
check("half strength lands between the voice and the note", 125 < f_half < 215, True)
vib = sing.sung(v, 220.0, len(x), vibrato=1.0)
tr = pitch.track_f0(vib)
spread = np.ptp(1200 * np.log2(tr.f0[tr.voiced][40:] / 220.0))
check("vibrato wobbles the note by about a semitone each way", 100 < spread < 400, True)
cute = sing.sung(v, 220.0, len(x), formant=1.3)
check("cute keeps the note", abs(1200 * np.log2(pitch.detect_pitch(cute).f0 / 220.0)) < 30, True)
S = lambda y: np.abs(np.fft.rfft(y)); fr = np.fft.rfftfreq(len(x), 1 / sr)
cen = lambda y: float((S(y) * fr).sum() / S(y).sum())
check("and brightens the voice", cen(cute) > cen(sing.sung(v, 220.0, len(x))) * 1.1, True)

check("^+2 is relative", sing.parse_mark("+2"), ("rel", 2.0))
check("^A3 is a note", sing.parse_mark("A3"), ("abs", 57.0))
check("^C#4 too", sing.parse_mark("C#4"), ("abs", 61.0))
check("a melody is moved by octaves to the voice", sing.fit([72, 74, 76], 4, 50.0), [48.0, 50.0, 52.0, 48.0])
import app.generate as g
toks = g.tokenize_full("hello^+3 world^A3. plain")
check("the markup comes off the word", [t["word"] for t in toks], ["hello", "world", "plain"])
check("and is kept as its note", [t["note"] for t in toks], [("rel", 3.0), ("abs", 57.0), None])
check("string options are cleaned", g.generation_options({"sing_midi": "ab/../c d"})["sing_midi"], "ab..cd")

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
