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

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
