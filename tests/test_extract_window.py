#!/usr/bin/env python3
"""What span of audio each clip is pulled from, as plain asserts.

    python tests/test_extract_window.py

This is the calculation that decides what a clip sounds like, and until now
the only way to exercise it was to render a video and listen. It got the
"mother" in Tomato 1 wrong -- the clip ends at 33.122, the next word starts at
33.182, and the extraction ran to 33.282, so every sentence using it said
"mother like".

The rules being checked:

  - never start inside the previous word, never end inside the next one;
  - never come out shorter than the stored clip, whatever the padding says;
  - a word ending in a sonorant may reach for its decay, but not past the
    neighbour;
  - a unit cut to a phoneme boundary comes out exactly as cut;
  - a clip edited by hand is not extended at all.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.generate as g

failures: list[str] = []


def clip(**kw):
    base = {"word": "cat", "source_file": "nope.mp4",
            "start_time": 10.0, "end_time": 10.4,
            "prev_end": None, "next_start": None}
    base.update(kw)
    return base


def check(label, got, want, tol=1e-6):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want!r}")
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got!r}")


# Reading real audio is not the point here, and there is none: pin the
# sonorant walk to its maximum, which is the worst case and the one that
# caused the fault.
g._sound_ends = lambda f, s, e: e + g._SONORANT_MAX

# ── the reported fault ─────────────────────────────────────────────────────
mother = clip(word="mother", start_time=32.885, end_time=33.122,
              prev_end=32.8, next_start=33.182)
start, end = g.extract_window(mother)
check("mother stops before 'like' starts", end <= 33.182, True)
check("and keeps the whole word", end >= 33.122, True)
check("landing on the gap minus the tail gap", end, 33.182 - g._TAIL_GAP)

# ── the padding still works where there is room ────────────────────────────
roomy = clip(word="mother", start_time=10.0, end_time=10.4, next_start=12.0)
_s, end = g.extract_window(roomy)
# Whichever is longer: the ordinary tail pad, or the walk to where the sound
# actually stops.
check("a sonorant with room reaches its decay", end,
      10.4 + max(g._PAD_END, g._SONORANT_MAX))

plain = clip(word="cat", start_time=10.0, end_time=10.4, next_start=12.0)
_s, end = g.extract_window(plain)
check("a plain word takes the ordinary pad", end, 10.4 + g._PAD_END)

# ── never shorter than the clip ────────────────────────────────────────────
tight = clip(word="cat", start_time=10.0, end_time=10.4, next_start=10.41)
_s, end = g.extract_window(tight)
check("a close neighbour never trims the word", end, 10.4)

overlap = clip(word="mother", start_time=10.0, end_time=10.4, next_start=10.3)
_s, end = g.extract_window(overlap)
check("nor does an overlapping one", end, 10.4)

# ── the start side ─────────────────────────────────────────────────────────
lead = clip(start_time=10.0, end_time=10.4, prev_end=9.0)
start, _e = g.extract_window(lead)
check("lead-in with room", start, 10.0 - g._PAD_START)

crowded = clip(start_time=10.0, end_time=10.4, prev_end=9.99)
start, _e = g.extract_window(crowded)
check("never starts inside the word before", start, 10.0)

check("nor before the file begins", g.extract_window(
    clip(start_time=0.01, end_time=0.4))[0], 0.0)

# ── cut units come out exactly as cut ──────────────────────────────────────
unit = clip(word="mother", start_time=10.0, end_time=10.06, subword=True,
            prev_end=10.0, next_start=10.06)
start, end = g.extract_window(unit)
check("a butt-joined unit keeps its start", start, 10.0)
check("and its end, with no sonorant nursing", end, 10.06)

short = clip(word="cat", start_time=10.0, end_time=10.03, subword=True,
             prev_end=10.0, next_start=10.03)
check("and is not padded up to a minimum length",
      g.extract_window(short)[1], 10.03)

# ── a hand-set boundary is final ───────────────────────────────────────────
edited = clip(word="mother", start_time=10.0, end_time=10.4, edited=1,
              next_start=12.0)
_s, end = g.extract_window(edited)
check("an edited clip gets no tail at all", end, 10.4)

check("nor any lead-in", g.extract_window(edited)[0], 10.0)

# A short edited clip stays short: it is 20ms because somebody made it 20ms.
edited_short = clip(word="mother", start_time=10.0, end_time=10.02, edited=1,
                    next_start=12.0)
es, ee = g.extract_window(edited_short)
check("a short edited clip is not padded out", (round(es, 4), round(ee, 4)),
      (10.0, 10.02))

# ── a vowel-initial neighbour is given a wider berth ───────────────────────
bleed = clip(word="hot", start_time=10.0, end_time=10.4, next_start=10.5,
             bleed_risk=True)
_s, end = g.extract_window(bleed)
check("bleed risk keeps further clear", end, max(10.4, 10.5 - g._BLEED_TAIL))

# ── the window is always usable ────────────────────────────────────────────
for name, c in [("plain", plain), ("mother", mother), ("unit", unit),
                ("overlap", overlap), ("edited", edited)]:
    s, e = g.extract_window(c)
    if not (e > s and s >= 0):
        failures.append(f"{name}: degenerate window {s}->{e}")

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
