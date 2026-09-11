#!/usr/bin/env python3
"""Pitch perfect, as plain asserts.

    python tests/test_pitch.py

The claim the YTPMV makes is that a word lands *on* the note -- not near it,
not on average. So the check is the one an ear makes: take a voice that
wobbles, sing it on a note, measure the result frame by frame, and require
every voiced frame to be there. A synthetic vowel stands in for speech: a
pulse train through two formants, with an 80-cent vibrato that a shift by the
average pitch would leave in place.

Needs numpy only.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.ytpmv import pitch as p

failures: list[str] = []
SR = p.SR


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def vowel(f0=150.0, dur=0.6, vib_cents=80.0, vib_hz=5.5, sr=SR):
    """A wobbly 'ah': harmonics shaped by formants near 700 and 1200 Hz."""
    t = np.arange(int(dur * sr)) / sr
    f = f0 * 2 ** (vib_cents / 1200.0 * np.sin(2 * np.pi * vib_hz * t))
    phase = 2 * np.pi * np.cumsum(f) / sr
    y = np.zeros_like(t)
    for k in range(1, 40):
        fk = k * f0
        if fk > sr / 2 - 1000:
            break
        amp = (1.0 / k) * (1.0 + 3.0 * math.exp(-((fk - 700) / 150) ** 2)
                           + 2.0 * math.exp(-((fk - 1200) / 200) ** 2))
        y += amp * np.sin(k * phase)
    env = np.minimum(1.0, np.minimum(t / 0.02, (dur - t) / 0.03))
    return (0.3 * y / np.abs(y).max() * env).astype(np.float32)


def cents(a, b):
    return 1200.0 * math.log2(a / b)


def voiced_cents(y, target):
    tr = p.track_f0(y)
    f = tr.f0[tr.voiced]
    return 1200.0 * np.log2(f / target)


print("detection")
src = vowel()
info = p.detect_pitch(src)
check("finds the median pitch of a wobbling vowel",
      info.f0 is not None and abs(cents(info.f0, 150.0)) < 10,
      f"{info.f0:.2f} Hz")
check("calls it D3", info.note == "D3", info.note)
check("sees the vibrato as instability", info.stability > 30,
      f"{info.stability:.1f} cents std")
check("sees a vowel as voiced", info.voiced_ratio > 0.9, f"{info.voiced_ratio:.2f}")

noise = (np.random.default_rng(1).standard_normal(SR // 2) * 0.2).astype(np.float32)
check("sees noise as unvoiced", p.detect_pitch(noise).voiced_ratio < 0.2,
      f"{p.detect_pitch(noise).voiced_ratio:.2f}")

import json
try:
    json.dumps(p.detect_pitch(noise).as_dict(), allow_nan=False)
    ok = True
except ValueError:
    ok = False
check("a sound with no pitch still describes itself as valid JSON", ok)

check("midi / hz / names agree", p.note_name(p.hz_to_midi(p.midi_to_hz(64))) == "E4")

print("perfect")
voice = p.Voice(src)
for target_midi in (57, 50, 45, 62):          # A3 up, D3 same, A2 down, D4 up
    target = p.midi_to_hz(target_midi)
    y, _w = voice.render(target, 0.5, "perfect")
    c = voiced_cents(y, target)
    med, spread = float(np.median(c)), float(np.std(c))
    check(f"lands on {p.note_name(target_midi)}", abs(med) < 10 and spread < 12,
          f"median {med:+.1f}c, spread {spread:.1f}c")
    check(f"{p.note_name(target_midi)} is exactly the note long", len(y) == int(round(0.5 * SR)))

y, w = voice.render(p.midi_to_hz(57), 1.8, "perfect")
check("a long note draws the vowel out", w.s > 2.0 and np.abs(y[int(1.5 * SR):]).max() > 0.01,
      f"stretch x{w.s:.2f}")
y, w = voice.render(p.midi_to_hz(57), 1.8, "perfect", stretch=False)
check("without stretch it ends with the sample",
      np.abs(y[int(0.7 * SR):]).max() == 0.0)

print("tape")
target = p.midi_to_hz(62)
y, w = voice.render(target, 0.4, "tape")
info2 = p.detect_pitch(y[: int(0.3 * SR)])
check("tape moves the median by the ratio", abs(cents(info2.f0, target)) < 15,
      f"{info2.f0:.1f} Hz vs {target:.1f}")
check("tape's warp runs at that speed", abs(w.rate - target / info.f0) < 1e-9)
check("tape keeps the wobble", info2.stability > 30, f"{info2.stability:.1f}c")

print("raw")
y, w = voice.render(None, 0.2, "raw")
check("raw is the sample, cut to the note",
      len(y) == int(0.2 * SR) and np.allclose(y[200:5000], src[200:5000], atol=1e-6))

print("warp")
w = p.Warp(0.1, 0.3, 3.0)
check("before the vowel is 1:1", w.src(0.05) == 0.05)
check("inside the vowel is slowed", abs(w.src(0.1 + 0.3) - 0.2) < 1e-9)
check("after the vowel is 1:1 again", abs(w.src(0.1 + 0.6 + 0.05) - 0.35) < 1e-9)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
