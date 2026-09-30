#!/usr/bin/env python3
"""A held note keeps singing, on pitch, however long it is.

    python tests/test_hold.py

A ~0.5 s synthetic vowel (glottal pulses through two formants, a little pitch
glide, noise consonants at both ends) is sung at 220 Hz for 5 s. Past the
sample's own length the note used to go silent (the stretch was capped) and
the stretched glides/gaps were heard as pitch jumps.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.ytpmv import pitch as p

SR = p.SR
rng = np.random.default_rng(1)


def formant(x, f, bw):
    r = np.exp(-np.pi * bw / SR)
    a1, a2 = -2 * r * np.cos(2 * np.pi * f / SR), r * r
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = x[i] - a1 * (y[i - 1] if i > 0 else 0) - a2 * (y[i - 2] if i > 1 else 0)
    return y


n_v = int(0.40 * SR)
f0 = 150.0 * 2 ** (np.linspace(0.6, 0.0, n_v) / 12)          # gliding down 60 cents
ph = np.cumsum(f0 / SR)
pulses = (np.diff(np.floor(ph), prepend=0) > 0).astype(np.float64)
vowel = formant(formant(pulses, 700, 90), 1100, 110)
vowel *= 0.5 / np.abs(vowel).max()
noise = lambda: 0.05 * rng.standard_normal(int(0.05 * SR))
src = np.concatenate([noise(), vowel, noise()]).astype(np.float32)

voice = p.Voice(src)
target, dur = 220.0, 5.0
y, w = voice.render(target, dur, "perfect")

fails = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   ({detail})" if detail else ""))
    if not ok:
        fails.append(label)


win = int(0.1 * SR)
rms = np.sqrt((y[:len(y) // win * win].reshape(-1, win) ** 2).mean(axis=1))
body = rms[2:-1]
check("no silent gap in the held note", body.min() > 0.3 * np.median(body),
      f"min {body.min():.4f} vs median {np.median(body):.4f}")

tr = p.track_f0(y[int(0.6 * SR):int(4.5 * SR)])
c = 1200 * np.log2(tr.f0[tr.voiced] / target)
check("held part is voiced throughout", tr.voiced.mean() > 0.95, f"{tr.voiced.mean():.2f}")
check("held part stays within 20 cents", np.abs(c).max() < 20, f"max {np.abs(c).max():.1f}c")

t = np.linspace(0, dur, 2001)
s = np.array([w.src(x) for x in t])
check("src stays inside the sample", s.min() >= 0 and s.max() <= voice.duration + 1e-6)
check("src ends where the word ends", abs(w.src(dur) - voice.duration) < 0.02,
      f"{w.src(dur):.3f} vs {voice.duration:.3f}")
held = (t > w.v0) & (t < w.v0 + w.hold)
check("src holds inside the nucleus", (s[held] >= w.v0 - 1e-9).all() and (s[held] <= w.v1 + 1e-9).all())
check("src is continuous", np.abs(np.diff(s)).max() < 0.01, f"{np.abs(np.diff(s)).max():.4f}")

print()
sys.exit(1 if fails else 0)
