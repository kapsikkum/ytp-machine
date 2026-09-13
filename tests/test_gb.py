#!/usr/bin/env python3
"""The Game Boy's sound chip. Plain asserts.

    python tests/test_gb.py

Needs numpy. Figures from the gbdev Pan Docs for the channels and SameBoy
for the output capacitor.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ytpmv import gb
from app.ytpmv.pitch import SR, midi_to_hz

failures: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


def near(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}   {got:+.3f} (want {want:+.3f} ±{tol})")
    if not ok:
        failures.append(label)


def peak_hz(x, lo, hi, n=1 << 16):
    s = np.pad(x, (0, max(0, n - len(x))))[:n] * np.hanning(n)
    sp = np.abs(np.fft.rfft(s))
    fr = np.fft.rfftfreq(n, 1.0 / SR)
    k = int(np.argmax(sp * ((fr >= lo) & (fr < hi))))
    a, b, c = float(sp[k - 1]), float(sp[k]), float(sp[k + 1])
    den = a - 2 * b + c
    return (k + (0.5 * (a - c) / den if abs(den) > 1e-20 else 0.0)) * SR / n


print("the channels' ranges")
# Pan Docs: period $500 is 170.67 Hz on a pulse channel and 85.333 on the wave.
near("pulse period $500 is 170.67 Hz", 131072.0 / (2048 - 0x500), 170.667, 0.01)
near("wave period $500 is 85.33 Hz", 65536.0 / (2048 - 0x500), 85.333, 0.01)
near("so the pulses bottom out at 64 Hz", gb.floor_hz("pulse"), 64.0, 0.01)
near("and the wave an octave under", gb.floor_hz("wave"), 32.0, 0.01)
check("noise has no floor", gb.floor_hz("noise"), 0.0)

print("the shift register")
# Fifteen bits, or seven in short mode, both maximal length.
check("the long register comes round after 32767 steps", len(gb.lfsr_bits(False)), 32767)
check("the short one after 127", len(gb.lfsr_bits(True)), 127)

print("the DAC and capacitor")
check("digital 0 is analog +1", float(gb.dac(np.array([0]))[0]), 1.0)
check("digital 15 is analog -1: the slope is negative", float(gb.dac(np.array([15]))[0]), -1.0)
k = gb.CHARGE ** (gb.CLOCK / SR)
near("the capacitor's corner is about 28 Hz", -math.log(k) * SR / (2 * math.pi), 28.0, 1.0)
held = gb.render_note(gb.VOICES["gb-pulse-full"], 110.0, 0.8)
cycle = held[int(0.5 * SR): int(0.5 * SR) + int(SR / 110.0)]
near("a held square is pulled to the centre", float(cycle.mean()), 0.0, 0.01)
check("and sags from where it started",
      float(np.abs(held[:400]).max()) > float(np.abs(held[int(0.6 * SR):int(0.6 * SR) + 400]).max()), True)
# A note must not start with a thump: the capacitor is charged between notes.
first = float(np.abs(held[:40]).max())
check("no step at the front of a note", first <= float(np.abs(held[:2000]).max()) + 1e-6, True)

print("tuning")
for voice, midis in (("gb-pulse", (45, 57, 69)), ("gb-wave", (33, 45, 57))):
    for m in midis:
        hz = midi_to_hz(m)
        got = peak_hz(gb.render_note(gb.VOICES[voice], hz, 0.4), hz * 0.75, hz * 1.3)
        near(f"{voice} at MIDI {m}", 1200 * math.log2(got / hz), 0.0, 6.0)

print("the envelope")
env = gb.envelope(SR, SR, start=15, pace=1)
check("it steps at 64 Hz, 15 down to 0 in a quarter second", len(np.unique(env[: SR // 4])), 16)
check("and a pace of 0 holds", len(np.unique(gb.envelope(SR, SR, 9, 0))), 1)

print("dynamics")
loud = float(np.abs(gb.render_note(gb.VOICES["gb-pulse"], 220.0, 0.3, level=1.0)).max())
soft = float(np.abs(gb.render_note(gb.VOICES["gb-pulse"], 220.0, 0.3, level=30 / 127)).max())
check("a soft note is well under a hard one", soft < loud * 0.4, True)

print("every voice")
for name, voice in gb.VOICES.items():
    x = gb.render_note(voice, 220.0, 0.3)
    check(f"{name} is finite and audible", bool(np.isfinite(x).all() and np.abs(x).max() > 0.1), True)
    check(f"{name} does not clip", bool(np.abs(x).max() < 1.0), True)
check("a bass part gets the wave channel", gb.voice_for("bass").name, "gb-wave")
check("drums come off the noise register", gb.voice_for("drums:snare").name, "gb-snare")
out = gb.pcm(np.sin(np.linspace(0, 200, SR // 2)).astype(np.float32))
check("the voice through the wave table comes back finite", bool(np.isfinite(out).all()), True)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
