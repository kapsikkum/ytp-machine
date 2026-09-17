#!/usr/bin/env python3
"""Measuring how loud a part is, and moving it. Plain asserts.

    python tests/test_master.py

Needs numpy. A loudness meter is easy to write and easy to write wrongly,
and a wrong one still returns a number that looks like decibels -- so these
check it against signals whose loudness is known from arithmetic rather than
from this implementation.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ytpmv import master
from app.ytpmv.pitch import SR

failures: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


def near(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}   {got:+.2f} (want {want:+.2f} ±{tol})")
    if not ok:
        failures.append(label)


rng = np.random.default_rng(20260913)


def sine(db, hz=1000.0, secs=4.0):
    amp = 10.0 ** (db / 20.0)
    return amp * np.sin(2 * np.pi * hz * np.arange(int(SR * secs)) / SR)


print("the meter reads what arithmetic says it should")
# A sine's RMS is 3.01 dB under its peak, and the weighting is flat at 1 kHz,
# so a sine peaking at -20 dBFS has to land within a whisker of -23 LUFS.
near("a 1 kHz sine peaking at -20 dBFS", master.loudness(sine(-20.0)), -23.01, 0.5)
near("the same sine 12 dB down", master.loudness(sine(-32.0)), -35.01, 0.5)
# Loudness is a ratio, so doubling the amplitude must move it by exactly 6.02.
a = master.loudness(sine(-30.0))
b = master.loudness(sine(-24.0))
near("twice the amplitude is 6 dB more, exactly", b - a, 6.02, 0.05)
check("silence has no loudness", math.isinf(master.loudness(np.zeros(SR))), True)
check("and neither has an empty buffer", math.isinf(master.loudness(np.zeros(0))), True)

print("the weighting")
# The shelf lifts the top by about 4 dB and the high-pass throws away the
# bottom. Both are the point of using this measure rather than plain RMS.
low = master.loudness(sine(-20.0, hz=30.0))
mid = master.loudness(sine(-20.0, hz=1000.0))
high = master.loudness(sine(-20.0, hz=8000.0))
# The high-pass is gentle, not a brick wall: 8.8 dB off at 30 Hz, which is
# enough to stop rumble deciding a balance without pretending it is absent.
near("30 Hz is discounted", low - mid, -8.8, 1.5)
near("8 kHz is lifted by the shelf", high - mid, 3.6, 1.0)
# Straight off the filter, where the standard's own numbers can be checked.
curve = 20 * np.log10(np.abs(master._k_response(
    np.array([38.13, 1000.0, 16000.0]), SR)))
near("the shelf tops out at exactly +4 dB", float(curve[2]), 4.0, 0.05)
near("and the high-pass is 6 dB down at its corner", float(curve[0]), -6.0, 0.2)

print("gating: what stops a sparse part being cranked")
hit = np.concatenate([
    rng.standard_normal(int(0.25 * SR)) * np.exp(-np.arange(int(0.25 * SR)) / (0.06 * SR)) * 0.5,
    np.zeros(int(0.05 * SR))])
busy = np.tile(hit, 40)
sparse = np.zeros(len(busy))
for i in range(4):                       # the same hit, four times in the same span
    sparse[i * len(busy) // 4: i * len(busy) // 4 + len(hit)] = hit
rms = lambda x: 20 * math.log10(math.sqrt(float((x ** 2).mean())))
check("plain RMS says the sparse one is far quieter", rms(busy) - rms(sparse) > 8.0, True)
near("the gated measure says they are about as loud while playing",
     master.loudness(busy) - master.loudness(sparse), 0.0, 3.0)

print("moving a part to where its job wants it")
check("a part already on target is left alone",
      round(master.gain_for(master.REFERENCE, "lead"), 3), 1.0)
# Near enough is left alone on purpose: the gaps between parts of one kind
# are usually an arrangement, and closing them is not mastering.
check("so is one that is only a little out",
      round(master.gain_for(master.REFERENCE - master.DEADBAND + 0.5, "lead"), 3), 1.0)
near("one well too quiet is brought part of the way up",
     20 * math.log10(master.gain_for(master.REFERENCE - 9.0, "lead")),
     (9.0 - master.DEADBAND) * master.STRENGTH, 0.01)
near("one well too loud is brought all the way down, bar a little leeway",
     20 * math.log10(master.gain_for(master.REFERENCE + 9.0, "lead")),
     -(9.0 - master.DEADBAND_DOWN) * master.STRENGTH_DOWN, 0.01)
check("and a correction is never the whole of the error",
      abs(20 * math.log10(master.gain_for(master.REFERENCE - 9.0, "lead"))) < 9.0, True)
near("a trim the person asked for is added on top",
     20 * math.log10(master.gain_for(master.REFERENCE, "lead", trim_db=-3.0)), -3.0, 0.01)
# Each role is judged against its own place, not against the lead's.
for role in ("bass", "rhythm", "drums:hats", "drums:cymbals"):
    on_target = master.REFERENCE + master.TARGETS[role]
    check(f"{role} sitting where it belongs is left alone",
          round(master.gain_for(on_target, role), 3), 1.0)
    check(f"{role} well above its place is brought down",
          master.gain_for(on_target + 9.0, role) < 1.0, True)
check("a part nobody can measure is not moved at all",
      master.gain_for(-math.inf, "lead"), 1.0)
check("and nothing is ever moved further than the limit",
      round(20 * math.log10(master.gain_for(-200.0, "lead")), 1), master.MAX_MOVE)
print("balancing a whole song against itself")
sr = SR
t = np.arange(8 * sr) / sr
tone = lambda hz, db: (10 ** (db / 20) * np.sin(2 * np.pi * hz * t)).astype(np.float32)
# A song 30 dB under the fixed reference, with its rhythm part 10 dB too loud.
lead, loud = tone(440, -50), tone(330, -45)
got = master.balance([{"id": "lead", "role": "lead", "power": master.block_power(lead)},
                      {"id": "pad", "role": "rhythm", "power": master.block_power(loud)}])
after = {k: v[0] + 20 * math.log10(v[1]) for k, v in got.items()}
check("a quiet song is balanced against its own level, not a fixed one",
      abs((after["lead"] - after["pad"]) - (-master.TARGETS["rhythm"])) < master.DEADBAND + 0.5, True)
# Two identical pads playing together share their place.
pads = [{"id": f"p{i}", "role": "chords", "power": master.block_power(tone(220, -30))} for i in range(2)]
shared = master.balance(pads + [{"id": "lead", "role": "lead", "power": master.block_power(tone(440, -30))}])
check("parts playing together share their role's place",
      shared["p0"][1] < 1.0 and abs(shared["p0"][1] - shared["p1"][1]) < 1e-6, True)

print("mastering")
mix = np.stack([tone(220, -20), tone(221, -20)], axis=1)
mix[sr:sr + 50] = 0.99                               # a spike
master.finish(mix)
check("the mix is limited under the ceiling", float(np.abs(mix).max()) <= master.CEILING + 1e-3, True)
near("and sits at the target loudness", master.loudness(mix), master.TARGET_LUFS, 1.0)

check("a kit sits well below a melody, as a kit measures",
      master.TARGETS["drums:hats"] < master.TARGETS["lead"] - 15.0, True)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
