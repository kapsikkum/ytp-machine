#!/usr/bin/env python3
"""The emulated Mega Drive sound chip, as plain asserts.

    python tests/test_ym2612.py

Needs numpy. These check the things a wrong FM implementation gets wrong
quietly -- a chip that is a semitone out, or whose envelope rates mean
nothing, still makes a noise, and the noise is the only thing anyone looks
at. So each figure here is one that can be worked out from the datasheet
rather than read off this implementation.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ytpmv import ym2612 as chip
from app.ytpmv.pitch import SR, describe, midi_to_hz, track_f0

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


MUTE = chip.Op(tl=127, ar=31)


def only(op, alg=7, fb=0, **kw):
    """A patch that is one operator and three silent ones."""
    return chip.Patch("test", "", alg, fb, (op, MUTE, MUTE, MUTE), **kw)


def db_at(x, t, win=0.02):
    a = x[int(t * SR):int((t + win) * SR)]
    return 20 * math.log10(max(float(np.abs(a).max(initial=0.0)), 1e-9))


def spectrum(x, n=1 << 13):
    seg = np.pad(x, (0, max(0, n - len(x))))[:n] * np.hanning(n)
    return np.abs(np.fft.rfft(seg)), np.fft.rfftfreq(n, 1.0 / SR)


print("the clock")
near("53267 samples a second, being the NTSC clock over 144", chip.FM_RATE, 53267.0, 1.0)
near("the envelope ticks a third as often", chip.EG_RATE, chip.FM_RATE / 3.0, 0.01)

print("pitch")
sine = only(chip.Op(tl=0, ar=31))
# The pitch tracker is built for speech and stops at 600 Hz, so anything
# higher is measured off the spectrum instead of asked of it.
for midi in (40, 52, 60, 64):
    hz = midi_to_hz(midi)
    info = describe(track_f0(chip.render_note(sine, hz, 0.4)), 0.4)
    near(f"a note at MIDI {midi} comes out on the note", 1200 * math.log2((info.f0 or hz) / hz), 0.0, 6.0)

print("one operator is a sine and nothing else")
s, f = spectrum(chip.render_note(sine, 440.0, 0.5))
k = int(np.argmax(s))
near("the peak is where the note is", f[k], 440.0, 6.0)
ok = 20 * math.log10(s[2 * k] / s[k]) < -60
check("its second harmonic is buried (a sine, not a saw)", ok, True)

print("the envelope generator")
# Sustain level is 4 steps of 3 dB, which is where the first decay stops.
# A decibel of slack: the output stage's dead band is a fixed size, so it
# reads as a fraction more level the quieter the thing being measured.
held = chip.render_note(only(chip.Op(tl=0, ar=31, d1r=20, sl=4, d2r=0)), 440.0, 0.5)
near("sustain level 4 holds 12 dB down", db_at(held, 0.3) - db_at(held, 0.0), -12.0, 1.5)
held = chip.render_note(only(chip.Op(tl=0, ar=31, d1r=20, sl=8, d2r=0)), 440.0, 0.5)
near("sustain level 8 holds 24 dB down", db_at(held, 0.3) - db_at(held, 0.0), -24.0, 2.0)
# Total level is 7 bits of 0.75 dB.
quiet = chip.render_note(only(chip.Op(tl=16, ar=31)), 440.0, 0.2)
loud = chip.render_note(only(chip.Op(tl=0, ar=31)), 440.0, 0.2)
near("total level 16 is 12 dB down", db_at(quiet, 0.05) - db_at(loud, 0.05), -12.0, 1.0)
# A rate of 0 never arrives anywhere.
check("attack rate 0 never sounds", db_at(chip.render_note(only(chip.Op(tl=0, ar=0)), 440.0, 0.3), 0.2) < -80, True)
fast = chip.render_note(only(chip.Op(tl=0, ar=31)), 440.0, 0.2)
slow = chip.render_note(only(chip.Op(tl=0, ar=10)), 440.0, 0.2)
check("a slow attack is still climbing where a fast one has arrived",
      db_at(slow, 0.004) < db_at(fast, 0.004) - 20, True)
# Key scaling: the same patch decays faster the higher it is played.
# A rate slow enough that the bottom of the keyboard is still sounding at
# 0.2 s; at key scaling 3 the top of it should be long gone.
ks = only(chip.Op(tl=0, ar=31, d1r=8, sl=15, ks=3))
low = chip.render_note(ks, midi_to_hz(36), 0.5)
high = chip.render_note(ks, midi_to_hz(84), 0.5)
check("key scaling makes high notes decay faster", db_at(high, 0.2) < db_at(low, 0.2) - 20, True)
flat = only(chip.Op(tl=0, ar=31, d1r=8, sl=15, ks=0))
check("with key scaling off the note makes little difference",
      abs(db_at(chip.render_note(flat, midi_to_hz(84), 0.5), 0.2)
          - db_at(chip.render_note(flat, midi_to_hz(36), 0.5), 0.2)) < 6, True)

print("frequency multipliers")
for mul in (0, 1, 2, 4, 15):
    s_, f_ = spectrum(chip.render_note(only(chip.Op(tl=0, ar=31, mul=mul)), 220.0, 0.4))
    near(f"multiplier {mul} plays {mul or 0.5} times the note",
         f_[int(np.argmax(s_))] / 220.0, mul or 0.5, 0.03)

print("feedback leans a sine over towards a saw")
prev = -1e9
for fb in (0, 3, 5, 7):
    s, f = spectrum(chip.render_note(only(chip.Op(tl=0, ar=31), fb=fb), 440.0, 0.4))
    k = int(np.argmax(s))
    h2 = 20 * math.log10(s[2 * k] / s[k])
    check(f"feedback {fb} has more second harmonic than the setting below", h2 > prev, True)
    prev = h2

print("algorithms")
# 7 is four operators added together; 0 is all four in a line, so three of
# them are silent modulators and the sound is only the last one's.
four = chip.Patch("t", "", 7, 0, tuple(chip.Op(tl=0, ar=31) for _ in range(4)))
s7, _ = spectrum(chip.render_note(four, 440.0, 0.3))
chain = chip.Patch("t", "", 0, 0, tuple(chip.Op(tl=0, ar=31) for _ in range(4)))
s0, _ = spectrum(chip.render_note(chain, 440.0, 0.3))
check("adding four operators is not the same as stacking them", float(s7.sum()) != float(s0.sum()), True)
check("a stack of four is far richer than a single sine",
      (s0 > s0.max() * 0.01).sum() > (spectrum(chip.render_note(sine, 440.0, 0.3))[0] > s0.max() * 0.01).sum(), True)
for alg in range(8):
    pa = chip.Patch("t", "", alg, 0, tuple(chip.Op(tl=0, ar=31) for _ in range(4)))
    x = chip.render_note(pa, 440.0, 0.1)
    check(f"algorithm {alg} makes a sound, and no infinities", bool(np.isfinite(x).all() and np.abs(x).max() > 0), True)

print("the sample channel")
noise = np.sin(2 * np.pi * 300 * np.arange(SR // 2) / SR).astype(np.float32)
# The stages on their own: the coupling capacitor comes after both of these,
# so a finished sample is no longer exactly eight bits or exactly flat, and
# asserting that of chip.dac would only be asserting the filter away.
held = chip.hold(noise, 13300.0)
check("the sample channel holds its level between writes",
      float(np.abs(np.diff(held[:100])).min()) == 0.0, True)
check("and writes about 13300 times a second",
      round(SR / max(1, np.diff(np.flatnonzero(np.diff(held))).min())), 14700)
check("eight bits means at most 256 different values",
      len(np.unique(np.round(held * 128.0) / 128.0)) <= 257, True)
check("the ladder never lets a sound sit at zero",
      bool(np.abs(chip.ladder(np.zeros(10) + 0.001)).min() >= chip.LADDER_GAP), True)
check("but silence stays silent", float(np.abs(chip.ladder(np.zeros(10))).max()), 0.0)
check("and the gap is about one bit of the nine the DAC has",
      0.0015 < chip.LADDER_GAP < 0.006, True)

print("the output stage")
# FM puts a sideband on DC whenever a modulator matches its carrier, and
# eight parts of that is a rumble under a song written nowhere near that low.
sub = chip.render_note(chip.PATCHES["bass"], midi_to_hz(38), 0.4)
sp_, fr_ = spectrum(sub, 1 << 14)
check("the coupling capacitor keeps the sub-bass out",
      float(sp_[fr_ < 50].sum() / sp_.sum()) < 0.01, True)
check("and leaves the note itself alone",
      float(sp_[(fr_ >= 50) & (fr_ < 400)].sum() / sp_.sum()) > 0.3, True)
check("nothing comes out with a DC offset", abs(float(chip.dac(noise).mean())) < 1e-3, True)

print("a note stops when it is over")
# Every note used to be cut off partway down its own decay, and a few hundred
# clicks a minute is a rumble, not a song.
tailed = chip.render_note(chip.PATCHES["bass"], midi_to_hz(38), 0.136)
check("a note is rendered past the key coming up", len(tailed) > int(0.136 * SR) + 100, True)
# Not bit-exact zero: the resampling leaves a dusting of 1e-11 behind it,
# which is 200-odd dB down and is the fade having worked, not having failed.
check("and it has reached silence by the end", float(np.abs(tailed[-64:]).max()) < 1e-6, True)

print("the patches")
for name, patch in chip.PATCHES.items():
    hz = midi_to_hz(38 if name == "bass" else 55)
    x = chip.render_note(patch, hz, 0.4)
    check(f"{name} is finite and audible", bool(np.isfinite(x).all() and np.abs(x).max() > 0.2), True)
    check(f"{name} does not clip on its own", bool(np.abs(x).max() < 1.0), True)
    info = describe(track_f0(x), 0.4)
    near(f"{name} plays the note it was given", 1200 * math.log2((info.f0 or hz) / hz), 0.0, 15.0)
check("a bass part is played by the bass patch", chip.patch_for("bass", 33).name, "bass")
check("drums have no program, and the role decides", chip.patch_for("chords", None).name, "organ")
check("with no role the program decides", chip.patch_for("", 56).name, "brass")
check("the same note twice is the same sound",
      bool((chip.render_note(chip.PATCHES["bass"], 110.0, 0.2)
            == chip.render_note(chip.PATCHES["bass"], 110.0, 0.2)).all()), True)

print("the song-wide switch")
from app.ytpmv import tone as tones
# The page hands every part back the settings it was given, tone and all, so
# the default "clean" arrives looking exactly like a choice. Counting it as
# one turned the switch off for every song rendered from the web page.
check("the default tone does not outrank the switch", tones.overrides_switch("clean"), False)
check("nor does no tone at all", tones.overrides_switch(None), False)
check("nor does something that is not a tone", tones.overrides_switch("trombone"), False)
check("a chosen patch does", tones.overrides_switch("organ"), True)
check("so does asking for the voice off the sample channel", tones.overrides_switch("dac"), True)
check("and so does an old name for one", tones.overrides_switch("megadrive"), True)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f_ in failures:
    print(" ", f_)
sys.exit(1 if failures else 0)
