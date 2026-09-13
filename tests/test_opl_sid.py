#!/usr/bin/env python3
"""The PC's FM chips and the C64's SID. Plain asserts.

    python tests/test_opl_sid.py

Needs numpy. Figures from Nuked-OPL3 for the OPL, Freedoom's GENMIDI for the
patches, and reSID for the SID.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ytpmv import opl, sid, tone, ym2612
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


print("drums synthesise, through the path render really takes")
# Every chip's kit used to fall through to the voice on the Mega Drive's
# sample channel, because drums are handed no note and synthesis wanted one.
clip = (0.3 * np.sin(2 * np.pi * 200 * np.arange(SR // 4) / SR)).astype(np.float32)
as_voice = ym2612.dac(clip / float(np.abs(clip).max()), tone.DAC_RATE, SR) * float(np.abs(clip).max())
for name in ("md-kick", "nes-snare", "sms-hat", "gb-kick", "sid-snare"):
    got = tone.apply(clip, name, hz=None)
    check(f"{name}, handed no note, is synthesised and not the voice",
          bool(len(got) == len(as_voice) and np.allclose(got, as_voice)), False)
for key in (36, 38, 42):
    got = tone.apply(clip, "opl2", hz=None, key=key)
    check(f"an OPL drum picks its patch by key {key}", bool(np.isfinite(got).all() and np.abs(got).max() > 0.01), True)

print("the OPL's ROMs")
# Nuked-OPL3's dumps, spot-checked: the formulas reproduce all of them.
check("log-sine ROM entry 0 is 0x859", int(opl.LOGSIN[0]), 0x859)
check("and entry 255 is 0", int(opl.LOGSIN[255]), 0)
check("exponent ROM entry 0 is 0x7fa", int(opl.EXP[0]), 0x7FA)
check("and entry 1 is 0x7f5", int(opl.EXP[1]), 0x7F5)
check("the multipliers are the chip's, in halves", opl.MULT[0], 1)
check("with its repeats at 10 and 12 and 15", (opl.MULT[10], opl.MULT[11], opl.MULT[14]), (20, 20, 30))

print("the OPL's envelope timing, against published figures")
near("attack rate 1 takes 2.84 s", opl.ATTACK_SAMPLES[4] / opl.RATE, 2.83, 0.05)
near("decay rate 1 falls 96 dB in about 40 s", 511 / opl.DECAY_PER_SAMPLE[4] / opl.RATE, 40.0, 3.0)
check("rate 15 attack is instant", opl.ATTACK_SAMPLES[60], 0)
check("rate 0 never arrives", opl.ATTACK_SAMPLES[0], None)

print("GENMIDI")
bank = opl.bank()
check("175 instruments: 128 programs and 47 drums", len(bank), 175)
check("program 0 is the piano", bank[0].name, "Acoustic Grand Piano")
check("drum key 36 is a bass drum", opl.instrument_for(None, 36).name, "Bass Drum 1")
check("drums play a fixed pitch", opl.instrument_for(None, 36).fixed_note is not None, True)

print("the OPL plays in tune")
for prog in (0, 33, 56):
    inst = opl.instrument_for(prog)
    for opl3 in (False, True):
        note = 57
        hz = midi_to_hz(note + 12 + inst.voices[0].note_offset)
        got = peak_hz(opl.render_note(inst, note, 0.5, opl3=opl3), hz * 0.8, hz * 1.25)
        near(f"{inst.name} on the OPL{'3' if opl3 else '2'}", 1200 * math.log2(got / hz), 0.0, 10.0)

print("Doom's note numbering")
# DMX's note 57 is 440 Hz, an octave under MIDI. Getting that wrong put
# E1M1's bass at 20 Hz.
for prog, note in ((34, 28), (0, 69)):
    inst = opl.instrument_for(prog)
    hz = midi_to_hz(note)
    got = peak_hz(opl.render_note(inst, note, 0.5), hz * 0.8, hz * 1.25)
    near(f"{inst.name} plays MIDI {note} where MIDI says it is", 1200 * math.log2(got / hz), 0.0, 10.0)

print("OPL2 and OPL3 are different chips")
lead = opl.instrument_for(80)                      # a square lead wants waveform 6
a = opl.render_note(lead, 60, 0.4, opl3=False)
b = opl.render_note(lead, 60, 0.4, opl3=True)
check("a patch using waveform 6 sounds different on each", bool(np.allclose(a, b, atol=1e-3)), False)
# The OPL2 only reads two waveform bits, so 6 becomes 2, an absolute sine,
# which never goes negative -- and the card's output coupling must remove it.
check("the OPL2's fallback waveform leaves no DC", abs(float(a.mean())) < 0.002, True)
def dac_steps(lo, hi):
    """The distinct output steps of the YM3014 over a run of integer inputs."""
    return np.diff(np.unique(np.round(opl.ym3014(np.arange(lo, hi) / 32767.0) * 32767)))
check("the YM3014 steps by one when quiet", float(dac_steps(0, 400).max()), 1.0)
check("and by 64 when loud: ten bits of mantissa cannot resolve more", float(dac_steps(30000, 32767).min()), 64.0)

print("the SID")
d6, d8 = sid.dac_table("6581"), sid.dac_table("8580")
lin = np.arange(4096) / 4095
check("the 6581's DAC is not linear", float(np.abs(d6 - lin).max()) > 0.005, True)
near("the 8580's is", float(np.abs(d8 - lin).max()), 0.0, 1e-9)
check("the 6581's cutoff floors at 220 Hz", sid.cutoff_hz(0, "6581"), 220.0)
check("and drops back as the register passes 0x7f",
      sid.cutoff_hz(1024, "6581") < sid.cutoff_hz(1023, "6581"), True)
check("the 8580 runs to 12.5 kHz", sid.cutoff_hz(2047, "8580"), 12500.0)
near("8580 resonance 4 is a Q of 1", sid.inv_q(4, "8580"), 1.0, 1e-9)
env = sid.envelope(int(0.2 * SR), SR, 2, 0, 15, 0, gate_samples=int(0.2 * SR))
near("attack rate 2 reaches the top in 16 ms", float(np.argmax(env >= 255)) / SR * 1000, 16.0, 1.0)
held = sid.envelope(int(2 * SR), SR, 0, 9, 10, 9, gate_samples=int(2 * SR))
check("sustain 10 holds at 0x11 x 10", int(held[int(1.5 * SR)]), 170)
bits = sid.noise_bits()
check("the shift register follows bit 22 XOR bit 17",
      bool((bits[100:5000] == (bits[100 - 23:5000 - 23] ^ bits[100 - 18:5000 - 18])).all()), True)
for name in ("sid-lead", "sid-bass", "sid-rhythm", "sid-bell"):
    for model in ("6581", "8580"):
        hz = midi_to_hz(45)
        x = sid.render_note(sid.VOICES[name], hz, 0.4, model=model)
        near(f"{name} on the {model}", 1200 * math.log2(peak_hz(x, hz * 0.8, hz * 1.25) / hz), 0.0, 8.0)
for name, voice in sid.VOICES.items():
    x = sid.render_note(voice, 220.0, 0.3)
    check(f"{name} is finite and audible", bool(np.isfinite(x).all() and np.abs(x).max() > 0.05), True)
quiet = float(np.abs(sid.digi(clip, model="8580")).max())
loud = float(np.abs(sid.digi(clip, model="6581")).max())
check("volume-register speech barely works on an 8580", quiet < loud * 0.3, True)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
