#!/usr/bin/env python3
"""The Master System's chip and the SNES's. Plain asserts.

    python tests/test_sms_snes.py

Needs numpy. Figures come from the hardware notes rather than from these
implementations: SMSPower's for the SN76489, fullsnes for the S-DSP.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ytpmv import sms, snes
from app.ytpmv.pitch import SR, midi_to_hz

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


def peak_hz(x, lo, hi, n=1 << 16):
    s = np.pad(x, (0, max(0, n - len(x))))[:n] * np.hanning(n)
    sp = np.abs(np.fft.rfft(s))
    band = sp * ((np.fft.rfftfreq(n, 1.0 / SR) >= lo) & (np.fft.rfftfreq(n, 1.0 / SR) < hi))
    k = int(np.argmax(band))
    a, b, c = float(sp[k - 1]), float(sp[k]), float(sp[k + 1])
    den = a - 2 * b + c
    return (k + (0.5 * (a - c) / den if abs(den) > 1e-20 else 0.0)) * SR / n


print("the SN76489's dividers")
# The figure the chip notes give to check an implementation against.
near("register $0FE is 440.4 Hz", sms.CLOCK / (32 * 0xFE), 440.4, 0.1)
near("the lowest note, $3FF, is 109 Hz", sms.CLOCK / (32 * 0x3FF), 109.3, 0.5)
near("and that is the floor", sms.FLOOR_HZ, 109.3, 0.5)
check("the divider is ten bits", sms.divider(1.0), 1023)

print("its volume")
# Two decibels a step, and 15 is off rather than merely quiet.
near("one step down is 2 dB", 20 * math.log10(sms.attenuation(1)), -2.0, 0.01)
near("seven steps down is 14 dB", 20 * math.log10(sms.attenuation(7)), -14.0, 0.01)
check("fifteen is silence", sms.attenuation(15), 0.0)

print("its shift register")
# The documented taps are not a maximal-length polynomial, and pretending
# otherwise would mean using the wrong ones.
check("the white sequence is 57337 long", len(sms.lfsr_bits(False)), 57337)
check("which is not maximal, and is not meant to be",
      len(sms.lfsr_bits(False)) == 65535, False)
check("the other mode is one in sixteen", len(sms.lfsr_bits(True)), 16)
check("and that is a pitch, not a hiss", int(sms.lfsr_bits(True).sum()), 1)

print("what it plays")
for name, voice in sms.VOICES.items():
    x = sms.render_note(voice, midi_to_hz(45), 0.3)
    check(f"{name} is finite and audible", bool(np.isfinite(x).all() and np.abs(x).max() > 0.2), True)
    check(f"{name} does not clip", bool(np.abs(x).max() < 1.0), True)
for midi in (45, 57, 69):
    hz = midi_to_hz(midi)
    got = peak_hz(sms.render_note(sms.VOICES["psg"], hz, 0.4), hz * 0.75, hz * 1.3)
    near(f"MIDI {midi} comes out on the note", 1200 * math.log2(got / hz), 0.0, 8.0)
# Below the floor it octaves up rather than landing flat on 109 Hz.
low = sms.render_note(sms.VOICES["psg-bass"], midi_to_hz(33), 0.4)
got = peak_hz(low, 80.0, 400.0)
# Distance to the nearest octave, not the remainder: a cent under one is a
# cent out, and the remainder calls it 1199.
_off = 1200 * math.log2(got / midi_to_hz(33)) % 1200.0
near("a note under the floor is raised by whole octaves",
     min(_off, 1200.0 - _off), 0.0, 10.0)
check("and it does land above the floor", got > sms.FLOOR_HZ - 1.0, True)
check("a bass part gets a square", sms.voice_for("bass").name, "psg-bass")
check("drums come off the shift register", sms.voice_for("drums:snare").name, "sms-snare")

print("dynamics survive")
# Every note used to be levelled to the same peak at the very end, after the
# velocity had been applied, so velocity 30 came out as loud as 127.
from app.ytpmv import nes
for mod, voice in ((sms, sms.VOICES["psg"]), (nes, nes.VOICES["pulse"])):
    name = mod.__name__.split(".")[-1]
    loud = float(np.abs(mod.render_note(voice, 220.0, 0.3, level=1.0)).max())
    soft = float(np.abs(mod.render_note(voice, 220.0, 0.3, level=30 / 127)).max())
    check(f"{name}: a soft note is quieter than a hard one", soft < loud * 0.5, True)
check("psg-soft is softer than psg, as its name says",
      float(np.abs(sms.render_note(sms.VOICES["psg-soft"], 220.0, 0.3)).max())
      < float(np.abs(sms.render_note(sms.VOICES["psg"], 220.0, 0.3)).max()) * 0.7, True)
check("the triangle has no volume control, so velocity cannot touch it",
      round(float(np.abs(nes.render_note(nes.VOICES["triangle"], 110.0, 0.3, level=0.2)).max()), 3),
      round(float(np.abs(nes.render_note(nes.VOICES["triangle"], 110.0, 0.3, level=1.0)).max()), 3))
near("an NES pulse stops at 55 Hz, eleven bits of timer",
     nes.floor_hz(nes.VOICES["pulse"]), 54.6, 0.2)
near("and the triangle an octave under that", nes.floor_hz(nes.VOICES["triangle"]), 27.3, 0.2)
check("noise has no floor to speak of", nes.floor_hz(nes.VOICES["nes-snare"]), 0.0)

print("the SNES's interpolation table")
g = snes._GAUSS
check("512 entries", len(g), 512)
# The real check: the four taps at any phase sum to unity gain. A table
# typed in wrongly would not.
sums = {int(g[0xFF - i] + g[0x1FF - i] + g[0x100 + i] + g[i]) for i in range(256)}
check("its four taps sum to about 2048 at every phase", max(sums) - min(sums) <= 2, True)
near("and that sum is 2048", sum(sums) / len(sums), 2048.0, 1.5)
check("it is a low-pass, not a neutral resampler", float(g[0]) < float(g[255]), True)

print("BRR")
check("four predictors", len(snes.BRR_FILTERS), 4)
check("the first does not predict", snes.BRR_FILTERS[0], (0.0, 0.0))
near("the second leans 15/16 on the last sample", snes.BRR_FILTERS[1][0], 0.9375, 1e-9)
near("the third, 61/32 and -15/16", snes.BRR_FILTERS[2][0], 1.90625, 1e-9)
near("the fourth, 115/64 and -13/16", snes.BRR_FILTERS[3][0], 1.796875, 1e-9)
# Four bits with a per-block scale and a predictor: good enough to be
# recognisable, bad enough to hear.
sine = (0.5 * np.sin(2 * np.pi * 220 * np.arange(SR // 2) / SR)).astype(np.float32)
at32 = snes._resample(sine, SR, snes.DSP_RATE)
err = snes.brr(at32) - at32
snr = 20 * math.log10(math.sqrt(float((err ** 2).mean()))
                      / math.sqrt(float((at32 ** 2).mean())))
check("it loses something", snr > -70.0, True)
check("but not everything", snr < -20.0, True)

print("the whole SNES path")
out = snes.play(sine, SR)
check("it comes back finite", bool(np.isfinite(out).all()), True)
check("and does not clip", bool(np.abs(out).max() <= 1.0), True)
check("the echo rings on past the note", len(out) > len(sine), True)
# Storing a sample lower is most of why the machine sounds muffled.
def above(x, hz=4000.0, n=1 << 13):
    sp = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
    fr = np.fft.rfftfreq(n, 1.0 / SR)
    return float(sp[fr > hz].sum() / max(sp.sum(), 1e-9))
sq = (0.5 * np.sign(np.sin(2 * np.pi * 180 * np.arange(SR // 2) / SR))).astype(np.float32)
bright = above(snes.play(sq, SR, stored_hz=32000, feedback=0.0, tail=0.0))
dark = above(snes.play(sq, SR, stored_hz=8000, feedback=0.0, tail=0.0))
check("a sample kept at a lower rate comes back darker", dark < bright / 3.0, True)
check("silence survives it", snes.play(np.zeros(1000, dtype=np.float32)).max(), 0.0)
check("and so does an empty buffer", snes.play(np.zeros(0, dtype=np.float32)).size, 0)

print("the SNES's instruments")
# A General MIDI SoundFont cut down to what a SNES game could carry.
b = snes.bank()
check("every General MIDI program has a sample", all(f"p{p}" in b for p in range(128)), True)
check("and the drum keys GM names", all(f"d{k}" in b for k in range(35, 82)), True)
check("and it says which font it was cut from", bool(snes.source().get("file")), True)
b = {k: v for k, v in b.items() if not k.startswith("_")}
cap = float(snes.source().get("built", {}).get("store_hz", 16000.0))
check("none kept above the rate it was built for, but for a loop too short for one block",
      max(i.rate for i in b.values() if not i.loop or i.loop[1] - i.loop[0] > 16) <= cap + 0.5, True)
check("and never above the DSP's own", max(i.rate for i in b.values() if not i.loop or i.loop[1] - i.loop[0] > 16)
      <= snes.DSP_RATE + 0.5, True)
looped = [i for i in b.values() if i.loop]
check("every loop is whole BRR blocks", all((i.loop[1] - i.loop[0]) % 16 == 0 for i in looped), True)
check("and starts on a block", all(i.loop[0] % 16 == 0 for i in looped), True)
check("most instruments loop, as a held note needs", len(looped) > 100, True)
check("a program with no sample falls back to the piano", snes.instrument_for(500).name, b["p0"].name)
for prog, note in ((0, 60), (33, 40), (56, 64), (73, 79), (80, 69)):
    inst = snes.instrument_for(prog)
    hz = midi_to_hz(note)
    x = snes.render_note(inst, note, 0.8)
    got = peak_hz(np.pad(x[:int(0.6 * SR)], (0, 1 << 17)), hz * 0.85, hz * 1.18, n=1 << 18)
    # Within a SNES's own error: a loop ripped from a game is a whole number
    # of samples, and a short one cannot land nearer its note than this.
    near(f"{inst.name} plays in tune", 1200 * math.log2(got / hz), 0.0, 12.0)
held = snes.render_note(snes.instrument_for(48), 55, 2.0)
check("a looped sample holds a note longer than the sample is",
      float(np.abs(held[int(1.7 * SR):int(1.9 * SR)]).max()) > 0.05, True)
check("and stops after its release", float(np.abs(held[-int(0.05 * SR):]).max()) < 0.05, True)
kick = snes.render_note(snes.instrument_for(None, 36), 36, 0.2)
check("a drum sounds", bool(np.isfinite(kick).all() and np.abs(kick).max() > 0.1), True)
from app.ytpmv import tone
check("and the SNES plays through tone.apply, drums included",
      bool(np.abs(tone.apply(np.full(4000, 0.5, np.float32), "snes", hz=None, key=38)).max() > 0.01), True)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
