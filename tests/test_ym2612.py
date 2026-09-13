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
    if patch.base_hz:
        # A drum is not playing a note. Asking it to be in tune with one is
        # asking the wrong thing: what it promises is to ignore it.
        other = chip.render_note(patch, hz * 2.0, 0.4)
        check(f"{name} sounds the same whatever note it is handed",
              bool(np.array_equal(x, other)), True)
    else:
        info = describe(track_f0(x), 0.4)
        near(f"{name} plays the note it was given",
             1200 * math.log2((info.f0 or hz) / hz), 0.0, 15.0)
check("a bass part is played by the bass patch", chip.patch_for("bass", 33).name, "bass")
check("drums have no program, and the role decides", chip.patch_for("chords", None).name, "organ")
check("with no role the program decides", chip.patch_for("", 56).name, "brass")
check("the same note twice is the same sound",
      bool((chip.render_note(chip.PATCHES["bass"], 110.0, 0.2)
            == chip.render_note(chip.PATCHES["bass"], 110.0, 0.2)).all()), True)

print("the NES")
from app.ytpmv import nes
# The shift register is periodic, and how long it takes to come round is why
# the short mode has a pitch and the long one does not.
check("the long shift register runs 32767 steps", len(nes.noise_bits(0)), 32767)
check("and the short one 93", len(nes.noise_bits(1)), 93)
check("the noise period table is the hardware's",
      nes.NOISE_PERIODS[0] == 4 and nes.NOISE_PERIODS[-1] == 4068, True)
check("so is the DMC's", nes.DMC_PERIODS[0] == 428 and nes.DMC_PERIODS[-1] == 54, True)
near("the NTSC clock", nes.CPU, 1789773.0, 1.0)
# The duty widths are 1 << the setting, the fourth being the quarter negated.
for duty, want in ((0, 1 / 8), (1, 2 / 8), (2, 4 / 8), (3, 6 / 8)):
    on = float((nes.pulse(220.0, SR, duty, SR) > 0).mean())
    near(f"duty {duty} is high {want * 100:.0f}% of the time", on, want, 0.02)
# The triangle divides by 32 where the pulses divide by 16, so for the same
# timer it sounds an octave lower -- it is the bass voice for a reason.
tri = nes.triangle(110.0, SR // 4, SR)
check("the triangle has 16 levels each way", len(np.unique(tri)), 16)
check("and no volume control to apply", float(tri.max()), 15.0)
# Every voice, in tune and audible.
for name, voice in nes.VOICES.items():
    x = nes.render_note(voice, midi_to_hz(45), 0.3)
    check(f"{name} is finite and audible", bool(np.isfinite(x).all() and np.abs(x).max() > 0.2), True)
    check(f"{name} does not clip", bool(np.abs(x).max() < 1.0), True)
check("a bass part takes the triangle", nes.voice_for("bass").name, "triangle")
check("drums come off the noise channel", nes.voice_for("drums:snare").name, "nes-snare")
# The DMC can only move two steps at a time, which is the whole character.
ramp = np.linspace(-1.0, 1.0, SR // 10).astype(np.float32)
out = nes.dpcm(ramp, 10)
check("the DMC output stays inside its 7 bits",
      bool(out.min() >= 0.0 and out.max() <= 127.0), True)
check("and never moves more than two steps at once",
      bool(np.abs(np.diff(np.unique(out))).max() <= 2.0), True)

print("both machines, each of two minds")
_r = __import__("app.ytpmv.render", fromlist=["CHIPS"])
for name in ("md", "md-voice", "nes", "nes-voice", "sms", "sms-voice", "snes", "snes-voice",
             "gb", "gb-voice", "gbc", "gbc-voice"):
    check(f"{name} is a setting we have", name in _r.CHIPS, True)
check("true still means the Mega Drive, as it did when this was a tickbox",
      _r.which_chip(True), "md")
check("and the long name still works for anyone who typed it",
      _r.which_chip("megadrive"), "md")
check("and false still means none of them", _r.which_chip(False), None)
check("a machine we do not have is not one", _r.which_chip("amiga"), None)
# The point of the sampled settings: the voice survives them.
from app.ytpmv import music as _m
_part = _m.Part(id="x", name="n", track=0, channel=0, program=33, notes=[], role="bass")
check("synthesised, the Mega Drive plays it itself",
      _r.chip_tone(_part, "md"), "bass")
check("but its voice setting keeps him, off the sample channel",
      _r.chip_tone(_part, "md-voice"), "dac")
check("synthesised, the NES gives the bass its triangle",
      _r.chip_tone(_part, "nes"), "triangle")
check("and its voice setting keeps him, delta-modulated",
      _r.chip_tone(_part, "nes-voice"), "dpcm")
check("the Master System gives the bass a square",
      _r.chip_tone(_part, "sms"), "psg-bass")
check("and hammers the voice out of its volume register when asked",
      _r.chip_tone(_part, "sms-voice"), "psg-pcm")
# The SNES has no synthesiser, but it has instruments: the sample is chosen
# per note from the part's program, as the OPL's patch is.
check("the SNES plays the part on its instruments",
      _r.chip_tone(_part, "snes"), "snes")
check("and keeps him through its sampler when asked",
      _r.chip_tone(_part, "snes-voice"), "brr")
# The picture goes with the sound, not beside it.
for chip, want in (("md", "md"), ("md-voice", "md"), ("nes", "nes"), ("nes-voice", "nes"),
                   ("sms", "sms"), ("sms-voice", "sms"), ("snes", "snes"), ("snes-voice", "snes"),
                   ("gb", "gb"), ("gb-voice", "gb"), ("gbc", "gbc"), ("gbc-voice", "gbc"),
                   (None, "none")):
    check(f"{chip} puts the picture through {want}", _r.screen_for(chip), want)

print("a part that cannot go as low as it is written")
from app.ytpmv.music import Note, Song
from app.ytpmv.pitch import midi_to_hz as _hz
# The Megalovania riff, which crosses the Master System's 109 Hz floor.
riff = [Note(i * 0.2, 0.2, m, 90) for i, m in enumerate((38, 38, 50, 45, 44, 43, 41))]
up = _r.octaves_to_fit(riff, 109.3, 0)
check("the whole part is raised by one octave", up, 1)
played = [n.pitch + 12 * up for n in riff]
check("so every note clears the floor", min(_hz(m) for m in played) >= 109.3 - 1e-6, True)
# The thing raising notes one at a time broke: the shape of the line.
steps = lambda ms: [b - a for a, b in zip(ms, ms[1:])]
check("and every interval is exactly as written", steps(played), steps([n.pitch for n in riff]))
check("a part already above the floor is not moved", _r.octaves_to_fit(riff, 50.0, 0), 0)
check("nor is anything with no floor", _r.octaves_to_fit(riff, 0.0, 0), 0)

print("finding the bass when nothing says which it is")
def _part(pid, role, pitches, drums=False):
    # A part is drums by its drum group, not by its channel.
    return _m.Part(id=pid, name=pid, track=0, channel=9 if drums else 0, program=None if drums else 0,
                   notes=[Note(i * 0.2, 0.2, m, 90) for i, m in enumerate(pitches)], role=role,
                   drum_group=role.split(":")[-1] if drums else None)
tune = _part("tune", "lead", [72, 74, 76, 79])
low = _part("low", "rhythm", [38, 45, 50, 43])
kit = _part("kit", "drums:kick", [36, 36], drums=True)
song = Song([tune, low, kit], 2.0, 120.0, (4, 4), "D minor", 0.9)
check("the lowest line is the bass", _r.bass_line(song), "low")
check("which the NES gives its triangle", _r.chip_tone(low, "nes", as_bass=True), "triangle")
check("the Master System its bass square", _r.chip_tone(low, "sms", as_bass=True), "psg-bass")
check("but drums are never the bass", _r.chip_tone(kit, "nes", as_bass=True), "nes-kick")
named = Song([tune, _part("b", "bass", [40, 43]), low], 2.0, 120.0, (4, 4), "D minor", 0.9)
check("when a part is already called the bass, nothing is guessed", _r.bass_line(named), None)
high = Song([tune, _part("h", "rhythm", [67, 69, 71])], 2.0, 120.0, (4, 4), "C", 0.9)
check("and a line that is not low is not made one", _r.bass_line(high), None)

print("the bot knows every tone there is")
# The bot keeps its own list so it can be tested without numpy, which means
# it can fall behind -- it did, and quietly refused every Master System tone.
import re as _re
from app.ytpmv import tone as _tone
_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "app", "matrix_commands.py"), encoding="utf-8").read()
_block = _src[_src.index("_TONES = {"):_src.index("}", _src.index("_TONES = {"))]
check("every tone is one the bot will accept",
      sorted(set(_tone.TONES) - set(_re.findall(r'"([a-z0-9-]+)"', _block))), [])
_chips = _src[_src.index("_CHIPS = {"):_src.index("}", _src.index("_CHIPS = {"))]
check("and every chip setting", sorted(set(_r.CHIPS) - set(_re.findall(r'"([a-z0-9-]+)"', _chips))), [])

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
# "clean" cannot mean "keep him" because the page used to echo it; "voice"
# is the same sound, only ever sent by somebody choosing it.
check("voice does outrank it", tones.overrides_switch("voice"), True)
check("and is the voice untouched", tones.resolve("voice"), "clean")

print("choosing a part's instrument")
_song = Song([tune, low, kit], 2.0, 120.0, (4, 4), "D minor", 0.9)
def _chipped(given, chip):
    """Settings for every part of _song after the switch, as the render sees them."""
    st = {p.id: _r._merge(_r._default_settings_for(p, {"text": "x", "octave": -2}), given.get(p.id))
          for p in _song.parts}
    _r._through_chip(_song, st, given, chip)
    return st
st = _chipped({}, "sid")
check("left alone, a part plays what the chip picks", st["tune"]["tone"], "sid-lead")
check("and a chip-played part goes back to its written octave", st["tune"]["octave"], 0)
st = _chipped({"tune": {"tone": "pulse"}, "low": {"tone": "voice"}}, "sid")
check("a chosen instrument outlives the switch, from another machine too", st["tune"]["tone"], "pulse")
check("voice keeps a part as him on a chipped song", st["low"]["tone"], "clean")
check("and leaves the octave the voice wanted", st["low"]["octave"], -2)
check("the kit still follows the chip", st["kit"]["tone"], "sid-kick")
check("every part knows the chip, chosen or not, so a SID voice gets the right SID",
      _chipped({"tune": {"tone": "sid-bell"}}, "sid8580")["tune"]["chip"], "sid8580")
st = _chipped({"tune": {"tone": "organ"}}, None)
check("with no chip at all, a part can still be given one", st["tune"]["tone"], "organ")
check("and is put back in its written octave like any chip part", st["tune"]["octave"], 0)
check("while the rest stay him", st["low"]["tone"], "clean")
check("an asked octave is kept", _chipped({"tune": {"tone": "organ", "octave": 1}}, None)["tune"]["octave"], 1)

print("a part's program")
st = _chipped({"low": {"program": 29}}, "opl2")
check("a program can be swapped", st["low"]["program"], 29)
check("and is clamped to General MIDI", _chipped({"low": {"program": 400}}, None)["low"]["program"], 127)
check("drums have none to swap", _chipped({"kit": {"program": 5}}, None)["kit"]["program"], None)
# Most roles pick their voice outright; a part with none the chips know is
# chosen by its instrument, and that is where a swapped program shows.
_prog = _m.Part(id="p", name="p", track=0, channel=0, program=0, notes=[], role="piano")
check("the program reaches the choice where the chip chooses by it",
      (_r.chip_tone(_prog, "md"), _r.chip_tone(_prog, "md", program=9)), ("piano", "bell"))
check("on the NES a bass program is the triangle",
      (_r.chip_tone(_m.Part(id="p", name="p", track=0, channel=0, program=0, notes=[], role="piano"), "nes", program=33)),
      "triangle")
_clip = (0.3 * np.sin(2 * np.pi * 220 * np.arange(8000) / 44100)).astype(np.float32)
check("and on an OPL it is a different patch, and a different sound",
      bool(np.allclose(tones.apply(_clip, "opl2", hz=220.0, program=0),
                       tones.apply(_clip, "opl2", hz=220.0, program=29), atol=1e-3)), False)

print("the page's list of instruments")
_cat = tones.catalogue()
for m, info in _cat["machines"].items():
    for kind in ("melodic", "drums"):
        check(f"{info['name']} has {kind} to offer, all of them real synth tones",
              bool(info[kind]) and all(t in tones.SYNTH for t, _ in info[kind]), True)
check("the voice list is all voice", all(tones.resolve(t) and tones.resolve(t) not in tones.SYNTH
                                         for t, _ in _cat["voice"]), True)
# What "auto" says it is has to be on the chip's own list, or the page offers
# a machine's instruments that do not include the one it is playing.
for chip in _r.CHIPS:
    machine = _cat["machine_of"].get(chip.split("-")[0])
    for p in (tune, low, kit):
        auto = _r.chip_tone(p, chip)
        if chip.endswith("-voice"):
            ok = auto in dict(_cat["voice"])
        else:
            ok = auto in dict(_cat["machines"][machine]["drums" if p.is_drums else "melodic"])
        check(f"on {chip}, auto for {p.id} ({auto}) is on the list it is offered from", ok, True)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f_ in failures:
    print(" ", f_)
sys.exit(1 if failures else 0)
