#!/usr/bin/env python3
"""No voice: a song played by the chip alone, drawn as oscilloscopes. Plain asserts.

    python tests/test_synth_mode.py

Needs numpy and mido; the last section needs ffmpeg, and is skipped without it.
"""
import io
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ytpmv import render, scope, screen
from app.ytpmv.music import Note, Part, Song
from app.ytpmv.pitch import SR, midi_to_hz

failures: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


def part(pid, role, pitches, drums=False, program=0):
    return Part(id=pid, name=pid, track=0, channel=9 if drums else 0,
                program=None if drums else program,
                notes=[Note(i * 0.25, 0.2, m, 100) for i, m in enumerate(pitches)], role=role,
                drum_group=role.split(":")[-1] if drums else None)


tune = part("tune", "lead", [72, 74, 76, 79])
low = part("low", "bass", [36, 43, 36, 43])
kit = part("kit", "drums:kick", [36, 36, 36, 36], drums=True)
song = Song([tune, low, kit], 1.2, 120.0, (4, 4), "C", 0.9)

print("which chip plays a song with no voice")
check("none means the Mega Drive", render.synth_chip(None), "md")
check("a -voice setting means the same machine playing", render.synth_chip("nes-voice"), "nes")
check("a synthesising setting stays", render.synth_chip("sid8580"), "sid8580")
check("the SNES plays its instruments", render.synth_chip("snes-voice"), "snes")


def settings_for(given, chip):
    st = {p.id: render._merge(render._default_settings_for(p, {}), given.get(p.id))
          for p in song.parts}
    render._through_chip(song, st, given, chip, synth=True)
    return st


print("the parts, with nothing to say")
st = settings_for({}, "nes")
check("a part with no word is not muted for it", st["tune"]["mute"], False)
check("and is shown", st["tune"]["visible"], True)
check("every part is played by the chip", [st[p]["tone"] for p in ("tune", "low", "kit")],
      ["pulse", "triangle", "nes-kick"])
check("and labelled with what plays it", st["low"]["text"], "triangle")
st = settings_for({"tune": {"tone": "voice"}, "low": {"tone": "dac"}, "kit": {"tone": "sid-kick"}}, "nes")
check("his voice cannot be kept: there is none", st["tune"]["tone"], "pulse")
check("nor his voice off a sample channel", st["low"]["tone"], "triangle")
check("a chosen synthesised instrument can", st["kit"]["tone"], "sid-kick")
check("a part muted on purpose stays muted", settings_for({"tune": {"mute": True}}, "nes")["tune"]["mute"], True)

print("playing a part from nothing")
s = settings_for({}, "nes")["low"]
mix = np.zeros((int(1.2 * SR), 2), dtype=np.float32)
hits = render._play(low, s, [], mix, 0.0, 1.2, 1.0, 0, False, vary="random")
check("every note is a hit", len(hits), 4)
check("and made a sound", float(np.abs(mix).max()) > 0.05, True)
check("which starts on the note", float(np.abs(mix[:int(0.01 * SR)]).max()) > 0.01, True)
# the triangle's pitch, near the first note, straight off the mix
seg = mix[int(0.02 * SR):int(0.18 * SR), 0]
spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 16))
freqs = np.fft.rfftfreq(1 << 16, 1.0 / SR)
band = (freqs > 40) & (freqs < 120)
got = float(freqs[band][np.argmax(spec[band])])
check("in tune with the part, octave and all", abs(1200 * np.log2(got / midi_to_hz(36))) < 25, True)

print("the oscilloscope")
n_frames, spf, width = 12, SR // 30, 64
t = np.arange(n_frames * spf + SR) / SR
sine = np.sin(2 * np.pi * 220 * t + 0.7).astype(np.float32)
tr = scope.traces(sine, n_frames, spf, width, hz=220.0).astype(np.float32)
check("one row of columns per frame", tr.shape, (n_frames, width, 2))
# Triggering holds a steady tone still, wherever each frame happens to start.
check("a steady tone stands still from frame to frame",
      float(np.abs(tr[1:] - tr[:-1]).max()) < 0.12, True)
check("and starts on its way up from nothing",
      bool(abs(float(tr[0, 0].mean())) < 0.2 and tr[0, 2, 1] > tr[0, 0, 1]), True)
check("columns meet their neighbours, so the trace never breaks",
      bool((np.maximum(tr[:, :-1, 0], tr[:, 1:, 0]) <= np.minimum(tr[:, :-1, 1], tr[:, 1:, 1]) + 1e-6).all()),
      True)
check("silence is a flat line", float(np.abs(scope.traces(np.zeros(SR, np.float32), 2, spf, width)).max()), 0.0)
img = scope.draw(tr[0], width, 40)
check("a tile is the size asked for", img.shape, (40, width, 3))
check("and lit where the trace is", int((img[:, :, 0] > 200).sum()) > width, True)
dim = scope.draw(tr[0], width, 40, level=0.35)
check("dimmed, it is darker", int(dim.max()) < int(img.max()), True)

print("neon red, in every console's own colours")
for name, (w, h) in screen.SIZES.items():
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:40, :width] = scope.draw(tr[0], width, 40, screen=name)
    lit = screen.apply(frame, name)
    colours = {tuple(int(v) for v in c) for c in lit.reshape(-1, 3)}
    trace = max(colours, key=lambda c: sum(c))
    if name == "gb":
        check("the Game Boy draws in its lightest shade, having no red",
              trace, tuple(int(v) for v in screen.GB_SHADES[0]))
    else:
        check(f"{name}'s trace is a red: more red in it than anything else",
              trace[0] > 1.5 * max(trace[1], trace[2]), True)

print("a whole song, rendered")
if shutil.which("ffmpeg") is None:
    print("  skip  no ffmpeg")
else:
    import mido
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("program_change", program=33, channel=0, time=0))
    for m in (36, 43, 36, 43):
        track.append(mido.Message("note_on", note=m, velocity=100, channel=0, time=0))
        track.append(mido.Message("note_off", note=m, velocity=0, channel=0, time=240))
    buf = io.BytesIO()
    mid.save(file=buf)
    midi_id = render.save_midi(buf.getvalue(), "synth-mode-test.mid")
    out = None
    try:
        r = render.render_ytpmv({"midi_id": midi_id, "parts": [],
                                 "options": {"synth": True, "chip": "sms-voice", "max_seconds": 3}})
        out = os.path.join("output", os.path.basename(r["video_url"]))
        check("it renders with no voice anywhere", os.path.exists(out), True)
        check("on the machine playing, not its sample channel", r["options"]["chip"], "sms")
        check("through that machine's screen", r["options"]["screen"], "sms")
        check("its one part played by the chip", r["parts"][0]["tone"], "psg-bass")
        frame = subprocess.run(["ffmpeg", "-v", "error", "-ss", "0.5", "-i", out, "-frames:v", "1",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               capture_output=True, check=True).stdout
        px = np.frombuffer(frame, dtype=np.uint8).reshape(-1, 3).astype(int)
        check("and the picture is a red trace on black",
              bool(((px[:, 0] > 150) & (px[:, 1] < 90)).sum() > 100 and np.median(px.sum(axis=1)) < 60), True)
    finally:
        for f in (os.path.join(render.MIDI_DIR, midi_id + ".mid"),
                  os.path.join(render.MIDI_DIR, midi_id + ".mid.title"), out):
            if f and os.path.exists(f):
                os.remove(f)

# The MIDI library: every uploaded file described, and the description cached.
import mido, tempfile, time as _time
_lib = tempfile.mkdtemp()
_was = render.MIDI_DIR
render.MIDI_DIR = _lib
try:
    mf = mido.MidiFile()
    tr = mido.MidiTrack()
    tr += [mido.Message("note_on", note=60, velocity=90, time=0), mido.Message("note_off", note=60, time=960)]
    mf.tracks.append(tr)
    buf = io.BytesIO(); mf.save(file=buf)
    mid = render.save_midi(buf.getvalue(), "little tune.mid")
    lib = render.library()
    check("an upload is in the library", [m["midi_id"] for m in lib], [mid])
    check("with its title and what is in it", (lib[0]["title"], lib[0]["parts"], lib[0]["notes"]), ("little tune", 1, 1))
    check("and its description is cached beside it", os.path.exists(os.path.join(_lib, mid + ".mid.info.json")), True)
    check("which a second read uses", render.library(), lib)
finally:
    render.MIDI_DIR = _was
    shutil.rmtree(_lib, ignore_errors=True)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
