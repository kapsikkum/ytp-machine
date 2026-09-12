#!/usr/bin/env python3
"""Reading a MIDI file, as plain asserts.

    python tests/test_music.py

Everything a YTPMV decides downstream -- which word plays which part, what
octave it sings in, where each note lands in time -- comes from this reading
of the file, so a mistake here is a video that is wrong in a way that looks
deliberate. Files are built in memory with mido; needs mido and numpy.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mido

from app.ytpmv import music

failures: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


def near(label, got, want, tol=1e-6):
    ok = abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


def track(channel, notes, program=None, tpb=480, name=None):
    """notes: (start_beat, length_beats, pitch). Returns a MidiTrack."""
    t = mido.MidiTrack()
    if name:
        t.append(mido.MetaMessage("track_name", name=name, time=0))
    if program is not None:
        t.append(mido.Message("program_change", channel=channel, program=program, time=0))
    ev = []
    for start, length, pitch in notes:
        ev.append((int(start * tpb), 1, pitch))
        ev.append((int((start + length) * tpb), 0, pitch))
    ev.sort()
    now = 0
    for tick, on, pitch in ev:
        t.append(mido.Message("note_on" if on else "note_off", channel=channel, note=pitch,
                              velocity=100 if on else 0, time=tick - now))
        now = tick
    return t


def song(*tracks, tempo=None, tpb=480):
    mf = mido.MidiFile(ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    if tempo is not None:
        for tick, bpm in tempo:
            meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=tick))
    meta.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    mf.tracks.append(meta)
    mf.tracks.extend(tracks)
    buf = io.BytesIO()
    mf.save(file=buf)
    return buf.getvalue()


# C major: a tune on a flute, a bass line, a chord pad and a drum kit.
scale = [60, 62, 64, 65, 67, 69, 71, 72]
tune = track(0, [(i * 0.5, 0.5, p) for i, p in enumerate(scale * 2)], program=73)
bass = track(1, [(i * 2, 2, p) for i, p in enumerate([36, 41, 43, 36])], program=33)
pad = track(2, [(i * 2, 2, p) for i in range(4) for p in (60, 64, 67)], program=89)
kit = track(9, [(b, 0.1, 36) for b in range(8)] + [(b + 0.5, 0.1, 42) for b in range(8)]
            + [(b, 0.1, 38) for b in range(1, 8, 2)])

print("tempo")
s = music.parse(song(tune, bass, pad, kit, tempo=[(0, 90)]))
check("tempo read from the file", round(s.bpm), 90)
near("seconds follow the tempo: beat 1 at 90 BPM", s.part("t1c0").notes[1].start, 60 / 90 * 0.5)
check("time signature", s.time_signature, (3, 4))

s0 = music.parse(song(tune))
check("no tempo event means 120 BPM (as in E1M1)", round(s0.bpm), 120)
near("... and half a beat is a quarter second", s0.part("t1c0").notes[1].start, 0.25)

sc = music.parse(song(track(0, [(b, 1, 60) for b in range(8)]), tempo=[(0, 120), (4 * 480, 60)]))
near("a tempo change slows what follows", sc.parts[0].notes[5].start, 4 * 0.5 + 1 * 1.0)
check("the main tempo is the one it spends longest in", round(sc.bpm), 60)

print("broken files")
raw = bytearray(song(track(0, [(0, 1, 60)], program=0)))
raw[raw.index(b"MTrk") + 12] = 200          # a data byte no MIDI file should have
try:
    over = music.parse(bytes(raw))
    check("a byte out of range is clamped, not fatal", bool(over.parts), True)
except Exception as exc:
    check("a byte out of range is clamped, not fatal", f"{type(exc).__name__}: {exc}", True)

print("key")
check("C major", s.key, "C major")
minor = [57, 59, 60, 62, 64, 65, 67, 69]            # A natural minor, sitting on A
am = music.parse(song(track(0, [(i * 0.5, 0.5, p) for i, p in enumerate(minor * 2)] +
                           [(0, 8, 45)], program=0)))
check("A minor", am.key, "A minor")
# A drone that makes the profiles hesitate: E all the time, G natural, never G#.
drone = [(i * 0.25, 0.25, 40) for i in range(64)] + [(i * 2 + 0.5, 0.25, 43) for i in range(8)] \
    + [(i * 2 + 1.0, 0.25, 47) for i in range(8)]
check("a riff on a drone with a minor third is minor", music.parse(song(track(0, drone, program=30))).key,
      "E minor")

# Megalovania's shape: only F major's notes, but it starts on D, over a bass on D.
dm_tune = [(i * 0.5, 0.5, p) for i, p in enumerate([62, 62, 74, 69, 67, 65, 62, 65, 67, 72, 70, 69, 65, 62] * 2)]
dm_bass = [(i * 2, 2, p) for i, p in enumerate([38, 36, 34, 33, 38, 36, 34, 38])]
check("the relative minor wins when the music starts and ends on its tonic",
      music.parse(song(track(0, dm_tune, program=0), track(1, dm_bass, program=33))).key, "D minor")

print("roles")
roles = {p.id: p.role for p in s.parts}
check("the flute is the tune", roles["t1c0"], "lead")
check("the GM bass is the bass", roles["t2c1"], "bass")
check("three notes at once is chords", roles["t3c2"], "chords")
check("the whole GM kit is mapped, not just its middle",
      [music.drum_group(n) for n in (36, 31, 54, 66, 62, 55, 34, 82)],
      ["kick", "snare", "hats", "toms", "toms", "cymbals", "perc", "hats"])
check("drums split into kit pieces",
      sorted(p.drum_group for p in s.parts if p.is_drums), ["hats", "kick", "snare"])
check("the kit sorts after the band", [p.is_drums for p in s.parts], [False] * 3 + [True] * 3)
gtr = track(0, drone, program=30)
gtr2 = track(1, [(i * 0.5, 0.5, 40 + (i % 5)) for i in range(32)], program=29)
check("low guitars are not basses when a bass exists",
      sorted(p.role for p in music.parse(song(gtr, gtr2, bass)).parts), ["bass", "lead", "rhythm"])
check("with no bass program, the lowest single-note part is the bass",
      {p.program: p.role for p in music.parse(song(track(0, [(i, 1, 64 + i % 3) for i in range(8)], program=80),
                                                   track(1, [(i, 1, 36) for i in range(8)], program=81))).parts},
      {80: "lead", 81: "bass"})

print("parts")
fl = s.part("t1c0").summary()
check("range in note names", fl["range"], ["C4", "C5"])
check("instrument name", fl["instrument"], "Flute")
check("chord polyphony", s.part("t3c2").summary()["polyphony"], 3)
check("a friendly unique name", s.part("t1c0").name, "Flute (lead)")
check("the summary names the key and tempo", "C major" in s.describe() and "90 BPM" in s.describe(), True)

print("auto octave")
check("speech at C3 and a bass at C2: up one", music.auto_octave(36, 48), 1)
check("a bass at E1 under a voice at C3: up two", music.auto_octave(28, 48), 2)
check("a tune at C5 for a voice at C3: down two", music.auto_octave(72, 48), -2)
check("close enough stays put", music.auto_octave(50, 48), 0)
check("no pitch, no move", music.auto_octave(50, None), 0)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
