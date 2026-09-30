#!/usr/bin/env python3
"""A MIDI file whose writer counted past channel 16, as plain asserts.

    python tests/test_midi_overflow.py

The writer ORed its channel count into the status byte, so channel 0x34's
program change is 0xF4 -- a status MIDI never defined -- and its note-on and
note-off are both 0xB4, which reads as a controller. That track must still
come out as notes, at the right times, and a note nobody ever released must
not make the song last as long as its track does. Files are built by hand.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ytpmv import music

failures: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


def mtrk(*events: bytes, end: bytes = bytes(1)) -> bytes:
    body = b"".join(events) + end + bytes((0xFF, 0x2F, 0x00))
    return b"MTrk" + len(body).to_bytes(4, "big") + body


def midi(*tracks: bytes) -> bytes:           # format 1, 480 ticks a beat, 120 BPM
    return b"MThd" + (6).to_bytes(4, "big") + bytes((0, 1, 0, len(tracks), 0x01, 0xE0)) + b"".join(tracks)


# A plain piano on channel 0, one beat of middle C.
piano = mtrk(bytes((0x00, 0xC0, 0x00, 0x00, 0x90, 60, 90, 0x83, 0x60, 0x80, 60, 0)))
# Channel 0x34: program change 0xC0|0x34 = 0xF4 (Fingered Bass), notes 0x90|0x34
# = 0x80|0x34 = 0xB4, released with velocity 0. One beat of E2, then two of G2,
# struck with a velocity of 138 -- this writer's go over 127 too.
bass = mtrk(b"\x00\xFF\x03\x04Bass",
            bytes((0x00, 0xF4, 33,
                   0x00, 0xB4, 40, 100, 0x83, 0x60, 0xB4, 40, 0,
                   0x00, 0xB4, 43, 0x8A, 0x87, 0x40, 0xB4, 43, 0)))
s = music.parse(midi(piano, bass), "overflow")
b = next((p for p in s.parts if p.track == 1), None)
check("the overflowed track is read", b is not None, True)
if b:
    check("its notes, times and lengths", [(n.start, n.dur, n.pitch) for n in b.notes],
          [(0.0, 0.5, 40), (0.5, 1.0, 43)])
    check("its program change", b.program, 33)
    check("a loud note is clamped", b.notes[-1].velocity, 127)
    check("it is not a drum kit", b.is_drums, False)
check("the piano is untouched", [(n.start, n.dur, n.pitch) for p in s.parts if p.track == 0 for n in p.notes],
      [(0.0, 0.5, 60)])
check("the song lasts as long as its notes", s.duration, 1.5)

# One note let go, one never, and the track running on for 100 seconds.
stuck = mtrk(bytes((0x00, 0x90, 60, 90, 0x83, 0x60, 0x80, 60, 0, 0x00, 0x90, 62, 90)),
             end=bytes((0x85, 0xEA, 0x20)))                   # 199 beats of nothing
s = music.parse(midi(stuck), "stuck")
check("a stuck note does not stretch the song", s.duration, 4.5)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
