"""Singing: every word of a finished video put on a note of a tune.

The generator makes the video as it always does; this then takes its sound,
tunes each word onto the next note of a melody with TD-PSOLA, and puts it
back. Durations do not change, so the picture is untouched and still in sync.

The tune is either a *shape* walked through the major scale of *key*, or the
top line of a MIDI file, one note a word. Either way it is pitched round where
the voice already sits, so nobody is dragged an octave out of their range. A
word can be moved off its note by the markup: hello^+2 is two semitones above
the tune, hello^A3 is an A3 whatever the tune says.

The PSOLA here is pitch.Voice's with three things added, which is why it is
its own loop rather than a call: a target that moves (vibrato, and partial
correction that leaves some of the speech's own glide in), and grains squeezed
in time, which moves the formants -- the "cute" knob. Squeezing a grain raises
the vocal tract it came from while the pulse spacing, and so the note, stays
where it was put.
"""

from __future__ import annotations

import logging
import math
import os
import random
import subprocess
import tempfile
import wave

import numpy as np

from app.ytpmv import pitch

log = logging.getLogger(__name__)

MAJOR = (0, 2, 4, 5, 7, 9, 11, 12, 14, 16)
SHAPES = ("arch", "rising", "falling", "wander", "twinkle")
_TWINKLE = (0, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0)
# How unperiodic a frame may be and still count as voiced. The tracker's own
# 0.35 suits a clean note; this corpus is live recordings, room and audience
# and all, where a loud vowel measured 0.75-0.85 and nothing was tuned at all.
_APERIODIC = 0.8
_VIBRATO_HZ = 5.5
_VIBRATO_ONSET = 0.25     # seconds for the vibrato to grow in, as a singer's does
_UNVOICED = 0.010

def degrees(n: int, shape: int, seed: int = 0) -> list[int]:
    """Scale degrees, one a word."""
    name = SHAPES[max(0, min(len(SHAPES) - 1, int(shape)))]
    if name == "arch":
        cycle = (0, 1, 2, 3, 4, 3, 2, 1)
        return [cycle[i % len(cycle)] for i in range(n)]
    if name == "rising":
        return [i % 8 for i in range(n)]
    if name == "falling":
        return [7 - i % 8 for i in range(n)]
    if name == "twinkle":
        return [_TWINKLE[i % len(_TWINKLE)] for i in range(n)]
    rng, d, out = random.Random(seed), 2, []
    for _ in range(n):
        out.append(d)
        d = max(0, min(7, d + rng.choice((-2, -1, -1, 0, 1, 1, 2))))
    return out


def notes(n: int, key: int, shape: int, centre: float, seed: int = 0) -> list[float]:
    """MIDI notes for *n* words, the tune's middle placed near *centre*."""
    # The root half an octave below the voice, so the tune (0 to 7 degrees,
    # an octave) straddles where it speaks.
    root = key % 12 + 12 * round((centre - 6 - key % 12) / 12)
    return [float(root + MAJOR[d]) for d in degrees(n, shape, seed)]


def midi_melody(midi_id: str, part_id: str = "") -> list[int]:
    """The notes of one part of an uploaded MIDI file, in order, one at a time.

    *part_id* if it names a part with notes, else the part the analysis called
    the lead, else the highest. Chords are sung as their top note.
    """
    from app.ytpmv import render
    song = render.load_song(midi_id)
    parts = [p for p in song.parts if p.notes and not p.is_drums]
    if not parts:
        return []
    part = (next((p for p in parts if p.id == part_id), None)
            or next((p for p in parts if p.role == "lead"), None)
            or max(parts, key=lambda p: p.median_pitch()))
    out, last = [], -1.0
    for note in sorted(part.notes, key=lambda n: (n.start, -n.pitch)):
        if out and note.start - last < 0.03:
            continue
        out.append(note.pitch)
        last = note.start
    return out


def fit(melody: list[int], n: int, centre: float) -> list[float]:
    """*n* notes of *melody*, round again if it runs out, moved by whole
    octaves to where the voice is."""
    if not melody:
        return []
    shift = 12 * round((centre - float(np.median(melody))) / 12)
    return [float(melody[i % len(melody)] + shift) for i in range(n)]


def sung(v: pitch.Voice, hz: float, n_out: int, amount: float = 1.0,
         vibrato: float = 0.0, formant: float = 1.0) -> np.ndarray:
    """*v* sung on *hz* for *n_out* samples.

    amount   1 puts every pulse on the note; less leaves that much of the
             speech's own pitch in, geometrically, so 0.5 is halfway in cents
    vibrato  depth in semitones either side
    formant  above 1 squeezes each grain, raising the voice's character
    """
    sr, x = v.sr, v.x
    marks, periods, mv = v.marks, v.periods, v.mvoiced
    y = np.zeros(n_out, dtype=np.float32)
    if not len(marks):
        k = min(n_out, len(x))
        y[:k] = x[:k]
        return y
    longest = max(float(periods.max(initial=0.0)), sr / hz) * 2.0 ** (1 + vibrato / 12)
    unv = int(_UNVOICED * sr)
    pad = int(longest) + unv + 4
    out = np.zeros(n_out + 2 * pad, dtype=np.float64)
    t = 0.0
    while t < n_out and t < len(x):
        k = int(np.searchsorted(marks, t))
        if k >= len(marks) or (k > 0 and t - marks[k - 1] < marks[k] - t):
            k -= 1
        c = int(round(t)) + pad
        if mv[k]:
            P = float(periods[k])
            h = (sr / P) ** (1.0 - amount) * hz ** amount
            if vibrato:
                ts = t / sr
                h *= 2.0 ** (vibrato * min(1.0, ts / _VIBRATO_ONSET)
                             * math.sin(2 * math.pi * _VIBRATO_HZ * ts) / 12.0)
            Pt = sr / h
            half = max(2, int(round(P)))
            g = pitch._grain(x, int(marks[k]), half)
            h2 = max(2, int(round(half / formant)))
            if h2 != half:
                g = np.interp(np.linspace(0.0, 2 * half - 1, 2 * h2), np.arange(2 * half), g)
            # Hann grains of half-length h2 laid every Pt add up to h2/Pt.
            out[c - h2:c + h2] += g * min(1.0, Pt / h2)
            t += Pt
        else:
            out[c - unv:c + unv] += pitch._grain(x, int(round(t)), unv)
            t += unv
    y[:] = out[pad:pad + n_out]
    return pitch._fade(y, sr, min(n_out, len(x)))


def sing(path: str, spans: list[dict], options: dict, marks: list | None = None,
         seed: int = 0) -> None:
    """Re-sing the video at *path* in place.

    *spans* is generate.timeline(); *options* is generation_options(); *marks*
    is each token's parse_mark(), by token index, where the text had any.
    """
    if not spans:
        return
    sr = pitch.SR
    x = pitch.decode_audio(path, sr=sr)
    cuts = []
    for s in spans:
        a, b = int(s["start"] * sr), min(int(s["end"] * sr), len(x))
        if b - a > sr // 50:
            cuts.append((s["i"], a, b))
    voices = [pitch.Voice(x[a:b], sr, aperiodic=_APERIODIC) for _, a, b in cuts]
    f0s = [v.info.f0 for v in voices if v.info.f0]
    if not f0s:
        return
    cute = float(options.get("sing_cute", 0.0))
    # Somebody small has a higher voice as well as a smaller throat.
    centre = pitch.hz_to_midi(float(np.median(f0s))) + 9.0 * cute

    tune: list[float] = []
    if options.get("sing_midi"):
        try:
            tune = fit(midi_melody(options["sing_midi"], options.get("sing_part", "")),
                       len(cuts), centre)
        except Exception as exc:              # a file since removed: sing the shape instead
            log.warning("sing: MIDI %s unusable (%s); using the tune shape",
                        options["sing_midi"], exc)
    if not tune:
        tune = notes(len(cuts), int(options.get("sing_key", 0)),
                     int(options.get("sing_shape", 0)), centre, seed)

    y = x.copy()
    for (i, a, b), v, m in zip(cuts, voices, tune):
        mark = marks[i] if marks and i < len(marks) else None
        if mark:
            m = mark[1] if mark[0] == "abs" else m + mark[1]
        if not v.info.f0:
            continue                          # nothing voiced to put on a note
        y[a:b] = sung(v, pitch.midi_to_hz(m), b - a,
                      amount=float(options.get("sing_amount", 1.0)),
                      vibrato=float(options.get("sing_vibrato", 0.0)),
                      formant=1.0 + 0.3 * cute)

    # Beside the video, so the finished file is renamed into place rather than
    # copied across filesystems (/tmp and output/ are different mounts in the container).
    tmp = tempfile.mkdtemp(prefix="sing_", dir=os.path.dirname(os.path.abspath(path)))
    wav, mp4 = os.path.join(tmp, "sung.wav"), os.path.join(tmp, "sung.mp4")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-i", wav,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "192k", "-ac", "2", "-movflags", "+faststart", mp4],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg failed putting the singing back:\n{r.stderr[-1000:]}")
    os.replace(mp4, path)
    os.remove(wav)
    os.rmdir(tmp)
