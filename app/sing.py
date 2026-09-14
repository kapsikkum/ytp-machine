"""Singing: every word of a finished video put on a note of a tune.

The generator makes the video as it always does; this then takes its sound,
hard-tunes each word onto the next note of a melody with the same PSOLA the
YTPMV uses (app/ytpmv/pitch.py), and puts it back. Durations do not change, so
the picture is untouched and still in sync.

A tune is a *shape* walked through the major scale of *key*, pitched round
where the voice already sits, so nobody is dragged an octave out of their range.
"""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
import wave

import numpy as np

from app.ytpmv import pitch

MAJOR = (0, 2, 4, 5, 7, 9, 11, 12, 14, 16)
SHAPES = ("arch", "rising", "falling", "wander", "twinkle")
_TWINKLE = (0, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0)
# How unperiodic a frame may be and still count as voiced. The tracker's own
# 0.35 suits a clean note; this corpus is live recordings, room and audience
# and all, where a loud vowel measured 0.75-0.85 and nothing was tuned at all.
_APERIODIC = 0.8


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
    # The root an octave's worth of choices below the voice, so the tune
    # (0 to 7 degrees, an octave) straddles where it speaks.
    root = key % 12 + 12 * round((centre - 6 - key % 12) / 12)
    return [root + MAJOR[d] for d in degrees(n, shape, seed)]


def sing(path: str, spans: list[dict], key: int = 0, shape: int = 0, seed: int = 0) -> None:
    """Re-sing the video at *path* in place. *spans* is generate.timeline()."""
    if not spans:
        return
    sr = pitch.SR
    x = pitch.decode_audio(path, sr=sr)
    cuts = [(int(s["start"] * sr), int(s["end"] * sr)) for s in spans]
    cuts = [(a, min(b, len(x))) for a, b in cuts if min(b, len(x)) - a > sr // 50]
    voices = [pitch.Voice(x[a:b], sr, aperiodic=_APERIODIC) for a, b in cuts]
    f0s = [v.info.f0 for v in voices if v.info.f0]
    if not f0s:
        return
    centre = pitch.hz_to_midi(float(np.median(f0s)))
    y = x.copy()
    for (a, b), v, m in zip(cuts, voices, notes(len(cuts), key, shape, centre, seed)):
        if not v.info.f0:
            continue                          # nothing voiced to put on a note
        out, _ = v.render(pitch.midi_to_hz(m), (b - a) / sr, "perfect", stretch=False)
        y[a:a + len(out)] = out[:b - a]

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
