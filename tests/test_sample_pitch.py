#!/usr/bin/env python3
"""A short clip's sample keeps the pitch the recommender measured.

    python tests/test_sample_pitch.py

The recommender measures a take from its recording; the renderer used to take
the sound from the take's mp4, whose audio is cut to whole 23ms AAC frames. On
a word that is a few milliseconds nobody hears. On an 80ms phoneme it left one
frame -- no pitch -- so on Morshu every part was recommended a pitched vowel
and then played one with none. An 80ms clip of a 150Hz buzz, built the way a
take is, stands in for Morshu's.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import database  # noqa: E402

database.set_active("michael-rosen")

from app.ytpmv import samples  # noqa: E402
from app.ytpmv.pitch import SR  # noqa: E402

fails = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   ({detail})" if detail else ""))
    if not ok:
        fails.append(label)


tmp = tempfile.mkdtemp(prefix="sample_pitch_")
src = os.path.join(tmp, "buzz.mp4")
# A sawtooth-ish buzz (sine plus harmonics) behind a flat picture, one second.
subprocess.run(["ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", "color=c=gray:s=480x270:r=25:d=1",
                "-f", "lavfi", "-i", "aevalsrc='0.4*sin(2*PI*150*t)+0.2*sin(2*PI*300*t)"
                                     "+0.1*sin(2*PI*450*t)':s=44100:d=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src],
               check=True)
clip = {"id": None, "word": "/zz/", "start_time": 0.4, "end_time": 0.48,
        "source_file": src, "edited": True, "prev_end": 0.4, "next_start": 0.48}
samples.unit_takes = lambda _u: [clip]

v = samples.build("/zz/", 0).voice
check("an 80ms clip is not cut to one AAC frame", len(v.x) / SR > 0.06,
      f"{len(v.x) / SR * 1000:.0f}ms")
check("and keeps its pitch", v.info.f0 is not None and abs(v.info.f0 - 150) < 3,
      f"{v.info.f0 and round(v.info.f0, 1)}Hz")

try:
    os.remove(samples.build("/zz/", 0).mp4)
except OSError:
    pass
print()
sys.exit(1 if fails else 0)
