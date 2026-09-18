#!/usr/bin/env python3
"""YTP glitches: markup, the chaos sprinkler, the boomerang, and a real render.

    python tests/test_glitch.py

The render is the test that matters. Every effect goes through ffmpeg on a
synthetic clip, in one chunk, because a filter that changes the frame size
or has a typo fails the whole concat rather than one word.
"""
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.generate as g

failures: list[str] = []


def check(label, got, want, tol=None):
    ok = (abs(got - want) <= tol) if tol is not None else got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


print("markup")
check("flags become glitches", g.parse_fx("shake,earrape,pitch=2"),
      {"glitch": ["shake", "earrape"], "pitch": 2.0})
check("boom is a flag", g.parse_fx("boom"), {"boom": True})
check("carried through effective_fx", g.effective_fx(g.parse_fx("invert,boom")),
      {"glitch": ["invert"], "boom": True})
check("unknown names ignored", g.parse_fx("explode"), {})

print("boomerang")
seg = {"_tok": 0, "pause_after": 0.3, "reverse": False}
b = g._boom_copies(seg)
check("three copies", len(b), 3)
check("middle one backwards", [c["reverse"] for c in b], [False, True, False])
check("only the last keeps the gap", [c["pause_after"] for c in b], [0.0, 0.0, 0.3])

print("chaos")
def sprinkled(seed):
    random.seed(seed)
    segs = [{"_tok": i} for i in range(40)] + [{"idle": True}]
    g._sprinkle(segs, 0.6)
    return segs
a1, a2, other = sprinkled("7"), sprinkled("7"), sprinkled("8")
check("same seed, same wreckage", a1, a2)
check("another seed, other wreckage", a1 != other, True)
check("idle clip left alone", a1[-1], {"idle": True})
check("some words wrecked", sum(1 for s in a1 if s.get("glitch")) > 5, True)

print("render")
tmp = tempfile.mkdtemp()
src = os.path.join(tmp, "src.mp4")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=480x270:r=25:d=3",
                # silent for its first second: acrusher's log mode turns silence into NaN
                "-f", "lavfi", "-i", "aevalsrc='if(lt(t,1),0,sin(1382*t))':d=3", "-shortest", "-c:v", "libx264",
                "-preset", "ultrafast", "-c:a", "aac", src], check=True)
names = [n for n in g.GLITCH if g._glitch_ok(n)]
check("every effect available here", sorted(names), sorted(g.GLITCH))
segs = []
for k, name in enumerate(names):
    s = {"source_file": src, "start_time": 0.2 + 0.1 * (k % 5), "end_time": 0.6 + 0.1 * (k % 5),
         "prev_end": None, "next_start": None, "edited": True, "glitch": [name]}
    if k % 3 == 0:
        s["stretch"] = 1.5
    segs.append(s)
segs += g._boom_copies(dict(segs[0], glitch=[]))
opts = g.generation_options({})
want = sum(g.segment_durations(s, g.extract_window(s), opts)[2] for s in segs)
out = os.path.join(tmp, "out.mp4")
# Three times over: an effect that fails only on some runs (vibrato did,
# handing the encoder NaN) fails the whole video when it does.
for _ in range(3):
    g._encode_chunk(segs, out, final_tail=False, subtitles=False, options=opts)
check("rendered", os.path.exists(out), True)
got = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", out], capture_output=True, text=True).stdout)
check("effects did not change the length", got, want, tol=0.1)

print("joined")
# A video long enough to be encoded in parts and joined. Copying the parts
# together left repeated and missing frames at every join on ffmpeg 7.1 (the
# server's): a corrupt-looking picture in every long video.
many = [dict(segs[k % len(segs)], glitch=[]) for k in range(g._MAX_INPUTS_PER_CALL + 6)]
joined = os.path.join(tmp, "joined.mp4")
g._build_video(many, joined, options=opts)
rows = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "packet=pts_time", "-of", "csv=p=0", joined], capture_output=True, text=True).stdout.split()
pts = sorted(float(r.strip(",")) for r in rows)
check("every frame 40ms after the last, joins included",
      [round(b - a, 3) for a, b in zip(pts, pts[1:]) if not 0.035 < b - a < 0.045], [])

print()
print(f"{len(failures)} failure(s)" if failures else "all ok")
sys.exit(1 if failures else 0)
