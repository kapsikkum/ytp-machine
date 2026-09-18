#!/usr/bin/env python3
"""Effects on a whole phrase keep it a phrase. Against the michael-rosen corpus.

    python tests/test_run_fx.py

"what are you doing" is one real recording in the corpus. Giving every word
the same effects should still play that recording, carrying them; giving the
words different effects has to split it, and pinning one word's recording
has to take that word out of it.
"""
import os
import random
import sys

os.environ["MRS_CORPUS"] = "michael-rosen"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.generate as g

failures: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


OPTS = g.generation_options({"sentence_pause": 0, "word_gap": 0})


def resolve(text):
    random.seed(1)
    return g.resolve_text(text, options=OPTS)


segs, rep = resolve("what are you doing")
check("plain: one run", rep["runs"], ["what are you doing"])

segs, rep = resolve("what{pitch=3,shake} are{pitch=3,shake} you{pitch=3,shake} doing{pitch=3,shake}")
check("same effects: still one run", rep["runs"], ["what are you doing"])
run = next(s for s in segs if s.get("_tokens"))
check("the run carries the pitch", run.get("pitch"), 3.0)
check("and the glitch", run.get("glitch"), ["shake"])

segs, rep = resolve("what{pitch=3} are{pitch=3} you{pitch=5} doing{pitch=5}")
check("different effects: split where they change", rep["runs"], ["what are", "you doing"])

segs, rep = resolve("what are{clip=1} you doing")
check("a pinned recording leaves the run", "are" in " ".join(rep["runs"]), False)

# A run played at half speed: each word's highlight moves with the sound.
segs, rep = resolve("what{rate=0.5} are{rate=0.5} you{rate=0.5} doing{rate=0.5}")
run = next(s for s in segs if s.get("_tokens"))
check("the run is stretched", run.get("stretch"), 2.0)
slow = g.timeline([run], OPTS)
segs, rep = resolve("what are you doing")
plain = g.timeline([next(s for s in segs if s.get("_tokens"))], OPTS)
check("last word starts twice as late", round(slow[-1]["start"], 2), round(plain[-1]["start"] * 2, 2))

# What a full stop does: footage of silence, a held frame, or nothing.
def stop(style):
    random.seed(1)
    segs, _ = g.resolve_text("what are you doing. what", options={"pause_style": style, "sentence_pause": 0.6})
    silent = [s.get("word") == "" and s.get("_tok") is None and not s.get("_tokens") for s in segs]
    return silent, max(s.get("pause_after", 0.0) for s in segs)
idle0, held0 = stop(0)
idle1, held1 = stop(1)
idle2, held2 = stop(2)
check("full stop: a silent clip by default", any(idle0), True)
check("freeze: no clip, the frame held instead", (any(idle1), held1 >= 0.6), (False, True))
check("none: neither", (any(idle2), held2), (False, 0.0))

print()
print(f"{len(failures)} failure(s)" if failures else "all ok")
sys.exit(1 if failures else 0)
