#!/usr/bin/env python3
"""Takes that sound alike, chosen together. See app/prosody.py.

    python tests/test_prosody.py

Two made-up recordings of the same three words: one low, quiet and dull, one
high, loud and bright. Measured, they must come out as different tones; asked for a
sentence, the generator must keep to one of them far more often than chance
-- and a corpus never measured, or tone_match at 0, must pick as before.
Also the schema: a v1 corpus migrates to v2 by gaining tables only.
"""
import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile

data = tempfile.mkdtemp(prefix="ytp-prosody-")
os.environ["MRS_DATA_DIR"] = data
os.environ["MRS_CORPUS"] = "toned"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from app import database as db  # noqa: E402
from app import prosody  # noqa: E402
import app.generate as g  # noqa: E402

failures: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   ({detail})" if detail and not ok else ""))
    if not ok:
        failures.append(label)


WORDS = ("hello", "there", "friend")


def voice(path, hz, amp, harmonics):
    """Three 'words' of a buzz at *hz*, at 0.5s, 1.5s and 2.5s."""
    tone = "+".join(f"{amp / k:.3f}*sin(2*PI*{hz * k}*t)" for k in range(1, harmonics + 1))
    gate = "+".join(f"between(t,{0.5 + i},{1.0 + i})" for i in range(3))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"aevalsrc='({tone})*({gate})':s=16000:d=3.5", path], check=True)


# ── A v1 corpus, as the last release shipped them ─────────────────────────────
corpus = os.path.join(data, "corpora", "toned")
os.makedirs(os.path.join(corpus, "downloads"))
voice(os.path.join(corpus, "downloads", "calm.wav"), 110, 0.05, 1)
voice(os.path.join(corpus, "downloads", "loud.wav"), 220, 0.6, 20)
conn = sqlite3.connect(os.path.join(corpus, "corpus.db"))
conn.executescript("""
    CREATE TABLE sources (id INTEGER PRIMARY KEY, video_id TEXT, source_file TEXT, title TEXT, url TEXT);
    CREATE TABLE word_clips (id INTEGER PRIMARY KEY, source_id INTEGER, word TEXT,
                             start_time REAL, end_time REAL, source_file TEXT,
                             edited INTEGER DEFAULT 0, phones TEXT);
    CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
    INSERT INTO settings VALUES ('schema_version', '1');
    INSERT INTO sources VALUES (1, 'c', 'downloads/calm.wav', 'calm', '');
    INSERT INTO sources VALUES (2, 'l', 'downloads/loud.wav', 'loud', '');
""")
for sid, name in ((1, "calm"), (2, "loud")):
    for k, w in enumerate(WORDS):
        conn.execute("INSERT INTO word_clips (source_id, word, start_time, end_time, source_file) "
                     "VALUES (?, ?, ?, ?, ?)", (sid, w, 0.5 + k, 1.0 + k, f"downloads/{name}.wav"))
conn.commit()
before = {t: [c[1] for c in conn.execute(f"PRAGMA table_info({t})")]
          for t in ("sources", "word_clips", "settings")}
conn.close()

print("schema")
db.init_db()
with db.get_db() as c:
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    after = {t: [col[1] for col in c.execute(f"PRAGMA table_info({t})")] for t in before}
check("a v1 corpus migrates to the current version", db.get_schema_version() == db.SCHEMA_VERSION == 2)
check("v2 adds the tone tables", {"clip_tone", "tone_categories"} <= tables)
# What lets a v2 corpus open in code that only knows v1: nothing it reads changed.
check("and alters nothing that was there", after == before, f"{before} -> {after}")
db.forget_ready()
db.init_db()
check("migrating again is a no-op", db.get_schema_version() == 2)
meta = db.corpus_meta()
check("coverage says nothing is measured yet", meta["tone_clips"] == 0 and not meta["needs_upgrade"])

print("unmeasured")
check("an unmeasured corpus loads no tone", all("tone" not in c for cs in g._pool(db.active())["clips_by_word"].values() for c in cs))
check("and weighs every take the same", prosody.weight(prosody.context(1.0), {"source_id": 1}) == 1.0)

print("measuring")
from scripts import measure_tone  # noqa: E402
done, failed = measure_tone.measure()
check("every clip measured", (done, failed) == (6, 0), f"{done}, {failed}")
names = measure_tone.categorise()
check("categories made and named", len(names) >= 2 and all(n for n, _s in names), str(names))
with db.get_db() as c:
    rows = {r["clip_id"]: dict(r) for r in c.execute(
        "SELECT t.*, w.source_id FROM clip_tone t JOIN word_clips w ON w.id = t.clip_id")}
    stats = json.loads(c.execute("SELECT value FROM settings WHERE key='tone_stats'").fetchone()[0])
calm = [r for r in rows.values() if r["source_id"] == 1]
loud = [r for r in rows.values() if r["source_id"] == 2]
gap = np.mean([r["pitch"] for r in loud]) - np.mean([r["pitch"] for r in calm])
check("an octave up measures 12 semitones up", abs(gap - 12) < 0.5, f"{gap:.2f}")
# Against its own recording: a hotter upload or a closer mic is not a louder
# delivery, and every word of an evenly loud recording is ordinary for it.
check("loudness is relative to the recording", max(abs(r["loudness"]) for r in rows.values()) < 1.0)
check("and brighter", min(r["brightness"] for r in loud) > max(r["brightness"] for r in calm))
check("each recording is one category", len({r["category"] for r in calm}) == 1 and len({r["category"] for r in loud}) == 1
      and calm[0]["category"] != loud[0]["category"])
check("join pitches share the pitch scale", stats["pitch_start"] == stats["pitch"] == stats["pitch_end"])

print("weighing")
g.invalidate_cache()
pool = g._pool(db.active())
takes = pool["clips_by_word"]["hello"]
check("measured takes carry their tone", all(c.get("tone") is not None for c in takes))
ctx = prosody.context(1.0)
a, b = sorted(takes, key=lambda c: c["source_id"])
ctx["target"] = a["tone_cat"]
check("the sentence's category is favoured", prosody.weight(ctx, a) > prosody.weight(ctx, b))
prosody.follow(ctx, a)
ctx["target"] = None
check("so is joining on to the take before", prosody.weight(ctx, a) > prosody.weight(ctx, b))
check("but nothing is ruled out", prosody.weight(ctx, b) > 0)
check("strength 0 is no context at all", prosody.context(0.0) is None)
check("target is the category covering most words",
      prosody.target([[a, b], [a], [a, b]]) == a["tone_cat"])

print("choosing")


def together(strength, tries=60):
    """How often a sentence comes out of one recording."""
    same = 0
    for k in range(tries):
        random.seed(k)
        segs, _ = g.resolve_text("hello there friend", options={
            "tone_match": strength, "phrases": False, "word_gap": 0, "sentence_pause": 0})
        same += len({s["source_id"] for s in segs}) == 1
    return same / tries


off, on = together(0.0), together(1.0)
# Three words from two equally likely recordings agree a quarter of the time.
check("off, takes are mixed about as chance has it", off < 0.5, f"{off:.2f}")
check("on, a sentence keeps to one tone", on > 0.9, f"{on:.2f}")

print("realigned")
with db.get_db() as c:
    c.execute("DELETE FROM clip_tone WHERE clip_id=1")
g.invalidate_cache()
check("a clip without a row is simply unmeasured",
      sum("tone" not in c for c in g._pool(db.active())["clips_by_word"]["hello"]) == 1)
check("and is measured on the next run", measure_tone.measure() == (1, 0))

if failures:
    sys.exit(f"\n{len(failures)} failed: {failures}")
print("\nall ok")
