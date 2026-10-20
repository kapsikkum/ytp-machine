#!/usr/bin/env python3
"""Whisper's words checked against a second listener. See scripts/second_opinion.py.

    python tests/test_second_opinion.py

No model: what Qwen "heard" is given, and what is checked is what is done
with it -- the reported "and" that was really "only thing that happened",
spellings left alone, edited clips untouched, nothing deleted that only one
of them heard, and the database left consistent.
"""
import json
import os
import sqlite3
import sys
import tempfile

data = tempfile.mkdtemp(prefix="ytp-second-opinion-")
os.environ["MRS_DATA_DIR"] = data
os.environ["MRS_CORPUS"] = "heard"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs(os.path.join(data, "corpora", "heard"))

from app import database as db  # noqa: E402
from scripts import second_opinion as so  # noqa: E402

failures: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   ({detail})" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def clips(text, start=0.0, step=0.3, edited=()):
    return [{"id": i + 1, "word": w, "start_time": start + i * step,
             "end_time": start + i * step + step * 0.9, "edited": int(i in edited)}
            for i, w in enumerate(text.split())]


def kinds(actions):
    return [next(k for k in ("relabel", "replace", "insert", "dispute") if k in a) for a in actions]


# ── Deciding ──────────────────────────────────────────────────────────────────
ws = clips("and he looks at us all fed up and he says hes all down hes all you see")
heard = so._tokens("And he looks at us, all fed up, and he says, \"Only thing that happened "
                   "was I got a rotten headache.\" You see")
acts = so.plan(ws, heard)
check("the garbled stretch is replaced by what was said", kinds(acts) == ["replace"], kinds(acts))
check("with the words that were said",
      acts and acts[0]["words"][:4] == ["only", "thing", "that", "happened"], acts and acts[0]["words"])
check("and only the garbled clips go",
      acts and [c["word"] for c in acts[0]["replace"]] == "hes all down hes all".split())

ws = clips("one two cat four five six")
check("a word swapped one for one is relabelled where it stands",
      [(a["relabel"]["id"], a["word"]) for a in so.plan(ws, "one two hat four five six".split())] == [(3, "hat")])

ws = clips("one two play time on to four five six")
check("a different spelling of the same sound is left alone",
      so.plan(ws, "one two playtime onto four five six".split()) == [])
ws = clips("we can hypnotise people right now")
check("hypnotise / hypnotize too", so.plan(ws, "we can hypnotize people right now".split()) == [])
ws = clips("i was born on the 8th of may")
check("a number and its words are the same word",
      so.plan(ws, "i was born on the eighth of may".split()) == [])

ws = clips("you know go to bed go to bed brian i cant")
acts = so.plan(ws, "you know go to bed brian i cant".split())
check("a word only Whisper heard is disputed, not deleted", kinds(acts) == ["dispute"], kinds(acts))

ws = clips("go to bed brian i cant unlock")
acts = so.plan(ws, "go to bed brian erm i cant unlock".split())
check("a word only Qwen heard is added between its neighbours",
      kinds(acts) == ["insert"] and acts[0]["after"]["word"] == "brian" and acts[0]["before"]["word"] == "i")

ws = clips("one two three four five six", edited=(3,))
check("nothing beside a hand-edited clip is changed",
      so.plan(ws, "one two three bore five six".split()) == []
      and so.plan(ws, "one two tree four five six".split()) == [])

ws = clips("cat two three four five hat")
check("a disagreement at a window's edge is taken for the cut, not the words",
      so.plan(ws, "at two three four five ha".split()) == [])

ws = clips(" ".join(["word"] * 200), step=0.4)
w = so.windows(ws)
check("windows cover every word once, in order",
      [i for a, b in w for i in range(a, b)] == list(range(200)))
check("and each is about the length asked for",
      all(ws[b - 1]["end_time"] - ws[a]["start_time"] <= so._WINDOW_MAX for a, b in w[:-1])
      and all(ws[b - 1]["end_time"] - ws[a]["start_time"] >= so._WINDOW_MIN - 1 for a, b in w[:-1]))

times = so._spread(["a", "rotten", "headache"], 1.0, 2.0)
check("added words fill the gap they were heard in, in order",
      times[0][0] == 1.0 and abs(times[-1][1] - 2.0) < 1e-9
      and all(a[1] <= b[0] + 1e-9 for a, b in zip(times, times[1:])))

# ── Applying ──────────────────────────────────────────────────────────────────
db.init_db()
with db.get_db() as conn:
    conn.execute("INSERT INTO sources (id, video_id, source_file) VALUES (1, 'v', 'downloads/v.mp4')")
    words = "he says hes all down then go to bed go to bed brian i cant".split()
    for i, t in enumerate(words):
        conn.execute("INSERT INTO word_clips (id, source_id, word, start_time, end_time, source_file, phones) "
                     "VALUES (?, 1, ?, ?, ?, 'downloads/v.mp4', 'x')", (i + 1, t, i * 0.5, i * 0.5 + 0.4))
    conn.execute("INSERT INTO clip_tone (clip_id, pitch) VALUES (3, 1.0)")
    conn.execute("INSERT INTO boundary_reports (clip_id, kind, count) VALUES (4, 'clipped', 1)")
    conn.execute("INSERT INTO splice_recipes (word, recipe) VALUES ('hesall', ?)",
                 (json.dumps([{"clip_id": 3, "start": 1.0, "end": 1.2}, {"clip_id": 1}]),))
with db.get_db() as conn:
    ws = [dict(r) for r in conn.execute("SELECT * FROM word_clips ORDER BY start_time")]
acts = so.plan(ws, "he says only thing that happened then go to bed brian erm i cant".split())
got = so.apply(1, "downloads/v.mp4", acts)
with db.get_db() as conn:
    now = [r["word"] for r in conn.execute("SELECT word FROM word_clips WHERE source_id=1 ORDER BY start_time, id")]
    disputed = {r[0] for r in conn.execute("SELECT clip_id FROM boundary_reports WHERE kind='disputed'")}
    orphans = conn.execute("SELECT COUNT(*) FROM boundary_reports b LEFT JOIN word_clips w ON w.id=b.clip_id "
                           "WHERE w.id IS NULL").fetchone()[0]
    tone = conn.execute("SELECT COUNT(*) FROM clip_tone WHERE clip_id=3").fetchone()[0]
    recipe = json.loads(conn.execute("SELECT recipe FROM splice_recipes WHERE word='hesall'").fetchone()[0])
    new_paths = {r[0] for r in conn.execute("SELECT DISTINCT source_file FROM word_clips")}
check("the words read as they were said, in time order",
      [w for w in now if w] == "he says only thing that happened then go to bed go to bed brian erm i cant".split(),
      " ".join(now))
check("the duplicated words are kept, and marked", len(disputed) == 3, disputed)
check("nothing is left pointing at a removed clip", orphans == 0 and tone == 0)
check("a recipe pinned to a removed clip chooses again",
      "clip_id" not in recipe[0] and "start" not in recipe[0] and recipe[1] == {"clip_id": 1})
check("added clips point at the source's file", new_paths == {"downloads/v.mp4"}, new_paths)
check("and the counts say so", got == {"relabelled": 0, "replaced": 3, "added": 5, "disputed": 3}, got)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
sys.exit(1 if failures else 0)
