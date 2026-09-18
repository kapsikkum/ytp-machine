#!/usr/bin/env python3
"""Two voices in one sentence, without switching the active corpus.

    python tests/test_voices.py

Two made-up corpora, "a" saying "alpha" and "b" saying "beta", each with its
own downloads directory. A word marked {v=b} must come out of b's clips and
b's files, while the active corpus stays a throughout -- switching it for a
read is what once sent ffmpeg to one voice's video inside another's folder.
"""
import os
import sqlite3
import sys
import tempfile

data = tempfile.mkdtemp(prefix="ytp-voices-")
os.environ["MRS_DATA_DIR"] = data
os.environ["MRS_CORPUS"] = "a"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import database as db  # noqa: E402
import app.generate as g  # noqa: E402

failures: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


for slug, word in (("a", "alpha"), ("b", "beta")):
    d = os.path.join(data, "corpora", slug)
    os.makedirs(os.path.join(d, "downloads"))
    conn = sqlite3.connect(os.path.join(d, "corpus.db"))
    conn.executescript(f"""
        CREATE TABLE sources (id INTEGER PRIMARY KEY, video_id TEXT, source_file TEXT, title TEXT, url TEXT);
        CREATE TABLE word_clips (id INTEGER PRIMARY KEY, source_id INTEGER, word TEXT,
                                 start_time REAL, end_time REAL, source_file TEXT);
        INSERT INTO sources VALUES (1, 'v', 'downloads/v.mp4', '{slug} video', '');
        INSERT INTO word_clips VALUES (1, 1, '{word}', 1.0, 1.5, 'downloads/v.mp4');
        CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO settings VALUES ('splice_mode', '{"strict" if slug == "a" else "loose"}');
    """)
    conn.commit()
    conn.close()

opts = g.generation_options({"sentence_pause": 0, "word_gap": 0})
segs, rep = g.resolve_text("alpha beta{v=b}", options=opts)
check("both words found", [t["status"] for t in rep["tokens"]], ["found", "found"])
check("alpha from a's files", os.path.join("corpora", "a") in segs[0]["source_file"], True)
check("beta from b's files", os.path.join("corpora", "b") in segs[1]["source_file"], True)
check("the active corpus never moved", db.active()["slug"], "a")
check("a's clips installed again after", sorted(g._clips_by_word_cache), ["alpha"])
check("beta marked as another voice", rep["tokens"][1].get("voice"), "b")
check("alpha not", rep["tokens"][0].get("voice"), None)

check("without the voice, a cannot say beta",
      [t["status"] for t in g.resolve_text("beta", options=opts)[1]["tokens"]], ["missing"])
check("an unknown voice is the active one",
      [t["status"] for t in g.resolve_text("alpha{v=nobody}", options=opts)[1]["tokens"]], ["found"])

# Anything that asks for the cache mid-sentence -- vote penalties, the
# splicer's phrase lookup -- must get the voice being spoken, not the active
# corpus: that swap made every word after a splice come from the wrong voice.
g._use_voice("b")
g._ensure_cache()
check("asking for the cache keeps the voice in use", sorted(g._clips_by_word_cache), ["beta"])
g._use_voice(None)
g._ensure_cache()
check("and the active one once it is put back", sorted(g._clips_by_word_cache), ["alpha"])
real_penalty = g._penalty_for
g._penalty_for = lambda w: (g._ensure_cache(), real_penalty(w))[1]
segs, rep = g.resolve_text("beta{v=b} zeta{v=b} beta{v=b}", options=opts)   # zeta must be spliced
g._penalty_for = real_penalty
check("words after a splice stay in the voice",
      os.path.join("corpora", "b") in segs[-1]["source_file"], True)
check("active again after a sentence", sorted(g._clips_by_word_cache), ["alpha"])

check("each voice's own settings", db.splice_mode(g._use_voice("b")["settings"]), "loose")
g._use_voice(None)
check("its recordings, with its titles", [c["source"] for c in g.clip_choices("beta", voice="b")], ["b video"])
check("and back afterwards", sorted(g._clips_by_word_cache), ["alpha"])
check("its source video found by id", os.path.normpath(g.voice_source("b", 1)).endswith(os.path.join("b", "downloads", "v.mp4")), True)

print()
print(f"{len(failures)} failure(s)" if failures else "all ok")
sys.exit(1 if failures else 0)
