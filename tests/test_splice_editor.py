#!/usr/bin/env python3
"""Structural checks for the splice editor API, against a synthetic corpus.

    python tests/test_splice_editor.py

No audio, so anything that has to cut a clip (preview) is out of reach here --
that is checked on the server. This covers the parts that are pure logic:
which words can supply a run of phonemes, and whether a recipe is accepted,
stored, read back and obeyed.
"""
import importlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

d = tempfile.mkdtemp()
os.makedirs(os.path.join(d, "corpora", "t"), exist_ok=True)
os.environ["MRS_DATA_DIR"] = d
os.environ["MRS_CORPUS"] = "t"
os.environ.pop("MRS_DB_PATH", None)

import app.database as db
importlib.reload(db)
db.init_db()

WORDS = ["big", "catch", "bit", "itch", "rich", "beach", "shell", "the"]
with db.get_db() as c:
    c.execute("INSERT INTO sources (id, video_id, title, source_file) "
              "VALUES (1,'v','T','v.mp4')")
    for i, w in enumerate(WORDS, start=1):
        for k in range(2):
            c.execute("INSERT INTO word_clips (source_id, word, start_time, "
                      "end_time, source_file) VALUES (1,?,?,?, 'v.mp4')",
                      (w, i + k * 0.5, i + k * 0.5 + 0.3))

import app.generate as g
from app import phonemes as ph
for m in (g, ph):
    importlib.reload(m)

# The endpoints need FastAPI, which the byte-compile-only CI job does not
# install by default. The phoneme half of this file is the half that decides
# what a splice can be made of, and it runs on a bare interpreter -- so the
# API checks step aside rather than taking the rest down with them.
try:
    import app.editor as ed
    importlib.reload(ed)
except ModuleNotFoundError as exc:
    ed = None
    print(f"SKIP api checks: {exc}")

g._ensure_cache()
CBW = g._clips_by_word_cache or {}

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
    print(f"{'ok ' if got == want else 'FAIL'} {label}: {got!r}")


# ── what the splicer is aiming at ──────────────────────────────────────────
check("phones(bitch)", ph.canonical_phones("bitch"), ["B", "IH", "CH"])

# ── who can supply a run of sounds ─────────────────────────────────────────
srcs = ph.group_sources(["B", "IH"], CBW)
# both are one cut off a two-phone word, so either order is right
check("B IH sources", sorted(s["word"] for s in srcs), ["big", "bit"])
check("B IH is a front cut", [s["cuts"] for s in srcs], [1, 1])

srcs = ph.group_sources(["CH"], CBW)
check("CH sources", sorted(s["word"] for s in srcs), ["beach", "catch", "itch", "rich"])

check("nothing supplies ZZ", ph.group_sources(["ZZ"], CBW), [])

# ── the endpoint ───────────────────────────────────────────────────────────
if ed is None:
    print()
    print(f"{len(fails)} failures" if fails else "ALL PASS (phonemes only)")
    sys.exit(1 if fails else 0)

out = ed.splice_word("Bitch!")
check("endpoint word", out["word"], "bitch")
check("endpoint phones", out["phones"], ["B", "IH", "CH"])
check("endpoint knows it", out["known"], True)
check("no exact clips", out["exact_total"], 0)
check("no recipe yet", out["recipe"], None)

out = ed.splice_sources("b,ih1")
check("sources endpoint (ih1 too)",
      sorted(s["word"] for s in out["sources"]), ["big", "bit"])
check("sources carry clips", len(out["sources"][0]["clip_list"]), 2)

# ── saving one ─────────────────────────────────────────────────────────────
Plan, Group = ed.SplicePlan, ed.SpliceGroup
plan = Plan(groups=[Group(phones=["B", "IH"], source="bit"),
                    Group(phones=["CH"], source="itch")])
check("save", ed.save_recipe("bitch", plan)["saved"], True)
check("read back", ph.user_recipe("bitch"), [(["B", "IH"], ["bit"]), (["CH"], ["itch"])])
check("listed", [r["word"] for r in ed.recipes()["recipes"]], ["bitch"])
check("endpoint shows it", ed.splice_word("bitch")["recipe"],
      [{"phones": ["B", "IH"], "from": ["bit"]}, {"phones": ["CH"], "from": ["itch"]}])

# a recipe that does not spell the word is refused rather than stored dead
for label, bad in [
        ("wrong phones", Plan(groups=[Group(phones=["B", "IH"], source="bit")])),
        ("no source", Plan(groups=[Group(phones=["B", "IH"], source="bit"),
                                   Group(phones=["CH"])])),
]:
    try:
        ed.save_recipe("bitch", bad)
        fails.append(f"{label}: accepted")
        print(f"FAIL {label}: accepted")
    except Exception as exc:
        print(f"ok  {label}: {getattr(exc, 'status_code', '')} "
              f"{str(getattr(exc, 'detail', exc))[:60]}")

check("delete", ed.delete_recipe("bitch")["saved"], False)
check("gone", ph.user_recipe("bitch"), None)

# ── an unknown word ────────────────────────────────────────────────────────
out = ed.splice_word("qqzzq")
check("unknown has no phones", out["phones"], None)

print()
print(f"{len(fails)} failures" if fails else "ALL PASS")
sys.exit(1 if fails else 0)
