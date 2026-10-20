#!/usr/bin/env python3
"""
A second listener: check Whisper's words against Qwen3-ASR's, and fix them.

Whisper is the corpus's only ear, and where it mishears it does not say so.
It garbles a stretch -- "and he says, only thing that happens, I've got a
rotten headache" came out as "and he says hes all down hes all" -- and every
later stage believes it. Worse, the boundary refinement then forces those
wrong words onto the audio, so "and" ended up cut from "only thing that".

Qwen3-ASR transcribes the same audio a second time. Where the two agree the
clip stands. Where they disagree Qwen is taken as the better ear (it got both
reported faults right, word for word), with care for what that could cost:

  - the same number of words: each clip is relabelled where it stands, so
    the cut is kept and only the label changes;
  - a different number: Whisper's clips are replaced by Qwen's words, spread
    over the same stretch -- the refinement that runs next places them;
  - words only Qwen heard are added between their neighbours;
  - words only Whisper heard are not deleted (Qwen drops the odd filler
    too) but marked 'disputed', which makes a take all but never picked and
    keeps it out of the refinement's alignment, where a word that is not
    there drags its neighbours on to the wrong sound;
  - nothing near a hand-edited clip is touched: somebody listened to it.

Run it after transcribing and before refine_boundaries.py. It needs the
qwen-asr package, which pins its own transformers, so it is usually run from
an environment of its own:

    pip install qwen-asr
    python scripts/second_opinion.py --all            # dry run: what would change
    python scripts/second_opinion.py --all --apply
    python scripts/second_opinion.py --source-id 37 --apply

Resumable: sources already checked are skipped (--redo to check again).
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_db, init_db

MODEL = "Qwen/Qwen3-ASR-1.7B"
_SR = 16000
_DONE_KEY = "second_opinion"

# Qwen hears a window at a time. Long enough to give it the sentence, short
# enough that it does not start summarising; cut at the widest gap Whisper
# left near the end, so a word is rarely split between two windows.
_WINDOW_MIN = 20.0
_WINDOW_MAX = 30.0
_WINDOW_PAD = 0.3
# Words this close to a window's edge may be half in it; a disagreement
# touching them is more likely the cut than the transcript, and is left be.
_EDGE_WORDS = 2
_DISPUTED_REPORTS = 3           # as the refinement's 'unheard': 0.2 ** 3


def _fold(w: str) -> str:
    return unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()


def _tokens(text: str) -> list[str]:
    """Words as clips are labelled: lower case, punctuation removed within."""
    out = []
    for tok in text.replace("-", " ").split():
        w = re.sub(r"[^\w]", "", _fold(tok)).lower()
        if w:
            out.append(w)
    return out


def _key(w: str) -> str:
    """What a word is compared by: "8th" and "eighth" are the same word."""
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)?", w)
    if m:
        try:
            from num2words import num2words
            return re.sub(r"[^a-z]", "", num2words(
                int(m.group(1)), to="ordinal" if m.group(2) else "cardinal"))
        except Exception:
            return w
    return w


def _same_sound(have: list[str], heard: list[str]) -> bool:
    """Two spellings of one stretch: "play time"/"playtime", "hypnotise"/"-ize".

    Not a mishearing, and relabelling would only move the word to a spelling
    the corpus is not looked up by.
    """
    a, b = "".join(have), "".join(heard)
    return len(a) == len(b) and all(x == y or {x, y} == {"s", "z"} for x, y in zip(a, b))


def windows(words: list[dict]) -> list[tuple[int, int]]:
    """Index ranges [i, j) of *words* to hear at once."""
    out, i, n = [], 0, len(words)
    while i < n:
        start = words[i]["start_time"]
        j = i + 1
        while j < n and words[j]["end_time"] - start < _WINDOW_MIN:
            j += 1
        cut, widest, k = j, float("-inf"), j
        while k < n and words[k]["end_time"] - start < _WINDOW_MAX:
            gap = words[k]["start_time"] - words[k - 1]["end_time"]
            if gap > widest:
                cut, widest = k, gap
            k += 1
        out.append((i, cut))
        i = cut
    return out


def plan(words: list[dict], heard: list[str]) -> list[dict]:
    """What to change so *words* (one window, in order) says *heard*.

    Returns actions: {"relabel": clip, "word"}, {"replace": [clips], "words"},
    {"insert": words, "after": clip|None, "before": clip|None},
    {"dispute": [clips]}.
    """
    have = [_key(w["word"]) for w in words]
    want = [_key(h) for h in heard]
    n = len(words)
    actions: list[dict] = []
    ops = difflib.SequenceMatcher(None, have, want, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        # At a window edge the disagreement is usually the cut.
        if i1 < _EDGE_WORDS or i2 > n - _EDGE_WORDS:
            continue
        clips = words[i1:i2]
        # Hand-edited clips and their immediate neighbours are final.
        near = words[max(0, i1 - 1):min(n, i2 + 1)]
        if any(c.get("edited") for c in near):
            continue
        new = heard[j1:j2]
        if tag == "replace" and _same_sound(have[i1:i2], [_key(h) for h in new]):
            continue
        if tag == "replace" and len(clips) == len(new):
            for c, h in zip(clips, new):
                if _key(c["word"]) != _key(h):
                    actions.append({"relabel": c, "word": h})
        elif tag == "replace":
            actions.append({"replace": clips, "words": new})
        elif tag == "delete":
            actions.append({"dispute": clips})
        elif tag == "insert":
            actions.append({"insert": new,
                            "after": words[i1 - 1] if i1 > 0 else None,
                            "before": words[i1] if i1 < n else None})
    return actions


def _spread(words: list[str], a: float, b: float) -> list[tuple[float, float]]:
    """Times for *words* across [a, b], by letter count: a guess to be refined.

    Never past *b*: the order of start times is the order of the words."""
    b = max(b, a + 0.001 * len(words))
    weights = [max(1, len(w)) for w in words]
    total, t, out = sum(weights), a, []
    for w in weights:
        d = (b - a) * w / total
        out.append((t, t + d))
        t += d
    return out


# ── The model ─────────────────────────────────────────────────────────────────

_model = None


def _load(device: str | None):
    global _model
    if _model is None:
        import torch
        from qwen_asr import Qwen3ASRModel
        dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if dev == "cuda":
            dev = "cuda:0"
        _model = Qwen3ASRModel.from_pretrained(
            MODEL, dtype=torch.bfloat16 if dev.startswith("cuda") else torch.float32,
            device_map=dev, max_inference_batch_size=8)
        print(f"  second opinion from {MODEL} on {dev}")
    return _model


def _decode(path: str):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(_SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def hear(model, x, spans: list[tuple[float, float]]) -> list[list[str]]:
    """Qwen's words for each (start, end) of the decoded source *x*."""
    clips = [(x[int(max(0.0, a) * _SR):int(b * _SR)], _SR) for a, b in spans]
    out = model.transcribe(clips, language="English")
    return [_tokens(r.text) for r in out]


# ── Applying ──────────────────────────────────────────────────────────────────

def _forget(conn, ids: list[int]) -> None:
    """Drop what was measured about clips whose label or existence changed."""
    for table in ("clip_alignments", "clip_cache", "clip_tone"):
        conn.executemany(f"DELETE FROM {table} WHERE clip_id=?", [(i,) for i in ids])
    conn.executemany("UPDATE word_clips SET phones=NULL WHERE id=?", [(i,) for i in ids])


def _release_recipes(conn, gone: set[int]) -> int:
    """Splice recipes pinned to a removed clip go back to choosing their own."""
    released = 0
    for row in conn.execute("SELECT word, recipe FROM splice_recipes").fetchall():
        try:
            groups = json.loads(row["recipe"])
        except (ValueError, TypeError):
            continue
        if not isinstance(groups, list):
            continue
        changed = False
        for g in groups:
            if isinstance(g, dict) and g.get("clip_id") in gone:
                for field in ("clip_id", "start", "end"):
                    g.pop(field, None)
                changed = True
                released += 1
        if changed:
            conn.execute("UPDATE splice_recipes SET recipe=? WHERE word=?",
                         (json.dumps(groups), row["word"]))
    return released


def apply(sid: int, source_file: str, actions: list[dict]) -> dict:
    counts = {"relabelled": 0, "replaced": 0, "added": 0, "disputed": 0}
    with get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        changed: list[int] = []
        gone: list[int] = []
        for a in actions:
            if "relabel" in a:
                conn.execute("UPDATE word_clips SET word=? WHERE id=?", (a["word"], a["relabel"]["id"]))
                changed.append(a["relabel"]["id"])
                counts["relabelled"] += 1
            elif "dispute" in a:
                conn.executemany(
                    "INSERT INTO boundary_reports (clip_id, kind, count) VALUES (?, 'disputed', ?) "
                    "ON CONFLICT(clip_id, kind) DO UPDATE SET count = excluded.count",
                    [(c["id"], _DISPUTED_REPORTS) for c in a["dispute"]])
                counts["disputed"] += len(a["dispute"])
            else:
                if "replace" in a:
                    clips = a["replace"]
                    lo, hi = clips[0]["start_time"], max(c["end_time"] for c in clips)
                    gone += [c["id"] for c in clips]
                    counts["replaced"] += len(clips)
                else:
                    before, after = a["before"], a["after"]
                    lo = after["end_time"] if after else (before["start_time"] - 0.3)
                    hi = before["start_time"] if before else lo + 0.3
                    if after and hi <= lo:      # neighbours overlap: fit between their starts
                        lo = after["start_time"] + (hi - after["start_time"]) / 2
                counts["added"] += len(a.get("words") or a.get("insert"))
                words = a.get("words") or a["insert"]
                conn.executemany(
                    "INSERT INTO word_clips (source_id, word, start_time, end_time, source_file) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [(sid, w, s, e, source_file) for w, (s, e) in zip(words, _spread(words, max(0.0, lo), hi))])
        _forget(conn, changed + gone)
        if gone:
            for table in ("boundary_reports", "splice_ratings"):
                conn.executemany(f"DELETE FROM {table} WHERE clip_id=?", [(i,) for i in gone])
            conn.executemany("DELETE FROM word_clips WHERE id=?", [(i,) for i in gone])
            _release_recipes(conn, set(gone))
    return counts


def _done(conn) -> set[int]:
    row = conn.execute("SELECT value FROM settings WHERE key=?", (_DONE_KEY,)).fetchone()
    try:
        return set(json.loads(row[0])) if row else set()
    except (TypeError, ValueError):
        return set()


def check_source(model, sid: int, do_apply: bool, verbose: bool) -> dict:
    from app.database import resolve_path
    with get_db() as conn:
        src = conn.execute("SELECT source_file FROM sources WHERE id=?", (sid,)).fetchone()
        words = [dict(r) for r in conn.execute(
            "SELECT id, word, start_time, end_time, edited, source_file FROM word_clips "
            "WHERE source_id=? ORDER BY start_time, id", (sid,))]
    if not src or not words:
        return {}
    x = _decode(resolve_path(src["source_file"]))
    groups = windows(words)
    spans = [(words[i]["start_time"] - _WINDOW_PAD, words[j - 1]["end_time"] + _WINDOW_PAD)
             for i, j in groups]
    actions: list[dict] = []
    for (i, j), heard in zip(groups, hear(model, x, spans)):
        if not heard:
            continue                # silence or a refusal: no opinion
        actions += plan(words[i:j], heard)
    if verbose:
        for a in actions:
            if "relabel" in a:
                print(f"    {a['relabel']['start_time']:8.2f}  {a['relabel']['word']} -> {a['word']}")
            elif "replace" in a:
                print(f"    {a['replace'][0]['start_time']:8.2f}  "
                      f"{' '.join(c['word'] for c in a['replace'])} -> {' '.join(a['words'])}")
            elif "insert" in a:
                at = a["before"]["start_time"] if a["before"] else a["after"]["end_time"]
                print(f"    {at:8.2f}  + {' '.join(a['insert'])}")
            else:
                print(f"    {a['dispute'][0]['start_time']:8.2f}  ? {' '.join(c['word'] for c in a['dispute'])}")
    counts = {"actions": len(actions)}
    if do_apply:
        counts = apply(sid, words[0]["source_file"], actions)
        with get_db() as conn:
            done = _done(conn) | {sid}
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                         (_DONE_KEY, json.dumps(sorted(done))))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--source-id", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--redo", action="store_true", help="check sources already checked")
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet", action="store_true", help="totals only, not every change")
    args = ap.parse_args()
    if not args.all and args.source_id is None:
        ap.error("--source-id or --all")
    init_db()
    with get_db() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM sources ORDER BY id")]
        done = _done(conn)
    if args.source_id is not None:
        ids = [args.source_id]
    elif not args.redo:
        ids = [i for i in ids if i not in done]
    model = _load(args.device)
    totals: dict[str, int] = {}
    for n, sid in enumerate(ids, 1):
        print(f"  [{n}/{len(ids)}] source {sid}", flush=True)
        for k, v in check_source(model, sid, args.apply, not args.quiet).items():
            totals[k] = totals.get(k, 0) + v
    print("  " + (", ".join(f"{v} {k}" for k, v in totals.items()) or "nothing to change")
          + ("" if args.apply else "  (dry run)"))
    if args.apply and totals:
        print("  Now run refine_boundaries.py --all --apply: added and replaced words "
              "are only roughly placed until it does.")


if __name__ == "__main__":
    main()
