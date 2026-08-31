#!/usr/bin/env python3
"""
Transcript-based word correction.

Whisper sometimes mishears a word (the audio is correct, the *label* is wrong).
This aligns a source's stored transcription against the real poem text and
relabels only high-confidence 1:1 substitutions — where the misheard word and
the true word are phonetically similar, i.e. the same audio, wrong spelling.
Timestamps are never touched.

Insertions/deletions (performance ad-libs, repeats, sound effects) are left
alone — only same-length, phonetically-close swaps are corrected.

Usage:
    python scripts/correct.py --source-id 16 --ref-file hotfood.txt          # dry run
    python scripts/correct.py --source-id 16 --ref-file hotfood.txt --apply  # write
"""
import argparse
import difflib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db, get_db

_SIM_NONWORD = 0.5   # min similarity to fix a Whisper non-word (e.g. "heared")
_SIM_CLOSE   = 0.8   # min similarity to fix a valid-word swap (e.g. "potatoes")
_SIM_NW_TARGET = 0.77  # min similarity to relabel TO a non-CMU word (proper
                       # nouns / UK spellings: euston, conker, scheddle)

_WORDS: set[str] | None = None


def _is_word(w: str) -> bool:
    """True if *w* is a real English word (in the CMU pronouncing dictionary)."""
    global _WORDS
    if _WORDS is None:
        try:
            from nltk.corpus import cmudict
            _WORDS = set(cmudict.dict().keys())
        except LookupError:
            import nltk
            nltk.download("cmudict", quiet=True)
            from nltk.corpus import cmudict
            _WORDS = set(cmudict.dict().keys())
    return w in _WORDS


def _norm(text: str) -> list[str]:
    """Normalise exactly like ingest does: strip all non-word chars WITHIN each
    token (so "don't" → "dont", matching the stored clip labels) rather than
    splitting on them (which would yield "don t" and break 1:1 alignment)."""
    out = []
    for tok in text.split():
        w = re.sub(r"[^\w]", "", tok).lower()
        if w:
            out.append(w)
    return out


def _fold(w: str) -> str:
    """Strip accents (café → cafe) for comparing against stored labels."""
    import unicodedata
    return unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()


def _spoken_number(w: str) -> str | None:
    """'4' → 'four', '15' → 'fifteen', '7th' → 'seventh' (single-token only)."""
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)?", w)
    if not m:
        return None
    from num2words import num2words
    spoken = num2words(int(m.group(1)), to="ordinal" if m.group(2) else "cardinal")
    return re.sub(r"[^\w]", "", spoken)


def _phones(w: str) -> list[str] | None:
    try:
        from app.phonemes import word_to_phonemes
        return word_to_phonemes(w)
    except Exception:
        return None


def _one_phone_sub(old: str, new: str) -> bool:
    """True when the two words differ by at most ONE substituted phoneme —
    i.e. the same audio with a different spelling (sex/six, full/fool,
    mom/mum, hear/here).  Char similarity misses these."""
    po, pn = _phones(old), _phones(new)
    if not po or not pn or len(po) != len(pn):
        return False
    return sum(1 for a, b in zip(po, pn) if a != b) <= 1


# Spelling-convention swaps that aren't mislabels: the audio matches the stored
# label fine, and the reference's spelling is a rarer input token.
_SKIP_PAIRS = {("miss", "ms")}


def _accept(old: str, new: str, sim: float, old_nonword: bool) -> bool:
    """Decide whether relabelling *old* → *new* is safe."""
    if (old, new) in _SKIP_PAIRS:
        return False
    if new.isdigit():
        return False              # digit labels never match tokenised input
    if _fold(new) == old:
        return False              # accent-only difference (café) — keep ascii
    if _spoken_number(old) == new:
        return True               # '4' → 'four': digit labels are dead clips
    if _is_word(new) and (sim >= _SIM_CLOSE or (old_nonword and sim >= _SIM_NONWORD)):
        return True
    if _one_phone_sub(old, new):
        return True
    if not _is_word(new) and sim >= _SIM_NW_TARGET:
        return True               # proper noun / UK spelling missing from CMU
    return False


def plan_corrections(source_id: int, reference_text: str):
    """Return (corrections, stats).  corrections = list of dicts with the row id,
    old word, new word and similarity.

    A swap is accepted when it is phonetically close (sim >= _SIM_CLOSE) OR when
    Whisper produced a non-word (not in the dictionary) and the fix is at least
    loosely similar (sim >= _SIM_NONWORD)."""
    with get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT id, word, start_time FROM word_clips WHERE source_id=? ORDER BY start_time",
            (source_id,)).fetchall()]

    whisper = [r["word"] for r in rows]
    ref = _norm(reference_text)

    sm = difflib.SequenceMatcher(a=whisper, b=ref, autojunk=False)
    corrections = []
    n_replace_blocks = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        n_replace_blocks += 1
        # Only handle equal-length blocks so the 1:1 pairing is unambiguous.
        if (i2 - i1) != (j2 - j1):
            continue
        for k in range(i2 - i1):
            old = whisper[i1 + k]
            new = ref[j1 + k]
            if old == new:
                continue
            sim = difflib.SequenceMatcher(None, old, new).ratio()
            nonword = not _is_word(old)
            if _accept(old, new, sim, nonword):
                corrections.append({
                    "id": rows[i1 + k]["id"],
                    "old": old,
                    "new": new,
                    "sim": sim,
                    "nonword": nonword,
                    "t": rows[i1 + k]["start_time"],
                })

    stats = {
        "whisper_words": len(whisper),
        "ref_words": len(ref),
        "ratio": sm.ratio(),
        "replace_blocks": n_replace_blocks,
    }
    return corrections, stats


def _apply(corrections: list[dict]) -> None:
    with get_db() as conn:
        for c in corrections:
            # The phoneme times describe the sounds of the *old* word, so
            # they cannot survive a correction. Dropped rather than moved:
            # there is nothing to move them to until the clip is realigned.
            conn.execute("UPDATE word_clips SET word=?, phones=NULL WHERE id=?",
                         (c["new"], c["id"]))


def _run_one(source_id: int, reference: str, apply: bool, verbose: bool = True) -> int:
    corrections, stats = plan_corrections(source_id, reference)
    if verbose:
        print(f"source {source_id}: {stats['whisper_words']} whisper vs "
              f"{stats['ref_words']} reference words  (match {stats['ratio']*100:.0f}%)  "
              f"-> {len(corrections)} corrections")
        for c in sorted(corrections, key=lambda x: x["t"]):
            tag = "non-word" if c["nonword"] else "close   "
            print(f"    [{c['t']:7.2f}s]  {c['old']!r:16s} -> {c['new']!r:16s}  ({tag} sim {c['sim']:.2f})")
    if apply and corrections:
        _apply(corrections)
    return len(corrections)


def main() -> None:
    ap = argparse.ArgumentParser(description="Correct misheard word labels from a reference transcript.")
    ap.add_argument("--source-id", type=int, help="single source id")
    ap.add_argument("--ref-file", help="UTF-8 text file with the real poem (single mode)")
    ap.add_argument("--all", action="store_true",
                    help="batch: correct every source that has transcripts/<video_id>.txt")
    ap.add_argument("--transcripts-dir", default="transcripts")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    init_db()

    total = 0
    if args.all:
        with get_db() as conn:
            sources = [dict(r) for r in conn.execute(
                "SELECT id, video_id, title FROM sources ORDER BY id").fetchall()]
        done = 0
        for s in sources:
            path = os.path.join(args.transcripts_dir, f"{s['video_id']}.txt")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                ref = f.read()
            print(f"\n[{s['video_id']}] {(s['title'] or '')[:50]}")
            total += _run_one(s["id"], ref, args.apply)
            done += 1
        print(f"\n{'-'*60}\nProcessed {done} source(s) with transcripts, "
              f"{total} corrections {'applied' if args.apply else '(dry run)'}.")
    else:
        if args.source_id is None or not args.ref_file:
            ap.error("provide --source-id and --ref-file, or use --all")
        with open(args.ref_file, encoding="utf-8") as f:
            reference = f.read()
        total = _run_one(args.source_id, reference, args.apply)
        if not total:
            print("  (no corrections)")

    if args.apply and total:
        try:
            from app.generate import invalidate_cache
            invalidate_cache()
        except Exception:
            pass
        print("Done. Call POST /api/reload (or restart) so the server picks up changes.")
    elif not args.apply:
        print("DRY RUN — re-run with --apply to write.")


if __name__ == "__main__":
    main()
