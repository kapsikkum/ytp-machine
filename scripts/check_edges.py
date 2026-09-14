#!/usr/bin/env python3
"""How often a picked word comes out cut off, or runs into its neighbours.

    python scripts/check_edges.py --corpus michael-rosen -n 350

Draws words as often as the corpus says them, picks a take for each the way
the generator does -- once preferring takes with a pause around them, once
not -- and judges every cut against wav2vec2's own per-letter alignment of
that word and of its neighbours:

  end cut      the window stops before the word's last letter ends
  start cut    it starts after the first letter begins
  into next    it reaches past where the next word's first letter begins
  from prev    it starts before the previous word's last letter ends

The aligner is not ground truth, and is noisy to about 20ms; what it is good
for is comparing two ways of cutting the same words. Needs torch (the same
model the splicer uses). Alignments are slow, so they are kept in a pickle
in the temp directory between runs.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("-n", type=int, default=350)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    from app import database, forced_align
    if args.corpus:
        database.set_active(args.corpus)
    import app.generate as g
    if not forced_align._ensure_model():
        raise SystemExit("needs the forced-alignment model (torch, torchaudio)")

    # In the temp directory, not the corpus: corpora are committed, and this is not.
    import tempfile
    cache_path = os.path.join(tempfile.gettempdir(), f"edge_alignments_{database.active()['slug']}.pkl")
    fa = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}

    def letters(c):
        if c["id"] not in fa:
            fa[c["id"]] = forced_align.char_times(c)
        return fa[c["id"]]

    g._ensure_cache()
    seqs, cbw = g._ordered_by_source, g._clips_by_word_cache
    where = {c["id"]: (s, i) for s, seq in seqs.items() for i, c in enumerate(seq)}
    uses = [c["word"] for seq in seqs.values() for c in seq if c["word"] in cbw]
    rng = random.Random(args.seed)
    draws = [rng.choice(uses) for _ in range(args.n)]

    def score(clean: bool):
        random.seed(args.seed)
        tally = dict(n=0, adjacent=0, end_cut=0, start_cut=0, into_next=0, from_prev=0)
        for w in draws:
            c = g.pick_clip(cbw[w], w, clean=clean)
            a = letters(c)
            if not a:
                continue
            ws, we = g.extract_window(c)
            tally["n"] += 1
            tally["adjacent"] += c["next_start"] is not None and c["next_start"] - c["end_time"] < g._CLEAN_AFTER
            tally["end_cut"] += we < c["start_time"] + a[-1][2] - 0.02
            tally["start_cut"] += ws > c["start_time"] + a[0][1] + 0.02
            s, i = where[c["id"]]
            seq = seqs[s]
            if i + 1 < len(seq) and seq[i + 1]["start_time"] - c["end_time"] < 0.3:
                b = letters(seq[i + 1])
                if b:
                    tally["into_next"] += we > seq[i + 1]["start_time"] + b[0][1] + 0.02
            if i > 0 and c["start_time"] - seq[i - 1]["end_time"] < 0.3:
                b = letters(seq[i - 1])
                if b:
                    tally["from_prev"] += ws < seq[i - 1]["start_time"] + b[-1][2] - 0.02
        return tally

    rows = {"any take": score(False), "clean takes first": score(True)}
    pickle.dump(fa, open(cache_path, "wb"))
    print(f"{args.n} words from {database.active()['slug']}")
    print(f"  {'':<20}" + "".join(f"{k:>12}" for k in ("adjacent", "end_cut", "start_cut", "into_next", "from_prev")))
    for name, t in rows.items():
        n = max(t["n"], 1)
        print(f"  {name:<20}" + "".join(f"{100 * t[k] / n:11.1f}%"
                                         for k in ("adjacent", "end_cut", "start_cut", "into_next", "from_prev")))


if __name__ == "__main__":
    main()
