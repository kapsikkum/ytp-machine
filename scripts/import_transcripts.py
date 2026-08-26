#!/usr/bin/env python3
"""
Import poem transcripts from the Michael Rosen fandom wiki into transcripts/.

Parses a MediaWiki API wikitext dump (==Title== sections), cleans the markup,
matches each poem to a DB source by title, and writes transcripts/<video_id>.txt.

Fetch the dump first (browser UA avoids the bot block):
    curl -s -A "Mozilla/5.0 ... Chrome/120.0 Safari/537.36" \
      "https://michaelrosen.fandom.com/api.php?action=parse&page=<PAGE>&prop=wikitext&format=json&formatversion=2" \
      -o wiki.json

Then:
    python scripts/import_transcripts.py --json wiki.json            # dry run (show matches)
    python scripts/import_transcripts.py --json wiki.json --write    # write transcript files
"""
import argparse
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db, get_db

_NUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
_STOP = {"the", "a", "poem", "kids", "poems", "stories", "with", "michael", "rosen"}


def clean_markup(text: str) -> str:
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)  # [[a|b]] -> b
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)            # [[a]]  -> a
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)                  # {{tpl}}
    text = re.sub(r"<[^>]+>", " ", text)                        # html
    text = text.replace("'''", "").replace("''", "")           # bold/italic
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def norm_title(t: str) -> str:
    t = t.split("|")[0]                       # drop "| POEM | ..."
    t = re.split(r"\s[-–]\s", t)[0]           # drop " - Kids' Poems ..."
    t = re.sub(r"\(.*?\)", " ", t.lower())    # drop parentheticals
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    words = [_NUM.get(w, w) for w in t.split()]
    words = [w for w in words if w not in _STOP]
    return " ".join(words).strip()


def parse_wiki(wikitext: str) -> dict[str, str]:
    poems: dict[str, str] = {}
    parts = re.split(r"^==\s*(.+?)\s*==\s*$", wikitext, flags=re.M)
    # parts = [pre, title1, body1, title2, body2, ...]
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = clean_markup(parts[i + 1])
        if body:
            poems[title] = body
    return poems


def match(db_norm: str, wiki_norms: dict[str, str]) -> tuple[str, float] | None:
    """Return (wiki_title, score) for the best match, or None."""
    best, best_score = None, 0.0
    for wt, wn in wiki_norms.items():
        score = difflib.SequenceMatcher(None, db_norm, wn).ratio()
        # containment bonus (e.g. "harrybo" vs "harrybos grandad")
        if wn and (wn in db_norm or db_norm in wn or db_norm.split()[:1] == wn.split()[:1] and abs(len(wn) - len(db_norm)) < 12):
            score = max(score, 0.9)
        if score > best_score:
            best, best_score = wt, score
    return (best, best_score) if best else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="MediaWiki API json dump")
    ap.add_argument("--out", default="transcripts")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.6)
    args = ap.parse_args()

    init_db()
    data = json.load(open(args.json, encoding="utf-8"))
    poems = parse_wiki(data["parse"]["wikitext"])
    wiki_norms = {t: norm_title(t) for t in poems}

    with get_db() as conn:
        sources = [dict(r) for r in conn.execute(
            "SELECT id, video_id, title FROM sources ORDER BY id").fetchall()]

    if args.write:
        os.makedirs(args.out, exist_ok=True)

    matched = unmatched = 0
    for s in sources:
        dn = norm_title(s["title"])
        m = match(dn, wiki_norms)
        clean_title = s["title"].split("|")[0].strip().encode("ascii", "ignore").decode()
        if m and m[1] >= args.min_score:
            wt, score = m
            flag = "" if score >= 0.85 else "  <-- CHECK"
            print(f"  OK  {clean_title:32s} -> {wt!r} ({score:.2f}){flag}")
            if args.write:
                with open(os.path.join(args.out, f"{s['video_id']}.txt"), "w", encoding="utf-8") as f:
                    f.write(poems[wt])
            matched += 1
        else:
            sc = f" best={m[1]:.2f} ({m[0]!r})" if m else ""
            print(f"  ??  {clean_title:32s} -> NO MATCH{sc}")
            unmatched += 1

    print(f"\n{matched} matched, {unmatched} unmatched. "
          f"{'WROTE files.' if args.write else 'Dry run — add --write.'}")


if __name__ == "__main__":
    main()
