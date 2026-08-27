#!/usr/bin/env python3
"""Tokeniser cases, as plain asserts.

    python tests/test_tokenize.py

No pytest: this runs in CI next to `compileall`, and a test that needs a
dependency installed is a test that gets skipped.

Everything here is a case that was once wrong. The tokeniser decides which
words get looked up, so a mistake in it is either a word reported missing that
the corpus has, or -- worse -- a different word said confidently.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An empty corpus, so the expansion rules are tested on their own. _expand_token
# prefers a word the corpus actually has, and pointing this at the real corpus
# would make the expected values depend on what someone happened to say.
os.environ["MRS_DATA_DIR"] = tempfile.mkdtemp(prefix="ytp-tokenize-")

from app import generate  # noqa: E402
from app.generate import _expand_token, tokenize_marked  # noqa: E402

CASES: list[tuple[str, list[str]]] = [
    # Letters and digits run together. Left whole these match nothing and have
    # no pronunciation to splice from either, so they could only ever go
    # missing -- which is what "i30" did.
    ("i30",           ["i", "thirty"]),
    ("v8",            ["v", "eight"]),
    ("mk2",           ["mark", "two"]),
    ("330i",          ["three", "hundred", "and", "thirty", "i"]),
    ("4x4",           ["four", "by", "four"]),

    # Units. "km" has no CMU pronunciation at all; the expansion has to land on
    # a spelling the dictionary knows, which is American -- "kilometres" is not
    # in it, so expanding to that swaps one miss for another.
    ("km",            ["kilometers"]),
    ("50km",          ["fifty", "kilometers"]),
    ("100kg",         ["one", "hundred", "kilograms"]),
    ("50mm",          ["fifty", "millimeters"]),
    ("2l",            ["two", "litres"]),

    # Decades, not units. "s" is deliberately not seconds: reading it that way
    # turned "the 80s" into "the eighty seconds".
    ("80s",           ["eighties"]),
    ("90s",           ["nineties"]),
    ("1980s",         ["nineteen", "eighties"]),
    ("5s",            ["five", "s"]),

    # Rates: the slash is the word "per".
    ("km/h",          ["kilometers", "per", "hour"]),
    ("80kmh",         ["eighty", "kilometers", "per", "hour"]),
    ("l/100km",       ["litres", "per", "hundred", "kilometers"]),
    # ...but not every slash is a rate.
    ("and/or",        ["and", "or"]),

    # Decimals. Stripping the point first read "2.0" as "20" and said "twenty":
    # not a miss but a wrong answer, which is worse because nothing reports it.
    ("2.0",           ["two", "point", "zero"]),
    ("3.5l",          ["three", "point", "five", "litres"]),

    # Thousands separators, which used to split into "one" and "five hundred".
    ("1,500",         ["one", "thousand", "five", "hundred"]),

    # The unit outside the word characters was silently dropped.
    ("$50",           ["fifty", "dollars"]),
    ("50%",           ["fifty", "percent"]),

    # Hyphens separated words rather than vanishing: stripping them glued
    # "four-cylinder" into "fourcylinder", which nobody has ever said.
    ("four-cylinder", ["four", "cylinder"]),
    ("turbo-charged", ["turbo", "charged"]),

    ("1st",           ["first"]),
    ("22nd",          ["twenty", "second"]),

    # Ordinary words and empty tokens are left alone.
    ("hello",         ["hello"]),
    ("",              []),
    ("...",           []),
]


def main() -> int:
    failures = []
    for token, expected in CASES:
        got = _expand_token(token)
        if got != expected:
            failures.append(f"  {token!r}: expected {expected}, got {got}")

    # A full stop only ends a sentence at the end of a token. Any period used
    # to count, so "2.0" planted a pause in the middle of a phrase.
    marked = tokenize_marked("the 2.0 litre engine")
    if any(ends for _w, ends, _n, _r in marked[:-1]):
        failures.append(f"  a decimal ended a sentence: {marked}")
    if not marked[-1][1]:
        failures.append("  the last token should always end a sentence")
    if not tokenize_marked("it died. then what")[1][1]:
        failures.append("  a real full stop stopped ending a sentence")

    # ~word~ plays that word backwards. The marker has to come off before
    # anything else looks at the token, or the word never matches the corpus
    # and a marked full stop stops ending its sentence.
    def marks(text):
        return [(w, e, n, r) for w, e, n, r in tokenize_marked(text)]

    got = marks("say ~hello~ now")
    if got[1][:1] != ("hello",) or not got[1][3]:
        failures.append(f"  ~hello~ should be the word 'hello', reversed: {got}")
    if got[0][3] or got[2][3]:
        failures.append(f"  the mark leaked onto its neighbours: {got}")

    got = marks("it is ~gone~.")
    if not got[-1][1]:
        failures.append(f"  ~gone~. should still end a sentence: {got}")
    if not got[-1][3]:
        failures.append(f"  ~gone~. should still be reversed: {got}")

    # A marked token that expands reverses every word it became.
    got = marks("~i30~")
    if [w for w, _e, _n, _r in got] != ["i", "thirty"]:
        failures.append(f"  ~i30~ should still expand: {got}")
    if not all(r for _w, _e, _n, r in got):
        failures.append(f"  every word of ~i30~ should be reversed: {got}")

    # An unmarked token is never reversed, and a stray tilde is not a marker.
    if any(r for _w, _e, _n, r in marks("plain words here")):
        failures.append("  an unmarked token came back reversed")
    if any(r for _w, _e, _n, r in marks("~half marked")):
        failures.append("  a single tilde should not mark anything")

    # What the corpus says beats what the rules would make of it. The
    # transcriber writes some tokens in a form the rules would expand right
    # past -- a real corpus has clips labelled "v8" and "80s" -- and expanding
    # those walks away from a recording of the exact thing being asked for.
    saved = generate._clips_by_word_cache
    try:
        generate._clips_by_word_cache = {"v8": [{}], "80s": [{}]}
        for token, expected in (("v8", ["v8"]), ("80s", ["80s"]),
                                ("V8.", ["v8"]),          # punctuation and case
                                ("i30", ["i", "thirty"])):  # still expands
            got = _expand_token(token)
            if got != expected:
                failures.append(
                    f"  corpus-first {token!r}: expected {expected}, got {got}")
    finally:
        generate._clips_by_word_cache = saved

    if failures:
        print(f"FAILED ({len(failures)}):")
        print("\n".join(failures))
        return 1
    print(f"ok: {len(CASES)} token cases, 3 sentence-boundary cases, "
          f"6 reverse-marker cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
