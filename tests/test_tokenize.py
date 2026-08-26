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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    ("5s",            ["five", "seconds"]),

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
    if any(ends for word, ends, _ in marked[:-1]):
        failures.append(f"  a decimal ended a sentence: {marked}")
    if not marked[-1][1]:
        failures.append("  the last token should always end a sentence")
    if not tokenize_marked("it died. then what")[1][1]:
        failures.append("  a real full stop stopped ending a sentence")

    if failures:
        print(f"FAILED ({len(failures)}):")
        print("\n".join(failures))
        return 1
    print(f"ok: {len(CASES)} token cases + 3 sentence-boundary cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
