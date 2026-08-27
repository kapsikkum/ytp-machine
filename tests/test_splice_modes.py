#!/usr/bin/env python3
"""Splice-mode plumbing, as plain asserts.

    python tests/test_splice_modes.py

Deliberately does not call find_phoneme_splice: realising a splice imports
app.forced_align, which needs torch, and CI installs neither. What is covered
here is everything that decides *whether* a word can be attempted -- the
pronunciation guess, the phoneme substitution table, and the mode setting --
which is where the new logic lives.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["MRS_DATA_DIR"] = tempfile.mkdtemp(prefix="ytp-splice-")

from app import database  # noqa: E402
from app.phonemes import (_AFFRICATES, _HALF_COST, _SKIP_COST, _SUB_COST,
                          _near, guess_phonemes, word_to_phonemes)  # noqa: E402

failures: list[str] = []


def check(label, got, want):
    if got != want:
        failures.append(f"  {label}: expected {want}, got {got}")


# A guess is only ever consulted for a word the dictionary does not have, so
# anything CMU knows must come back identical -- a guess that overrode a real
# pronunciation would make every mode but strict say words wrong on purpose.
for word in ("ring", "hyundai", "credit", "sorry"):
    check(f"guess matches CMU for {word!r}", guess_phonemes(word), word_to_phonemes(word))

# Real chunks are preferred over spelling. "nurburgring" ends in the actual
# word "ring", and spelling it out letter by letter would lose that.
guess = guess_phonemes("nurburgring")
check("nurburgring ends on a real 'ring'", guess[-3:], ["R", "IH", "NG"])
check("nurburgring starts on an N", guess[0], "N")

# Clusters, not letters: "sch" is one sound and "ck" is one sound.
check("sch is SH", guess_phonemes("schleife")[0], "SH")
check("ck is one K", guess_phonemes("brick"), ["B", "R", "IH", "K"])

# Nothing pronounceable in, nothing out.
check("empty", guess_phonemes(""), None)
check("punctuation only", guess_phonemes("!!!"), None)

# Substitutions are symmetric and never include the phoneme itself: a table
# where P offers B but B does not offer P would make a splice depend on which
# direction it happened to be searched from.
for a, b in (("P", "B"), ("S", "Z"), ("M", "N"), ("IY", "IH")):
    if b not in _near(a):
        failures.append(f"  {a} should accept {b}")
    if a not in _near(b):
        failures.append(f"  {b} should accept {a} (table is not symmetric)")
for p in ("P", "AA", "NG"):
    if p in _near(p):
        failures.append(f"  {p} lists itself as a substitute")

# Far-apart phonemes are not substitutes. Swapping these does not give a word
# said oddly, it gives a different word.
for a, b in (("P", "S"), ("IY", "UW"), ("M", "K")):
    if b in _near(a):
        failures.append(f"  {a} should not accept {b}")

# Cost ordering is the whole safety property of the non-strict modes. A real
# match costs about 1.0 per unit, so each fallback has to sit far enough above
# that no chain of them can undercut a plan made of real matches:
#
#   real match  <  affricate half  <  substitution  <  dropped phoneme
#
# Get this wrong and turning the mode up silently rewrites words the corpus
# could already say exactly -- which is what happened when a substitution was
# priced at 1.6 and bought its way in to save two joins.
if not 1.0 < _HALF_COST < _SUB_COST < _SKIP_COST:
    failures.append(f"  cost ordering broken: half={_HALF_COST} "
                    f"sub={_SUB_COST} skip={_SKIP_COST}")

# Both affricates offer their fricative half. ZH is not listed as a target
# because _EQUIV folds ZH into SH before the target is ever looked up.
check("affricate halves", _AFFRICATES, {"CH": "SH", "JH": "SH"})

# The mode setting: strict unless a corpus says otherwise, and unknown values
# fall back rather than reaching the splicer as something it cannot handle.
database.init_db()
check("default mode", database.splice_mode(), "strict")
for value, expected in (("loose", "loose"), ("DESPERATE", "desperate"),
                        ("nonsense", "strict"), ("", "strict")):
    database.set_setting("splice_mode", value)
    check(f"mode {value!r}", database.splice_mode(), expected)

if failures:
    print(f"FAILED ({len(failures)}):")
    print("\n".join(failures))
    sys.exit(1)
print("ok: pronunciation guess, substitution table, and mode setting")
