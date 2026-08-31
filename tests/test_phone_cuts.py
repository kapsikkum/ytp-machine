#!/usr/bin/env python3
"""Cutting a sub-word unit out of a clip with aligned phoneme times.

    python tests/test_phone_cuts.py

The splicer takes pieces of words by phoneme index. Where a corpus has been
through scripts/align_phones.py those indices are real boundaries measured
against the audio; where it has not, they are inferred from the spelling by
splitting the word into letter groups.

The letter method is what this replaces, and it was wrong in a way nothing
caught: on a real corpus the group count disagrees with the phoneme count for
a third of words, and on twelve Rosen clips it placed the final phoneme 8 to
135ms early -- the K of "like" arriving with the diphthong still running.

No model and no audio here: the times are the ones the build stored, so this
is the arithmetic that turns them into a cut.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.forced_align import (cut_end_after_phonemes,          # noqa: E402
                              cut_start_before_phonemes, phone_times)

failures: list[str] = []


def check(label, got, want, tol=1e-6):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) and got is not None else got == want
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want!r}")
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got!r}")


# "because", as measured: B IH K AO Z, the Z running well past the stored end.
BECAUSE = {
    "word": "because", "source_file": "nope.mp4",
    "start_time": 10.0, "end_time": 10.78,
    "phones": json.dumps([["B", 0.0, 0.02], ["IH", 0.02, 0.14],
                          ["K", 0.14, 0.24], ["AO", 0.24, 0.58],
                          ["Z", 0.58, 0.82]]),
}

check("times are read back", len(phone_times(BECAUSE) or []), 5)

# ── taking the front off ───────────────────────────────────────────────────
check("skip nothing starts at the word", cut_start_before_phonemes(BECAUSE, 0), 0.0)
check("skip B starts at IH", cut_start_before_phonemes(BECAUSE, 1), 0.02)
check("skip B-IH starts at K", cut_start_before_phonemes(BECAUSE, 2), 0.14)
check("skip to the Z", cut_start_before_phonemes(BECAUSE, 4), 0.58)
# Asking past the end is a caller error; answer with the last phoneme rather
# than raising in the middle of a generation.
check("skip past the end", cut_start_before_phonemes(BECAUSE, 9), 0.58)

# ── taking the back off ────────────────────────────────────────────────────
check("first phoneme only", cut_end_after_phonemes(BECAUSE, 1, last_is_vowel=False),
      0.055)                       # a stop keeps its burst: 20ms would vanish
check("through the vowel", cut_end_after_phonemes(BECAUSE, 4, last_is_vowel=True), 0.58)
check("the whole word", cut_end_after_phonemes(BECAUSE, 5, last_is_vowel=False), 0.82)

# A consonant's minimum never runs past the phoneme that follows it.
SHORT = {
    "word": "at", "source_file": "nope.mp4",
    "start_time": 0.0, "end_time": 0.2,
    "phones": json.dumps([["AE", 0.0, 0.10], ["T", 0.10, 0.12]]),
}
end = cut_end_after_phonemes(SHORT, 2, last_is_vowel=False)
check("a short final stop is stretched", end > 0.12, True)
check("but not past its own phoneme's end", end <= 0.12 + 0.06, True)

# ── a corpus that has never been aligned ───────────────────────────────────
# No stored times and no readable audio: it must decline, not invent a number.
UNALIGNED = {"word": "because", "source_file": "nope.mp4",
             "start_time": 10.0, "end_time": 10.78}
check("no times, no audio, no answer", cut_start_before_phonemes(UNALIGNED, 2), None)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
