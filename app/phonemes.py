"""
Sub-word phoneme splice planning.

Build a missing word from the fewest source units, where each unit is the
leading phonemes of a source word.  When a unit uses only part of a word it is
CUT at an accurate sub-word boundary via forced alignment — so "pee" comes from
"peach", not the whole word.  (This relies on frame-accurate clip boundaries;
see scripts/refine_boundaries.py.)

Splices are capped at _MAX_UNITS pieces; anything needing more is reported as
missing rather than emitted as gibberish.

Example
-------
Target "penis" (typed "pee nes") → "pee"(peach, cut) + "nes"(…)
"""

import re
import random
from collections import defaultdict
from functools import lru_cache
from typing import Any

_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
           "IH", "IY", "OW", "OY", "UH", "UW"}

# Function words tend to be spoken with a reduced, throwaway vowel — a poor
# source for a stressed target vowel (the "at" in "cat" wants "fat", not "at").
_FUNCTION_WORDS = {
    "a", "an", "the", "to", "of", "and", "is", "it", "in", "on", "or", "as",
    "at", "but", "for", "so", "if", "by", "be", "he", "we", "do", "my", "me",
    "you", "your", "his", "her", "i", "am", "are", "was", "that", "this",
    "with", "they", "them", "then", "than",
}


_EQUIV = {"ZH": "SH"}   # phonemes with no corpus coverage → nearest available
_MIN_VOWEL_CUT = 0.16   # a vowel-ending cut unit must be at least this long
_STOPS = {"B", "D", "G", "P", "T", "K", "CH", "JH"}   # need a release to be heard

# Phonemes close enough to stand in for one another when the corpus has no
# coverage of the one actually wanted. Only consulted outside strict mode.
#
# Grouped by how they are made rather than how they sound to a listener: the
# pairs here differ in one feature -- voicing (P/B, S/Z), place (M/N/NG), or a
# neighbouring vowel height -- so substituting one leaves a word that is still
# recognisably the word, said oddly. Anything further apart is not a
# substitution, it is a different word.
_SIMILAR_GROUPS = [
    # vowels
    {"IY", "IH"}, {"IH", "EH"}, {"EY", "EH"}, {"EH", "AE"}, {"AE", "AA"},
    {"AA", "AO"}, {"AO", "OW"}, {"OW", "UH"}, {"UH", "UW"}, {"AH", "ER"},
    {"AH", "AA"}, {"AY", "AA"}, {"AW", "AA"}, {"OY", "AO"},
    # consonants: voiced/voiceless pairs
    {"P", "B"}, {"T", "D"}, {"K", "G"}, {"F", "V"}, {"S", "Z"},
    {"SH", "ZH"}, {"CH", "JH"}, {"TH", "DH"},
    # consonants: same manner, near place
    {"M", "N"}, {"N", "NG"}, {"L", "R"}, {"S", "SH"}, {"Z", "ZH"},
    {"CH", "SH"}, {"JH", "ZH"}, {"TH", "F"}, {"DH", "V"}, {"W", "V"},
    {"HH", "F"}, {"Y", "IY"}, {"R", "ER"},
]


@lru_cache(maxsize=256)
def _near(phone: str) -> tuple[str, ...]:
    """Phonemes that may stand in for *phone*, nearest-first is not attempted --
    they are all one feature away, so the DP's costs decide between them."""
    out: set[str] = set()
    for group in _SIMILAR_GROUPS:
        if phone in group:
            out |= group
    out.discard(phone)
    return tuple(sorted(out))


# What a substitution and a dropped phoneme cost the DP.
#
# These are not "expensive", they are prohibitive, and deliberately so. A join
# costs about 1.0, so a substitution priced at 1.6 buys its way into a word the
# corpus could already say exactly, just to save two joins -- which is a worse
# answer to a question nobody asked. Priced this far above any achievable
# saving, an exact splice always wins where one exists, and these appear only
# where the alternative is reporting the word missing.
_SUB_COST = 10.0
_SKIP_COST = 25.0

# An affricate is a stop welded to a fricative: CH is [t] then [sh], JH is [d]
# then [zh]. So a corpus with no SH at all still physically contains one, in the
# back half of every "ch" -- the splicer just could not see it, because CH is a
# single ARPAbet symbol and units are whole phonemes.
#
# Taking that half is a real SH, not a stand-in, so it is priced well below a
# substitution. It is still above an ordinary join because it is a cut inside a
# single phoneme, which is finer than the alignment was ever tuned for.
_AFFRICATES = {"CH": "SH", "JH": "SH"}      # ZH targets are mapped to SH by _EQUIV
_HALF_COST = 2.0

# Where the frication starts inside the affricate, as a fraction of its span.
# The stop closure and burst come first and are the part that must not survive:
# leave any of the burst in and "wish" keeps sounding like "witch".
_AFFRICATE_SPLIT = 0.45


def _is_vowel(phone: str) -> bool:
    return phone in _VOWELS


# ── CMU dict ──────────────────────────────────────────────────────────────────

def _load_cmudict() -> dict[str, list[list[str]]]:
    """Return all pronunciations per word (list of phoneme lists)."""
    try:
        from nltk.corpus import cmudict
        raw = cmudict.dict()
    except LookupError:
        import nltk
        nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict
        raw = cmudict.dict()
    return {
        word: [[re.sub(r"\d+", "", p) for p in pron] for pron in prons]
        for word, prons in raw.items()
    }


_DICT: dict[str, list[str]] | None = None

def _dict() -> dict[str, list[str]]:
    global _DICT
    if _DICT is None:
        _DICT = _load_cmudict()
    return _DICT


_MAX_UNITS = 8   # max pieces in a splice


# Pronunciations for words the CMU dict lacks (slang, irregulars).  Extend freely.
_OVERRIDES = {
    "clit": ["K", "L", "IH", "T"],
    "shat": ["SH", "AE", "T"],
    "arse": ["AA", "R", "S"],
    "arsehole": ["AA", "R", "S", "HH", "OW", "L"],
    "wank": ["W", "AE", "NG", "K"],
    "cum": ["K", "AH", "M"],          # = "come"
    "skyfoogle": ["S", "K", "AY", "F", "UW", "G", "AH", "L"],
    "oclock": ["AH", "K", "L", "AA", "K"],    # tokenizer strips the apostrophe
    # "sex" sounds soft when spliced; the corpus "six" clips were literally
    # misheard AS "sex" by Whisper, so matching six's phonemes makes the DP
    # use one of those whole clips — no cuts, and it already sounds right.
    "sex": ["S", "IH", "K", "S"],
    "shhhh": ["SH"],                  # the long shush in Harrybo's Grandad

    # Australian and workshop vocabulary. CMU is an American dictionary, so
    # none of these are in it -- and a word with no pronunciation cannot be
    # spliced at all, however many of its sounds the corpus happens to hold.
    # They were simply unsayable, which for a channel with a video called
    # "Dodgy Bogan Exhaust Repairs" is a real gap rather than a nicety.
    #
    # This makes a splice possible, not certain: the corpus still has to
    # contain the sounds.
    "ute":       ["Y", "UW", "T"],
    "utes":      ["Y", "UW", "T", "S"],
    "dyno":      ["D", "AY", "N", "OW"],
    "esky":      ["EH", "S", "K", "IY"],
    "tradie":    ["T", "R", "EY", "D", "IY"],
    "dodgy":     ["D", "AA", "JH", "IY"],
    "donk":      ["D", "AA", "NG", "K"],
    "knackered": ["N", "AE", "K", "ER", "D"],
    "shonky":    ["SH", "AA", "NG", "K", "IY"],
    "bodgy":     ["B", "AA", "JH", "IY"],
}


# Hand-tuned splice plans for words whose DP-chosen splice sounds weak.
# target → list of (phoneme_group, preferred_source_words).  The groups must
# concatenate to the target's phonemes; each group is cut from the first
# preferred source word that aligns (then any other word containing it).
# A preferred entry with spaces ("ive got to do it") means: the clip of the
# LAST word as spoken right after the preceding words in one source — for
# picking a specific delivery of a common word.  A leading "*" ("*shhhh")
# means: use the clip verbatim, no forced-alignment trim (for noise-like words
# FA can't segment); "*shhhh:0.10-0.22" takes a random-length slice (seconds)
# from a random position instead of the whole clip — varies every generation.
_RECIPES: dict[str, list[tuple[list[str], list[str]]]] = {
    # a short random slice of the shhhh + an "it" from the "do it" deliveries
    # (several sources, picked per-generation for variety)
    "shit":  [(["SH"], ["*shhhh:0.10-0.22"]),
              (["IH", "T"], ["do it"])],
    # f + the "uck" of ducks — much harder attack than the DP's f|stuck pick
    "fuck":  [(["F"], ["four", "fool", "five", "food"]),
              (["AH", "K"], ["ducks"])],
    "fucks": [(["F"], ["four", "fool", "five", "food"]),
              (["AH", "K", "S"], ["ducks"])],
    # k + a word-final "-ock" (clean stop — the old unlock pick had an echo tail)
    "cock":  [(["K"], ["catch", "key"]),
              (["AA", "K"], ["lock", "oclock"])],
}


def _base_phones(b: str) -> list[str] | None:
    prons = _dict().get(b)
    return prons[0] if prons else None


def _direct(w: str) -> list[str] | None:
    """Phonemes from the override table or CMU dict only (no fallbacks)."""
    if w in _OVERRIDES:
        return list(_OVERRIDES[w])
    prons = _dict().get(w)
    return prons[0] if prons else None


def _resolve_simple(w: str) -> list[str] | None:
    """Direct lookup, then inflection rules (plurals / 3rd-person / -ing / -ed)."""
    r = _direct(w)
    if r:
        return r
    if len(w) > 3 and w.endswith("s"):
        if w.endswith("es"):
            bp = _base_phones(w[:-2])
            if bp:
                last = bp[-1]
                if last in ("S", "Z", "SH", "ZH", "CH", "JH"):
                    return bp + ["IH", "Z"]
                return bp + (["S"] if last in ("P", "T", "K", "F", "TH") else ["Z"])
        bp = _base_phones(w[:-1])
        if bp:
            last = bp[-1]
            return bp + (["S"] if last in ("P", "T", "K", "F", "TH") else ["Z"])
    if len(w) > 4 and w.endswith("ing"):
        bp = _base_phones(w[:-3]) or _base_phones(w[:-3] + "e")
        if bp:
            return bp + ["IH", "NG"]
    if len(w) > 3 and w.endswith("ed"):
        bp = _base_phones(w[:-2]) or _base_phones(w[:-2] + "e") or _base_phones(w[:-1])
        if bp:
            last = bp[-1]
            if last in ("T", "D"):
                return bp + ["IH", "D"]
            return bp + (["T"] if last in ("P", "K", "F", "S", "SH", "CH", "TH") else ["D"])
    return None


def word_to_phonemes(word: str) -> list[str] | None:
    """Return stripped ARPAbet phonemes for *word*, or None.

    Tries: overrides → CMU → inflections → compound split (run-together words
    like "dumbass" = dumb + ass, "asshole" = ass + hole).
    """
    w = word.lower()
    r = _resolve_simple(w)
    if r:
        return r
    # compound: longest balanced split where both halves are real words
    best = None
    for i in range(len(w) - 2, 1, -1):       # prefer a longer first half
        a, b = _direct(w[:i]), _direct(w[i:])
        if a and b:
            best = a + b
            break
    return best


# Letter clusters to ARPAbet, longest cluster first, for words no dictionary
# knows. English spelling being what it is, this is an approximation and often
# a poor one -- it exists so that "nordschleife" can be attempted at all rather
# than pronounced correctly.
_SPELL: list[tuple[str, list[str]]] = [
    ("sch", ["SH"]), ("tch", ["CH"]), ("igh", ["AY"]), ("ough", ["AH", "F"]),
    ("ch", ["CH"]), ("sh", ["SH"]), ("th", ["TH"]), ("ph", ["F"]),
    ("ck", ["K"]), ("ng", ["NG"]), ("qu", ["K", "W"]), ("wh", ["W"]),
    ("gh", ["G"]), ("kn", ["N"]), ("wr", ["R"]), ("ps", ["S"]),
    ("ee", ["IY"]), ("ea", ["IY"]), ("oo", ["UW"]), ("ou", ["AW"]),
    ("ow", ["AW"]), ("ai", ["EY"]), ("ay", ["EY"]), ("oi", ["OY"]),
    ("oy", ["OY"]), ("au", ["AO"]), ("aw", ["AO"]), ("ei", ["EY"]),
    ("ie", ["IY"]), ("ue", ["UW"]), ("ar", ["AA", "R"]), ("er", ["ER"]),
    ("ir", ["ER"]), ("ur", ["ER"]), ("or", ["AO", "R"]),
    ("a", ["AE"]), ("b", ["B"]), ("c", ["K"]), ("d", ["D"]), ("e", ["EH"]),
    ("f", ["F"]), ("g", ["G"]), ("h", ["HH"]), ("i", ["IH"]), ("j", ["JH"]),
    ("k", ["K"]), ("l", ["L"]), ("m", ["M"]), ("n", ["N"]), ("o", ["AA"]),
    ("p", ["P"]), ("q", ["K"]), ("r", ["R"]), ("s", ["S"]), ("t", ["T"]),
    ("u", ["AH"]), ("v", ["V"]), ("w", ["W"]), ("x", ["K", "S"]),
    ("y", ["IY"]), ("z", ["Z"]),
]


def guess_phonemes(word: str) -> list[str] | None:
    """A pronunciation for a word nothing in the dictionary covers.

    Real pronunciations come first: the longest prefix CMU actually knows is
    consumed as a chunk, so "nurburgring" gives up its real "ring" rather than
    being spelled out letter by letter. Only what is left over falls back to
    the spelling table.

    This is a guess and is only reached outside strict mode. Strict mode
    reports the word missing instead, which is the honest answer -- an
    approximated pronunciation spliced out of approximate phonemes is a
    plausible-sounding noise, not the word.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return None

    out: list[str] = []
    i = 0
    while i < len(w):
        # A real chunk, longest first. Three letters is the floor: shorter
        # "words" are mostly CMU's single letters and abbreviations, which
        # spell the word out loud instead of pronouncing it.
        chunk = None
        for j in range(len(w), i + 2, -1):
            got = _direct(w[i:j])
            if got:
                chunk = (got, j)
                break
        if chunk:
            out.extend(chunk[0])
            i = chunk[1]
            continue
        for cluster, phones in _SPELL:
            if w.startswith(cluster, i):
                out.extend(phones)
                i += len(cluster)
                break
        else:
            i += 1                      # nothing matched: skip the character
    return out or None


# ── Sub-word splice DP ─────────────────────────────────────────────────────────

def _best_clip(clips: list[dict], word: str, penalty: dict[int, int] | None):
    """Pick a clip for *word*, preferring ones the user hasn't down-rated for
    the current target.  Falls back to the full pool when every clip is rated
    down (so a splice is still produced — just from the least-bad option)."""
    from app.generate import pick_clip
    if penalty:
        clean = [c for c in clips if penalty.get(c.get("id"), 0) <= 0]
        if clean:
            return pick_clip(clean, word)
    return pick_clip(clips, word)


def find_phoneme_splice(
    target_word: str,
    clips_by_word: dict[str, list[dict[str, Any]]],
    penalty: dict[int, int] | None = None,
    mode: str = "strict",
) -> list[dict[str, Any]] | None:
    """
    Cover *target_word*'s phonemes with the fewest source units and return a list
    of clip segment dicts, or None.

    A unit uses the leading ``matched`` phonemes of a source word.  When that is
    only part of the word, the segment is cut at an accurate sub-word boundary
    (forced alignment); otherwise the whole word clip is used.

    Returned segment dicts are ordinary word_clips rows with start_time/end_time
    set to the chosen span and prev_end/next_start clamped so the video builder
    extracts exactly that span (sub-word units) or applies normal padding
    (whole-word units).
    """
    raw = word_to_phonemes(target_word)
    if not raw and mode != "strict":
        raw = guess_phonemes(target_word)
    if not raw:
        return None
    approximate = mode != "strict" and not word_to_phonemes(target_word)
    target_phones = [_EQUIV.get(p, p) for p in raw]   # ZH→SH etc. (close enough)

    n = len(target_phones)
    d = _dict()

    # Index: phoneme → [(word, phones, clips, pos)] at EVERY position, so we can
    # match a source word's prefix, suffix, or interior (the "at" in "fat").
    # Source phonemes are canonicalised the same way so e.g. SH sources can cover
    # a ZH target (no clip in the corpus actually contains ZH).
    index: defaultdict[str, list] = defaultdict(list)
    half_index: defaultdict[str, list] = defaultdict(list)
    for word, clips in clips_by_word.items():
        w = word.lower()
        if w in _OVERRIDES:               # non-CMU corpus words (shhhh, oclock…)
            cphones = [_EQUIV.get(p, p) for p in _OVERRIDES[w]]
        else:
            prons = d.get(w)
            if not prons:
                continue
            # Only the PRIMARY pronunciation — secondary CMU variants often
            # don't match the audio (e.g. "get" has a G-IH-T variant, but
            # Michael says "g-eh-t", so using it for "shit"'s IH-T gives
            # "sh-et").
            cphones = [_EQUIV.get(p, p) for p in prons[0]]
        for pos in range(len(cphones)):
            index[cphones[pos]].append((word, cphones, clips, pos))
            half = _AFFRICATES.get(cphones[pos])
            if half:
                half_index[half].append((word, cphones, clips, pos))

    # Hand-tuned recipe?  Build the unit list directly from it when every
    # group can be sourced; otherwise fall through to the DP.
    recipe = _RECIPES.get(target_word.lower())
    if recipe and [p for grp, _pref in recipe for p in grp] == target_phones:
        chosen = _chosen_from_recipe(recipe, index, penalty)
        if chosen is not None:
            return _realise(chosen, target_phones, index, clips_by_word, penalty)

    INF = float("inf")
    dp: list[float]           = [INF] * (n + 1)
    prev: list[tuple | None]  = [None] * (n + 1)
    dp[0] = 0.0

    for i in range(n):
        if dp[i] == INF:
            continue

        first = target_phones[i]
        candidates = list(index.get(first, []))
        if mode != "strict":
            # Nothing in the corpus makes this sound. Let a unit *start* on a
            # near-enough one, or the DP can never leave this position and the
            # whole word is reported missing over a single phoneme.
            for alt in _near(first):
                candidates.extend(index.get(alt, []))

        for word, phones, clips, pos in candidates:
            matched = subs = 0
            while pos + matched < len(phones) and i + matched < n:
                have, want = phones[pos + matched], target_phones[i + matched]
                if have == want:
                    pass
                elif mode != "strict" and have in _near(want):
                    subs += 1
                else:
                    break
                matched += 1
            if matched == 0:
                continue

            front_cut = pos > 0
            end_cut   = pos + matched < len(phones)

            # Fewest units (joins); reward longer matches; penalise each cut;
            # mildly prefer common words; strongly discourage reduced
            # function-word vowels; and — crucially — discourage cutting the
            # source right AFTER a vowel (an imprecise vowel→consonant edge that
            # smears the join, e.g. "bitch" as big|catch).  Cutting at consonant
            # boundaries keeps a vowel bound to its consonant ("b"+"itch").
            cost = dp[i] + 1.0 - 0.08 * matched + _SUB_COST * subs
            if front_cut:
                cost += 0.12
            if end_cut:
                cost += 0.12
            cost -= 0.04 * min(len(clips), 10) / 10.0
            if word.lower() in _FUNCTION_WORDS and any(_is_vowel(p) for p in phones[pos:pos + matched]):
                cost += 0.60
            if end_cut and _is_vowel(phones[pos + matched - 1]):
                cost += 0.25       # end-cut sits just after a vowel
            if front_cut and _is_vowel(phones[pos - 1]):
                cost += 0.25       # front-cut sits just after a vowel

            # User feedback: if the chosen clip for this source word was rated
            # down for this target, make the unit costlier so the DP prefers a
            # different source — but still allow it if nothing else covers these
            # phonemes ("only when it needs to").
            clip = _best_clip(clips, word, penalty) or random.choice(clips)
            if penalty:
                cost += 0.7 * penalty.get(clip.get("id"), 0)

            j = i + matched
            if cost < dp[j]:
                dp[j] = cost
                prev[j] = (i, word, phones, clip, matched, pos, subs, None)

        # The back half of an affricate, for a target phoneme the corpus has
        # no clean source of. One phoneme only: the front half is a stop that
        # belongs to a different sound, so this can never extend.
        if mode != "strict":
            for word, phones, clips, pos in half_index.get(first, []):
                clip = _best_clip(clips, word, penalty) or random.choice(clips)
                cost = dp[i] + 1.0 + _HALF_COST
                if penalty:
                    cost += 0.7 * penalty.get(clip.get("id"), 0)
                if cost < dp[i + 1]:
                    dp[i + 1] = cost
                    prev[i + 1] = (i, word, phones, clip, 1, pos, 0, "half")

        # Desperate: step over a phoneme nothing can cover. The word comes out
        # missing that sound, which is worse than a substitution and better
        # than nothing at all -- which is the whole point of the mode.
        if mode == "desperate" and dp[i] + _SKIP_COST < dp[i + 1]:
            dp[i + 1] = dp[i] + _SKIP_COST
            prev[i + 1] = ("skip", i)

    if dp[n] == INF:
        return None

    # Reconstruct chosen units left-to-right (capture each unit's target span).
    chosen: list[tuple] = []
    p = n
    while p > 0:
        entry = prev[p]
        if entry is None:
            return None
        if entry[0] == "skip":
            approximate = True
            p = entry[1]                    # contributes no audio
            continue
        prev_pos, word, phones, clip, matched, pos, subs, kind = entry
        if subs:
            approximate = True
        unit = [prev_pos, matched, word, clip, pos, len(phones)]
        if kind:
            unit.append(kind)
        chosen.append(tuple(unit))
        p = prev_pos
    chosen.reverse()
    if not chosen:
        return None
    segs = _realise(chosen, target_phones, index, clips_by_word, penalty)
    if segs and approximate:
        # Marked so the caller can say this one was approximated rather than
        # let it pass as a clean splice.
        for seg in segs:
            seg["approx"] = True
    return segs


def _phones_for(w: str) -> list[str] | None:
    if w in _OVERRIDES:
        return [_EQUIV.get(p, p) for p in _OVERRIDES[w]]
    prons = _dict().get(w)
    return [_EQUIV.get(p, p) for p in prons[0]] if prons else None


def _find_sub(cph: list[str], group: list[str]) -> int | None:
    for pos in range(len(cph) - len(group) + 1):
        if cph[pos:pos + len(group)] == group:
            return pos
    return None


def _phrase_clips(words: list[str]) -> list[dict]:
    """Clips of words[-1] spoken right after words[:-1] in one source —
    a context-specific delivery of a common word."""
    import app.generate as g
    g._ensure_cache()
    res = []
    for src, idx in (g._word_positions or {}).get(words[0], []):
        seq = g._ordered_by_source[src]          # type: ignore[index]
        if all(idx + k < len(seq) and seq[idx + k]["word"] == words[k]
               for k in range(len(words))):
            c = seq[idx + len(words) - 1]
            if c["end_time"] - c["start_time"] > 0.04:
                res.append(c)
    return res


def _chosen_from_recipe(
    recipe: list[tuple[list[str], list[str]]],
    index,
    penalty: dict[int, int] | None = None,
) -> list[tuple] | None:
    """Turn a hand-tuned recipe into a `chosen` unit list, preferring the
    listed sources (in order) and then any word with the fewest cuts.
    Returns None if some group has no source at all."""
    chosen: list[tuple] = []
    t = 0
    for group, preferred in recipe:
        unit = None
        for pref in preferred:
            if " " in pref:                       # phrase-context pick
                words = pref.split()
                w = words[-1]
                cph = _phones_for(w)
                pos = _find_sub(cph, group) if cph else None
                clips = _phrase_clips(words) if pos is not None else []
                if clips:
                    clip = _best_clip(clips, w, penalty) or random.choice(clips)
                    # verbatim (True): clip bounds are frame-accurate already;
                    # FA trimming short function words eats their substance
                    unit = (t, len(group), w, clip, pos, len(cph), True)
                    break
            else:
                flag: Any = None                  # "*word" / "*word:LO-HI"
                name = pref
                if pref.startswith("*"):
                    m = re.fullmatch(r"\*([^:]+)(?::([\d.]+)-([\d.]+))?", pref)
                    name = m.group(1)
                    flag = (float(m.group(2)), float(m.group(3))) if m.group(2) else True
                for w2, cph, cl, ps in index.get(group[0], []):
                    if w2 == name and cph[ps:ps + len(group)] == group:
                        clip = _best_clip(cl, w2, penalty) or random.choice(cl)
                        unit = (t, len(group), w2, clip, ps, len(cph), flag)
                        break
                if unit:
                    break
        if unit is None:                          # any word, fewest cuts
            cands = []
            for w2, cph, cl, ps in index.get(group[0], []):
                if cph[ps:ps + len(group)] == group:
                    ncuts = (1 if ps > 0 else 0) + (1 if ps + len(group) < len(cph) else 0)
                    cands.append((ncuts, w2, cph, cl, ps))
            if not cands:
                return None
            cands.sort(key=lambda x: x[0])
            _nc, w2, cph, cl, ps = cands[0]
            clip = _best_clip(cl, w2, penalty) or random.choice(cl)
            unit = (t, len(group), w2, clip, ps, len(cph))
        chosen.append(unit)
        t += len(group)
    return chosen


def _realise(
    chosen: list[tuple],
    target_phones: list[str],
    index,
    clips_by_word: dict[str, list[dict[str, Any]]],
    penalty: dict[int, int] | None = None,
) -> list[dict[str, Any]] | None:
    """Build playable segment dicts from a `chosen` unit list
    (tstart, matched, word, clip, pos, total) — shared by the DP and recipe
    paths."""
    if len(chosen) > _MAX_UNITS:
        return None

    from app.forced_align import cut_end_after_phonemes, cut_start_before_phonemes

    def _try(w, clip_list, ps, total, sub, last_v):
        """Realise one unit from a source word, trimmed to its forced-aligned
        CONTENT bounds (so stored leading/trailing silence isn't dragged in).
        Returns a seg dict or None if no clip aligns."""
        front_cut = ps > 0
        end_cut   = ps + len(sub) < total
        cut = front_cut or end_cut
        for cand in clip_list[:6]:
            dur = cand["end_time"] - cand["start_time"]
            try:
                cs = cut_start_before_phonemes(cand, ps)              # FA onset of kept part
                ce = cut_end_after_phonemes(cand, ps + len(sub), last_v)  # FA end of matched part
            except Exception:
                cs = ce = None
            if cs is None or ce is None or (ce - cs) <= 0.04:
                continue
            # Release: ONLY a word-final consonant gets extended into the trailing
            # audio (where its burst/decay lives).  A *medial* consonant's
            # "release" would be the very next phoneme we're cutting off, so
            # extending there drags it in ("prick"→"prickley", "and"→D→"ed").
            if not last_v and not end_cut:
                ce = min(ce + 0.06, dur)
            # Lead-in: ONLY a word-initial piece gets extended back to capture its
            # onset.  A front-cut piece's lead would be the skipped phoneme's tail
            # ("and"→D picking up the "n" → "ed").
            if not front_cut:
                cs = max(0.0, cs - 0.02)
            s = dict(cand)
            s["start_time"] = cand["start_time"] + cs
            s["end_time"]   = cand["start_time"] + ce
            s["subword"]    = cut
            s["_cut"]       = cut
            s["spliced_from"] = w
            s["matched"]      = len(sub)
            return s
        return None

    def _sources_for(sub):
        res = []
        for w, cph, cl, ps in index.get(sub[0], []):
            if cph[ps:ps + len(sub)] == sub:
                ncuts = (1 if ps > 0 else 0) + (1 if ps + len(sub) < len(cph) else 0)
                res.append((ncuts, w, cl, ps, len(cph)))
        res.sort(key=lambda x: x[0])           # fewest cuts first
        return res

    # Build segments.  Try the DP-chosen source first; if its clips won't cut
    # cleanly, re-search OTHER words for the same phonemes rather than falling
    # back to a whole word that drags in an extra vowel ("zoe" → "shove-es").
    segments: list[dict] = []
    for tstart, matched, cword, cclip, cpos, ctotal, *flags in chosen:
        sub = target_phones[tstart:tstart + matched]
        last_v = _is_vowel(sub[-1])

        flag = flags[0] if flags else None

        if flag == "half":
            # The fricative tail of an affricate. Both edges come from forced
            # alignment -- the affricate's own span -- and then the front is
            # dropped, because that front is the stop burst that makes a
            # borrowed "ch" still sound like "ch".
            seg = None
            for cand in clips_by_word.get(cword, [cclip])[:6]:
                try:
                    a = cut_start_before_phonemes(cand, cpos)
                    b = cut_end_after_phonemes(cand, cpos + 1, False)
                except Exception:
                    a = b = None
                if a is None or b is None or (b - a) <= 0.03:
                    continue
                st = a + (b - a) * _AFFRICATE_SPLIT
                if (b - st) <= 0.02:              # nothing audible left to take
                    continue
                seg = dict(cand)
                seg["start_time"] = cand["start_time"] + st
                seg["end_time"]   = cand["start_time"] + b
                seg["subword"] = seg["_cut"] = True
                seg["spliced_from"] = cword
                seg["matched"] = 1
                break
            if seg is None:
                return None
            segments.append(seg)
            continue

        if flag:                                  # no FA trim: verbatim / slice
            seg = dict(cclip)
            sliced = isinstance(flag, tuple)
            if sliced:                            # random slice within the clip
                lo, hi = flag
                cdur = cclip["end_time"] - cclip["start_time"]
                dur = min(random.uniform(lo, hi), cdur)
                span = cdur - dur
                st = cclip["start_time"] + (random.uniform(0, span) if span > 0 else 0)
                seg["start_time"] = st
                seg["end_time"]   = st + dur
            seg["subword"] = sliced
            seg["_cut"] = sliced
            seg["spliced_from"] = cword
            seg["matched"] = matched
            segments.append(seg)
            continue

        clips_chosen = [cclip] + [c for c in clips_by_word.get(cword, []) if c is not cclip]
        if penalty:                               # down-rated clips tried last
            clips_chosen.sort(key=lambda c: penalty.get(c.get("id"), 0))
        seg = _try(cword, clips_chosen, cpos, ctotal, sub, last_v)
        if seg is None:
            for _ncuts, w, cl, ps, total in _sources_for(sub):
                if w == cword:
                    continue
                seg = _try(w, cl, ps, total, sub, last_v)
                if seg is not None:
                    break
        if seg is None:                          # last resort: whole word clip
            # Use the LONGEST available clip, not the DP pick — that one may be a
            # degenerate ~10ms alignment artefact that would be inaudible.
            pool = clips_by_word.get(cword) or [cclip]
            best = max(pool, key=lambda c: c["end_time"] - c["start_time"])
            seg = dict(best); seg["_cut"] = False
            seg["spliced_from"] = cword; seg["matched"] = matched
        segments.append(seg)

    # Safety: never emit a degenerate (zero/negative/inaudible) span — it would
    # crash the FFmpeg trim.  Clamp the end forward within the clip.
    for seg in segments:
        if seg["end_time"] - seg["start_time"] < 0.04:
            seg["end_time"] = seg["start_time"] + 0.06

    last = len(segments) - 1
    for idx, seg in enumerate(segments):
        seg["prev_end"] = seg["start_time"]          # tight lead (no unrelated pre-roll)
        # Tight join everywhere except the final WHOLE-word unit, which keeps its
        # natural tail so word-final sounds aren't clipped.  Cut units stay tight
        # (their tail would be the dropped phonemes).
        if not (idx == last and not seg["_cut"]):
            seg["next_start"] = seg["end_time"]
        seg["fade_in"]  = 0.0
        seg["fade_out"] = 0.0
    for seg in segments:
        seg.pop("_cut", None)
    if segments:
        segments[0].pop("fade_in", None)             # natural fade at word start
        segments[-1].pop("fade_out", None)           # and word end

    return segments
