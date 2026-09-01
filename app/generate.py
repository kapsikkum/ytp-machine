import logging
import os
import re
import uuid
import random
import subprocess
from functools import lru_cache
from typing import Any

from app.database import get_db, resolve_path

log = logging.getLogger(__name__)

_PAD_START  = 0.05   # seconds before word start (into preceding silence only)
_PAD_END    = 0.18   # seconds after word end (into trailing silence only)
_PAD_END_LAST = 0.45 # more tail for the final word so its release isn't clipped
_SAFETY     = 0.05   # gap to leave before a neighbouring word (start side)
_TAIL_GAP   = 0.05   # gap before next word on the tail (runs are clamped precisely
                     # at the last word's content end, so this can stay modest)
# How far a word ending in a sonorant may be extended to finish its consonant,
# and how quiet the audio has to get before it counts as finished.
#
# M, N, NG, L and R fade out rather than stopping, and an aligner puts the
# boundary where it loses confidence, not where the sound ends. A fixed 40ms
# was the first attempt and was not enough: "rosen" is stored ending at 1.640
# and is still at full volume at 1.680, only dying away around 1.73. The
# stored next_start, 1.660, is early for the same reason and cannot be used as
# the limit either.
#
# So the end is measured rather than guessed -- extend while the sound is
# still there, up to a cap. The cap matters: without one, a word running
# straight into the next would swallow it.
_SONORANT_TAIL = 0.04     # floor, when the audio cannot be read
_SONORANT_MAX = 0.16      # never add more than this
_SONORANT_QUIET = 0.22    # fraction of the word's own peak that counts as over
_FINAL_SONORANTS = {"M", "N", "NG", "L", "R", "ER"}

_BLEED_TAIL = 0.16   # bigger tail gap when a run ends on a word whose next source
                     # word is an adjacent vowel — its onset glide bleeds backward
                     # (e.g. "full of" + "oil" leaking the "oi"); cut well clear of it
_MIN_DUR    = 0.05   # discard clips shorter than this (bad timestamps)
_FADE_IN    = 0.012  # audio fade-in per clip (removes start click)
_FADE_OUT   = 0.015  # audio fade-out per clip (smooths end boundary)
_MAX_RUN_GAP = 0.60  # don't merge a phrase across pauses longer than this (s)

# Pacing
_WORD_GAP    = 0.03  # brief freeze-frame pause between separate words (s)
_STOP_PAUSE  = 0.50  # idle silent clip inserted on full stops (. ! ?) (s)
_IDLE_MIN_GAP = 0.45 # a source gap this long counts as on-screen idle/silence

# ── Global cache ──────────────────────────────────────────────────────────────
_clips_by_word_cache: dict[str, list[dict[str, Any]]] | None = None
_ordered_by_source:   dict[int, list[dict[str, Any]]] | None = None  # source_id → clips in spoken order
_word_positions:      dict[str, list[tuple[int, int]]] | None = None  # word → [(source_id, idx)]
_source_quality:      dict[int, float] = {}  # source_id → fraction of well-aligned clips
_idle_clips:          list[dict[str, Any]] = []  # on-screen silent gaps (for pauses)
_noise_by_word:       dict[str, list[dict[str, Any]]] = {}  # "click"/"spew"/… → noise clips
_all_noises:          list[dict[str, Any]] = []  # every noise clip (for the "noise" token)
_splice_scores:       dict[tuple[str, int], int] | None = None  # (word, clip_id) → user score

# How a vote bends the odds. Each net vote multiplies or divides a clip's
# selection weight by this, so one downvote makes a clip half as likely, three
# make it eight times less likely, and the same in reverse for upvotes.
_VOTE_BASE = 2.0

# A downvoted clip is never ruled out entirely, only starved: when it is the
# only clip a word has, a quiet 50:1 outsider still beats reporting the word
# missing. Upvotes saturate so one enthusiastic clip cannot crowd out the
# variety that makes repeated generations differ.
_VOTE_FLOOR = 0.02
_VOTE_CEIL = 8.0


# Net score at which a clip stops being used as a real recording at all.
#
# Starving a clip is the right response to one downvote -- the clip may be fine
# and the vote a mood -- but it is the wrong response to a clip somebody has
# rejected twice, because starving still plays it when nothing else is
# available, which is exactly when a bad take is most annoying. Past this, the
# word is spliced from other words instead: a synthetic word beats a recording
# that has been turned down repeatedly.
#
# It applies to runs too. A run is one clip covering several words, so a
# rejected word inside it cannot be swapped out -- the run has to end there.
_VETO_SCORE = -2


def _vote_weight(score: int) -> float:
    """A net vote count as a multiplier on a clip's chance of being picked."""
    if not score:
        return 1.0
    return min(_VOTE_CEIL, max(_VOTE_FLOOR, _VOTE_BASE ** score))


def _vetoed(word: str, clip: dict) -> bool:
    """Has this clip been voted down far enough to stop using it for *word*?"""
    cid = clip.get("id")
    if cid is None:
        return False
    return (_splice_scores or {}).get((word.lower(), int(cid)), 0) <= _VETO_SCORE


def _ensure_cache() -> None:
    """Build all lookup structures from the DB in a single pass.

    Structures (sharing the same clip dicts):
      • _clips_by_word_cache : word → clips usable solo (≥ _MIN_DUR)
      • _ordered_by_source   : source_id → clips in spoken order
      • _word_positions      : word → (source_id, index) for phrase matching
      • _source_quality      : source_id → fraction of clips that are well-aligned

    Degenerate zero-duration clips (Whisper dumping several words on one
    timestamp — common in fast/rapped videos) cannot be extracted meaningfully
    and would corrupt adjacency, so they are dropped from every structure.
    Adjacency and the position index are otherwise built from the full set so
    short function words ("i", "it", "a") don't break contiguous-phrase runs.
    """
    global _clips_by_word_cache, _ordered_by_source, _word_positions, _source_quality, _idle_clips
    global _noise_by_word, _all_noises, _splice_scores
    if _clips_by_word_cache is not None:
        return

    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM word_clips ORDER BY source_id, start_time"
        ).fetchall()
        try:
            rating_rows = conn.execute(
                "SELECT word, clip_id, score FROM splice_ratings"
            ).fetchall()
        except Exception:
            rating_rows = []
        try:
            noise_rows = conn.execute("SELECT * FROM noise_clips").fetchall()
        except Exception:
            noise_rows = []

    # Per-source raw counts (for a quality score) and the cleaned ordered list.
    src_total: dict[int, int] = {}
    src_good:  dict[int, int] = {}
    ordered:   dict[int, list[dict]] = {}
    for r in rows:
        clip = dict(r)
        clip["source_file"] = resolve_path(clip["source_file"])   # → absolute
        sid  = clip["source_id"]
        dur  = clip["end_time"] - clip["start_time"]
        src_total[sid] = src_total.get(sid, 0) + 1
        if dur >= _MIN_DUR:
            src_good[sid] = src_good.get(sid, 0) + 1
        if dur > 0.0:                       # drop degenerate zero-duration clips
            ordered.setdefault(sid, []).append(clip)

    positions: dict[str, list[tuple[int, int]]] = {}
    by_word:   dict[str, list[dict]] = {}
    idle:      list[dict] = []
    from app.phonemes import word_to_phonemes, _is_vowel
    for src, seq in ordered.items():
        for idx, clip in enumerate(seq):
            clip["prev_end"]   = seq[idx - 1]["end_time"]   if idx > 0            else None
            clip["next_start"] = seq[idx + 1]["start_time"] if idx + 1 < len(seq) else None
            # Does the next word start with a vowel right up against this one?
            # Its gradual onset bleeds in ("hot" → "hot air"), so flag it.
            clip["bleed_risk"] = False
            if idx + 1 < len(seq):
                gap = seq[idx + 1]["start_time"] - clip["end_time"]
                if gap < 0.12:
                    nph = word_to_phonemes(seq[idx + 1]["word"])
                    if nph and _is_vowel(nph[0]):
                        clip["bleed_risk"] = True
            positions.setdefault(clip["word"], []).append((src, idx))
            if clip["end_time"] - clip["start_time"] >= _MIN_DUR:
                by_word.setdefault(clip["word"], []).append(clip)
            # A long gap before the next word = Michael on screen but silent.
            if clip["next_start"] is not None:
                gap = clip["next_start"] - clip["end_time"]
                if gap >= _IDLE_MIN_GAP:
                    idle.append({
                        "source_file": clip["source_file"],
                        "start": clip["end_time"] + 0.06,
                        "end":   clip["next_start"] - 0.06,
                    })

    # Noise clips (clicks/spews/…) — kept out of the word flow; extracted tight.
    nbw: dict[str, list[dict]] = {}
    alln: list[dict] = []
    for r in noise_rows:
        c = dict(r)
        c["source_file"] = resolve_path(c["source_file"])
        c["prev_end"]   = c["start_time"]
        c["next_start"] = c["end_time"]
        nbw.setdefault(c["word"], []).append(c)
        alln.append(c)

    _clips_by_word_cache = by_word
    _ordered_by_source   = ordered
    _word_positions      = positions
    _idle_clips          = idle
    _noise_by_word       = nbw
    _all_noises          = alln
    _splice_scores       = {(r["word"], r["clip_id"]): r["score"] for r in rating_rows}
    _source_quality      = {
        s: src_good.get(s, 0) / src_total[s] for s in src_total
    }


def _get_clips_by_word() -> dict[str, list[dict[str, Any]]]:
    _ensure_cache()
    return _clips_by_word_cache  # type: ignore[return-value]


def invalidate_cache(alignments: bool = True) -> None:
    global _clips_by_word_cache, _ordered_by_source, _word_positions, _source_quality, _idle_clips
    global _noise_by_word, _all_noises, _splice_scores
    _clips_by_word_cache = None
    _ordered_by_source   = None
    _word_positions      = None
    _source_quality      = {}
    _idle_clips          = []
    _noise_by_word       = {}
    _all_noises          = []
    _splice_scores       = None
    # The CSV is per-corpus, so switching corpus or reloading must drop it too.
    try:
        from app.phonemes import invalidate_user_dict
        invalidate_user_dict()
    except Exception:
        pass
    try:
        from app.phonemes import invalidate_recipes
        invalidate_recipes()
    except Exception:
        pass
    # Alignments are keyed on clip id, and ids mean different clips in
    # different corpora. Editing one clip passes alignments=False and drops
    # just that one instead: re-aligning is a model forward pass per clip, and
    # throwing the lot away every time a boundary is nudged by 10ms would make
    # the next sentence crawl.
    if alignments:
        try:
            from app.forced_align import invalidate as invalidate_alignment
            invalidate_alignment()
        except Exception:
            pass


def _penalty_for(word: str) -> dict[int, int]:
    """clip_id → cost adjustment for splicing *word*; unvoted clips are omitted.

    Positive means avoid (downvotes), negative means prefer (upvotes) — the
    splice planner adds this straight into a candidate's cost, so an upvoted
    clip becomes cheaper by the same amount a downvoted one becomes dearer.
    Upvotes used to be dropped here, which meant they were recorded, reported
    back to the user, and then had no effect on anything.
    """
    _ensure_cache()
    word = word.lower()
    return {cid: -score for (w, cid), score in (_splice_scores or {}).items()
            if w == word and score}


def rate_splice(word: str, clip_ids: list[int], delta: int = -1) -> dict[str, int]:
    """Record user feedback (default a downvote) for the clips that made up a
    splice of *word*.  Persisted and applied to the live cache immediately so
    the next generation reflects it without a full reload."""
    global _splice_scores
    word = word.lower()
    ids = sorted({int(c) for c in clip_ids})
    if not ids:
        return {}
    with get_db() as conn:
        for cid in ids:
            if delta == 0:
                # An explicit "unvote": back to neutral in one step, rather than
                # leaving the caller to guess how many votes it takes to undo.
                conn.execute(
                    "INSERT INTO splice_ratings (word, clip_id, score) VALUES (?, ?, 0) "
                    "ON CONFLICT(word, clip_id) DO UPDATE SET score = 0",
                    (word, cid),
                )
            else:
                conn.execute(
                    "INSERT INTO splice_ratings (word, clip_id, score) VALUES (?, ?, ?) "
                    "ON CONFLICT(word, clip_id) DO UPDATE SET score = score + ?",
                    (word, cid, delta, delta),
                )
        new = {cid: conn.execute(
            "SELECT score FROM splice_ratings WHERE word=? AND clip_id=?",
            (word, cid)).fetchone()[0] for cid in ids}
    if _splice_scores is not None:
        for cid, score in new.items():
            _splice_scores[(word, cid)] = score
    return new


def _find_noise(word: str) -> dict[str, Any] | None:
    """Return a non-verbal noise clip for 'noise' (any) or a specific kind
    like 'click' / 'spew', or None."""
    if word == "noise":
        return pick_clip(_all_noises, word) if _all_noises else None
    pool = _noise_by_word.get(word)
    return pick_clip(pool, word) if pool else None


# Loudness a stretch has to stay under to count as silence, and how many
# candidates to audition before giving up.
#
# An idle region is a gap between two *transcribed* words, and a gap in a
# transcript is not silence -- it is where nothing was transcribed. On a corpus
# of poetry read aloud those coincide, which is why this went unnoticed; on
# commentary videos they do not, because the gaps are full of music, b-roll and
# speech Whisper skipped. Measured on one such corpus, 22 of 25 sampled "idle"
# regions had audible sound in them, so a full stop played somebody saying
# "300" as its moment of quiet.
#
# find_noises.py has always known this -- it mines exactly these gaps for
# clicks and spews. The two modules held opposite beliefs about the same audio.
_IDLE_MAX_RMS = 400.0
_IDLE_TRIES = 12
_idle_rms_cache: dict[tuple, float] = {}


def _region_rms(source_file: str, start: float, duration: float) -> float:
    """Loudness of a slice, or 0.0 if it cannot be read."""
    key = (source_file, round(start, 2), round(duration, 2))
    if key in _idle_rms_cache:
        return _idle_rms_cache[key]
    import subprocess as _sp, tempfile as _tf, wave as _wave, os as _os
    tmp = _os.path.join(_tf.gettempdir(), f"_idle_{_os.getpid()}.wav")
    rms = 0.0
    try:
        _sp.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}",
                 "-t", f"{duration:.3f}", "-i", source_file,
                 "-ac", "1", "-ar", "16000", tmp], capture_output=True)
        with _wave.open(tmp, "rb") as w:
            raw = w.readframes(w.getnframes())
        if raw:
            import array, math
            samples = array.array("h")
            samples.frombytes(raw[: len(raw) // 2 * 2])
            if samples:
                rms = math.sqrt(sum(v * v for v in samples) / len(samples))
    except Exception:
        rms = 0.0                      # unreadable: treat as quiet, not as loud
    finally:
        try:
            _os.remove(tmp)
        except OSError:
            pass
    _idle_rms_cache[key] = rms
    return rms


def _pick_idle(duration: float) -> dict[str, Any] | None:
    """Return a clip dict for a *duration*-second slice of on-screen silence
    (Michael idle between lines), or None if no idle region is available."""
    if not _idle_clips:
        return None
    cands = [r for r in _idle_clips if (r["end"] - r["start"]) >= duration]
    if cands:
        # Audition a few, take the first that is actually quiet. Returning None
        # is a real answer: the caller falls back to a frozen frame and padded
        # silence, which is silent by construction rather than by assumption.
        for r in random.sample(cands, min(_IDLE_TRIES, len(cands))):
            span = (r["end"] - r["start"]) - duration
            st = r["start"] + (random.uniform(0, span) if span > 0 else 0)
            if _region_rms(r["source_file"], st, duration) <= _IDLE_MAX_RMS:
                return {
                    "source_file": r["source_file"],
                    "start_time": st, "end_time": st + duration,
                    "prev_end": st, "next_start": st + duration,
                    "word": "", "id": None,
                    "fade_in": 0.0, "fade_out": 0.0,
                }
        return None
    else:
        r = max(_idle_clips, key=lambda r: r["end"] - r["start"])
        duration = min(duration, r["end"] - r["start"])
        if duration < 0.08:
            return None
    span = (r["end"] - r["start"]) - duration
    st = r["start"] + (random.uniform(0, span) if span > 0 else 0)
    end = st + duration
    return {
        "source_file": r["source_file"],
        "start_time": st, "end_time": end,
        "prev_end": st, "next_start": end,   # tight: extract exactly this slice
        "word": "", "idle": True,
        "fade_in": 0.0, "fade_out": 0.0,
    }


# ── Tokeniser ─────────────────────────────────────────────────────────────────

# Units as they are said out loud. Expanding these is not cosmetic: "km" has no
# CMU pronunciation, so it cannot even be phoneme-spliced -- it can only ever
# come out missing. "kilometres" can be built from parts even when the corpus
# has never said the whole word.
#
# Australian spellings, because that is what the speaker says and the lookup is
# against words someone actually spoke.
_UNITS = {
    "km": "kilometres", "kmh": "kilometres per hour", "kph": "kilometres per hour",
    "mph": "miles per hour", "kg": "kilograms", "hp": "horsepower",
    "kw": "kilowatts", "nm": "newton metres", "mm": "millimetres",
    "cm": "centimetres", "ml": "millilitres",
}

# Only expanded when written against a number, because standing alone they are
# ordinary letters or words: "l" is a letter, "t" is a letter, "m" could be
# metres or the letter m. "5l" is unambiguous; a bare "l" is not.
_UNITS_AFTER_NUMBER = {
    "l": "litres", "m": "metres", "g": "grams",
    "k": "thousand", "t": "tonnes",
}
# Deliberately no "s" for seconds: "80s" is a decade far more often than it is
# eighty seconds, and reading it as a unit turned "the 80s" into "the eighty
# seconds". A bare "5s" now splits into "five" and "s", which is a visible miss
# rather than a confident wrong answer.

_DECADES = {"20": "twenties", "30": "thirties", "40": "forties", "50": "fifties",
            "60": "sixties", "70": "seventies", "80": "eighties", "90": "nineties"}

# Model-code shorthands that are pronounced as a word rather than as letters.
_ABBREV = {"mk": "mark"}

# The denominator of a rate, said after "per". Only consulted for a token with
# a slash in it, so a bare "h" or "s" is never dragged into this.
_PER_UNITS = {
    "h": "hour", "hr": "hour", "s": "second", "sec": "second",
    "min": "minute", "km": "kilometre", "100km": "hundred kilometres",
    "l": "litre", "kg": "kilogram", "m": "metre",
}

_CURRENCY = {"$": "dollars", "£": "pounds", "€": "euros"}


def _split_words(s: str) -> list[str]:
    """A spoken phrase into lookup words: no hyphens, no commas."""
    return [w for w in re.split(r"[\s-]+", s.replace(",", "")) if w]


# British spelling -> American, for the endings that differ. Ordered longest
# first so "-res" is tried before "-re".
_SPELLING_SWAPS = (("res", "ers"), ("re", "er"), ("our", "or"),
                   ("ise", "ize"), ("ised", "ized"), ("yse", "yze"))


@lru_cache(maxsize=512)
def _known_spelling(word: str) -> str:
    """The spelling the pronunciation dictionary actually knows.

    The unit names below are written the way the speaker says them, but the
    lookup key is the spelling the *transcriber* wrote, and CMU is American.
    "kilometres" has no pronunciation at all, so expanding "km" to it swapped a
    missing token for a differently-missing one that could not even be spliced.
    "kilometers" is in the dictionary -- and, as it turns out, is what Whisper
    wrote for this corpus, twelve times.

    Only used as a fallback, so a word the dictionary already knows is left
    exactly as written.
    """
    from app.phonemes import word_to_phonemes
    if word_to_phonemes(word):
        return word
    for british, american in _SPELLING_SWAPS:
        if word.endswith(british):
            alt = word[: -len(british)] + american
            if word_to_phonemes(alt):
                return alt
    return word


def _unit_words(phrase: str) -> list[str]:
    """A unit's spoken form, each word in a spelling that can be looked up."""
    return [_known_spelling(w) for w in _split_words(phrase)]


def _say_number(n: int, ordinal: bool = False) -> list[str]:
    from num2words import num2words
    return _split_words(num2words(n, to="ordinal" if ordinal else "cardinal"))


def _expand_atom(atom: str) -> list[str]:
    """One separator-free token into the words a person would say for it."""
    if not atom:
        return []

    out: list[str] = []

    # Currency and percent live outside the word characters, so they have to be
    # read before anything strips punctuation -- which is exactly how "$50" and
    # "50%" both used to come out as a bare "fifty", quietly losing the unit.
    suffix: list[str] = []
    if atom[0] in _CURRENCY:
        suffix.append(_CURRENCY[atom[0]])
        atom = atom[1:]
    if atom.endswith("%"):
        suffix.append("percent")
        atom = atom[:-1]

    atom = atom.strip(".,!?;:'\"()[]").lower()
    if not atom:
        return suffix

    # A possessive or a contraction is one word, not two.
    #
    # The ingest stores words with the punctuation stripped, so the corpus
    # holds "rosens" and "dont" -- and the token is joined the same way here.
    # Without this the letters-and-digits split further down took "rosen's"
    # apart into "rosen" and "s", and the stray "s" was looked up as a word in
    # its own right and said out loud. "dad's" escaped it only by accident:
    # the corpus happens to contain "dads", so the joined form was recognised
    # before the split could happen.
    #
    # Only the endings that really are suffixes. "y'know" is left alone
    # because the half after the apostrophe is a word, and joining it would
    # invent one nobody says.
    m = re.fullmatch(r"([a-z]+)'(s|re|ve|ll|d|t|m)", atom)
    if m:
        atom = m.group(1) + m.group(2)

    # Thousands separators, before anything treats the comma as a boundary:
    # "1,500" split into "1" and "500" and said "one five hundred".
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", atom):
        atom = atom.replace(",", "")

    # 2.0 -> two point zero. Stripping punctuation first turned this into "20"
    # and said "twenty" -- not a miss but a wrong answer, which is worse.
    # The trailing unit is part of the same match because "3.5l" is the way an
    # engine size is actually written, and matching only a bare decimal sent it
    # down the letters-and-digits path, which dropped the "point".
    m = re.fullmatch(r"(\d+)\.(\d+)([a-z]+)?", atom)
    if m:
        out += _say_number(int(m.group(1))) + ["point"]
        for digit in m.group(2):
            out += _say_number(int(digit))
        unit = m.group(3)
        if unit:
            out += _unit_words(_UNITS.get(unit)
                                or _UNITS_AFTER_NUMBER.get(unit)
                                or unit)
        return out + suffix

    # 1st, 2nd, 21st
    m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", atom)
    if m:
        return _say_number(int(m.group(1)), ordinal=True) + suffix

    # Decades: 80s, 1980s. Without this the letters-and-digits split read them
    # as a number followed by a stray letter.
    m = re.fullmatch(r"(?:(19|20))?(\d0)s", atom)
    if m and m.group(2) in _DECADES:
        century = _say_number(int(m.group(1))) if m.group(1) else []
        return century + [_DECADES[m.group(2)]] + suffix

    # 4x4 is said "four by four", not "four x four".
    m = re.fullmatch(r"(\d+)x(\d+)", atom)
    if m:
        return (_say_number(int(m.group(1))) + ["by"]
                + _say_number(int(m.group(2))) + suffix)

    if atom.isdigit():
        return _say_number(int(atom)) + suffix

    if atom in _UNITS:
        return _unit_words(_UNITS[atom]) + suffix

    if atom in _ABBREV:
        return [_ABBREV[atom]] + suffix

    # Letters and digits run together: i30, v8, mk2, 50km, 330i. Said as their
    # parts, so they are looked up as their parts. Left whole, a token like
    # "i30" matches nothing and has no pronunciation to splice from either.
    parts = re.findall(r"\d+|[a-z]+", atom)
    if len(parts) > 1:
        after_number = False
        for part in parts:
            if part.isdigit():
                out += _say_number(int(part))
                after_number = True
                continue
            if part in _UNITS:
                out += _unit_words(_UNITS[part])
            elif after_number and part in _UNITS_AFTER_NUMBER:
                out += _unit_words(_UNITS_AFTER_NUMBER[part])
            elif part in _ABBREV:
                out.append(_ABBREV[part])
            else:
                out.append(part)
            after_number = False
        return out + suffix

    return [re.sub(r"[^\w]", "", atom)] + suffix if re.sub(r"[^\w]", "", atom) else suffix


def _known_word(word: str) -> bool:
    """Does the corpus hold a clip labelled exactly this?"""
    if not word:
        return False
    try:
        _ensure_cache()
    except Exception:
        return False                      # no corpus to consult; just expand
    return bool(_clips_by_word_cache) and word in _clips_by_word_cache


def _expand_token(token: str) -> list[str]:
    """Return one or more lowercase words for a single input token.

    Hyphens and slashes separate words rather than disappearing: stripping them
    joined "four-cylinder" into "fourcylinder", a word nobody has ever said, so
    every hyphenated compound was a guaranteed miss.
    """
    # What the corpus actually says wins over what the rules would make of it.
    # The transcriber writes some tokens in a form the rules would expand right
    # past: this corpus has nineteen clips labelled "v8" and seventeen labelled
    # "80s", and expanding those to "v eight" and "eighty ..." walked away from
    # real recordings of the exact thing being asked for. Expansion is the
    # fallback for tokens nobody has said, not the first move.
    raw = re.sub(r"[^\w]", "", token).lower()
    if _known_word(raw):
        return [raw]
    # A rate: km/h, l/100km. Both halves are units and the slash is the word
    # "per", so it has to be read before the generic split, which would
    # otherwise leave the denominator as a bare letter nobody has ever said.
    # Deliberately narrow -- only when both sides are known units -- because
    # a slash is not always "per", and "and/or" is not "and per or".
    m = re.fullmatch(r"([a-z]+)/(\d*[a-z]+)", token.strip(".,!?;:").lower())
    if m and m.group(2) in _PER_UNITS:
        # The numerator may be a single letter -- "l/100km" is the standard way
        # fuel economy is written -- so both unit maps are in play here. A
        # slash makes it a unit unambiguously, which a bare "l" would not be.
        top = _UNITS.get(m.group(1)) or _UNITS_AFTER_NUMBER.get(m.group(1))
        if top:
            return (_unit_words(top) + ["per"]
                    + _unit_words(_PER_UNITS[m.group(2)]))

    out: list[str] = []
    for atom in re.split(r"[/\-–—]+", token):
        out.extend(_expand_atom(atom))
    return out


def tokenize(text: str) -> list[str]:
    result: list[str] = []
    for token in re.split(r"\s+", text.strip()):
        result.extend(_expand_token(token))
    return result


def tokenize_marked(text: str) -> list[tuple[str, bool, bool, bool]]:
    """Tokenise into (word, ends_sentence, is_noise, is_reversed) tuples.

    * ``*spew*`` / ``*click*`` → a non-verbal noise token (is_noise=True), so a
      plain "spew"/"click" is still spoken as a word.
    * ``~word~`` → played backwards (is_reversed=True). The clip is reversed at
      cut time rather than anything being stored backwards: reversed speech is
      not speech, so a corpus built from it would be labelled nonsense, while
      reversing on the way out works for any word in any corpus.
    * ``ends_sentence`` marks a . ! ? for pause insertion; the very last token
      is always treated as a sentence end (buffer tail for editing).
    """
    out: list[tuple[str, bool, bool, bool]] = []
    for token in re.split(r"\s+", text.strip()):
        if not token:
            continue
        # Strip the reverse marker first so everything below -- sentence ends,
        # noise markers, number expansion -- sees an ordinary token.
        rev = False
        m = re.fullmatch(r"~(.+?)~([.!?\"')\]]*)", token)
        if m:
            rev = True
            token = m.group(1) + m.group(2)
        # Trailing only. Any period anywhere used to end a sentence, so "2.0"
        # planted a full stop -- and its pause -- in the middle of a phrase.
        ends = bool(re.search(r"[.!?][\"')\]]*$", token))
        m = re.fullmatch(r"\*([A-Za-z]+)\*[.!?]*", token)
        if m:
            out.append((m.group(1).lower(), ends, True, rev))
            continue
        words = _expand_token(token)
        for k, w in enumerate(words):
            # A marked token that expands ("~i30~") reverses every word it
            # became, not just the first.
            out.append((w, ends and k == len(words) - 1, False, rev))

    if out:                                   # always full-stop the end
        w, _e, nz, rv = out[-1]
        out[-1] = (w, True, nz, rv)
    return out


# ── Lookup ────────────────────────────────────────────────────────────────────

_MIN_PHONE_DUR = 0.045   # rough minimum seconds of audio per phoneme


def _duration_floor(word: str) -> float:
    """Minimum plausible duration for a fully-articulated *word*.

    Uses the CMU phoneme count when available so we can reject clips whose
    timestamps are too tight to actually contain the whole word.
    """
    try:
        from app.phonemes import word_to_phonemes
        phones = word_to_phonemes(word)
        if phones:
            return len(phones) * _MIN_PHONE_DUR
    except Exception:
        pass
    # Fallback: ~3 chars per phoneme-ish
    return max(0.08, len(word) * 0.025)


def pick_clip(rows: list[dict], word: str) -> dict[str, Any] | None:
    """Choose a clip for *word* whose stored duration is *plausible*.

    Bad timestamps come in two flavours, both of which mangle the audio:
      • too short  → the word is cut off          (e.g. 120 ms "doing")
      • too long   → word + a long pause / bleed   (e.g. 2.1 s "and")

    The median duration across all clips of a word is a robust estimate of the
    true length, so we keep clips within a band around it (also respecting a
    phoneme-based absolute floor) and choose randomly among the survivors for
    variety.  Falls back to the clip closest to the median.
    """
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]

    durs = sorted(r["end_time"] - r["start_time"] for r in rows)
    median = durs[len(durs) // 2]

    floor   = max(_duration_floor(word), 0.5 * median)
    ceiling = max(2.5 * median, 0.45)

    good = [r for r in rows
            if floor <= (r["end_time"] - r["start_time"]) <= ceiling]
    if good:
        # Bias toward well-aligned sources (quality**3) and away from clips whose
        # next word is a vowel butted right up against this one ("hot air"),
        # which bleeds — but still allow them if that's all there is.
        # User votes are folded in here rather than as a filter, so that a clip
        # you disliked becomes rare instead of forbidden. The old behaviour only
        # applied to phoneme splices, which is why a merely bad *found* clip
        # kept coming back no matter how often it was rated down.
        scores = _splice_scores or {}
        key = word.lower()
        weights = []
        for r in good:
            w = _source_quality.get(r["source_id"], 1.0) ** 3 if _source_quality else 1.0
            if r.get("bleed_risk"):
                w *= 0.05
            cid = r.get("id")
            if cid is not None:
                w *= _vote_weight(scores.get((key, int(cid)), 0))
            weights.append(max(w, 1e-6))
        if sum(weights) > 0:
            return random.choices(good, weights=weights, k=1)[0]
        return random.choice(good)

    # Nothing in band — pick the clip closest to the median length.
    return min(rows, key=lambda r: abs((r["end_time"] - r["start_time"]) - median))


def _find_clip(word: str, cbw: dict) -> dict[str, Any] | None:
    rows = cbw.get(word) or []
    usable = [r for r in rows if not _vetoed(word, r)]
    if rows and not usable:
        # Every recording of this word has been rejected. Returning None sends
        # the caller to the splicer, which is the point: build it out of other
        # words rather than play a take that was turned down.
        return None
    return pick_clip(usable, word)


# ── Contiguous-phrase detection ────────────────────────────────────────────────
# If a run of the requested words was actually spoken consecutively in one
# source video, we use that single continuous clip instead of splicing word by
# word — preserving Michael's real prosody with no internal cuts.

def _find_run(words: list[str], i: int) -> tuple[int, int, int, int] | None:
    """Longest source run matching ``words[i:]`` (length ≥ 2).

    Returns ``(source_id, start_idx, end_idx, length)`` or ``None``.  A run only
    extends across consecutive clips whose inter-word gap is ≤ _MAX_RUN_GAP, so
    we never merge across a long pause (which would play as dead air).
    """
    assert _word_positions is not None and _ordered_by_source is not None
    best: tuple[int, int, int, int] | None = None
    best_q = -1.0

    for src, idx in _word_positions.get(words[i], []):
        seq = _ordered_by_source[src]
        if _vetoed(words[i], seq[idx]):
            continue                 # this run opens on a take already rejected
        k = 1
        while (i + k < len(words)
               and idx + k < len(seq)
               and seq[idx + k]["word"] == words[i + k]
               and seq[idx + k]["start_time"] - seq[idx + k - 1]["end_time"] <= _MAX_RUN_GAP
               and not _vetoed(words[i + k], seq[idx + k])):
            k += 1
        if k < 2:
            continue
        q = _source_quality.get(src, 1.0)
        # Prefer the longest run; break ties toward better-aligned sources.
        if best is None or k > best[3] or (k == best[3] and q > best_q):
            best = (src, idx, idx + k - 1, k)
            best_q = q

    return best


def _merged_run_clip(src: int, s: int, e: int) -> dict[str, Any]:
    """Build a single clip dict spanning source words ``s..e`` (inclusive).

    The run ends at the last word's forced-aligned content end.  When the next
    source word is adjacent, the run is clamped tight there so its onset (e.g. a
    gradual vowel) can't bleed in — that's what made "into the" play "into the
    ai".  When there's a real gap after, the natural tail is kept.
    """
    seq = _ordered_by_source[src]            # type: ignore[index]
    first, last = seq[s], seq[e]
    merged = dict(first)
    merged["word"]     = " ".join(seq[j]["word"] for j in range(s, e + 1))
    merged["prev_end"] = first["prev_end"]   # clip before the run (same source)

    end = last["end_time"]
    real_next = last["next_start"]
    try:
        from app.forced_align import word_core_end
        rel = word_core_end(last)
        if rel is not None and last["start_time"] + rel > last["start_time"] + 0.05:
            end = last["start_time"] + rel
    except Exception:
        pass

    # Never let the end reach the next word's onset — FA over-runs on silent
    # finals (the 'e' in "shove" aligns onto the following "it").  When the next
    # source word starts with a vowel butted right up against this one, its onset
    # glide bleeds backward into our tail ("full of" picking up "oil"'s "oi"), so
    # cut further clear of it.
    if real_next is not None:
        gap = _BLEED_TAIL if last.get("bleed_risk") else _TAIL_GAP
        end = min(end, real_next - gap)
        end = max(end, last["start_time"] + 0.05)

    # Clamp tight at the content end so the builder extracts exactly to there and
    # can't re-extend the tail into the next word.
    merged["end_time"]   = end
    merged["next_start"] = end
    return merged


# ── Typing suggestions ────────────────────────────────────────────────────────

def suggest_next(context: str, prefix: str, limit: int = 10) -> dict[str, Any]:
    """Suggestions for the frontend autocomplete.

    *context* is the text already typed before the word in progress; *prefix*
    is the partial word being typed (may be empty).  Returns:

      • continuations — words that FOLLOW the longest matching suffix of the
        context contiguously in some source (i.e. accepting one keeps a real
        spoken run going), with how many sources continue that way;
      • words — corpus vocabulary matching the prefix, by clip count.
    """
    _ensure_cache()
    assert _word_positions is not None and _ordered_by_source is not None

    ctx = tokenize(context)[-4:]                       # longest suffix we try
    prefix = re.sub(r"[^\w]", "", prefix).lower()

    continuations: dict[str, int] = {}
    matched_len = 0
    for length in range(len(ctx), 0, -1):              # longest suffix first
        tail = ctx[-length:]
        found: dict[str, int] = {}
        for src, idx in _word_positions.get(tail[0], []):
            seq = _ordered_by_source[src]
            if idx + length >= len(seq) + 1:
                continue
            ok = True
            for k in range(1, length):
                if (idx + k >= len(seq)
                        or seq[idx + k]["word"] != tail[k]
                        or seq[idx + k]["start_time"] - seq[idx + k - 1]["end_time"] > _MAX_RUN_GAP):
                    ok = False
                    break
            if not ok:
                continue
            j = idx + length
            if (j < len(seq)
                    and seq[j]["start_time"] - seq[j - 1]["end_time"] <= _MAX_RUN_GAP):
                w = seq[j]["word"]
                if not prefix or w.startswith(prefix):
                    found[w] = found.get(w, 0) + 1
        if found:
            continuations = found
            matched_len = length
            break

    cont_list = sorted(continuations.items(), key=lambda x: -x[1])[:limit]

    words_list: list[tuple[str, int]] = []
    if prefix:
        cbw = _clips_by_word_cache or {}
        seen = {w for w, _n in cont_list}
        cands = [(w, len(cl)) for w, cl in cbw.items()
                 if w.startswith(prefix) and w not in seen]
        cands.sort(key=lambda x: (-x[1], x[0]))
        words_list = cands[:limit]

    return {
        "continuations": [{"word": w, "sources": n} for w, n in cont_list],
        "matched_len": matched_len,
        "words": [{"word": w, "clips": n} for w, n in words_list],
    }


# ── Single-pass FFmpeg ────────────────────────────────────────────────────────

# Windows' CreateProcess command line caps at ~32K chars.  Each input segment
# adds "-ss X -t Y -i <absolute path>" to the FFmpeg call, so long sentences
# must be encoded in chunks and the chunks concatenated (losslessly).
_CMD_BUDGET = 24000

# Most inputs to hand one ffmpeg call. Every input gets its own decoder with its
# own reference-frame buffers, so peak memory scales with this number and not
# with how long the sentence is. Sixty-one at once reached 2.3 GB resident and
# 26 GB virtual, and the container's memory cap killed it -- an OOM kill, so
# ffmpeg died without printing anything and the failure looked like nothing at
# all. Twenty keeps a batch to a few hundred megabytes, which makes the length
# of a sentence a question of how many batches rather than whether it works.
_MAX_INPUTS_PER_CALL = 20

# The output frame rate. Every clip's picture is resampled to it, which
# quantises the picture's length -- see the audio trim in _encode_chunk.
_FPS = 25


def _trim(stderr: str, limit: int = 2000) -> str:
    """Keep the beginning and the end of ffmpeg's complaint.

    The first error is the one that caused the failure; everything after it is
    usually a consequence. Keeping only the tail, as this used to, reported the
    consequences and threw away the cause.
    """
    text = (stderr or "").strip()
    if len(text) <= limit:
        return text
    head, tail = limit * 2 // 3, limit // 3
    marker = f"  … {len(text) - limit} chars elided …"
    return "\n".join([text[:head], marker, text[-tail:]])


@lru_cache(maxsize=20_000)
def _sound_ends(source_file: str, start: float, stored_end: float) -> float:
    """Where a word's audio actually stops, at most _SONORANT_MAX past the end."""
    import array, math, os, subprocess, tempfile, wave
    limit = stored_end + _SONORANT_MAX
    tmp = os.path.join(tempfile.gettempdir(), f"_st_{os.getpid()}.wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.4f}",
                        "-t", f"{limit - start:.4f}", "-i", source_file,
                        "-ac", "1", "-ar", "16000", tmp], capture_output=True)
        with wave.open(tmp, "rb") as w:
            raw = w.readframes(w.getnframes())
    except Exception:
        return stored_end + _SONORANT_TAIL
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) // 2 * 2]) if raw else None
    if not raw or not samples:
        return stored_end + _SONORANT_TAIL
    win = 160
    env = [math.sqrt(sum(v * v for v in samples[k:k + win]) / win)
           for k in range(0, len(samples) - win, win)]
    if not env:
        return stored_end + _SONORANT_TAIL
    peak = max(env)
    if peak <= 0:
        return stored_end + _SONORANT_TAIL
    floor = peak * _SONORANT_QUIET
    # Walk on from the stored end for as long as the sound is still going.
    i = int((stored_end - start) / (win / 16000.0))
    j = i
    while j < len(env) and env[j] >= floor:
        j += 1
    extra = (j - i) * (win / 16000.0)
    return min(limit, stored_end + max(_SONORANT_TAIL, extra))


def _ends_in_sonorant(seg: dict[str, Any]) -> bool:
    """Does this clip end on a sound that fades rather than stops?"""
    word = (seg.get("word") or "").strip().split()
    if not word:
        return False
    try:
        from app.phonemes import word_to_phonemes
        phones = word_to_phonemes(word[-1].lower())
    except Exception:
        return False
    return bool(phones) and phones[-1] in _FINAL_SONORANTS


# Where to find a bold font, in order of preference. The container has
# DejaVu; a Windows workstation building or testing locally has Arial.
_SUB_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


@lru_cache(maxsize=1)
def subtitle_font() -> str | None:
    """A font ffmpeg can draw with, or None if the box has none."""
    for p in _SUB_FONTS:
        if os.path.exists(p):
            return p
    log.warning("no font found for subtitles; captions will be skipped")
    return None


# The frame is 480 wide and a caption wants a margin, so this is the room a
# word has. DejaVu Sans Bold averages a bit over half its point size per
# character, which is close enough to decide when to shrink.
_CAP_WIDTH = 440
_CAP_SIZE = 26
_CAP_MIN = 12


def _cap_size(text: str) -> int:
    """Point size for *text*, shrunk if it would run off the frame."""
    if not text:
        return _CAP_SIZE
    fits = int(_CAP_WIDTH / (0.58 * len(text)))
    return max(_CAP_MIN, min(_CAP_SIZE, fits))


def _drawtext(text: str, font: str, window: tuple[float, float] | None = None) -> str:
    """A drawtext filter for one word, styled to be read over anything.

    White on a black outline rather than a box: a box on a 480x270 frame
    covers a third of the picture, and the picture is the joke.

    *window* limits it to part of the clip, which is what makes a run read
    word by word instead of as a wall of text.

    expansion=none because the corpus can contain a literal % or {} and
    drawtext would otherwise try to expand it -- a word should be drawn, not
    evaluated.
    """
    size = _cap_size(text)
    for ch in ("\\", "'", ":"):
        text = text.replace(ch, "\\" + ch)
    font = font.replace("\\", "/").replace(":", "\\:")
    f = (f"drawtext=fontfile='{font}':text='{text}':expansion=none"
         f":fontcolor=white:fontsize={size}:borderw=3:bordercolor=black"
         ":x=(w-tw)/2:y=h-th-12")
    if window:
        # Quoted: the value has commas in it, and the chain is comma-joined.
        f += f":enable='between(t,{window[0]:.3f},{window[1]:.3f})'"
    return f


def _build_video(segments: list[dict[str, Any]], out_path: str, progress=None,
                 subtitles: bool = False) -> None:
    """Encode *segments* to *out_path* in batches, then join them.

    Split on two limits: the command-line length, and the number of inputs one
    call may open. The second is the one that bounds memory -- see
    _MAX_INPUTS_PER_CALL.
    """
    def _say(stage: str, done: int, total: int) -> None:
        if progress:
            progress(stage, done, total)

    # Split by command-line cost and by input count, whichever bites first.
    chunks: list[list[dict]] = [[]]
    used = 0
    for seg in segments:
        cost = len(seg["source_file"]) + 40   # "-ss 123.4567 -t 12.3456 -i <path>"
        too_long = used + cost > _CMD_BUDGET
        too_many = len(chunks[-1]) >= _MAX_INPUTS_PER_CALL
        if chunks[-1] and (too_long or too_many):
            chunks.append([])
            used = 0
        chunks[-1].append(seg)
        used += cost

    if len(chunks) > 1:
        log.info("  BATCH    %d clips -> %d ffmpeg calls", len(segments), len(chunks))

    if len(chunks) == 1:
        _say("encoding", 0, 1)
        _encode_chunk(segments, out_path, final_tail=True, subtitles=subtitles)
        _say("encoding", 1, 1)
        return

    # Encode each chunk (identical codec settings), then stream-copy concat.
    part_paths: list[str] = []
    try:
        for ci, chunk in enumerate(chunks):
            _say("encoding", ci, len(chunks))
            part = f"{out_path}.part{ci}.mp4"
            _encode_chunk(chunk, part, final_tail=(ci == len(chunks) - 1),
                          subtitles=subtitles)
            part_paths.append(part)
        _say("joining", len(chunks), len(chunks))

        list_path = out_path + ".concat.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for p in part_paths:
                f.write(f"file '{os.path.abspath(p).replace(chr(92), '/')}'\n")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "concat", "-safe", "0", "-i", list_path,
               "-c", "copy", "-movflags", "+faststart", out_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("FFmpeg concat failed:\n%s", result.stderr[-2000:])
            raise RuntimeError(f"FFmpeg concat failed:\n{result.stderr[-3000:]}")
        log.info("  CONCAT   %d parts -> %s", len(part_paths), out_path)
    finally:
        for p in part_paths + [out_path + ".concat.txt"]:
            try:
                os.remove(p)
            except OSError:
                pass


def extract_window(seg: dict[str, Any], pad_end: float = _PAD_END) -> tuple[float, float]:
    """The span of source audio to pull for *seg*: (start, end).

    Pulled out of the encoder because it is the one calculation that decides
    what a clip actually sounds like, and it was only ever exercised by
    rendering a video and listening to it.

    Three things are being balanced:

      - a word wants a little air around it, or it starts and stops abruptly;
      - it must never reach into a neighbouring word, or you hear that word;
      - and it must never come out shorter than the clip it came from, which
        is a boundary somebody chose.

    The last two used to be in the wrong order. A word ending in a sonorant --
    ER, N, M, L, NG, R, which is a huge share of English -- gets its tail
    extended to where the sound really stops, because those fade rather than
    stop. That extension was applied *after* the clamp that keeps clear of the
    next word, so it overrode it: "mother" in Tomato 1 ends at 33.122, "like"
    starts at 33.182, and the tail ran to 33.282 -- a tenth of a second into
    the following word. And the closer the next word, the more likely the
    walk keeps going, so it failed hardest exactly where it mattered.

    An edited clip is not extended at all. Its boundaries were set by hand,
    against a waveform, and second-guessing them is how a fix in the editor
    ends up sounding like it did nothing.
    """
    edited = bool(seg.get("edited"))
    # An edited clip is taken exactly as stored, at both ends. The lead-in has
    # the same fault the tail had: back up 50ms from a start somebody trimmed
    # deliberately and you hand back the sound they trimmed off.
    start = seg["start_time"] - (0.0 if edited else _PAD_START)
    prev_end = seg.get("prev_end")
    if prev_end is not None:
        start = max(start, prev_end + _SAFETY)      # clear of the word before
    start = max(0.0, min(start, seg["start_time"]))  # never start after the word

    stored_end = seg["end_time"]
    next_start = seg.get("next_start")
    # A tight butt-join: a splice unit whose next_start is its own end. It was
    # cut to a phoneme boundary and has to come out exactly as cut, including
    # not being padded up to a minimum length -- the padding would be the next
    # phoneme, which is the one it was cut away from.
    butt = next_start is not None and next_start <= stored_end + 1e-3

    # Where we would like to reach: into the trailing silence, and for a
    # sonorant as far as the sound actually goes.
    want = stored_end if edited else stored_end + pad_end
    # `subword` alone: _realise pops its private _cut flag before returning, so
    # by the time a segment reaches the encoder that is the only mark a cut
    # unit still carries.
    if not edited and not seg.get("subword") and _ends_in_sonorant(seg):
        want = max(want, _sound_ends(seg["source_file"], seg["start_time"], stored_end))

    # Where we must stop: clear of the next word -- but never before the end
    # of the clip itself, which is what the stored boundary says the word is.
    # A word whose next neighbour starts before it ends (overlapping stored
    # times) therefore gets no extension at all rather than a negative one.
    if next_start is not None:
        gap = _BLEED_TAIL if seg.get("bleed_risk") else _TAIL_GAP
        want = min(want, max(stored_end, next_start - gap))

    # Never shorter than the clip itself. The 50ms minimum is for whole words
    # only: a butt-joined unit padded up to a length it was not cut to would
    # drag the next phoneme in behind it.
    floor = stored_end if (butt or edited) else max(stored_end, start + 0.05)
    return start, max(want, floor)


def _encode_chunk(segments: list[dict[str, Any]], out_path: str,
                  final_tail: bool, subtitles: bool = False) -> None:
    """
    One FFmpeg call for one chunk of segments.

    All source clips are H.264 480×270 25fps AAC — same format, so we skip
    scaling and only normalise framerate/SAR.  ultrafast + small frame size
    keeps encode time well under a second for typical sentences.

    The concat filter (not demuxer) is used so FFmpeg resets timestamps
    between segments internally — no A/V drift.  *final_tail* is True only for
    the chunk holding the overall last word, which gets the generous tail pad.
    """
    n = len(segments)
    # -loglevel error, because the captured stderr is the only thing we get to
    # diagnose a failure from. ffmpeg dumps a full stream description per input
    # and there is one input per clip, so on a long sentence the last 3000
    # characters were entirely other files' metadata -- the actual fatal line
    # had scrolled past thousands of lines earlier and never reached the log.
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    # ── Inputs  (precompute durations so we can use them in filter_complex) ──
    clip_durations: list[float] = []
    clip_starts: list[float] = []      # where in the source each extract began
    for idx, seg in enumerate(segments):
        # The final word of the output gets a more generous tail so its
        # release (e.g. a word-final "s") isn't clipped.
        pad_end = _PAD_END_LAST if (final_tail and idx == len(segments) - 1) else _PAD_END
        start, tail_end = extract_window(seg, pad_end)
        clip_starts.append(start)

        duration = tail_end - start
        clip_durations.append(duration)
        cmd += [
            "-ss", f"{start:.4f}",
            "-t",  f"{duration:.4f}",
            "-i",  seg["source_file"],
        ]

    # ── filter_complex ────────────────────────────────────────────────────────
    # Per-clip: normalise, add short audio fade-in/out to remove clicks at
    # boundaries, reset timestamps.  No scaling — keep native 480×270.
    # Sub-word phoneme units of the same word carry fade_in/fade_out = 0 at their
    # internal joins so the word plays as one continuous unit, not a stutter of
    # blips; only the outer edges of each word fade.
    font = subtitle_font() if subtitles else None
    parts: list[str] = []
    for i, seg in enumerate(segments):
        dur = clip_durations[i]
        fi = seg.get("fade_in",  _FADE_IN)
        fo = seg.get("fade_out", _FADE_OUT)
        if fi is None: fi = _FADE_IN
        if fo is None: fo = _FADE_OUT
        pause = seg.get("pause_after", 0.0)   # brief inter-word hold (freeze + silence)

        # A ~word~ token, played backwards. Reversing at cut time rather than
        # storing anything reversed: it works for any word in any corpus, and
        # costs nothing when unused. `reverse` buffers the whole segment, which
        # is fine here -- these are single words, a fraction of a second each.
        # It goes before the pause padding so the frozen tail stays at the end
        # rather than being flipped to the front.
        rev = bool(seg.get("reverse"))

        vfilters = [f"scale=480:270:force_original_aspect_ratio=disable",
                    f"fps={_FPS}", "setsar=1"]
        if rev:
            vfilters.append("reverse")
        if pause > 0:
            vfilters.append(f"tpad=stop_mode=clone:stop_duration={pause:.4f}")
        # After the pause padding, so a held frame keeps the word on screen,
        # and after any reversal -- the text is the same on every frame, so it
        # reads forwards either way.
        #
        # No timeline arithmetic anywhere: the video is built one word per
        # clip, so a segment already *is* its word for exactly its duration.
        # A caption is one more filter on a clip that is being encoded anyway,
        # rather than a second pass over the finished video.
        if subtitles and font:
            timed = seg.get("subtitle_words")
            if timed:
                # A run is one clip covering several words, and the corpus
                # knows where each of them falls inside it -- so each word
                # appears as it is said rather than the whole phrase sitting
                # there at once, which on a 480-wide frame ran off both edges.
                #
                # Times are relative to where this extract began, not to the
                # source, and the first and last are stretched to the ends of
                # the clip so the caption does not blink off during the
                # padding either side.
                base = clip_starts[i]
                for k, (w, a, b) in enumerate(timed):
                    lo = 0.0 if k == 0 else max(0.0, a - base)
                    hi = dur if k == len(timed) - 1 else max(lo, b - base)
                    if rev:                       # played backwards, so read backwards
                        lo, hi = max(0.0, dur - hi), max(0.0, dur - lo)
                    vfilters.append(_drawtext(w, font, (lo, hi)))
            elif seg.get("subtitle"):
                vfilters.append(_drawtext(seg["subtitle"], font))
        vfilters.append("setpts=PTS-STARTPTS")
        parts.append(f"[{i}:v]" + ",".join(vfilters) + f"[v{i}]")

        afilters = ["aformat=sample_rates=44100:channel_layouts=stereo"]
        if rev:
            # Before the fades, so they still land on the audible edges of what
            # actually plays rather than on what used to be the edges.
            afilters.append("areverse")
        if fi > 0:
            afilters.append(f"afade=t=in:st=0:d={fi:.4f}")
        if fo > 0:
            fo_start = max(0.0, dur - fo)
            afilters.append(f"afade=t=out:st={fo_start:.4f}:d={fo:.4f}")
        if pause > 0:
            afilters.append(f"apad=pad_dur={pause:.4f}")
        # The audio is made exactly as long as the video, which is not the
        # same as being as long as the source span.
        #
        # fps=25 quantises each clip's picture to whole frames while its sound
        # keeps its real length, so every segment ends with the two a few
        # milliseconds apart. concat joins the video streams and the audio
        # streams separately, so those differences add up: measured over a
        # 71-segment sentence the offset wandered to 84ms -- two frames, and
        # audible -- while ending at -44ms, which is why the ends of a long
        # video look fine and the middle does not.
        #
        # apad then atrim pins the sound to the picture's own length, so the
        # walk cannot start.
        quantised = round(dur * _FPS) / _FPS
        afilters.append("apad")
        afilters.append(f"atrim=end={quantised:.4f}")
        afilters.append("asetpts=PTS-STARTPTS")
        parts.append(f"[{i}:a]" + ",".join(afilters) + f"[a{i}]")

    concat_in = "".join(f"[v{i}][a{i}]" for i in range(n))
    parts.append(f"{concat_in}concat=n={n}:v=1:a=1[vout][aout]")

    # The filter graph is huge for long sentences and would blow past Windows'
    # ~32K command-line limit (WinError 206), so write it to a script file and
    # reference that instead of passing it inline.
    filter_path = out_path + ".filter.txt"
    with open(filter_path, "w", encoding="utf-8") as f:
        f.write(";".join(parts))

    cmd += [
        "-filter_complex_script", filter_path,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "aac",
        "-ar", "44100",
        "-ac", "2",
        # Both streams end together.
        #
        # AAC encodes in frames of 1024 samples -- 23ms at 44.1kHz -- so the
        # encoded audio finishes a frame or two short of the video however
        # exactly the filters lined them up. On its own that is inaudible. It
        # stops being inaudible when a long sentence is encoded in chunks and
        # the chunks are joined with a stream copy, because each chunk's
        # shortfall is added to the last: measured over 90 words in four
        # chunks, the sound ended up 240ms behind the picture, drifting
        # further the longer the video ran.
        "-shortest",
        "-movflags", "+faststart",
        out_path,
    ]

    import time as _time
    t0 = _time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        try:
            os.remove(filter_path)
        except OSError:
            pass
    elapsed = _time.perf_counter() - t0
    if result.returncode != 0:
        log.error("FFmpeg failed:\n%s", result.stderr[-2000:])
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr[-3000:]}")
    log.info("  FFMPEG   %d clips  %.2fs  -> %s", n, elapsed, out_path)


# ── Main generation ───────────────────────────────────────────────────────────

def generate_video(text: str, progress=None,
                   subtitles: bool = False) -> dict[str, Any]:
    """Turn *text* into a video.

    `progress`, if given, is called as progress(stage, done, total) at points
    where the work is measurable: once per word while resolving, then per
    encode chunk. It exists so a queued job can say what it is doing rather
    than leaving a request open for minutes -- which is what used to happen,
    until a long sentence outlasted the proxy and returned 504.
    """
    def _say(stage: str, done: int, total: int) -> None:
        if progress:
            progress(stage, done, total)

    marked = tokenize_marked(text)
    if not marked:
        return {"found": [], "spliced": [], "missing": [], "runs": [],
                "tokens": [], "video_url": None}
    words   = [w  for w, _e, _n, _r in marked]
    ends    = [e  for _w, e, _n, _r in marked]   # ends[i] = word i ends a sentence
    is_noise = [n for _w, _e, n, _r in marked]   # is_noise[i] = *wrapped* noise token
    is_rev   = [r for _w, _e, _n, r in marked]   # is_rev[i]   = ~wrapped~ reversed token

    _say("loading", 0, 1)
    _ensure_cache()
    cbw = _clips_by_word_cache  # type: ignore[assignment]
    _say("loading", 1, 1)

    found:    list[str]  = []
    spliced:  list[str]  = []
    missing:  list[str]  = []
    runs:     list[str]  = []   # contiguous phrases used as one clip
    tokens:   list[dict] = []   # every spoken word in order, with its status
    segments: list[dict] = []

    from app.phonemes import find_phoneme_splice
    from app.database import max_units, splice_mode
    mode = splice_mode()
    # Read once, not per word: it is the corpus's answer to how badly it wants
    # to say something it has no recording of.
    units_cap = max_units()
    i = 0
    n = len(words)
    while i < n:
        _say("resolving", i, n)
        seg_before = len(segments)
        used_run = False

        # 0) Explicit *noise* token (e.g. *spew*) → a non-verbal clip when the
        # corpus has one. When it does not, the word is spoken instead of being
        # reported missing: asterisks are not a rare piece of markup people
        # reach for deliberately, they are markdown emphasis, censoring and
        # stage directions, and they arrive in any pasted text. Treating
        # "*kill*" as a failed sound-effect lookup silently dropped a word the
        # corpus could say perfectly well -- and said nothing about why, since
        # the tag looks identical to a word the corpus really lacks.
        nz = _find_noise(words[i]) if is_noise[i] else None
        if nz is not None:
            word = words[i]
            found.append(word)
            tokens.append({"word": word, "status": "found"})
            # Copied, like every other segment. A noise carries no caption:
            # a spew is not a word, and writing "*spew*" across the picture
            # explains a joke that did not need it.
            segments.append(dict(nz))
            log.info("  NOISE    %-14s  (%s)", word,
                     nz["source_file"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1])
            if is_rev[i]:
                for seg in segments[seg_before:]:
                    seg["reverse"] = True
            if len(segments) > seg_before:
                if ends[i]:
                    idle = _pick_idle(_STOP_PAUSE)
                    if idle is not None:
                        segments.append(idle)
                    else:
                        # No quiet footage to hold on. Freeze the last frame and
                        # pad the audio instead -- silent because it is made,
                        # not found.
                        segments[-1]["pause_after"] = max(
                            segments[-1].get("pause_after", 0.0), _STOP_PAUSE)
                elif i < n - 1:
                    segments[-1]["pause_after"] = max(segments[-1].get("pause_after", 0.0), _WORD_GAP)
            i += 1
            continue

        # 1) Prefer a contiguous phrase already spoken in some source — but never
        # let a run cross a full stop (the pause belongs at the sentence end).
        run = _find_run(words, i)
        if run:
            src, s, _e, length = run
            cap = length
            for k in range(length - 1):       # all but the run's last word
                # A run is one clip, so it is reversed or not as a whole. Where
                # the mark changes, the run has to end -- otherwise marking one
                # word would quietly reverse the words either side of it.
                if ends[i + k] or is_rev[i + k] != is_rev[i + k + 1]:
                    cap = k + 1
                    break
            if cap >= 2:
                merged = _merged_run_clip(src, s, s + cap - 1)
                # One clip, several words, and each word's own span inside it
                # -- so the caption can follow the speech instead of dumping
                # the whole phrase on screen at once.
                _rseq = _ordered_by_source[src]        # type: ignore[index]
                merged["subtitle_words"] = [
                    [words[i + k], _rseq[s + k]["start_time"], _rseq[s + k]["end_time"]]
                    for k in range(cap)
                ]
                segments.append(merged)
                found.extend(words[i:i + cap])
                runs.append(merged["word"])
                # Each word of the run carries its own clip id rather than the
                # run's, so a vote lands on the (word, clip) pair the selector
                # actually looks up. A merged run inherits the id of its first
                # word, so rating "the run" would have silently rated one clip
                # under a multi-word key nothing ever reads back.
                _seq = _ordered_by_source[src]           # type: ignore[index]
                tokens.extend(
                    {"word": w, "status": "run",
                     "clips": ([int(_seq[s + k]["id"])]
                               if _seq[s + k].get("id") is not None else [])}
                    for k, w in enumerate(words[i:i + cap])
                )
                fname = merged["source_file"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
                log.info("  RUN      %-22s  %d words  (%s)", merged["word"], cap, fname)
                last_idx = i + cap - 1
                used_run = True

        if not used_run:
            cap = 1
            word = words[i]
            clip = _find_clip(word, cbw)   # noises only via explicit *word* tokens
            if clip:
                found.append(word)
                tokens.append({
                    "word": word, "status": "found",
                    "clips": ([int(clip["id"])] if clip.get("id") is not None else []),
                })
                # A copy, because this is the cache's own dict and segments
                # get written to -- `reverse` for a ~word~, and the caption
                # below. Appending it directly meant reversing a word once
                # left that clip reversed for every later generation in the
                # same process, silently and for as long as the process ran.
                clip = dict(clip, subtitle=word)
                segments.append(clip)
                fname = clip["source_file"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
                log.info("  FOUND    %-14s  %.3f->%.3f  (%s)",
                         word, clip["start_time"], clip["end_time"], fname)
            else:
                segs = find_phoneme_splice(word, cbw, _penalty_for(word), mode,
                                           units_cap)
                if segs:
                    spliced.append(word)
                    clip_ids = sorted({int(s["id"]) for s in segs if s.get("id") is not None})
                    # An approximated word used a stand-in phoneme, a dropped
                    # one, or a guessed pronunciation. It is still a splice, but
                    # saying so lets the UI show which words were only a best
                    # effort instead of implying every one of them is faithful.
                    approx = any(s.get("approx") for s in segs)
                    tokens.append({"word": word, "status": "spliced",
                                   "clips": clip_ids, **({"approx": True} if approx else {})})
                    # Every unit of a splice carries the word being built, not
                    # the word it was cut from, so the caption stays put while
                    # the pieces play rather than flickering through them.
                    for unit in segs:
                        unit["subtitle"] = word
                    segments.extend(segs)
                    log.info("  %-8s %-14s  -> %s", "APPROX" if approx else "SPLICE",
                             word, "+".join(s["word"] for s in segs))
                else:
                    missing.append(word)
                    tokens.append({"word": word, "status": "missing"})
                    log.warning("  MISSING  %s", word)
            last_idx = i

        if any(is_rev[i:i + cap]):
            for seg in segments[seg_before:]:
                seg["reverse"] = True

        # Pacing: a 0.5s idle clip on full stops, else a brief freeze between
        # words — but only if this word actually produced audio.
        if len(segments) > seg_before:
            if ends[last_idx]:
                idle = _pick_idle(_STOP_PAUSE)
                if idle is not None:
                    segments.append(idle)
                else:
                    segments[-1]["pause_after"] = max(
                        segments[-1].get("pause_after", 0.0), _STOP_PAUSE)
            elif last_idx < n - 1:
                segments[-1]["pause_after"] = max(segments[-1].get("pause_after", 0.0), _WORD_GAP)

        i += cap

    if not segments:
        return {"found": [], "spliced": [], "missing": missing,
                "runs": [], "tokens": tokens, "video_url": None}

    run_id = uuid.uuid4().hex[:10]
    os.makedirs("output", exist_ok=True)
    final_path = os.path.join("output", f"{run_id}.mp4")

    _say("resolving", n, n)
    _build_video(segments, final_path, progress=progress, subtitles=subtitles)

    return {
        "found":     found,
        "spliced":   spliced,
        "missing":   missing,
        "runs":      runs,
        "tokens":    tokens,
        "video_url": f"/output/{run_id}.mp4",
    }
