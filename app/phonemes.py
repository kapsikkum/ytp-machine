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

import logging
import os
import re
import random
from collections import defaultdict
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

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


# Contraction endings, for putting back the apostrophe the ingest stripped.
#
# transcribe() reduces every word to [^\w], so "don't" is stored as "dont" --
# and CMU has "don't", not "dont". The result was that the commonest words in
# any corpus had no pronunciation at all and were skipped by the splice index
# entirely: 2,879 clips in one corpus, 496 of them "dont" alone. They still
# matched as whole words, so nothing looked wrong; they simply never
# contributed a phoneme to anything, which quietly threw away the
# best-recorded material in the corpus.
# The apostrophe goes *inside* the n't family -- "don't", not "do'nt" -- so it
# cannot be handled by the same split as the rest.
_CONTRACTION_ENDINGS = ("ve", "ll", "re", "d", "s", "m")

# British spellings, which CMU (American) does not carry. Whisper writes what
# the speaker sounds like, so a British or Australian channel produces labels
# the dictionary has never seen: "honourable" had three clips and no
# pronunciation, while "honorable" had a pronunciation and no clips, leaving
# three real recordings unreachable from either spelling.
_BRITISH_ENDINGS = (("our", "or"), ("ise", "ize"), ("ised", "ized"),
                    ("ising", "izing"), ("isation", "ization"),
                    ("yse", "yze"), ("ogue", "og"))


# Acronyms said letter by letter. There is no rule that separates these from
# ordinary short words -- "usb" is "you ess bee" but "nug" is "nug", and "wii"
# is "wee" rather than three letters -- so the ones that actually turn up in
# corpora are listed instead of guessed at. Spelling every unknown short token
# out loud would mangle far more words than it rescued.
_SPELLED_OUT = {
    # computing
    "usb", "cpu", "gpu", "ssd", "hdd", "hdmi", "rgb", "rgba", "fps", "pc",
    "led", "lcd", "oled", "crt", "dvd", "cd", "vga", "dvi", "ram", "rom",
    "bios", "os", "url", "dns", "ip", "lan", "wan", "vpn", "ssh", "ftp",
    "html", "css", "sql", "api", "sdk", "ide", "cli", "gui", "wsl", "xp",
    "nt", "ms", "pdf", "png", "svg", "mp3", "mp4", "avi", "hd", "uhd", "sd",
    "psu", "pcb", "pci", "sata", "nvme", "emmc", "sdxc", "rtx", "gtx", "amd",
    "vsync", "aio", "rgbw", "kvm", "nas", "raid", "tb", "gb", "mb", "kb",
    # cars and workshop
    "lpg", "cvt", "abs", "ecu", "vin", "rpm", "suv", "awd", "fwd", "rwd",
    "atv", "utv", "hp", "bhp", "psi", "dpf", "egr", "obd", "vtec", "dohc",
    "sohc", "mpg", "kph", "mph", "bmw", "gmc", "gm", "vw", "cls", "amg",
    "rs", "gt", "gti", "sti", "wrx", "ute", "wd", "ba", "rt", "xr", "ss",
    # general
    "tv", "usa", "uk", "us", "eu", "nz", "atm", "id", "ai", "faq", "diy",
    "ceo", "cfo", "hr", "pr", "vip", "ufo", "dj", "mc", "kfc", "bbc", "cnn",
    "nasa", "fbi", "cia", "dna", "rna", "iq", "ok", "pm", "am", "bc", "ad",
    "asap", "etc", "ie", "eg", "aka", "brb", "lol", "wtf", "omg", "imo",
    "xx", "xxx", "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
    # Turned up by scripts/verify_corpus.py --unspliceable on real corpora.
    "exe", "gta", "psp", "tnt", "udp", "tcp", "vram", "zif", "dlss", "obs",
    "sdl", "rca", "rts", "efi", "esp", "ngk", "pcv", "pcp", "epa", "iso",
    "ui", "vm", "vr", "dc", "rf", "rj", "ep", "xl", "xt", "xf", "vl", "cf",
    "cv", "va", "yf", "av", "jb", "jdm", "mx", "px", "kc", "ltt", "mg",
    "mot", "hsv", "iga", "bfi", "brz", "sl", "sma", "ut", "uv", "ipd", "pmb",
    "pp", "gms", "evs", "cvt", "cvts", "bmws", "sma", "ecu",
}

# Acronyms and brand names said as a word rather than letter by letter. These
# cannot be told apart from the set above by any rule -- "wii" is "wee" and
# "wsl" is three letters, and nothing in the spelling says which -- so both
# lists are enumerated rather than inferred.
_SAID_AS_WORD = {
    "wii":     ["W", "IY"],
    "wifi":    ["W", "AY", "F", "AY"],
    "wi":      ["W", "AY"],
    "ubuntu":  ["UW", "B", "UU", "N", "T", "UW"],
    "github":  ["G", "IH", "T", "HH", "AH", "B"],
    "ipod":    ["AY", "P", "AA", "D"],
    "ipad":    ["AY", "P", "AE", "D"],
    "iphone":  ["AY", "F", "OW", "N"],
    "emulator": ["EH", "M", "Y", "AH", "L", "EY", "T", "ER"],
    "emulators": ["EH", "M", "Y", "AH", "L", "EY", "T", "ER", "Z"],
    "speedo":  ["S", "P", "IY", "D", "OW"],
    "turbo":   ["T", "ER", "B", "OW"],
    "vac":     ["V", "AE", "K"],
    "plonk":   ["P", "L", "AA", "NG", "K"],
    "booger":  ["B", "UH", "G", "ER"],
    "boogered": ["B", "UH", "G", "ER", "D"],
    "goobers": ["G", "UW", "B", "ER", "Z"],
    "turd":    ["T", "ER", "D"],
    "cack":    ["K", "AE", "K"],
    "manky":   ["M", "AE", "NG", "K", "IY"],
    "fugly":   ["F", "AH", "G", "L", "IY"],
    "friggin": ["F", "R", "IH", "G", "IH", "N"],
    "frigging": ["F", "R", "IH", "G", "IH", "NG"],
    "effed":   ["EH", "F", "T"],
    "cuz":     ["K", "AH", "Z"],
    "splines": ["S", "P", "L", "AY", "N", "Z"],
    "igniter": ["IH", "G", "N", "AY", "T", "ER"],
    "crusties": ["K", "R", "AH", "S", "T", "IY", "Z"],
    "gloopy":  ["G", "L", "UW", "P", "IY"],
    "eggy":    ["EH", "G", "IY"],
    "crinkling": ["K", "R", "IH", "NG", "K", "L", "IH", "NG"],

    # Vocalisations. Whisper writes these as words, and with no pronunciation
    # they were both unspliceable and unusable as splice material -- which for
    # a corpus full of reactions is a lot of very characteristic audio.
    "hmm":  ["HH", "M"],
    "mmm":  ["M"],
    "mm":   ["M"],
    "hm":   ["HH", "M"],
    "eww":  ["IY", "UW"],
    "ew":   ["IY", "UW"],
    "ugh":  ["AH", "G"],
    "argh": ["AA", "R"],
    "aah":  ["AA"],
    "ahh":  ["AA"],
    "ooh":  ["UW"],
    "oooh": ["UW"],
    "woo":  ["W", "UW"],
    "yay":  ["Y", "EY"],
    "huh":  ["HH", "AH"],
    "meh":  ["M", "EH"],
    "shh":  ["SH"],
    "psh":  ["P", "SH"],
    "brr":  ["B", "R"],
}


# The names of the letters, spelled out. Not taken from CMU, which holds the
# *word* each letter spells: its "a" is the article, so every acronym with an a
# in it came out saying "uh" -- "gta" as "gee tee uh". The distinction does not
# exist in the dictionary, so the table has to.
_LETTER_NAMES = {
    "a": ["EY"],       "b": ["B", "IY"],   "c": ["S", "IY"],
    "d": ["D", "IY"],  "e": ["IY"],        "f": ["EH", "F"],
    "g": ["JH", "IY"], "h": ["EY", "CH"],  "i": ["AY"],
    "j": ["JH", "EY"], "k": ["K", "EY"],   "l": ["EH", "L"],
    "m": ["EH", "M"],  "n": ["EH", "N"],   "o": ["OW"],
    "p": ["P", "IY"],  "q": ["K", "Y", "UW"], "r": ["AA", "R"],
    "s": ["EH", "S"],  "t": ["T", "IY"],   "u": ["Y", "UW"],
    "v": ["V", "IY"],  "w": ["D", "AH", "B", "AH", "L", "Y", "UW"],
    "x": ["EH", "K", "S"], "y": ["W", "AY"], "z": ["Z", "IY"],
}


def _letters_to_phones(letters: str) -> list[str] | None:
    """Phonemes for a run of letters read out one at a time: u-s-b."""
    out: list[str] = []
    for ch in letters:
        name = _LETTER_NAMES.get(ch)
        if not name:
            return None
        out.extend(name)
    return out or None


def _number_to_phones(digits: str) -> list[str] | None:
    """Phonemes for a run of digits, spoken as a number: 50 -> "fifty"."""
    try:
        from num2words import num2words
        spoken = num2words(int(digits))
    except Exception:
        return None
    out: list[str] = []
    for part in re.split(r"[\s-]+", spoken):
        part = re.sub(r"[^a-z]", "", part.lower())
        if not part:
            continue
        prons = _dict().get(part)
        if not prons:
            return None
        out.extend(prons[0])
    return out or None


def _alphanumeric_phones(w: str) -> list[str] | None:
    """Phonemes for a token mixing letters and digits: ps5, v8, xr6, i30.

    A token containing a digit is never an English word, so unlike the bare
    acronyms above this can be decided by rule rather than by list. Each run of
    letters is a word if the dictionary knows one ("i" in "i30") and read out
    letter by letter otherwise ("ps" in "ps5"); each run of digits is spoken as
    a number.
    """
    parts = re.findall(r"[a-z]+|\d+", w)
    if not parts or not any(p.isdigit() for p in parts) or not any(
            p.isalpha() for p in parts):
        return None
    out: list[str] = []
    for part in parts:
        got = (_number_to_phones(part) if part.isdigit()
               else ((_dict().get(part) or [None])[0] or _letters_to_phones(part)))
        if not got:
            return None
        out.extend(got)
    return out


# British spellings, applied wherever they appear rather than only at the end of
# a word. Suffix-only rules caught "colour" and missed "colours", "favourite",
# "kilometres" and "carburettor" -- the inflected forms, which is most of how
# words actually turn up in speech.
_BRITISH_ANYWHERE = (
    ("our", "or"),        # colour, favourite, vapours
    ("ise", "ize"),       # realise, organised
    ("isa", "iza"),       # organisation
    ("yse", "yze"),       # analyse
    ("ogue", "og"),       # catalogue
    ("aemi", "emi"),      # anaemic
    ("oeu", "eu"),        # manoeuvre
    ("tt", "t"),          # carburettor
    ("ll", "l"),          # travelling, cancelled
)

_RE_ENDING = re.compile(r"([bcdfghjklmnpqrstvwxz])re(s?)$")


def _americanised(w: str):
    """Plausible American spellings of *w*, tried when CMU has not got it."""
    seen = {w}
    for british, american in _BRITISH_ANYWHERE:
        if british in w:
            cand = w.replace(british, american)
            if cand not in seen:
                seen.add(cand)
                yield cand
    # -re -> -er after a consonant: metre, centres, kilometres. The plural has
    # to be part of the pattern, because "kilometres" ends in -res and a
    # suffix-only -re rule never sees it.
    cand = _RE_ENDING.sub(r"\1er\2", w)
    if cand not in seen:
        seen.add(cand)
        yield cand
    # Everything at once, for a label carrying more than one Briticism.
    both = w
    for british, american in _BRITISH_ANYWHERE:
        both = both.replace(british, american)
    both = _RE_ENDING.sub(r"\1er\2", both)
    if both not in seen:
        yield both


def _spelling_variants(w: str):
    """Other spellings of *w* worth trying when the dictionary has not got it."""
    # "dont" -> "don't". The apostrophe replaces nothing and sits before the
    # final t, so this is not the same operation as the endings below.
    if len(w) > 3 and w.endswith("nt"):
        yield w[:-1] + "'t"
    for suffix in _CONTRACTION_ENDINGS:
        # > len(suffix), not >= : "ive" is exactly one letter plus the ending,
        # and it is one of the commonest words a stripped apostrophe ruins.
        if len(w) > len(suffix) and w.endswith(suffix):
            yield w[: -len(suffix)] + "'" + suffix
    for british, american in _BRITISH_ENDINGS:
        if len(w) > len(british) + 1 and w.endswith(british):
            yield w[: -len(british)] + american
    # -re -> -er (centre, litre, theatre). Only after a consonant and only on a
    # long enough word, so "are" and "more" are left alone.
    if len(w) > 4 and w.endswith("re") and w[-3] not in "aeiou":
        yield w[:-2] + "er"


# ── User dictionary ───────────────────────────────────────────────────────────
#
# The tables in this file are a starting set, not an answer. Any channel brings
# words no dictionary holds -- names, in-jokes, brand names, coinages -- and
# editing Python to add one is the wrong ask, so they live in a CSV instead.
#
#     $MRS_DATA_DIR/pronunciations.csv        every corpus on this machine
#     corpora/<name>/pronunciations.csv       just this one, and packs with it
#
# The global file is the one to use. Most of what needs teaching is not specific
# to a speaker at all -- "usb", "wii", "kilometres" are the same words whoever
# is saying them -- and keeping a copy per corpus would mean maintaining the
# same entries once per channel ingested.
#
# The per-corpus file is for a speaker's own vocabulary, and it wins over the
# global one. It travels inside the bundle, so a corpus arrives able to say its
# own coinages on a machine that has never heard of them.
#
# Two columns, and the second may be written either way round:
#
#     nug,N AH G          ARPAbet, if you know it
#     wii,wee             or just a word that already sounds right
#     mcnug,mick nug      several words are fine
#     usb,=letters        read it out letter by letter
#     hevexum,=skip       leave it unpronounceable on purpose
#
# The second form is the one to use. Nobody should have to learn ARPAbet to
# tell a program that "wii" rhymes with "wee".
_USER_DICT_NAME = "pronunciations.csv"
_user_dict: dict[str, list[str] | None] | None = None

_ARPABET = _VOWELS | {"B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L",
                      "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V",
                      "W", "Y", "Z", "ZH"}


def global_dict_path() -> str:
    """The dictionary shared by every corpus in this data directory."""
    from app.database import DATA_DIR
    return os.path.join(DATA_DIR, _USER_DICT_NAME)


def user_dict_path() -> str:
    """The active corpus's own dictionary, which overrides the global one."""
    from app.database import active
    return os.path.join(active()["dir"], _USER_DICT_NAME)


def _dict_paths() -> list[str]:
    """Global first, corpus second -- later files win."""
    paths = []
    for get in (global_dict_path, user_dict_path):
        try:
            paths.append(get())
        except Exception:
            pass
    return paths


def _parse_user_value(value: str) -> list[str] | None | str:
    """One CSV value into phonemes, None to suppress, or "?" if unresolvable."""
    value = value.strip()
    if not value:
        return "?"
    if value.lower() in ("=skip", "=none", "-"):
        return None                    # deliberately left unpronounceable
    if value.lower() == "=letters":
        return "=letters"
    tokens = value.replace(",", " ").split()
    if tokens and all(t.upper() in _ARPABET for t in tokens):
        return [t.upper() for t in tokens]
    # Otherwise it names words that already sound right. Resolved through the
    # ordinary machinery, so "mick nug" works even though "nug" is only in this
    # same file -- as long as it was defined before it is used.
    out: list[str] = []
    for token in tokens:
        clean = re.sub(r"[^a-z']", "", token.lower())
        if not clean:
            continue
        # A lone consonant means its sound, not its name. "zoop,zoo p" is
        # plainly zoo + a p on the end, and reading it as "zoo pee" is never
        # what anybody meant.
        if len(clean) == 1 and clean not in "aeiou":
            out.append(clean.upper())
            continue
        got = _direct(clean)
        if not got:
            return "?"
        out.extend(got)
    return out or "?"


def _load_user_dict() -> dict[str, list[str] | None]:
    global _user_dict
    if _user_dict is not None:
        return _user_dict
    _user_dict = {}
    rejected: list[tuple[str, str, str]] = []
    import csv
    for path in _dict_paths():
        if not os.path.exists(path):
            continue
        loaded = 0
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                for row in csv.reader(f):
                    if not row or row[0].lstrip().startswith("#"):
                        continue
                    word = re.sub(r"[^\w']", "", row[0].strip().lower())
                    if not word or len(row) < 2:
                        continue
                    parsed = _parse_user_value(row[1])
                    if parsed == "=letters":
                        _user_dict[word] = _letters_to_phones(word)
                    elif parsed == "?":
                        # Silence here is the worst outcome: the entry looks
                        # applied, the word stays unsayable, and nothing
                        # explains why. Usually a typo, or a word defined in
                        # terms of another word nothing knows yet.
                        rejected.append((path, word, row[1].strip()))
                        continue
                    else:
                        _user_dict[word] = parsed   # list, or None to suppress
                    loaded += 1
        except Exception as exc:
            log.warning("could not read %s: %s", path, exc)
            continue
        if loaded:
            log.info("%d pronunciations from %s", loaded, path)
    for path, word, value in rejected:
        log.warning("%s: could not make sense of %r for %r -- use ARPAbet, "
                    "words that already sound right, =letters or =skip",
                    os.path.basename(path), value, word)
    return _user_dict


def invalidate_user_dict() -> None:
    """Forget the CSV so an edit takes effect on the next lookup."""
    global _user_dict
    _user_dict = None
    # Memoised answers were computed against the old file, and the index is
    # built from them.
    try:
        word_to_phonemes.cache_clear()
    except Exception:
        pass
    _direct.cache_clear() if hasattr(_direct, "cache_clear") else None


def _direct(w: str) -> list[str] | None:
    """Phonemes from the user CSV, the override table, or CMU."""
    user = _load_user_dict()
    if w in user:
        got = user[w]
        return list(got) if got else None      # an explicit None suppresses
    if w in _OVERRIDES:
        return list(_OVERRIDES[w])
    prons = _dict().get(w)
    if prons:
        return prons[0]
    # Only reached once the word itself is not in the dictionary, so none of
    # this can displace a real pronunciation -- it only fills a hole.
    for variant in _spelling_variants(w):
        prons = _dict().get(variant)
        if prons:
            return prons[0]
    if w in _SAID_AS_WORD:
        return list(_SAID_AS_WORD[w])
    if w.isdigit():
        return _number_to_phones(w)
    if w in _SPELLED_OUT:
        return _letters_to_phones(w)
    # Drawn-out vocalisations: "ohhhhh", "brrrrrrra", "hahahaha". Whisper
    # spells the length out, so every stretch is its own unique label. Collapse
    # the repeats and try again -- "ohhh" is "oh", said longer.
    for collapse in (r"\1", r"\1\1"):
        # Once down to a single letter, once to a doubled one: "ohhhhh" wants
        # "oh" and "brrrr" wants "brr", and no single rule gives both.
        squashed = re.sub(r"(.)\1{2,}", collapse, w)
        if squashed == w:
            continue
        got = _OVERRIDES.get(squashed) or _SAID_AS_WORD.get(squashed)
        if got:
            return list(got)
        prons = _dict().get(squashed)
        if prons:
            return prons[0]
    # "hahahaha", "lalala": a syllable repeated. Say it once.
    m = re.fullmatch(r"(.{2,3}?)\1{1,}", w)
    if m:
        prons = _dict().get(m.group(1))
        if prons:
            return prons[0]
        got = _SAID_AS_WORD.get(m.group(1))
        if got:
            return list(got)
    return _alphanumeric_phones(w)


# Endings CMU does not list separately, with the phonemes they add. An English
# dictionary holds "smelly" but not "smelliest", "snip" but not "snipped", and
# a corpus of ordinary speech is full of the derived forms -- they were a
# larger share of the unpronounceable words than every acronym put together.
#
# The base is found by undoing the three spelling changes English makes when it
# adds a suffix: a dropped silent e (save + able), a doubled final consonant
# (snip + ed), and y turning to i (smelly + est).
_SUFFIX_PHONES = (
    ("iest",  ["IY", "AH", "S", "T"]),
    ("est",   ["AH", "S", "T"]),
    ("ier",   ["IY", "ER"]),
    ("iness", ["IY", "N", "AH", "S"]),
    ("ness",  ["N", "AH", "S"]),
    ("ily",   ["AH", "L", "IY"]),
    ("ly",    ["L", "IY"]),
    ("able",  ["AH", "B", "AH", "L"]),
    ("ible",  ["AH", "B", "AH", "L"]),
    ("ish",   ["IH", "SH"]),
    ("y",     ["IY"]),
)

_DOUBLE = "bdfglmnprstz"


def _bases_for(stem: str):
    """Spellings the base might have had before a suffix was added."""
    yield stem
    yield stem + "e"                                  # save + able -> savable
    if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] in _DOUBLE:
        yield stem[:-1]                               # snipp + ed -> snip
    if stem.endswith("i"):
        yield stem[:-1] + "y"                         # smelli + est -> smelly


# Prefixes that leave the base word intact. "unplayable" is "un" + a word CMU
# does not carry either, so the prefix has to come off before the suffix rules
# get a look at what is left.
_PREFIX_PHONES = (
    ("un",     ["AH", "N"]),
    ("re",     ["R", "IY"]),
    ("over",   ["OW", "V", "ER"]),
    ("under",  ["AH", "N", "D", "ER"]),
    ("non",    ["N", "AA", "N"]),
    ("mis",    ["M", "IH", "S"]),
    ("pre",    ["P", "R", "IY"]),
)


def _voiced_ending(phones: list[str]) -> list[str]:
    """The phonemes -ed adds, which depend on the sound before it."""
    last = phones[-1]
    if last in ("T", "D"):
        return ["AH", "D"]
    return ["T"] if last in ("P", "K", "F", "S", "SH", "CH", "TH") else ["D"]


def _suffixed(w: str) -> list[str] | None:
    """Phonemes for a derived form: base pronunciation plus the suffix."""
    # -ed and -ing over a doubled consonant: "snipped" -> snip, "modded" -> mod.
    # The plain rules in _resolve_simple do not undo the doubling.
    for suffix in ("ed", "ing"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            stem = w[: -len(suffix)]
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] in _DOUBLE:
                got = _direct(stem[:-1])
                if got:
                    return got + (["IH", "NG"] if suffix == "ing"
                                  else _voiced_ending(got))
    for suffix, phones in _SUFFIX_PHONES:
        if len(w) <= len(suffix) + 2 or not w.endswith(suffix):
            continue
        stem = w[: -len(suffix)]
        for base in _bases_for(stem):
            got = _direct(base)
            if got:
                # A base ending in the suffix's own vowel would say it twice.
                if phones[0] == "IY" and got[-1] == "IY":
                    return got + phones[1:]
                return got + phones
    return None


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


@lru_cache(maxsize=100_000)
def word_to_phonemes(word: str) -> list[str] | None:
    """Return stripped ARPAbet phonemes for *word*, or None.

    Tries: overrides → CMU → inflections → compound split (run-together words
    like "dumbass" = dumb + ass, "asshole" = ass + hole).
    """
    w = word.lower()
    r = _resolve_simple(w)
    if r:
        return r
    # An American spelling, then the inflection rules on top of it, so
    # "kilometres" reaches "kilometers" and "vapours" reaches "vapors" through
    # the plural rule rather than each needing an entry of its own.
    for cand in _americanised(w):
        r = _resolve_simple(cand)
        if r:
            return r
    # Derived forms: "smelliest" from "smelly", "snipped" from "snip".
    r = _suffixed(w)
    if r:
        return r
    # A prefix on a word that itself needed deriving: "unplayable".
    for prefix, phones in _PREFIX_PHONES:
        if len(w) > len(prefix) + 2 and w.startswith(prefix):
            rest = w[len(prefix):]
            got = _resolve_simple(rest) or _suffixed(rest)
            if got:
                return phones + got
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
        # The same resolution the rest of the module uses, rather than a bare
        # CMU lookup.
        #
        # This decides what sounds the splicer believes a clip contains, and it
        # used to consult only _OVERRIDES and CMU -- so the per-corpus
        # dictionary, restored contractions, British spellings, acronyms,
        # numbers and derived forms all applied to the word being asked for and
        # not to the words it was built out of. Two consequences, both bad:
        #
        #   - every clip of "dont", "usb", "kilometres" and the rest stayed
        #     invisible to the splicer, even after they had pronunciations
        #   - a corpus could not correct the dictionary about its own speaker.
        #     CMU is American and says tom-AY-to; Rosen says tom-AH-to, so the
        #     splicer cut "mate" out of the middle of "tomato" and produced
        #     "mart". Writing the real pronunciation into pronunciations.csv
        #     did nothing, because this line never read it.
        #
        # word_to_phonemes only returns the primary pronunciation, which is
        # still the point of the old comment here: CMU's secondary variants
        # often do not match the audio ("get" has a G-IH-T variant, but Michael
        # says g-EH-t, so using it for "shit" gives "sh-et").
        raw = word_to_phonemes(w)
        if not raw:
            continue
        cphones = [_EQUIV.get(p, p) for p in raw]
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


# Roughly how long a phoneme needs to be heard, and the extra a vowel wants.
#
# Nothing in the splice path looked at duration before, so a unit was cut from
# whichever clip the DP happened to pick. That is fine for a word said once,
# and wrong for the common ones: "it" has 3,537 clips in one corpus with a
# median of 110ms, which for IH plus T leaves a vowel too short to hear. "shit"
# came out "shht" -- both consonants present, the vowel gone.
#
# A preference, not a filter: when nothing meets the floor the longest clip is
# still used, because a short vowel beats reporting the word missing.
_MIN_PER_PHONE = 0.05
_VOWEL_EXTRA = 0.05


def _clip_floor(phones) -> float:
    """The shortest a clip can plausibly be and still contain *phones*."""
    need = _MIN_PER_PHONE * len(phones)
    if any(_is_vowel(p) for p in phones):
        need += _VOWEL_EXTRA
    return need


def _by_audibility(clips: list[dict], phones) -> list[dict]:
    """Clips ordered so the ones long enough to hold *phones* come first.

    Among those, shortest-first: a clip with room for the sounds is wanted, not
    the longest one in the corpus, which is usually a drawn-out or mis-aligned
    outlier.
    """
    floor = _clip_floor(phones)
    def key(c):
        dur = c["end_time"] - c["start_time"]
        return (0, dur) if dur >= floor else (1, -dur)
    return sorted(clips, key=key)


# A splice unit is trusted to be sound all the way to its stored end, and it is
# not. An aligner marks a boundary where it stops being confident, which for a
# short word is often well past where the word stopped: Morshu's "i" is stored
# as 140ms, of which only the first ~40ms is the vowel and the rest is the
# silent closure of the "can't" that follows it.
#
# Spliced into the middle of another word, that silence is a hole. "time" came
# out as t + [i, long gap] + m, which the ear hears as two words rather than
# one -- and since the gap is the run-up to "can't", it sounds like the phrase
# it was cut from.
#
# Trimmed relative to the unit's own peak, so it adapts to a whisper as
# readily as a shout, and never below a floor -- a stop's closure is silence
# that belongs to the sound.
_TRIM_FLOOR = 0.18       # fraction of the unit's peak that still counts as sound
_TRIM_KEEP = 0.03        # always keep this much after the last audible moment
_TRIM_MIN = 0.045        # never shorten a unit below this


@lru_cache(maxsize=20_000)
def _audible_end(source_file: str, start: float, end: float) -> float:
    """Where the sound in this span actually stops.

    Memoised: the same handful of clips supply most splices, and the answer
    depends only on the audio, which does not change under us.
    """
    import array
    import math
    import os
    import subprocess
    import tempfile
    import wave

    dur = end - start
    if dur <= _TRIM_MIN:
        return end
    tmp = os.path.join(tempfile.gettempdir(), f"_tr_{os.getpid()}.wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.4f}",
                        "-t", f"{dur:.4f}", "-i", source_file,
                        "-ac", "1", "-ar", "16000", tmp], capture_output=True)
        with wave.open(tmp, "rb") as w:
            raw = w.readframes(w.getnframes())
    except Exception:
        return end
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    if not raw:
        return end
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) // 2 * 2])
    if not samples:
        return end

    win = 160                                   # 10ms
    env = [math.sqrt(sum(v * v for v in samples[k:k + win]) / win)
           for k in range(0, len(samples) - win, win)]
    if not env:
        return end
    peak = max(env)
    if peak <= 0:
        return end
    floor = peak * _TRIM_FLOOR
    last = 0
    for i, level in enumerate(env):
        if level >= floor:
            last = i
    trimmed = start + (last + 1) * (win / 16000.0) + _TRIM_KEEP
    return max(start + _TRIM_MIN, min(end, trimmed))


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
            # Drop any dead air the aligner left on the end. A unit is going
            # inside another word, where a silence reads as a word boundary.
            s["end_time"] = _audible_end(cand["source_file"],
                                         s["start_time"], s["end_time"])
            s["next_start"] = s["end_time"]      # nothing follows it in the join
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

        # Long enough to be heard first, then the DP's pick, then the rest.
        # The DP chooses a source *word* well and a clip of it arbitrarily, so
        # this is where a 110ms "it" gets passed over for one with a vowel in
        # it.
        pool = clips_by_word.get(cword) or [cclip]
        clips_chosen = _by_audibility(pool, sub)
        if penalty:                               # down-rated clips tried last
            clips_chosen.sort(key=lambda c: penalty.get(c.get("id"), 0))
        seg = _try(cword, clips_chosen, cpos, ctotal, sub, last_v)
        if seg is None:
            for _ncuts, w, cl, ps, total in _sources_for(sub):
                if w == cword:
                    continue
                seg = _try(w, _by_audibility(cl, sub), ps, total, sub, last_v)
                if seg is not None:
                    break
        if seg is None:                          # last resort: whole word clip
            # Use the LONGEST available clip, not the DP pick — that one may be a
            # degenerate ~10ms alignment artefact that would be inaudible.
            pool = clips_by_word.get(cword) or [cclip]
            best = max(pool, key=lambda c: c["end_time"] - c["start_time"])
            seg = dict(best); seg["_cut"] = False
            # Whole-word units get the dead-air trim as well, and need it most:
            # this branch is reached when alignment could not cut the word at
            # all, so nothing has looked at where the sound actually stops. It
            # is where Morshu's "i" came through carrying 100ms of the closure
            # before "can't", which turned "time" into "t-i-[gap]-m".
            seg["end_time"] = _audible_end(seg["source_file"],
                                           seg["start_time"], seg["end_time"])
            seg["next_start"] = seg["end_time"]
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
