"""
Forced alignment for sub-word splicing.

Uses torchaudio's CTC forced-alignment (Wav2Vec2 BASE 960h, English characters)
to find where each *character* of a word sits in time inside a clip.  That lets
us cut a source word at an accurate sub-word boundary — e.g. take just the "th"
onset of "thought" — instead of only using whole words.

The model is loaded lazily and cached; character timings are memoised for each
clip's source, label and boundaries so unchanged clips need only one pass.
"""
from __future__ import annotations

import os
from typing import Any

import logging

log = logging.getLogger(__name__)

_SR = 16000

# ── Lazy model ──────────────────────────────────────────────────────────────
_model = None
_labels: tuple[str, ...] | None = None
_dict: dict[str, int] | None = None
_device: str = "cpu"


def _ensure_model() -> bool:
    """Load the alignment model once.  Returns False if unavailable."""
    global _model, _labels, _dict, _device
    if _model is not None:
        return True
    try:
        import torch  # noqa: F401
        import torchaudio
        from app.device import get as _get_device, describe as _describe

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        _device = _get_device()
        # This pass is the long one when building a corpus -- one Wav2Vec2
        # forward per word clip, and a corpus is tens of thousands of them --
        # so leaving the model on the CPU was quietly costing hours.
        _model = bundle.get_model().eval().to(_device)
        _labels = bundle.get_labels()
        _dict = {c: i for i, c in enumerate(_labels)}
        log.info("  forced-aligner loaded (%d labels) on %s",
                 len(_labels), _describe(_device))
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("  forced-aligner unavailable: %s", exc)
        _model = None
        return False


# ── Per-clip character timings ───────────────────────────────────────────────
# Ids start again at 1 in every corpus, and a sentence can hold two voices.
# Include the source, label and bounds: the same id can describe a new word
# after an edit, even when the caller has not explicitly invalidated it yet.
_char_cache: dict[tuple, list[tuple[str, float, float]] | None] = {}

import json as _json


def _clip_key(clip: dict[str, Any]) -> tuple:
    """Source and clip fingerprint, keeping id at index 1 for invalidate()."""
    from app.database import normalise_sep, resolve_path

    source = clip.get("source_file")
    if source:
        source = normalise_sep(source)
        # Borrowed clips already have an absolute path checked by their pool.
        # Resolve relative paths only against the active corpus.
        source = source if os.path.isabs(source) else resolve_path(source)
        source = os.path.normcase(os.path.realpath(source))
    return (source, clip.get("id"), clip.get("word"),
            clip.get("start_time"), clip.get("end_time"))


def _active_clip(conn, clip: dict[str, Any]) -> dict[str, Any] | None:
    """The matching active-corpus row; borrowed and edited inputs decline.

    Persistent alignments are keyed by id within the active database. An id
    alone cannot identify a clip from another voice, so never consult or write
    that table unless the supplied source, word and bounds match its own row.
    """
    row = conn.execute(
        "SELECT id, source_file, word, start_time, end_time FROM word_clips WHERE id=?",
        (clip.get("id"),)).fetchone()
    if row is None:
        return None
    stored = dict(row)
    return stored if _clip_key(stored) == _clip_key(clip) else None


def _load_alignment(clip: dict[str, Any]) -> list[tuple[str, float, float]] | None:
    """Load a matching active clip's persisted alignment, or None."""
    try:
        from app.database import get_db
        with get_db() as conn:
            if _active_clip(conn, clip) is None:
                return None
            row = conn.execute(
                "SELECT alignment FROM clip_alignments WHERE clip_id=?",
                (clip["id"],)
            ).fetchone()
            if row:
                data = _json.loads(row["alignment"] if isinstance(row, dict) else row[0])
                return [(ch, float(s), float(e)) for ch, s, e in data]
    except Exception:
        pass
    return None


def _save_alignment(clip: dict[str, Any], alignment: list[tuple[str, float, float]]) -> None:
    """Persist a matching active clip's alignment. Best-effort, never raises."""
    try:
        from app.database import get_db
        data = _json.dumps([(ch, round(s, 6), round(e, 6)) for ch, s, e in alignment])
        with get_db() as conn:
            stored = _active_clip(conn, clip)
            if stored is None:
                return
            conn.execute(
                "INSERT OR REPLACE INTO clip_alignments (clip_id, alignment) "
                "SELECT id, ? FROM word_clips WHERE id=? AND source_file=? "
                "AND word=? AND start_time=? AND end_time=?",
                (data, stored["id"], stored["source_file"], stored["word"],
                 stored["start_time"], stored["end_time"])
            )
    except Exception:
        pass

def invalidate(clip_id: int | None = None) -> None:
    """Forget an alignment, or all of them.

    The cache is keyed on clip id and holds times relative to the clip's own
    start_time, so it is only true for the clip as it was when aligned. Two
    ways it goes wrong:

      - a clip is edited. Moving a boundary in the corpus editor changes both
        the window aligned and the origin the times are measured from, so
        every sub-word cut out of that clip afterwards lands somewhere else
        than it says. The word still comes out; it is just cut in the wrong
        place, until the process restarts.

      - the corpus is switched. Ids start again at 1 in each corpus, so clip 5
        of the new one inherits the alignment of clip 5 of the old -- a
        different word, in a different video.
    """
    if clip_id is None:
        _char_cache.clear()
        try:
            from app.database import get_db
            with get_db() as conn:
                conn.execute("DELETE FROM clip_alignments")
        except Exception:
            pass
    else:
        for key in [k for k in _char_cache if k[1] == clip_id]:
            del _char_cache[key]
        try:
            from app.database import get_db
            with get_db() as conn:
                conn.execute("DELETE FROM clip_alignments WHERE clip_id=?", (clip_id,))
        except Exception:
            pass


def invalidate_many(clip_ids, *, persisted: bool = True) -> None:
    """Drop several clip fingerprints without rescanning the cache per id."""
    ids = set(clip_ids)
    for key in list(_char_cache):
        if key[1] in ids:
            _char_cache.pop(key, None)
    if persisted and ids:
        try:
            from app.database import get_db
            with get_db() as conn:
                conn.executemany("DELETE FROM clip_alignments WHERE clip_id=?", [(cid,) for cid in ids])
        except Exception:
            pass


def char_times(clip: dict[str, Any]) -> list[tuple[str, float, float]] | None:
    """Return [(char, start_s, end_s), …] for *clip*'s word, times relative to
    the clip's own start_time.  None if alignment fails."""
    cid = clip.get("id")
    try:
        key = _clip_key(clip)
    except (OSError, ValueError, TypeError) as exc:
        log.warning("  invalid alignment source for %r: %s", clip.get("word"), exc)
        return None
    if cid is not None and key in _char_cache:
        return _char_cache[key]

    # Check persistent storage before running the model
    if cid is not None:
        result = _load_alignment(clip)
        if result is not None:
            _char_cache[key] = result
            return result

    result = _align(clip)
    if cid is not None:
        _char_cache[key] = result
        if result is not None:
            _save_alignment(clip, result)
    return result


def _align(clip: dict[str, Any]) -> list[tuple[str, float, float]] | None:
    if not _ensure_model():
        return None

    word = "".join(ch for ch in clip["word"].upper() if ch in _dict)  # type: ignore[operator]
    if not word:
        return None

    import torch
    import torchaudio
    from torchaudio.functional import merge_tokens

    # CTC needs at least as many audio frames (~1 per 20ms) as characters, so
    # short clips fail.  Pad the window with surrounding audio to guarantee
    # enough frames; times are converted back to be relative to the clip start.
    _PAD = 0.20
    pad = min(_PAD, clip["start_time"])
    from app.database import resolve_path
    start = clip["start_time"] - pad
    dur   = (clip["end_time"] - clip["start_time"]) + pad + _PAD
    try:
        from app.ytpmv.pitch import decode_audio
        # A clip from the generator's pool is already absolute, checked against
        # its own corpus -- which for a borrowed voice is not the active one.
        src = clip["source_file"]
        audio = decode_audio(src if os.path.isabs(src) else resolve_path(src), start, dur,
                             _SR, "s16le")
        wav = torch.from_numpy(audio).unsqueeze(0).to(_device)
        with torch.inference_mode():
            emission, _ = _model(wav)  # type: ignore[misc]
        targets = torch.tensor([[_dict[c] for c in word]], dtype=torch.int32,  # type: ignore[index]
                               device=emission.device)
        if emission.size(1) <= targets.size(1):
            return None  # still too short — caller falls back to whole clip
        aligned, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
        # Back to the host for merge_tokens: it walks the sequence element by
        # element, which on a GPU tensor is a synchronising read per step.
        spans = merge_tokens(aligned[0].cpu(), scores[0].cpu())
        ratio = wav.size(1) / emission.size(1) / _SR

        out: list[tuple[str, float, float]] = []
        for s in spans:
            if s.token == 0:
                continue
            # shift back so times are relative to the clip's own start_time
            out.append((_labels[s.token], s.start * ratio - pad, s.end * ratio - pad))  # type: ignore[index]
        return out or None
    except Exception as exc:
        log.warning("  alignment failed for %r: %s", clip.get("word"), exc)
        return None


def phone_times(clip: dict[str, Any]) -> list[tuple[str, float, float]] | None:
    """This clip's aligned phoneme boundaries, or None if it has none.

    Stored per clip by the build, in seconds from the clip's own start_time --
    the same frame of reference char_times uses, so the two are
    interchangeable to a caller.
    """
    raw = clip.get("phones")
    if not raw:
        return None
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    try:
        return [(p, float(a), float(b)) for p, a, b in raw]
    except Exception:
        return None


# ── Grapheme grouping (≈ one group per phoneme) ──────────────────────────────
# Common multi-letter graphemes that map to a single phoneme, so we can map a
# phoneme count onto a character index for cutting.
_DIGRAPHS = (
    "tch", "sch",
    "th", "sh", "ch", "ph", "wh", "ck", "ng", "qu", "gh",
    "oo", "ee", "ea", "ai", "ay", "oa", "ow", "ou", "oy", "oi",
    "au", "aw", "ew", "ie", "igh", "ar", "er", "ir", "or", "ur",
)


def _grapheme_groups(word: str) -> list[int]:
    """Split *word* (lowercase) into grapheme group lengths, greedily merging
    common digraphs so the group count roughly matches the phoneme count."""
    w = word.lower()
    groups: list[int] = []
    i = 0
    n = len(w)
    while i < n:
        matched = 1
        for dg in _DIGRAPHS:
            if w.startswith(dg, i):
                matched = len(dg)
                break
        groups.append(matched)
        i += matched
    return groups


def word_core_end(clip: dict[str, Any]) -> float | None:
    """Return the END time (relative to clip start) of the word's last character
    — i.e. where the word's actual content ends, independent of any padding or
    the next word.  None if alignment fails."""
    ct = char_times(clip)
    if not ct:
        return None
    return ct[-1][2]


def cut_start_before_phonemes(clip: dict[str, Any], skip: int) -> float | None:
    """Return the START time (relative to clip start) where the word's content
    begins after skipping its first *skip* phonemes — i.e. cut the front off so
    we can take a suffix like the "at" inside "fat".  None if alignment fails.
    """
    # Aligned phonemes if the corpus has them: then this is simply where that
    # phoneme starts, with nothing to infer.
    pt = phone_times(clip)
    if pt:
        return pt[0][1] if skip <= 0 else pt[min(skip, len(pt) - 1)][1]

    ct = char_times(clip)
    if not ct:
        return None
    if skip <= 0:
        return ct[0][1]                       # word onset (trim any leading silence)
    groups = _grapheme_groups(clip["word"])
    if skip >= len(groups):
        return ct[-1][1]
    char_idx = sum(groups[:skip])            # first character of the kept part
    if char_idx >= len(ct):
        return ct[-1][1]
    return ct[char_idx][1]                    # that character's start


def cut_end_after_phonemes(clip: dict[str, Any], k: int,
                           last_is_vowel: bool = True) -> float | None:
    """Return the END time (relative to clip start) for the first *k* phonemes
    of the clip's word, or None if it can't be determined.

    If the k-th phoneme is a VOWEL we cut at the start of the next character so
    sustained vowels aren't truncated.  If it's a CONSONANT we cut just after
    that consonant instead — forced alignment labels the vowel *letters* in the
    middle of a long diphthong (the 'I' in "pie" lands at 0.25s), so cutting at
    the next char would drag the whole vowel in ("rape" → "raypie").
    """
    pt = phone_times(clip)
    if pt:
        # The end of the k-th phoneme, as aligned. None of the compensation
        # below applies: it exists because a phoneme index had to be guessed
        # onto a letter, and a letter boundary is not a sound boundary.
        end = pt[min(k, len(pt)) - 1][2] if k > 0 else pt[0][1]
        if not last_is_vowel and k > 0:
            # A stop still wants its burst: alignment marks the closure, and
            # the release is the part you can hear.
            start = pt[min(k, len(pt)) - 1][1]
            end = max(end, start + 0.055)
            if k < len(pt):
                end = min(end, pt[k][2])      # never past the next phoneme
        return end

    ct = char_times(clip)
    if not ct:
        return None

    groups = _grapheme_groups(clip["word"])
    if k >= len(groups):
        return ct[-1][2]  # whole word

    char_idx = sum(groups[:k])           # first character AFTER the matched groups
    if char_idx >= len(ct) or char_idx == 0:
        return ct[-1][2] if char_idx >= len(ct) else ct[char_idx][1]

    if last_is_vowel:
        # Cut at the next character's start.  If the grapheme/phoneme mapping
        # under-counted (a consonant spelled with 2 letters, e.g. "ge"=JH in
        # "george"), char_idx lands ON a vowel letter — extend THROUGH the vowel
        # run so it isn't clipped, but never into the following consonant
        # (which would turn "bi" into "big").
        cut_t = ct[char_idx][1]
        j = char_idx
        while j < len(ct) and ct[j][0].lower() in "aeiouy":
            cut_t = ct[j][2]
            j += 1
        return cut_t
    # Consonant: cut just past the last matched character — but keep at least
    # ~55ms so a stop's burst/aspiration is audible (a 35ms "k" vanishes) —
    # never reaching the next (vowel) character's start.
    cstart, cend = ct[char_idx - 1][1], ct[char_idx - 1][2]
    cut = max(cend + 0.03, cstart + 0.055)
    return min(cut, ct[char_idx][1])
