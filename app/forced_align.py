"""
Forced alignment for sub-word splicing.

Uses torchaudio's CTC forced-alignment (Wav2Vec2 BASE 960h, English characters)
to find where each *character* of a word sits in time inside a clip.  That lets
us cut a source word at an accurate sub-word boundary — e.g. take just the "th"
onset of "thought" — instead of only using whole words.

The model is loaded lazily and cached; per-clip character timings are memoised
by clip id so a word is never aligned twice in a session.
"""
from __future__ import annotations

import os
import subprocess
import wave
from typing import Any

import logging

log = logging.getLogger(__name__)

_SR = 16000

# ── Lazy model ──────────────────────────────────────────────────────────────
_model = None
_labels: tuple[str, ...] | None = None
_dict: dict[str, int] | None = None


def _ensure_model() -> bool:
    """Load the alignment model once.  Returns False if unavailable."""
    global _model, _labels, _dict
    if _model is not None:
        return True
    try:
        import torch  # noqa: F401
        import torchaudio
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        _model = bundle.get_model()
        _labels = bundle.get_labels()
        _dict = {c: i for i, c in enumerate(_labels)}
        log.info("  forced-aligner loaded (%d labels)", len(_labels))
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("  forced-aligner unavailable: %s", exc)
        _model = None
        return False


def _load_wav_mono16k(path: str):
    import numpy as np
    import torch
    w = wave.open(path, "rb")
    data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype("float32") / 32768.0
    w.close()
    return torch.from_numpy(data).unsqueeze(0)


# ── Per-clip character timings ───────────────────────────────────────────────
_char_cache: dict[int, list[tuple[str, float, float]] | None] = {}


def char_times(clip: dict[str, Any]) -> list[tuple[str, float, float]] | None:
    """Return [(char, start_s, end_s), …] for *clip*'s word, times relative to
    the clip's own start_time.  None if alignment fails."""
    cid = clip.get("id")
    if cid is not None and cid in _char_cache:
        return _char_cache[cid]

    result = _align(clip)
    if cid is not None:
        _char_cache[cid] = result
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
    tmp = f"_fa_{os.getpid()}.wav"
    start = clip["start_time"] - pad
    dur   = (clip["end_time"] - clip["start_time"]) + pad + _PAD
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.4f}", "-t", f"{dur:.4f}",
             "-i", resolve_path(clip["source_file"]), "-ac", "1", "-ar", str(_SR), tmp],
            capture_output=True,
        )
        wav = _load_wav_mono16k(tmp)
        with torch.inference_mode():
            emission, _ = _model(wav)  # type: ignore[misc]
        targets = torch.tensor([[_dict[c] for c in word]], dtype=torch.int32)  # type: ignore[index]
        if emission.size(1) <= targets.size(1):
            return None  # still too short — caller falls back to whole clip
        aligned, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
        spans = merge_tokens(aligned[0], scores[0])
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
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


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
