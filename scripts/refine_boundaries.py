#!/usr/bin/env python3
"""
Refine word-clip boundaries with CTC forced alignment.

Whisper/stable-whisper word timestamps are shifted early and vary by 100-350ms,
so short clips grab the previous word's tail.  This re-aligns each source's word
sequence to its audio with wav2vec2 forced alignment (frame-accurate) and
rewrites start_time/end_time in the DB — fixing short-word offset and making
splices land cleanly.

Usage:
    python scripts/refine_boundaries.py --source-id 36          # one source (dry run)
    python scripts/refine_boundaries.py --source-id 36 --apply
    python scripts/refine_boundaries.py --all --apply
    python scripts/refine_boundaries.py --all --apply --start-from-id 1 --end-at-id 16
"""
import argparse
import math
import os
import re
import subprocess
import tempfile
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db, get_db

_SR = 16000
_model = None
_labels = None
_DICT = None
_device = "cpu"

# How much audio goes through the model at once. Wav2Vec2 emits a frame per
# 20ms and attends over all of them, so cost grows with the square of this
# number: 40s is ~2,000 frames, which is comfortable on a laptop GPU and still
# gives the aligner plenty of context either side of every word.
_WINDOW_SECONDS = 40.0

# Extra audio at each end of a window, so a word sitting on the boundary is not
# asked to align against a sound that has been cut in half.
_WINDOW_PAD = 0.5


def _group_words(words: list[dict], window: float) -> list[list[int]]:
    """Word indices split into windows of at most *window* seconds.

    Whole words only: a word split across two windows would be aligned twice
    against two half-sounds and land worse than the timestamp it started with.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    start = 0.0
    for i, w in enumerate(words):
        if not current:
            current, start = [i], w["start_time"]
        elif w["end_time"] - start > window:
            groups.append(current)
            current, start = [i], w["start_time"]
        else:
            current.append(i)
    if current:
        groups.append(current)
    return groups


def _load_model(device: str | None = None):
    global _model, _labels, _DICT, _device
    if _model is not None:
        return
    import torchaudio
    from app.device import get as get_device, describe

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    _device = get_device(device)
    _model = bundle.get_model().eval().to(_device)
    _labels = bundle.get_labels()
    _DICT = {c: i for i, c in enumerate(_labels)}
    print(f"  aligner on {describe(_device)}")


def _load_wav(path):
    import numpy as np
    import torch
    with wave.open(path, "rb") as w:
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype("float32") / 32768.0
    return torch.from_numpy(d).unsqueeze(0)


def _alignment_text(word: str) -> str | None:
    """English target text without guessing a partial alphanumeric word.

    Numbers can occupy one stored clip even when spoken as several words.
    Leading zeroes and mixed letters/digits have several possible readings,
    so retain their original timestamps instead of aligning a guessed prefix.
    """
    match = re.fullmatch(r"([0-9]+)(st|nd|rd|th)?", word.lower())
    if match:
        digits, suffix = match.groups()
        if len(digits) > 1 and digits.startswith("0"):
            return None
        try:
            from num2words import num2words
            spoken = num2words(int(digits), to="ordinal" if suffix else "cardinal")
        except (ImportError, ValueError, OverflowError, NotImplementedError):
            return None
        return "|".join(spoken.upper().replace(",", "").replace("-", " ").split())
    if any(ch.isdigit() for ch in word):
        return None
    return word.upper()


def _ctc_targets(words: list[dict]) -> tuple[list[int], list[tuple[int, int] | None]]:
    """Encode whole words, retaining the model's trained word separator.

    Without the separator, "a nice" and "an ice" are the same target string;
    the aligner has no evidence of which boundary the transcript intended.
    Ranges exclude separators so word bounds come from letters alone.
    """
    chars: list[int] = []
    ranges: list[tuple[int, int] | None] = []
    for word in words:
        text = _alignment_text(word["word"])
        if not text or any(c not in _DICT for c in text):
            ranges.append(None)
            continue
        cw = [_DICT[c] for c in text]
        if chars:
            chars.append(_DICT["|"])
        start = len(chars)
        chars.extend(cw)
        ranges.append((start, len(chars)))
    return chars, ranges


def _align_window(segment, words: list[dict]) -> list[tuple[float, float] | None]:
    """Align one audio window; return window-relative word cores."""
    import torch
    import torchaudio
    from torchaudio.functional import merge_tokens

    chars, ranges = _ctc_targets(words)
    empty = [None] * len(words)
    if not chars:
        return empty
    with torch.inference_mode():
        emission, _ = _model(segment)
        # forced_align expects log probabilities, rather than unnormalised
        # logits. Its returned scores then share the model's probability scale.
        logp = emission.log_softmax(dim=-1)
    # Repeated characters need an intervening blank in the CTC path.
    required = len(chars) + sum(a == b for a, b in zip(chars, chars[1:]))
    if emission.size(1) < required:
        return empty

    targets = torch.tensor([chars], dtype=torch.int32, device=emission.device)
    aligned, scores = torchaudio.functional.forced_align(logp, targets, blank=0)
    spans = merge_tokens(aligned[0].cpu(), scores[0].cpu())
    if len(spans) != len(chars):
        return empty
    ratio = segment.size(1) / emission.size(1) / _SR
    return [(spans[rng[0]].start * ratio, spans[rng[1] - 1].end * ratio)
            if rng is not None else None for rng in ranges]


def _boundary_updates(words: list[dict], fa: list[tuple[float, float] | None],
                      duration: float) -> tuple[list[tuple[float, float, int]], list[float]]:
    """Pad aligned cores within the source and the immediate next word.

    Hand-edited clips still supply transcript context, but their boundaries
    are final and also constrain the automatic clip immediately before them.
    """
    updates: list[tuple[float, float, int]] = []
    shifts: list[float] = []
    for i, (word, aligned) in enumerate(zip(words, fa)):
        if aligned is None or word.get("edited"):
            continue
        onset, core_end = aligned
        if not all(math.isfinite(t) for t in (onset, core_end)) or core_end <= onset:
            continue
        nxt = None
        if i + 1 < len(words):
            successor = words[i + 1]
            nxt = (fa[i + 1][0] if fa[i + 1] is not None and not successor.get("edited")
                   else successor["start_time"])
        new_start = max(0.0, onset)
        # Preserve the core where there is less than 20ms between words, but
        # never let a minimum tail override the next word's onset or EOF.
        new_end = min(core_end + 0.30, duration)
        if nxt is not None:
            new_end = min(new_end, max(core_end, nxt - 0.02), nxt)
        if new_end - new_start < 0.04:
            continue
        shifts.append(new_start - word["start_time"])
        updates.append((round(new_start, 4), round(new_end, 4), word["id"]))
    return updates, shifts


def refine_source(sid: int, apply: bool) -> tuple[int, int, float]:
    """Re-align one source. Returns (updated, total, median_shift_ms)."""
    with get_db() as conn:
        src = conn.execute("SELECT source_file FROM sources WHERE id=?", (sid,)).fetchone()
        words = [dict(r) for r in conn.execute(
            "SELECT id, word, start_time, end_time, edited FROM word_clips "
            "WHERE source_id=? ORDER BY start_time", (sid,)).fetchall()]
    if not src or not words:
        return (0, 0, 0.0)

    from app.database import resolve_path
    # Scratch audio goes to the system temp directory, not the working directory.
    # ffmpeg is handed a path and does not care, but the working directory is
    # usually the repo: a killed run left a 50 MB wav sitting in it, syncing to
    # OneDrive and one `git add -A` away from the history. The temp directory is
    # also the right place for something this size -- it is cleaned up by the
    # system if a process dies before its own cleanup runs.
    with tempfile.NamedTemporaryFile(prefix="mrs-refine-", suffix=".wav", delete=False) as scratch:
        tmp = scratch.name
    try:
        decoded = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", resolve_path(src["source_file"]),
             "-ac", "1", "-ar", str(_SR), tmp], capture_output=True, text=True)
        if decoded.returncode != 0:
            raise RuntimeError(f"Could not decode source {sid}: {(decoded.stderr or '').strip()[-500:]}")
        wav = _load_wav(tmp)
        # Aligned a window at a time, not a source at a time.
        #
        # This used to push the entire source through Wav2Vec2 in one forward
        # pass, which worked only because the original corpus was 22-second
        # poetry clips. Self-attention is quadratic in sequence length, and the
        # model emits a frame per 20ms: a 22-second clip is ~1,100 frames and
        # costs nothing, but a 13-minute video is ~39,000, and the attention
        # matrix alone would be over a billion entries. Long-form sources did
        # not align slowly, they died allocating -- on the GPU and on the CPU
        # both. Windowing makes the cost linear in length instead of quadratic.
        #
        # Words are never split across a window, and each window is padded so a
        # word at the edge still has its surrounding audio to align against.
        fa: list[tuple[float, float] | None] = [None] * len(words)
        groups = _group_words(words, _WINDOW_SECONDS)
        total_samples = wav.size(1)

        for group in groups:
            first, last = words[group[0]], words[group[-1]]
            w_start = max(0.0, first["start_time"] - _WINDOW_PAD)
            w_end = last["end_time"] + _WINDOW_PAD
            s0 = max(0, int(w_start * _SR))
            s1 = min(total_samples, int(w_end * _SR))
            if s1 - s0 < _SR // 10:       # under 100ms of audio, nothing to do
                continue
            segment = wav[:, s0:s1].to(_device)

            # Window-relative back to source-relative.
            for idx, aligned in zip(group, _align_window(segment, [words[i] for i in group])):
                if aligned is None:
                    continue
                fa[idx] = (s0 / _SR + aligned[0], s0 / _SR + aligned[1])

        if not any(f is not None for f in fa):
            return (0, len(words), 0.0)

        updates, shifts = _boundary_updates(words, fa, total_samples / _SR)

        if apply and updates:
            with get_db() as conn:
                conn.executemany(
                    "UPDATE word_clips SET start_time=?, end_time=? WHERE id=?", updates)

                # This is the other thing that moves a start_time, and aligned
                # phoneme times are measured from it. Run over an already
                # aligned corpus without this and every sub-word cut is out by
                # however far the boundary moved -- silently, because the
                # times are still there and still look reasonable.
                from app.phone_align import outside, shift_stored
                for (new_start, _e, cid), moved in zip(updates, shifts):
                    row = conn.execute(
                        "SELECT phones FROM word_clips WHERE id=?", (cid,)).fetchone()
                    if row and row["phones"]:
                        shifted = shift_stored(row["phones"], -moved)
                        if outside(shifted, _e - new_start):
                            shifted = None
                        conn.execute(
                            "UPDATE word_clips SET phones=? WHERE id=?",
                            (shifted, cid))
                    # Character times use the same relative origin, but this
                    # table has no shifting helper. Recompute after a change.
                    conn.execute("DELETE FROM clip_alignments WHERE clip_id=?", (cid,))

        import statistics
        med = statistics.median(shifts) * 1000 if shifts else 0.0
        return (len(updates), len(words), med)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def main() -> None:
    global _WINDOW_SECONDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start-from-id", type=int)
    ap.add_argument("--end-at-id", type=int)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--window", type=float, default=_WINDOW_SECONDS, metavar="SEC",
                    help=f"Seconds of audio aligned at once (default "
                         f"{_WINDOW_SECONDS:g}). Lower it if you run out of memory.")
    from app.device import add_argument as _device_arg
    _device_arg(ap)
    args = ap.parse_args()
    _WINDOW_SECONDS = args.window

    init_db()
    _load_model(args.device)

    with get_db() as conn:
        if args.source_id is not None:
            ids = [args.source_id]
        else:
            q = "SELECT id FROM sources WHERE 1=1"
            p: list = []
            if args.start_from_id is not None:
                q += " AND id>=?"; p.append(args.start_from_id)
            if args.end_at_id is not None:
                q += " AND id<=?"; p.append(args.end_at_id)
            q += " ORDER BY id"
            ids = [r["id"] for r in conn.execute(q, p).fetchall()]

    print(f"Refining {len(ids)} source(s){'  (DRY RUN)' if not args.apply else ''}\n{'-'*54}")
    tot_u = 0
    for sid in ids:
        u, t, med = refine_source(sid, args.apply)
        tot_u += u
        print(f"  source {sid:3d}: {u}/{t} clips refined  (median shift {med:+.0f}ms)")

    if args.apply and tot_u:
        try:
            from app.generate import invalidate_cache
            invalidate_cache()
        except Exception:
            pass
    print(f"\n{tot_u} clips {'updated' if args.apply else '(dry run)'}. "
          f"{'Call POST /api/reload.' if args.apply else 'Add --apply to write.'}")


if __name__ == "__main__":
    main()
