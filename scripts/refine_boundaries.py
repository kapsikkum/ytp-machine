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
import os
import subprocess
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
    _model = bundle.get_model().to(_device)
    _labels = bundle.get_labels()
    _DICT = {c: i for i, c in enumerate(_labels)}
    print(f"  aligner on {describe(_device)}")


def _load_wav(path):
    import numpy as np
    import torch
    w = wave.open(path, "rb")
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype("float32") / 32768.0
    w.close()
    return torch.from_numpy(d).unsqueeze(0)


def refine_source(sid: int, apply: bool) -> tuple[int, int, float]:
    """Re-align one source. Returns (updated, total, median_shift_ms)."""
    import torch
    import torchaudio
    from torchaudio.functional import merge_tokens

    with get_db() as conn:
        src = conn.execute("SELECT source_file FROM sources WHERE id=?", (sid,)).fetchone()
        words = [dict(r) for r in conn.execute(
            "SELECT id, word, start_time, end_time FROM word_clips "
            "WHERE source_id=? ORDER BY start_time", (sid,)).fetchall()]
    if not src or not words:
        return (0, 0, 0.0)

    from app.database import resolve_path
    tmp = f"_rb_{os.getpid()}.wav"
    subprocess.run(["ffmpeg", "-y", "-i", resolve_path(src["source_file"]), "-ac", "1", "-ar", str(_SR), tmp],
                   capture_output=True)
    wav = _load_wav(tmp)
    try:
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

            # Flatten this window's characters; remember each word's range.
            chars: list[int] = []
            ranges: list[tuple[int, int] | None] = []
            for idx in group:
                cw = [_DICT[c] for c in words[idx]["word"].upper() if c in _DICT]
                if not cw:
                    ranges.append(None)
                    continue
                s = len(chars)
                chars.extend(cw)
                ranges.append((s, len(chars)))
            if not chars:
                continue

            with torch.inference_mode():
                emission, _ = _model(segment)
            if emission.size(1) <= len(chars):
                continue                   # too short to align safely

            targets = torch.tensor([chars], dtype=torch.int32, device=emission.device)
            aligned, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
            spans = merge_tokens(aligned[0].cpu(), scores[0].cpu())
            if len(spans) != len(chars):
                continue                   # index mismatch — skip this window

            ratio = segment.size(1) / emission.size(1) / _SR
            # Window-relative back to source-relative.
            for idx, rng in zip(group, ranges):
                if rng is None:
                    continue
                fa[idx] = (w_start + spans[rng[0]].start * ratio,
                           w_start + spans[rng[1] - 1].end * ratio)

        if not any(f is not None for f in fa):
            return (0, len(words), 0.0)

        updates = []
        shifts = []
        for i, (w, f) in enumerate(zip(words, fa)):
            if f is None:
                continue
            onset, core_end = f
            # next aligned word's onset (same source order)
            nxt = next((fa[j][0] for j in range(i + 1, len(fa)) if fa[j] is not None), None)
            # Span from the true onset to just before the next word, but cap the
            # tail so a long pause after the word isn't swallowed.
            cap = core_end + 0.30
            new_end = min(cap, nxt - 0.02) if nxt is not None else cap
            new_end = max(new_end, core_end + 0.05)
            new_start = onset
            if new_end - new_start < 0.04:
                continue
            shifts.append(new_start - w["start_time"])
            updates.append((round(new_start, 4), round(new_end, 4), w["id"]))

        if apply and updates:
            with get_db() as conn:
                conn.executemany(
                    "UPDATE word_clips SET start_time=?, end_time=? WHERE id=?", updates)

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
