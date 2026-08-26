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


def _load_model():
    global _model, _labels, _DICT
    if _model is not None:
        return
    import torchaudio
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    _model = bundle.get_model()
    _labels = bundle.get_labels()
    _DICT = {c: i for i, c in enumerate(_labels)}


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
        with torch.inference_mode():
            emission, _ = _model(wav)

        # Flatten all words' characters; remember each word's char range.
        chars: list[int] = []
        ranges: list[tuple[int, int] | None] = []
        for w in words:
            cw = [_DICT[c] for c in w["word"].upper() if c in _DICT]
            if not cw:
                ranges.append(None)
                continue
            s = len(chars)
            chars.extend(cw)
            ranges.append((s, len(chars)))

        if not chars or emission.size(1) <= len(chars):
            return (0, len(words), 0.0)  # too short to align safely

        targets = torch.tensor([chars], dtype=torch.int32)
        aligned, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
        spans = merge_tokens(aligned[0], scores[0])
        if len(spans) != len(chars):
            return (0, len(words), 0.0)  # index mismatch — skip for safety

        ratio = wav.size(1) / emission.size(1) / _SR

        # FA onset/core-offset per word (None where the word had no usable chars)
        fa: list[tuple[float, float] | None] = []
        for rng in ranges:
            if rng is None:
                fa.append(None)
            else:
                fa.append((spans[rng[0]].start * ratio, spans[rng[1] - 1].end * ratio))

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start-from-id", type=int)
    ap.add_argument("--end-at-id", type=int)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    init_db()
    _load_model()

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
