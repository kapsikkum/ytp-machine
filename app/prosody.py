"""How a take sounds, as numbers, and choosing takes that sound alike.

A sentence put together from single words jumps about: one word is from a
calm aside, the next from a shouted punchline, the one after that from
another recording entirely. Each word is fine on its own; the line is not.

So every word clip gets a short description of its delivery, measured once
when the corpus is built (scripts/measure_tone.py) and stored in clip_tone:

  pitch         median pitch, in semitones
  pitch_start   at the start and the end -- where it joins its neighbours
  pitch_end
  movement      how much the pitch moves: animated or flat
  loudness      against the rest of its own recording, dB
  pace          seconds per phoneme
  brightness    share of energy above 2kHz, dB: a shout is bright, a murmur dark

Corpus-wide means and spreads are stored beside them, so the generator works
in standard units without measuring anything, and the clips are grouped into a
handful of tone categories ("high, lively, loud") by k-means.

Choosing then leans three ways, each a multiplier on the weights pick_clip
already uses, so nothing is ever ruled out:

  - towards the category that the most words of the sentence can be said in,
    so a line that can be said calmly throughout is;
  - towards a take that starts near the pitch and level the last one ended on;
  - towards staying in the same recording, which brings its room and mic.

A corpus measured before any of this has no rows, and picks as it always did.
"""
from __future__ import annotations

import json
import math
from contextvars import ContextVar
from typing import Any

# numpy is imported where something is measured, not here: the generator
# imports this module to choose takes, which is arithmetic on a few floats,
# and the generator must import without numpy (see the tokeniser check).

SR = 16000

FEATURES = ("pitch", "pitch_start", "pitch_end", "movement", "loudness", "pace", "brightness")
# What a category is made of. The join pitches belong to the edges, not the
# delivery, and are left to the continuity term.
_SHAPE = ("pitch", "movement", "loudness", "pace", "brightness")
_CATEGORIES = 5
_PER_CATEGORY = 100       # takes a category needs before there is room for another
_EDGE_FRAMES = 4          # voiced frames that count as a word's start or end pitch
_BRIGHT_HZ = 2000.0

# How hard each preference pulls at full strength; a strength between scales
# them geometrically. They have to be strong: pick_clip already weighs a take
# with a pause either side at 600 times one without, and tone that only
# doubled the odds changed a fifth of the joins and none of the sound.
_SAME_CATEGORY = 100.0    # a take in the sentence's category is this much likelier
_SAME_SOURCE = 20.0       # and one from the recording just used
_JOIN_WIDTH = 0.35        # a join this many standard units apart costs e^-0.5


# ── Measuring ─────────────────────────────────────────────────────────────────

def measure_source(x, clips: list[dict], phones=None) -> dict[int, dict]:
    """Tone features of every clip in one source, decoded at SR mono.

    *phones* maps a word to its phoneme count; without one, pace is per letter
    over three. Clips with too little voicing for a pitch get None there.
    """
    import numpy as np
    from app.ytpmv.pitch import hz_to_midi, track_f0

    x = np.asarray(x, dtype=np.float32)
    track = track_f0(x, SR)
    # The recording's own level, from its voiced frames: loudness is relative
    # to the room, the mic and the gain of the upload it came from.
    ref = track.rms[track.voiced] if track.voiced.any() else track.rms
    ref_db = 20.0 * math.log10(float(np.median(ref)) + 1e-9)
    out: dict[int, dict] = {}
    for clip in clips:
        s, e = clip["start_time"], clip["end_time"]
        a, b = int(s * SR), int(e * SR)
        seg = x[a:b]
        if len(seg) < SR // 50:
            continue
        lo, hi = track.frame_at(s), track.frame_at(e) + 1
        voiced = track.voiced[lo:hi]
        semis = np.array([hz_to_midi(f) for f in track.f0[lo:hi][voiced]], dtype=np.float64)
        row: dict[str, Any] = dict.fromkeys(FEATURES)
        if len(semis) >= 3:
            edge = min(_EDGE_FRAMES, len(semis))
            row["pitch"] = float(np.median(semis))
            row["pitch_start"] = float(np.median(semis[:edge]))
            row["pitch_end"] = float(np.median(semis[-edge:]))
            row["movement"] = float(np.std(semis))
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        row["loudness"] = 20.0 * math.log10(rms + 1e-9) - ref_db
        n = (phones(clip["word"]) if phones else 0) or max(1, len(clip["word"]) // 3)
        row["pace"] = (e - s) / n
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        freqs = np.fft.rfftfreq(len(seg), 1.0 / SR)
        total = float(spec.sum()) + 1e-12
        row["brightness"] = 10.0 * math.log10(float(spec[freqs >= _BRIGHT_HZ].sum()) / total + 1e-9)
        out[int(clip["id"])] = row
    return out


def _scaled(f: str, v: float) -> float:
    # Pace is a ratio and badly skewed: a held "ooooh" is twenty times the
    # usual seconds per phoneme, and on a linear scale those few takes were a
    # category of their own.
    return math.log(max(v, 1e-3)) if f == "pace" else v


_CAP = 3.0                # standard units; past this a take is unusual, not more so


def statistics(rows: list[dict]) -> dict[str, list[float]]:
    """Mean and spread of each feature over the corpus."""
    import numpy as np
    stats = {}
    for f in FEATURES:
        v = np.array([_scaled(f, r[f]) for r in rows if r.get(f) is not None], dtype=np.float64)
        if len(v) < 2:
            stats[f] = [0.0, 1.0]
        else:
            stats[f] = [float(v.mean()), float(max(v.std(), 1e-6))]
    # One word's end pitch is compared with the next one's start: they have
    # to be on one scale, the voice's own.
    stats["pitch_start"] = stats["pitch_end"] = stats["pitch"]
    return stats


def standard(row: dict, stats: dict) -> tuple:
    """*row* in standard units, None where it has no value."""
    return tuple(None if row.get(f) is None else
                 max(-_CAP, min(_CAP, (_scaled(f, row[f]) - stats[f][0]) / stats[f][1]))
                 for f in FEATURES)


def _shape(z: tuple):
    """The category features of a standardised row; missing ones sit at the mean."""
    import numpy as np
    return np.array([0.0 if z[FEATURES.index(f)] is None else z[FEATURES.index(f)]
                     for f in _SHAPE], dtype=np.float64)


def categorise(zs: list[tuple], k: int | None = None, seed: int = 0) -> tuple[list[int], Any]:
    """k-means over the delivery features: (label per row, centroids).

    Seeded and with k-means++ starts, so the same corpus always gets the same
    categories, numbered by size, largest first. A small corpus gets fewer:
    a category is only worth leaning a sentence towards if most words have a
    take in it, and five categories of a few dozen takes each would not.
    """
    import numpy as np
    X = np.array([_shape(z) for z in zs])
    n = len(X)
    if n == 0:
        return [], np.zeros((0, len(_SHAPE)))
    if k is None:
        k = max(2, min(_CATEGORIES, n // _PER_CATEGORY))
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centres = [X[rng.integers(n)]]
    for _ in range(1, k):
        d = np.min([((X - c) ** 2).sum(1) for c in centres], axis=0)
        centres.append(X[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else X[rng.integers(n)])
    C = np.array(centres)
    for _ in range(100):
        labels = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)
        moved = np.array([X[labels == j].mean(0) if (labels == j).any() else C[j] for j in range(k)])
        if np.allclose(moved, C):
            break
        C = moved
    labels = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)
    order = np.argsort(-np.bincount(labels, minlength=k), kind="stable")
    rank = {int(old): new for new, old in enumerate(order)}
    return [rank[int(l)] for l in labels], C[order]


_WORDS = {"pitch": ("high", "low"), "movement": ("lively", "flat"), "loudness": ("loud", "soft"),
          "pace": ("slow", "quick"), "brightness": ("bright", "dark")}


def name(centroid) -> str:
    """A category described by whatever sets it apart from the average take."""
    parts = [(abs(v), _WORDS[f][0] if v > 0 else _WORDS[f][1])
             for f, v in zip(_SHAPE, centroid) if abs(v) >= 0.5]
    return ", ".join(w for _a, w in sorted(parts, reverse=True)[:3]) or "even"


# ── Loading ───────────────────────────────────────────────────────────────────

def load(conn) -> dict[int, tuple[tuple, int | None]]:
    """clip id -> (standardised features, category), or {} for an unmeasured corpus.

    A corpus older than the tables, or one never measured, is simply empty:
    selection then behaves exactly as it did before tone existed.
    """
    try:
        raw = conn.execute("SELECT value FROM settings WHERE key='tone_stats'").fetchone()
        rows = conn.execute("SELECT clip_id, " + ", ".join(FEATURES) + ", category FROM clip_tone").fetchall()
    except Exception:
        return {}
    if not raw or not rows:
        return {}
    try:
        stats = json.loads(raw[0])
    except (TypeError, ValueError):
        return {}
    out = {}
    for r in rows:
        row = {f: r[i + 1] for i, f in enumerate(FEATURES)}
        out[int(r[0])] = (standard(row, stats), r[len(FEATURES) + 1])
    return out


# ── Choosing ──────────────────────────────────────────────────────────────────

# The sentence being put together, when there is one: set by the generator
# for the length of a resolve, read by every pick inside it -- including the
# splicer's, which picks through the same function with no idea of sentences.
current: ContextVar[dict | None] = ContextVar("prosody_context", default=None)


def context(strength: float) -> dict | None:
    return {"strength": strength, "target": None, "prev": None} if strength > 0 else None


def target(candidates: list[list[dict]]) -> int | None:
    """The category the most of these words can be said in.

    *candidates* is each word's usable takes. Ties go to the category with
    more takes overall, which is the safer one to commit a sentence to.
    """
    cover: dict[int, int] = {}
    count: dict[int, int] = {}
    for takes in candidates:
        cats = [c["tone_cat"] for c in takes if c.get("tone_cat") is not None]
        for cat in set(cats):
            cover[cat] = cover.get(cat, 0) + 1
        for cat in cats:
            count[cat] = count.get(cat, 0) + 1
    if not cover:
        return None
    return max(cover, key=lambda c: (cover[c], count[c], -c))


def follow(ctx: dict | None, clip: dict | None) -> None:
    """Record *clip* as the one the next pick should join on to."""
    if ctx is not None and clip is not None and clip.get("tone") is not None:
        ctx["prev"] = clip


def weight(ctx: dict | None, clip: dict) -> float:
    """How much likelier *clip* is to be picked here, for its tone alone."""
    if ctx is None or clip.get("tone") is None:
        return 1.0
    s = ctx["strength"]
    w = 1.0
    if ctx["target"] is not None and clip.get("tone_cat") == ctx["target"]:
        w *= _SAME_CATEGORY ** s
    prev = ctx.get("prev")
    if prev is not None:
        if prev.get("source_id") == clip.get("source_id"):
            w *= _SAME_SOURCE ** s
        a, b = prev["tone"], clip["tone"]
        d2, used = 0.0, 0
        for i, j in ((FEATURES.index("pitch_end"), FEATURES.index("pitch_start")),
                     (FEATURES.index("loudness"), FEATURES.index("loudness")),
                     (FEATURES.index("brightness"), FEATURES.index("brightness"))):
            if a[i] is not None and b[j] is not None:
                d2 += (a[i] - b[j]) ** 2
                used += 1
        if used:
            w *= math.exp(-s * d2 / used / (2 * _JOIN_WIDTH ** 2))
    return w
