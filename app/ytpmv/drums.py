"""Turn a spoken word into a drum hit, beatbox-style.

A kick drum played by a whole word is somebody saying "bum" on every beat --
which is what the first version did, and it sounds like it. A beatboxer does
not say words: a kick is the pop of a "b", a snare the burst of a "k" run into
a hiss, a hi-hat a clipped "ts". So each kit piece takes just that part of the
word and shapes it like the drum:

  kick     the attack, tuned down, low-passed, gone in ~0.1 s
  snare    the attack burst joined to the word's hiss ("kiss" -> "k-ss")
  hats     a sliver of the hiss, high-passed, very short
  toms     the attack and a little vowel, tuned by the drum's note
  cymbals  the hiss held long and let decay
  perc     the attack alone

The parts are found from the sound, not from a transcript: the attack is where
the level first jumps, the hiss is where the energy sits highest in the
spectrum. So it works on any corpus, aligned to phonemes or not.

numpy only.
"""

from __future__ import annotations

import numpy as np

from app.ytpmv.pitch import SR

GROUPS = ("kick", "snare", "hats", "toms", "cymbals", "perc")

_HOP = int(0.005 * SR)


def _frames_env(x: np.ndarray) -> np.ndarray:
    n = max(1, len(x) // _HOP)
    return np.sqrt(np.mean(x[: n * _HOP].reshape(n, _HOP) ** 2, axis=1)) if len(x) >= _HOP \
        else np.zeros(1)


def onset(x: np.ndarray) -> int:
    """Where the word starts: the first moment it reaches a tenth of its peak."""
    env = _frames_env(x)
    peak = float(env.max(initial=0.0))
    if peak <= 0:
        return 0
    return max(0, int(np.argmax(env >= 0.1 * peak)) * _HOP - int(0.003 * SR))


def hiss_profile(x: np.ndarray, win: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Per 5 ms: the share of energy above 4 kHz, and the frame's level."""
    w = int(win * SR)
    if len(x) < w:
        return np.zeros(1), np.zeros(1)
    starts = np.arange(0, len(x) - w, _HOP)
    idx = starts[:, None] + np.arange(w)[None, :]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(w), axis=1)) ** 2
    freqs = np.fft.rfftfreq(w, 1.0 / SR)
    total = spec.sum(axis=1) + 1e-12
    return spec[:, freqs >= 4000].sum(axis=1) / total, np.sqrt(total)


def find_hiss(x: np.ndarray, min_share: float = 0.35) -> tuple[int, int] | None:
    """The stretch of *x* that is most "s" -- start and end sample -- if any."""
    share, level = hiss_profile(x)
    if len(share) < 2:
        return None
    loud = level > 0.05 * float(level.max(initial=0.0))
    score = np.where(loud, share, 0.0)
    best = int(np.argmax(score))
    if score[best] < min_share:
        return None
    a = b = best
    while a > 0 and score[a - 1] >= 0.6 * score[best]:
        a -= 1
    while b + 1 < len(score) and score[b + 1] >= 0.6 * score[best]:
        b += 1
    return a * _HOP, min(len(x), (b + 4) * _HOP)


def _filter(y: np.ndarray, lo: float | None = None, hi: float | None = None,
            shelf: tuple[float, float] | None = None) -> np.ndarray:
    """Band-limit *y* with gentle (half-octave) edges. Short arrays only.

    *shelf* is (corner, gain): everything below the corner is lifted, which is
    how a kick gets its weight without a synthesiser under it.
    """
    if len(y) < 16:
        return y
    n = 1 << int(np.ceil(np.log2(len(y) * 2)))
    Y = np.fft.rfft(y, n)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    H = np.ones_like(f)
    if hi:
        H *= 1.0 / (1.0 + (f / hi) ** 4)
    if lo:
        H *= 1.0 - 1.0 / (1.0 + (f / lo) ** 4)
    if shelf:
        corner, gain = shelf
        H *= 1.0 + (gain - 1.0) / (1.0 + (f / corner) ** 2)
    return np.fft.irfft(Y * H, n)[: len(y)].astype(np.float32)


def _saturate(y: np.ndarray, drive: float) -> np.ndarray:
    """Round the peaks off. Adds harmonics of the low end, which is what makes
    a kick audible on a phone speaker that cannot reproduce the low end."""
    return (np.tanh(y * drive) / np.tanh(drive)).astype(np.float32)


def _decay(y: np.ndarray, tau: float, attack: float = 0.002) -> np.ndarray:
    t = np.arange(len(y)) / SR
    env = np.exp(-t / tau)
    a = min(len(y), int(attack * SR))
    if a:
        env[:a] *= np.linspace(0.0, 1.0, a)
    fo = min(len(y), int(0.004 * SR))          # never end on a click
    if fo:
        env[-fo:] *= np.linspace(1.0, 0.0, fo)
    return (y * env).astype(np.float32)


def _slice(x: np.ndarray, a: int, dur: float) -> np.ndarray:
    return x[a:a + int(dur * SR)].astype(np.float32)


def _tape(y: np.ndarray, semitones: float) -> np.ndarray:
    r = 2.0 ** (semitones / 12.0)
    pos = np.arange(0.0, len(y) - 1, r)
    return np.interp(pos, np.arange(len(y)), y).astype(np.float32) if len(pos) > 1 else y


def _match_peak(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    py, pr = float(np.abs(y).max(initial=0.0)), float(np.abs(ref).max(initial=0.0))
    return (y * (pr / py)).astype(np.float32) if py > 0 and pr > 0 else y


def shape(x: np.ndarray, group: str) -> tuple[np.ndarray, int]:
    """(the hit, where in *x* it starts -- for lining the picture up)."""
    x = np.asarray(x, dtype=np.float32)
    s = onset(x)
    hiss = find_hiss(x)

    if group == "kick":
        # Beef, in the order it is built: take longer than the hit will be so
        # there is decay to work with, drop it further than a voice can go,
        # lay an octave-down copy underneath for the weight a mouth has not
        # got, lift everything under 90 Hz, then saturate so the low end
        # survives a speaker that cannot reproduce it. A longer decay lets it
        # land rather than click.
        y = _tape(_slice(x, s, 0.24), -7.0)
        sub = _tape(y, -12.0) * 0.55                        # the octave below, underneath
        body = np.zeros(max(len(y), len(sub)), dtype=np.float32)
        body[:len(y)] += y
        body[:len(sub)] += sub
        body = _filter(body, hi=750.0, shelf=(90.0, 2.2))
        body = _saturate(body, 2.0)
        y = _decay(body, 0.075)[: int(0.2 * SR)]
        start = s
    elif group == "snare":
        head = _slice(x, s, 0.045)
        if hiss is not None and hiss[0] > s + int(0.02 * SR):
            tail = _slice(x, hiss[0], 0.10)
        else:
            tail = _slice(x, s + len(head), 0.09)
        xf = min(len(head), len(tail), int(0.006 * SR))     # a short crossfade, no click
        if xf:
            ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
            y = np.concatenate([head[:-xf], head[-xf:] * (1 - ramp) + tail[:xf] * ramp, tail[xf:]])
        else:
            y = np.concatenate([head, tail])
        y = _filter(y, lo=250.0)
        y = _decay(y, 0.05)
        start = s
    elif group == "hats":
        a = hiss[0] if hiss else s
        y = _filter(_slice(x, a, 0.07), lo=3500.0)
        y = _decay(y, 0.022)
        start = a
    elif group == "cymbals":
        if hiss:
            a, b = hiss
            y = _slice(x, a, min(0.6, max(0.25, (b - a) / SR + 0.1)))
        else:
            a = s
            y = _slice(x, a, 0.35)
        y = _filter(y, lo=2500.0)
        y = _decay(y, 0.18, attack=0.004)
        start = a
    elif group == "toms":
        y = _filter(_slice(x, s, 0.22), hi=2500.0)
        y = _decay(y, 0.08)
        start = s
    else:                                                   # perc, and anything unknown
        y = _decay(_slice(x, s, 0.08), 0.025)
        start = s
    if len(y) == 0:
        return x[: int(0.05 * SR)].copy(), 0
    return _match_peak(y, x), start
