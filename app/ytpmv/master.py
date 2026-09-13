"""Balance the parts in a mix by measuring them.

The parts used to be balanced by a table: a lead is this loud, hats are that
loud. A table cannot know what it is looking at. The same "rhythm" slot holds
a whispered vowel and a screaming FM brass patch, and both were given the
same 0.75, so one vanished and the other buried the song.

So measure each part instead, and move it to where its job says it should
sit. The measurement is the broadcast one (ITU-R BS.1770): the signal is
filtered to roughly match what the ear cares about -- a shelf lifting the top
and a high-pass throwing away the sub-bass that a meter hears and a listener
does not -- and then averaged in blocks.

The part that matters most here is the *gating*, and it is why plain RMS will
not do. A cymbal part might be four hits in three minutes. Averaged over the
whole song it looks almost silent, and anything balancing on that number
would crank it until those four hits took the roof off. So blocks below an
absolute floor are dropped, and then blocks far below the average of what
remains are dropped as well, leaving "how loud is it while it is actually
playing" -- which is the question worth asking.

What this does *not* do is force every part onto its target. Parts inside one
role are not meant to be equally loud -- a song might have five things this
code calls "rhythm", and the several decibels between them are somebody's
arrangement. Flattening those made mixes measurably tidier and audibly worse.
So a part near where its job would put it is left alone, and one well outside
is brought part of the way back. See DEADBAND and STRENGTH.

Setting the level of the finished mix is a separate job and is still done by
render's own _finish_mix.

numpy only, and no filter is run as a recurrence: the weighting is applied in
the frequency domain, which for measuring a finished buffer is the same thing
and is fast enough to do per part.
"""

from __future__ import annotations

import math

import numpy as np

from app.ytpmv.pitch import SR

# Where each kind of part sits, in loudness units relative to the lead.
#
# Not invented, and not the old volume table converted either -- that table
# held amplitude multipliers, which are not loudness offsets, and reading it
# as though it were put the hats twelve decibels adrift. These are measured:
# render a few songs the way the old table mixed them, measure what each part
# actually came to, and take the median. So the balance it arrives at is the
# one that already sounded right, and what has changed is that a part which
# lands nowhere near its neighbours now gets pulled in rather than left.
#
# A kit reads far quieter than a melody at the same apparent level, because a
# hit is mostly the silence after it; that is why these go down to -26 rather
# than the -6 that "sounds about half as loud" would suggest.
#
# Re-derive them with scripts/ytpmv.py if the drum shaping ever changes.
TARGETS = {
    "lead": 0.0, "bass": -2.9, "chords": -3.4, "rhythm": -5.1,
    "drums:kick": -4.0, "drums:snare": -10.7, "drums:hats": -23.1,
    "drums:toms": -4.0, "drums:cymbals": -25.8, "drums:perc": -26.3,
}
DEFAULT_TARGET = -5.1          # anything unrecognised is treated as backing

# What the lead is aimed at before the master stage sets the final level.
# Only the distances between parts matter to the result -- the master stage
# re-levels the sum regardless -- but keeping this near what a lead actually
# measures means most parts barely move, and nothing runs into the limit
# below for no reason.
REFERENCE = -20.0

# A part is only ever moved this far. Past here something is wrong with the
# part rather than with its level, and shouting at it will not help.
MAX_MOVE = 12.0

# How far out a part has to be before it is touched at all, and how much of
# the difference is then taken off.
#
# This is the whole difference between balancing a mix and flattening one.
# Parts inside one role are not meant to be equally loud: a song might have
# five things this code calls "rhythm", and the several decibels between them
# are an arrangement -- somebody decided which of them carries and which sits
# under it. Moving each one onto a target destroys exactly that, and the
# result is worse than leaving it alone, which is what it sounded like.
#
# So: anything within DEADBAND of where its job would put it is left exactly
# as it is, and anything past that is brought back by STRENGTH of the excess
# rather than all of it. What gets fixed is the part that is plainly in the
# wrong place; what survives is everything somebody meant.
DEADBAND = 3.0
STRENGTH = 0.6

_SILENCE = -70.0            # blocks quieter than this are not the performance
_RELATIVE = -10.0           # nor are blocks this far below the rest of it


def _biquad_response(b: tuple, a: tuple, freqs: np.ndarray, sr: int) -> np.ndarray:
    z = np.exp(-2j * np.pi * freqs / sr)
    return ((b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z))


def _k_response(freqs: np.ndarray, sr: int) -> np.ndarray:
    """BS.1770's two filters: a shelf up top, and the sub-bass thrown away."""
    # Stage 1: a high shelf, about +4 dB from 1.7 kHz up, standing in for the
    # way a head in the way of a loudspeaker lifts the treble.
    f0, gain, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    a_ = 10.0 ** (gain / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw, rt = math.cos(w0), 2.0 * math.sqrt(a_) * alpha
    b1 = (a_ * ((a_ + 1) + (a_ - 1) * cw + rt), -2 * a_ * ((a_ - 1) + (a_ + 1) * cw),
          a_ * ((a_ + 1) + (a_ - 1) * cw - rt))
    a1 = ((a_ + 1) - (a_ - 1) * cw + rt, 2 * ((a_ - 1) - (a_ + 1) * cw),
          (a_ + 1) - (a_ - 1) * cw - rt)
    # Stage 2: a high-pass near 38 Hz. A meter counts rumble the ear barely
    # registers, and this is what stops the bass deciding every balance.
    f0, q = 38.13547087602444, 0.5003270373238773
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b2 = ((1 + cw) / 2.0, -(1 + cw), (1 + cw) / 2.0)
    a2 = (1 + alpha, -2 * cw, 1 - alpha)
    return _biquad_response(b1, a1, freqs, sr) * _biquad_response(b2, a2, freqs, sr)


def k_weight(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """*x* filtered the way a loudness meter filters it."""
    if len(x) < 64:
        return x
    # Long enough that the filters' own ringing cannot wrap round onto the
    # start, and no longer: doubling the length instead costs a transform
    # twice the size for nothing.
    n = 1 << int(np.ceil(np.log2(len(x) + 8192)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    h = _k_response(freqs, sr)
    return np.fft.irfft(np.fft.rfft(x, n) * h, n)[: len(x)].real


def loudness(x: np.ndarray, sr: int = SR) -> float:
    """How loud *x* is while it is playing, in LUFS. -inf for silence.

    Gated, as the standard says: quiet blocks are not part of the
    performance, and for a part that plays four times in a song they are
    nearly all of it.
    """
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if len(x) < sr // 10 or not np.any(x):
        return -math.inf
    y = k_weight(x, sr)

    block, hop = int(0.4 * sr), int(0.1 * sr)
    if len(y) < block:
        y = np.pad(y, (0, block - len(y)))
    starts = np.arange(0, len(y) - block + 1, hop)
    if len(starts) == 0:
        return -math.inf
    # Every block's mean square in one pass, off a running total, rather than
    # a thousand overlapping slices that each re-add what the last one added.
    cs = np.concatenate(([0.0], np.cumsum(y * y)))
    power = (cs[starts + block] - cs[starts]) / block
    lk = -0.691 + 10.0 * np.log10(np.maximum(power, 1e-20))

    loud = lk > _SILENCE                       # not silence
    if not loud.any():
        return -math.inf
    gate = -0.691 + 10.0 * math.log10(float(power[loud].mean())) + _RELATIVE
    keep = loud & (lk > gate)                  # nor the quiet bits around it
    if not keep.any():
        keep = loud
    return -0.691 + 10.0 * math.log10(float(power[keep].mean()))


def gain_for(measured: float, role: str, trim_db: float = 0.0) -> float:
    """How far to move a part that measured *measured*, as a factor.

    Nudged rather than moved: see DEADBAND and STRENGTH. A part near enough
    to where its job would put it is left alone, because the distances
    between parts of the same kind are usually somebody's arrangement rather
    than a mistake.

    *trim_db* is whatever the person asked for on top, which is respected in
    full: the measurement only ever suggests, and anyone who disagrees wins.
    """
    if not math.isfinite(measured):
        return 10.0 ** (trim_db / 20.0)
    err = REFERENCE + TARGETS.get(role, DEFAULT_TARGET) - measured
    over = max(0.0, abs(err) - DEADBAND)            # only the part that is out
    move = math.copysign(over * STRENGTH, err)
    move = float(np.clip(move, -MAX_MOVE, MAX_MOVE)) + trim_db
    return float(10.0 ** (move / 20.0))
