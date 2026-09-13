"""The SNES's sound chip, the S-DSP. Nothing here synthesises anything.

The other machines make sound out of nothing -- squares, shift registers,
operators modulating each other. This one cannot. It is eight channels of
sample playback and that is all it is: it plays his voice, or it plays the
instrument samples a game would have carried, which here are a General MIDI
SoundFont cut down to size (see the bottom of this file).

What makes a SNES sound like a SNES is the damage done on the way through:

  BRR       samples are stored nine bytes to sixteen of them: four bits
            each, a shared scale, and one of four predictors that guess the
            next sample from the last two. Four bits is not many, and the
            grain that leaves is the sound of the machine
  Gaussian  played back through a four-point interpolation whose taps are a
            table burnt into the chip. It is not a neutral resampler -- it
            is a low-pass, and a fierce one, and it is why everything on
            this console sounds like it is playing through a blanket
  echo      an eight-tap filter feeding a delay line back into itself.
            Nearly every SNES soundtrack is drenched in this

The tables and coefficients are from fullsnes rather than remembered: the
four predictors, and the 512-entry interpolation table. That table is worth
a sanity check -- its four taps sum to 2048 at every phase, which is unity
gain once the chip's two shifts are done, and a wrong table would not.

Two liberties, both declared. Real BRR carries its predictor state from one
block into the next; this encodes each block from the true signal instead,
so all of them can be done at once rather than sixteen thousand times in a
Python loop. And the envelope and per-voice gain are left to the caller,
since the part doing the playing has already decided how loud it is.

numpy only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from app.ytpmv.pitch import SR

# What the DSP runs at. Everything here happens at this rate and is put back
# afterwards.
DSP_RATE = 32000

# What a decoded sample may reach. The chip works in fifteen-bit integers
# and its own documentation warns that decoding must stay inside this, so
# the scales below are powers of two in *this* domain -- feeding it floats
# between -1 and 1 leaves every step far too coarse and quantises the lot
# to silence.
BRR_FULL = 16376.0

# The four predictors, as coefficients on the last two decoded samples.
# Filter 0 does not predict at all; 3 leans hardest on what came before.
BRR_FILTERS = (
    (0.0, 0.0),
    (0.9375, 0.0),
    (1.90625, -0.9375),
    (1.796875, -0.8125),
)

# The chip's interpolation table, 512 entries.
_GAUSS = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
    2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
    11, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 15, 16, 16, 17, 17,
    18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25, 26, 27, 27,
    28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 36, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    58, 59, 60, 61, 62, 64, 65, 66, 67, 69, 70, 71, 73, 74, 76, 77,
    78, 80, 81, 83, 84, 86, 87, 89, 90, 92, 94, 95, 97, 99, 100, 102,
    104, 106, 107, 109, 111, 113, 115, 117, 118, 120, 122, 124, 126, 128, 130, 132,
    134, 137, 139, 141, 143, 145, 147, 150, 152, 154, 156, 159, 161, 163, 166, 168,
    171, 173, 175, 178, 180, 183, 186, 188, 191, 193, 196, 199, 201, 204, 207, 210,
    212, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242, 245, 248, 251, 254, 257,
    260, 263, 267, 270, 273, 276, 280, 283, 286, 290, 293, 297, 300, 304, 307, 311,
    314, 318, 321, 325, 328, 332, 336, 339, 343, 347, 351, 354, 358, 362, 366, 370,
    374, 378, 381, 385, 389, 393, 397, 401, 405, 410, 414, 418, 422, 426, 430, 434,
    439, 443, 447, 451, 456, 460, 464, 469, 473, 477, 482, 486, 491, 495, 499, 504,
    508, 513, 517, 522, 527, 531, 536, 540, 545, 550, 554, 559, 563, 568, 573, 577,
    582, 587, 592, 596, 601, 606, 611, 615, 620, 625, 630, 635, 640, 644, 649, 654,
    659, 664, 669, 674, 678, 683, 688, 693, 698, 703, 708, 713, 718, 723, 728, 732,
    737, 742, 747, 752, 757, 762, 767, 772, 777, 782, 787, 792, 797, 802, 806, 811,
    816, 821, 826, 831, 836, 841, 846, 851, 855, 860, 865, 870, 875, 880, 884, 889,
    894, 899, 904, 908, 913, 918, 923, 927, 932, 937, 941, 946, 951, 955, 960, 965,
    969, 974, 978, 983, 988, 992, 997, 1001, 1005, 1010, 1014, 1019, 1023, 1027, 1032, 1036,
    1040, 1045, 1049, 1053, 1057, 1061, 1066, 1070, 1074, 1078, 1082, 1086, 1090, 1094, 1098, 1102,
    1106, 1109, 1113, 1117, 1121, 1125, 1128, 1132, 1136, 1139, 1143, 1146, 1150, 1153, 1157, 1160,
    1164, 1167, 1170, 1174, 1177, 1180, 1183, 1186, 1190, 1193, 1196, 1199, 1202, 1205, 1207, 1210,
    1213, 1216, 1219, 1221, 1224, 1227, 1229, 1232, 1234, 1237, 1239, 1241, 1244, 1246, 1248, 1251,
    1253, 1255, 1257, 1259, 1261, 1263, 1265, 1267, 1269, 1270, 1272, 1274, 1275, 1277, 1279, 1280,
    1282, 1283, 1284, 1286, 1287, 1288, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1297, 1298,
    1299, 1300, 1300, 1301, 1302, 1302, 1303, 1303, 1303, 1304, 1304, 1304, 1304, 1304, 1305, 1305
], dtype=np.float64)

# The default echo: eight taps and a delay feeding back on itself. These are
# a plain single-tap setting rather than any particular game's.
ECHO_FIR = np.array([127, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64) / 128.0


def _resample(y: np.ndarray, src: float, dst: float) -> np.ndarray:
    if abs(src - dst) < 1e-6 or len(y) < 2:
        return y.astype(np.float32)
    n = max(2, int(round(len(y) * dst / src)))
    return np.interp(np.arange(n) * src / dst, np.arange(len(y)), y).astype(np.float32)


def brr(y: np.ndarray) -> np.ndarray:
    """*y* squeezed into four bits a sample and unpacked again.

    Sixteen samples share one scale and one predictor, chosen per block: the
    predictor that leaves the least behind wins, and the scale is whatever
    just fits what is left into four bits. Then it is decoded exactly as the
    chip decodes it, error and all, which is the point of doing it.
    """
    n = len(y)
    if n < 32:
        return y.astype(np.float32)
    peak = float(np.abs(y).max(initial=0.0))
    if peak <= 0:
        return y.astype(np.float32)
    y = np.asarray(y, dtype=np.float64) * (BRR_FULL / peak)
    blocks = n // 16
    t = y[: blocks * 16].reshape(blocks, 16).astype(np.float64)
    # The two samples before each block, which the predictor leans on.
    prev1 = np.concatenate(([0.0], y[15: blocks * 16 - 1: 16]))
    prev2 = np.concatenate(([0.0, 0.0], y[14: blocks * 16 - 2: 16]))[:blocks]

    # Which predictor to use, judged by what it leaves behind.
    best, best_err, best_res = 0, None, None
    for f, (a, b) in enumerate(BRR_FILTERS):
        d1 = np.concatenate([prev1[:, None], t[:, :-1]], axis=1)
        d2 = np.concatenate([prev2[:, None], prev1[:, None], t[:, :-2]], axis=1)
        res = t - a * d1 - b * d2
        err = np.abs(res).max(axis=1)
        if best_err is None:
            best, best_err, best_res = np.zeros(blocks, dtype=int), err, res
        else:
            take = err < best_err
            best = np.where(take, f, best)
            best_err = np.where(take, err, best_err)
            best_res = np.where(take[:, None], res, best_res)

    # The scale: four bits run -8..7, so the step has to cover the worst of
    # the block. The chip's steps are powers of two and nothing between.
    shift = np.clip(np.ceil(np.log2(np.maximum(best_err, 1e-9) / 7.0)) + 1.0, 1, 12)
    step = (2.0 ** shift) / 2.0
    a = np.array([BRR_FILTERS[f][0] for f in best])
    b = np.array([BRR_FILTERS[f][1] for f in best])

    # Decode it the way the chip would, sixteen steps, all blocks at once.
    # The error compounds inside a block because each sample is predicted
    # from the decoded one before it, not the clean one -- which is exactly
    # where the grain comes from.
    out = np.empty_like(t)
    d1, d2 = prev1.copy(), prev2.copy()
    for i in range(16):
        want = t[:, i] - a * d1 - b * d2
        nib = np.clip(np.round(want / step), -8.0, 7.0)
        d = nib * step + a * d1 + b * d2
        out[:, i] = d
        d2, d1 = d1, d
    tail = y[blocks * 16:]
    got = np.concatenate([out.reshape(-1), tail])
    return (got * (peak / BRR_FULL)).astype(np.float32)


def interpolate(y: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    """*y* played back through the chip's four-point interpolation.

    Not a neutral resampler. Even played at its own speed the four taps are
    a low-pass, and that is why everything on this machine sounds like it is
    coming through a wall. The taps at any phase sum to 2048, which after
    the chip's two shifts is unity gain.
    """
    if len(y) < 8:
        return y.astype(np.float32)
    n = max(2, int(len(y) / max(ratio, 1e-6)))
    pos = np.arange(n) * ratio
    idx = np.clip(pos.astype(np.int64), 3, len(y) - 1)
    phase = np.clip(((pos - np.floor(pos)) * 256).astype(np.int64), 0, 255)
    newest, old = y[idx], y[idx - 1]
    older, oldest = y[idx - 2], y[idx - 3]
    out = (_GAUSS[0xFF - phase] * oldest + _GAUSS[0x1FF - phase] * older
           + _GAUSS[0x100 + phase] * old + _GAUSS[phase] * newest)
    return (out / 2048.0).astype(np.float32)


def echo(y: np.ndarray, delay_ms: float = 48.0, feedback: float = 0.45,
         fir: np.ndarray = ECHO_FIR, sr: int = DSP_RATE) -> np.ndarray:
    """The delay line, fed back through its eight-tap filter.

    Done a delay's worth at a time. Each chunk only depends on the one
    before it, so the recurrence costs a few dozen steps rather than one per
    sample.
    """
    d = max(8, int(delay_ms / 1000.0 * sr))
    if len(y) <= d or feedback <= 0:
        return y.astype(np.float32)
    out = np.array(y, dtype=np.float64)
    for start in range(d, len(out), d):
        stop = min(start + d, len(out))
        past = out[start - d: start - d + (stop - start)]
        out[start:stop] += feedback * np.convolve(past, fir)[: stop - start]
    return out.astype(np.float32)


def play(y: np.ndarray, sr: int = SR, delay_ms: float = 48.0,
         feedback: float = 0.45, tail: float = 0.35,
         stored_hz: float = 16000.0) -> np.ndarray:
    """A recording as this machine would have played it back.

    Down to the rate it would have been stored at, through four-bit
    compression, back up through the interpolation, and into the echo -- in
    that order, because that is the order the hardware does it in and each
    stage feeds on what the last one did.

    *stored_hz* is not a property of the chip. It is the fact that the whole
    machine had 64 KB to hold every sample in a soundtrack, so nothing was
    kept at the DSP's own rate and most things were kept well under half of
    it. That, and the interpolation filling the gap back in, is most of why
    a SNES sounds muffled -- the chip alone would be far brighter than
    anyone remembers it.
    """
    if len(y) == 0:
        return y.astype(np.float32)
    peak = float(np.abs(y).max(initial=0.0))
    if peak <= 0:
        return y.astype(np.float32)
    store = float(np.clip(stored_hz, 2000.0, DSP_RATE))
    x = _resample(y / peak, sr, store)
    x = brr(x)
    # Back up to the DSP's rate, and it is the chip's own interpolation that
    # takes it there rather than anything cleaner -- filling in the gap it
    # was never going to fill neatly is the whole of the sound.
    x = interpolate(x, store / DSP_RATE)
    if tail > 0:
        x = np.concatenate([x, np.zeros(int(tail * DSP_RATE), dtype=np.float32)])
    x = echo(x, delay_ms, feedback)
    out = _resample(x, DSP_RATE, sr)
    top = float(np.abs(out).max(initial=0.0))
    return (out / top * peak).astype(np.float32) if top > 0 else out


# ── Instruments ───────────────────────────────────────────────────────────────
# The chip plays back whatever a game carried, so to play a song with no voice
# it needs something to carry. That is a General MIDI SoundFont cut down to
# what fits the machine -- one short BRR sample per instrument, loops on block
# boundaries, 16 kHz at most -- by scripts/build_snes_bank.py, which says how.
# Which font it was is recorded in the bank: see source().

_BANK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "snes_bank.npz")
_BANK: dict | None = None
_PCM: np.ndarray | None = None

# What the pitch register can ask for: 0x3FFF steps against 0x1000 at the
# sample's own rate, so a sample can go up two octaves and no further.
MAX_STEP = 0x3FFF / 0x1000


@dataclass
class Instrument:
    name: str
    pcm: np.ndarray                  # float32, -1..1, BRR already done
    rate: float                      # what the sample plays at its root
    loop: tuple[int, int] | None
    root: int
    cents: int
    scale: float                     # 0 for a drum that plays one pitch whatever the key
    attenuation_db: float
    env: tuple[float, float, float, float, float, float]  # delay attack hold decay sustain(dB) release


def bank() -> dict:
    """Everything in the shipped bank, loaded once: {"p0": Instrument, "d36": ...}."""
    global _BANK, _PCM
    if _BANK is None:
        with np.load(_BANK_PATH) as z:
            _PCM = z["pcm"].astype(np.float32) / 32767.0
            index = json.loads(bytes(z["index"]).decode())
        _BANK = {k: Instrument(e["name"], _PCM[e["offset"]:e["offset"] + e["length"]], e["rate"],
                               tuple(e["loop"]) if e["loop"] else None, e["root"], e["cents"],
                               e["scale"], e["attenuation_db"], tuple(e["env"]))
                 for k, e in index.items() if not k.startswith("_")}
        _BANK["_source"] = index.get("_source", {})
    return _BANK


def source() -> dict:
    """Which SoundFont the bank was cut from, and what it says about itself."""
    return bank()["_source"]


def instrument_for(program: int | None = None, drum_key: int | None = None) -> Instrument:
    """The sample a part is played with: its program's, or for a drum its key's."""
    b = bank()
    if drum_key is not None:
        return b.get(f"d{int(drum_key)}") or b["d38"]
    return b.get(f"p{int(program or 0)}") or b["p0"]


def _envelope(n: int, gate: int, env, sr: int) -> np.ndarray:
    """The font's volume envelope, in the decibels it is written in."""
    delay, attack, hold, decay, sustain_db, release = env
    t = np.arange(n) / sr
    db = np.full(n, -sustain_db)
    a0, a1 = delay, delay + attack
    h1 = a1 + hold
    d1 = h1 + decay * min(1.0, sustain_db / 96.0) if decay > 0 else h1
    db[t < d1] = -sustain_db * np.clip((t[t < d1] - h1) / max(d1 - h1, 1e-9), 0.0, 1.0)
    amp = 10.0 ** (db / 20.0)
    amp[t < a1] = np.clip((t[t < a1] - a0) / max(attack, 1e-9), 0.0, 1.0) if attack > 0 else (t[t < a1] >= a0)
    # Released: down 96 dB over the release time, from wherever it had got to.
    if gate < n:
        start = amp[gate - 1] if gate > 0 else 0.0
        fall = (t[gate:] - t[gate]) / max(release, 0.01) * 96.0
        amp[gate:] = start * 10.0 ** (-fall / 20.0)
    return amp


def render_note(inst: Instrument, midi_note: float, dur: float, sr: int = SR,
                level: float = 1.0, tail: float = 0.35) -> np.ndarray:
    """One note of *inst*, as the S-DSP plays it: pitched through the Gaussian
    table, looped if the sample loops, enveloped, and into the echo.

    One liberty: a note more than two octaves over its sample is played anyway,
    where the pitch register would stop at two. A game chose its samples to fit;
    a General MIDI part cannot.
    """
    semis = (midi_note - inst.root) * inst.scale + inst.cents / 100.0
    step = 2.0 ** (semis / 12.0) * inst.rate / DSP_RATE
    release = min(max(inst.env[5], 0.02), 1.0)
    gate = max(1, int(dur * DSP_RATE))
    n = gate + int((release + tail) * DSP_RATE)
    pos = np.arange(n) * step
    y = inst.pcm
    if inst.loop:
        ls, le = inst.loop
        over = pos >= le
        pos[over] = ls + np.mod(pos[over] - ls, le - ls)
        span = le
    else:
        span = len(y)
        n = min(n, int(max(0, span - 1) / step))
        pos = pos[:n]
    if n <= 0:
        return np.zeros(max(1, int(dur * sr)), dtype=np.float32)
    whole = np.floor(pos).astype(np.int64)
    phase = ((pos - whole) * 256).astype(np.int64)

    def at(k):
        # The four points the table blends, reaching back across the loop.
        k = whole + k
        if inst.loop:
            ls, le = inst.loop
            k = np.where(k >= le, ls + np.mod(k - ls, le - ls), k)
        return y[np.clip(k, 0, span - 1)]

    out = (_GAUSS[0xFF - phase] * at(-2) + _GAUSS[0x1FF - phase] * at(-1)
           + _GAUSS[0x100 + phase] * at(0) + _GAUSS[phase] * at(1)) / 2048.0
    gain = level * 10.0 ** (-0.4 * inst.attenuation_db / 20.0)
    out = out * _envelope(len(out), min(gate, len(out)), inst.env, DSP_RATE) * gain
    out = echo(out.astype(np.float32))
    return _resample(out, DSP_RATE, sr)
