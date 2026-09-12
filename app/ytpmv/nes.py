"""The NES and Famicom's sound chip: the 2A03's audio half.

Nothing like the Mega Drive's. There is no FM here and no way to make a new
timbre: the chip has five fixed voices and the only choices are which, how
loud, and how often. That constraint is the sound.

  two pulses    a square wave at one of four widths -- an eighth, a quarter,
                a half, or a quarter upside down. Sixteen volume steps, and
                a sweep unit for the drop at the front of a note
  triangle      a staircase of sixteen levels up and sixteen down, and no
                volume control whatsoever. It is on or it is off, which is
                why NES basslines sound the way they do
  noise         a shift register, not a random number generator: it repeats,
                after 32767 steps in its long mode and after 93 in its short
                one, which is why the short one has an audible pitch to it
  DMC           one-bit delta modulation at seven bits, stepping by two at a
                time, at one of sixteen rates. This is where the NES kept
                its drums and its speech, and it is where a corpus goes

Everything here is checked against Blargg's Nes_Snd_Emu rather than
remembered: the noise period table, the DMC rates, the duty widths (1 << the
setting, with the fourth being the quarter negated) and the mixing weights.

Two things are deliberately not faithful. The real mixer is non-linear
*across* channels -- the pulses share one curve and the triangle, noise and
DMC share another, so each gets quieter as the others get louder. This
renders one part at a time and sums them afterwards, so each channel goes
through its own curve and gets the compression but not the interaction. And
the frame counter's envelopes are computed as curves rather than clocked at
240 Hz. Both are audible only if you go looking.

numpy only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from app.ytpmv.pitch import SR

# NTSC. The PAL machine runs at 1662607 and is a different tuning entirely.
CPU = 1789773.0

# Steps the noise's shift register waits between clocks, by setting.
NOISE_PERIODS = (4, 8, 16, 32, 64, 96, 128, 160,
                 202, 254, 380, 508, 762, 1016, 2034, 4068)

# CPU cycles per bit for the DMC, by setting: 4182 Hz at the slowest,
# 33144 at the fastest, and nothing in between is available.
DMC_PERIODS = (428, 380, 340, 320, 286, 254, 226, 214,
               190, 160, 142, 128, 106, 84, 72, 54)

# The chip runs far faster than any sample rate we are going to hand back:
# the noise register clocks at up to 447 kHz and a square wave has harmonics
# all the way up. Generating straight at 44.1 kHz folds all of that back down
# as aliasing -- which ruins the short noise mode in particular, whose whole
# point is that its 93-step loop has an audible pitch. So everything is made
# at four times the rate and filtered on the way down.
OVERSAMPLE = 4

_NOISE_CACHE: dict[int, np.ndarray] = {}
_DPCM_CACHE: dict[tuple, np.ndarray] = {}


def noise_bits(mode: int = 0) -> np.ndarray:
    """The whole shift register sequence, which is short enough to keep.

    Fifteen bits, fed back from bit 0 against bit 1, or against bit 6 in the
    short mode. It is periodic -- 32767 steps, or 93 -- so rather than clock
    it a sample at a time, clock it once and index into the result. The
    short mode repeating 93 times a second is not a bug in anybody's
    emulator: it is why that setting sounds like a pitch rather than a hiss.
    """
    if mode in _NOISE_CACHE:
        return _NOISE_CACHE[mode]
    tap = 6 if mode else 1
    reg = 1
    seen, out = {}, []
    while reg not in seen:
        seen[reg] = len(out)
        out.append((~reg) & 1)                    # the output is bit 0, inverted
        reg = (reg >> 1) | (((reg ^ (reg >> tap)) & 1) << 14)
    _NOISE_CACHE[mode] = np.array(out, dtype=np.float32)
    return _NOISE_CACHE[mode]


def _timer(hz: float, div: int) -> int:
    """The 11-bit divider the chip would actually be set to for *hz*.

    Kept because it is audible: the steps get coarse high up, so the top of
    the range goes out of tune, and that mistuning is part of how the
    machine sounds.
    """
    if hz <= 0:
        return 2047
    return int(np.clip(round(CPU / (div * hz) - 1.0), 0, 2047))


def pulse(hz: float, n: int, duty: int, sr: int = SR,
          bend: float = 0.0, bend_time: float = 0.01) -> np.ndarray:
    """A square wave at one of the chip's four widths, 0 to 15."""
    t = np.arange(n) / sr
    f = hz * 2.0 ** (bend * np.exp(-t / bend_time) / 12.0) if bend else np.full(n, hz)
    # Quantised to the timer the chip can hold, moment to moment.
    div = np.maximum(np.round(CPU / (16.0 * np.maximum(f, 1e-6)) - 1.0), 8.0)
    real = CPU / (16.0 * (div + 1.0))
    step = np.floor(np.cumsum(real) / sr * 8.0).astype(np.int64) % 8
    width = (1, 2, 4, 2)[duty & 3]
    on = step < width
    if duty & 3 == 3:
        on = ~on                                   # the quarter, upside down
    return on.astype(np.float32) * 15.0


def triangle(hz: float, n: int, sr: int = SR) -> np.ndarray:
    """The staircase: sixteen levels up, sixteen down, and no volume control.

    An octave lower than you would expect for the same timer value, because
    it divides by 32 where the pulses divide by 16.
    """
    t = _timer(hz, 32)
    real = CPU / (32.0 * (t + 1.0))
    step = np.floor(np.arange(n) / sr * real * 32.0).astype(np.int64) % 32
    return np.where(step < 16, 15 - step, step - 16).astype(np.float32)


def noise(setting: int, n: int, sr: int = SR, mode: int = 0) -> np.ndarray:
    """The shift register, clocked at one of sixteen rates, 0 to 15."""
    bits = noise_bits(mode)
    rate = CPU / NOISE_PERIODS[int(np.clip(setting, 0, 15))]
    idx = (np.arange(n) / sr * rate).astype(np.int64) % len(bits)
    return bits[idx] * 15.0


def dpcm(y: np.ndarray, setting: int = 10, sr: int = SR) -> np.ndarray:
    """A sample as the DMC would play it back.

    Seven bits, and the level can only move by two per step, so anything
    that rises faster than the rate allows simply does not get there -- the
    channel slews. That, and the handful of playback rates, is the whole
    character of NES speech: it is not merely quiet or noisy, it is blunted.
    """
    if len(y) == 0:
        return y.astype(np.float32)
    key = (hashlib.blake2b(np.ascontiguousarray(y, dtype=np.float32).tobytes(),
                           digest_size=16).hexdigest(), setting, sr, len(y))
    hit = _DPCM_CACHE.get(key)
    if hit is not None:
        return hit
    rate = CPU / DMC_PERIODS[int(np.clip(setting, 0, 15))]
    steps = max(1, int(len(y) / sr * rate))
    want = np.interp(np.arange(steps) / rate, np.arange(len(y)) / sr, y)
    want = np.clip(want * 63.0 + 64.0, 0.0, 127.0)     # into the channel's 7 bits
    out = np.empty(steps, dtype=np.float32)
    level = 64.0
    for i in range(steps):                              # a delta modulator is a recurrence
        level = min(127.0, level + 2.0) if want[i] > level else max(0.0, level - 2.0)
        out[i] = level
    played = out[np.minimum((np.arange(len(y)) / sr * rate).astype(np.int64), steps - 1)]
    if len(_DPCM_CACHE) > 512:
        _DPCM_CACHE.clear()
    _DPCM_CACHE[key] = played
    return played


def _lowpass(y: np.ndarray, hz: float, sr: float) -> np.ndarray:
    if len(y) < 32 or hz >= sr / 2:
        return y
    n = 1 << int(np.ceil(np.log2(len(y) * 2)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(y, n) / (1.0 + (f / hz) ** 8), n)[: len(y)]


def _down(y: np.ndarray, sr: int) -> np.ndarray:
    """Back to *sr* from the oversampled rate, without the aliasing."""
    if OVERSAMPLE <= 1:
        return y.astype(np.float32)
    return _lowpass(y, 0.45 * sr, sr * OVERSAMPLE)[::OVERSAMPLE].astype(np.float32)


def mix_pulse(v: np.ndarray) -> np.ndarray:
    """One pulse channel through the chip's non-linear output curve."""
    return (95.88 / (8128.0 / np.maximum(v, 1e-9) + 100.0)).astype(np.float32)


def mix_tnd(tri: np.ndarray = 0.0, noi: np.ndarray = 0.0,
            dmc: np.ndarray = 0.0) -> np.ndarray:
    """The triangle, noise and DMC share the other curve."""
    inner = np.asarray(tri) / 8227.0 + np.asarray(noi) / 12241.0 + np.asarray(dmc) / 22638.0
    return (159.79 / (1.0 / np.maximum(inner, 1e-12) + 100.0)).astype(np.float32)


@dataclass(frozen=True)
class Voice:
    """One of the chip's voices, set up a particular way.

    *decay* is how long the note takes to fall to nothing, in seconds; the
    hardware does this in fifteen steps off the frame counter, and a curve
    through the same range is close enough that nobody has ever heard the
    difference. *base_hz* belongs to the drums, which are not playing a note
    and ignore the one they are given.
    """
    name: str
    about: str
    kind: str                      # pulse | triangle | noise
    duty: int = 2
    setting: int = 8               # noise: which of the sixteen rates
    mode: int = 0                  # noise: 0 long, 1 the short buzzy one
    decay: float = 0.0             # 0 holds for as long as the note does
    bend: float = 0.0
    bend_time: float = 0.01
    base_hz: float = 0.0
    transpose: int = 0
    gain: float = 1.0


def render_note(voice: Voice, hz: float, dur: float, sr: int = SR,
                level: float = 1.0) -> np.ndarray:
    """*voice* playing *hz* for *dur* seconds, as the chip would."""
    n = max(2, int(round(dur * sr)))
    hz = (voice.base_hz or hz) * 2.0 ** (voice.transpose / 12.0)
    isr = sr * OVERSAMPLE
    m = n * OVERSAMPLE

    if voice.kind == "triangle":
        out = mix_tnd(tri=triangle(hz, m, isr))
    elif voice.kind == "noise":
        out = mix_tnd(noi=noise(voice.setting, m, isr, voice.mode))
    else:
        out = mix_pulse(pulse(hz, m, voice.duty, isr, voice.bend, voice.bend_time))
    out = _down(out, sr)

    t = np.arange(n) / sr
    # The triangle has no volume control at all, so neither velocity nor a
    # decay envelope can touch it -- it only starts and stops. Faking either
    # would be inventing a register the machine does not have.
    if voice.kind != "triangle":
        if voice.decay > 0:
            out = out * np.maximum(0.0, 1.0 - t / voice.decay) ** 1.5
        # Velocity is the four-bit volume, so it is a staircase, not a slope.
        out = out * (max(1, round(level * 15.0)) / 15.0)

    out = out - out.mean()                          # the curves sit well above zero
    edge = min(n, int(0.002 * sr))
    if edge > 1:
        out[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
    peak = float(np.abs(out).max(initial=0.0))
    return (out / peak * 0.62 * voice.gain).astype(np.float32) if peak > 0 else out


VOICES: dict[str, Voice] = {
    # Melodic. A quarter-width pulse is the sound people think of as "NES".
    "pulse": Voice("pulse", "a quarter-width square, the NES lead", "pulse", duty=1),
    "pulse-thin": Voice("pulse-thin", "an eighth-width square, thin and reedy", "pulse", duty=0),
    "pulse-full": Voice("pulse-full", "a half-width square, round and hollow", "pulse", duty=2),
    # The bass. Also the only voice on the machine with no volume control.
    "triangle": Voice("triangle", "the triangle bass, on or off and nothing between",
                      "triangle"),
    # Drums, which on this machine are the noise channel and a thump.
    "nes-kick": Voice("nes-kick", "a pulse dropped hard, the NES kick", "pulse", duty=2,
                      decay=0.10, bend=26.0, bend_time=0.013, base_hz=62.0),
    "nes-snare": Voice("nes-snare", "noise, cut short", "noise", setting=5, decay=0.13),
    "nes-hat": Voice("nes-hat", "noise, cut very short", "noise", setting=1, decay=0.035),
    "nes-tom": Voice("nes-tom", "a pulse dropped less hard", "pulse", duty=2,
                     decay=0.16, bend=14.0, bend_time=0.03, base_hz=110.0),
    "nes-cymbal": Voice("nes-cymbal", "noise, left to ring", "noise", setting=0, decay=0.45),
    "nes-perc": Voice("nes-perc", "a tick off the short shift register", "noise",
                      setting=3, mode=1, decay=0.05),
}

# What each part is played by. The triangle takes the bass because that is
# what it is for and what every NES composer used it for.
_BY_ROLE = {"bass": "triangle", "lead": "pulse", "chords": "pulse-full",
            "rhythm": "pulse-thin"}
_BY_PROGRAM = (
    (0, 31, "pulse-full"), (32, 39, "triangle"), (40, 55, "pulse"),
    (56, 79, "pulse"), (80, 103, "pulse"), (104, 127, "pulse-thin"),
)
DRUMS = {"kick": "nes-kick", "snare": "nes-snare", "hats": "nes-hat",
         "toms": "nes-tom", "cymbals": "nes-cymbal", "perc": "nes-perc"}


def voice_for(role: str, program: int | None = None) -> Voice:
    """The NES voice to play a part with, from what the analysis found."""
    head = (role or "").split(":")[0]
    if head == "drums":
        return VOICES[DRUMS.get((role or "").split(":")[-1], "nes-perc")]
    name = _BY_ROLE.get(head)
    if name is None and program is not None:
        for lo, hi, guess in _BY_PROGRAM:
            if lo <= program <= hi:
                name = guess
                break
    return VOICES[name or "pulse"]
