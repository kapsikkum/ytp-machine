"""The Game Boy's sound chip, shared unchanged by the Game Boy Color.

Four channels, and unlike the NES the third one is not fixed: it plays a
thirty-two-step, four-bit waveform that the game writes itself. So a Game
Boy gets to choose one timbre, which is more than the NES ever had and far
less than the Mega Drive.

  pulse x2   a square at one of four widths, 131072 / (2048 - x) for an
             eleven-bit x -- 64 Hz at the bottom. The first has a sweep
             that walks the pitch while the note plays
  wave       thirty-two four-bit samples on a loop, 65536 / (2048 - x),
             so an octave lower than the pulses reach. Its volume is a
             shift, not a level: full, half, quarter or off
  noise      a fifteen-bit shift register fed back from the XNOR of its two
             lowest bits, or a seven-bit one in short mode, clocked at
             262144 / (divider x 2^shift) with a divider of 0 counting as a
             half
  envelope   four bits of volume, stepped at 64 Hz -- a staircase, audibly,
             not a fade

Then the part people forget. Each channel's DAC runs from +1 at digital 0
down to -1 at 15, so a square that never goes below zero digitally sits
well off centre in the analog world, and a capacitor on the output drags
it back towards the middle the whole time the note is held. Squares sag.
That sag is a large part of what a Game Boy sounds like.

Checked against the gbdev Pan Docs for the formulas and SameBoy for the
capacitor. The gbdev wiki gives a separate charge factor for the Color,
0.998943, which at 44.1 kHz works out to a 700 Hz high-pass -- the bass
would simply not be there. SameBoy uses 0.999958 for every model, a 28 Hz
corner, and so does this: the two machines sound the same, and differ in
their screens.

numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ytpmv.pitch import SR

CLOCK = 4194304.0

# The output capacitor's charge factor per CPU tick, from SameBoy, for every
# model. See the module notes for why not the wiki's Color figure.
CHARGE = 0.999958

# Made high and filtered down, as with the other chips: a square has
# harmonics all the way up and the noise clocks far past what we hand back.
OVERSAMPLE = 4

# Wave channel volume, as the shift it really is.
WAVE_SHIFT = {0: None, 1: 0, 2: 1, 3: 2}         # off, full, half, quarter

_LFSR_CACHE: dict[bool, np.ndarray] = {}


def _lowpass(y: np.ndarray, hz: float, sr: float) -> np.ndarray:
    if len(y) < 32 or hz >= sr / 2:
        return y
    n = 1 << int(np.ceil(np.log2(len(y) + 4096)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(y, n) / (1.0 + (f / hz) ** 8), n)[: len(y)]


def _down(y: np.ndarray, sr: int) -> np.ndarray:
    return _lowpass(y, 0.45 * sr, sr * OVERSAMPLE)[::OVERSAMPLE].astype(np.float32)


def capacitor(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """The output capacitor: out = in - charge, charge creeping towards in.

    A one-pole high-pass, (1 - z^-1) / (1 - k z^-1). It is a recurrence, but
    a linear one, so it is applied through its frequency response instead of
    a sample at a time.
    """
    if len(y) < 32:
        return y.astype(np.float32)
    k = CHARGE ** (CLOCK / sr)
    pad = int(8.0 / max(1.0 - k, 1e-9))              # long enough to settle
    n = 1 << int(np.ceil(np.log2(len(y) + min(pad, 1 << 20))))
    z = np.exp(-2j * np.pi * np.fft.rfftfreq(n, 1.0 / sr) / sr)
    h = (1.0 - z) / (1.0 - k * z)
    return np.fft.irfft(np.fft.rfft(y, n) * h, n)[: len(y)].astype(np.float32)


def dac(digital: np.ndarray) -> np.ndarray:
    """Digital 0..15 to analog +1..-1. The slope really is negative."""
    return (1.0 - np.asarray(digital, dtype=np.float32) / 7.5).astype(np.float32)


def lfsr_bits(short: bool = False) -> np.ndarray:
    """The noise register's output bit, over one full cycle."""
    if short in _LFSR_CACHE:
        return _LFSR_CACHE[short]
    reg, seen, out = 0, {}, []
    while reg not in seen:
        seen[reg] = len(out)
        bit = 1 - ((reg ^ (reg >> 1)) & 1)            # XNOR of bits 0 and 1
        reg = (reg & ~(1 << 15)) | (bit << 15)
        if short:
            reg = (reg & ~(1 << 7)) | (bit << 7)
        reg >>= 1
        out.append(reg & 1)
    start = seen[reg]                                 # skip the lead-in, keep the loop
    _LFSR_CACHE[short] = np.array(out[start:], dtype=np.float32)
    return _LFSR_CACHE[short]


def floor_hz(kind: str) -> float:
    """The lowest note a channel can play; 0 for noise."""
    return {"pulse": CLOCK / 32.0 / 2048.0, "wave": CLOCK / 64.0 / 2048.0}.get(kind, 0.0)


def _period(hz: np.ndarray, top: float) -> np.ndarray:
    """The eleven-bit value for *hz* on a channel whose rate is top / (2048 - x)."""
    return np.clip(np.round(2048.0 - top / np.maximum(hz, 1e-6)), 0.0, 2047.0)


def pulse(hz: float, n: int, sr: int, duty: int = 2, sweep: tuple = ()) -> np.ndarray:
    """A square at one of four widths, as digital 0/1.

    *sweep* is (pace, shift, down): the first channel's pitch walker. Every
    pace/128 s it moves the period by period >> shift, which is a curve, not
    a line, and it is what a Game Boy kick is made of.
    """
    x = np.full(n, float(_period(np.array([hz]), 131072.0)[0]))
    if sweep:
        pace, shift, down = sweep
        step = max(1, int(round(sr * pace / 128.0)))
        cur = x[0]
        for i in range(0, n, step):
            x[i:i + step] = cur
            delta = np.floor(cur / (2 ** shift))
            cur = max(0.0, cur - delta) if down else min(2047.0, cur + delta)
    real = 131072.0 / (2048.0 - x)
    pos = np.floor(np.cumsum(real) / sr * 8.0).astype(np.int64) % 8
    width = (1, 2, 4, 6)[duty & 3]
    return (pos < width).astype(np.float32)


def wave(hz: float, n: int, sr: int, table: tuple) -> np.ndarray:
    """The programmable channel: thirty-two four-bit steps on a loop."""
    x = _period(np.array([hz]), 65536.0)[0]
    real = 65536.0 / (2048.0 - x)
    idx = np.floor(np.arange(n) / sr * real * 32.0).astype(np.int64) % 32
    return np.asarray(table, dtype=np.float32)[idx]


def noise(n: int, sr: int, divider: int = 0, shift: int = 4, short: bool = False) -> np.ndarray:
    """The shift register as digital 0/1, clocked as NR43 would clock it."""
    div = 0.5 if divider == 0 else float(divider)
    rate = 262144.0 / (div * 2 ** int(np.clip(shift, 0, 13)))
    bits = lfsr_bits(short)
    return bits[(np.arange(n) / sr * rate).astype(np.int64) % len(bits)]


def envelope(n: int, sr: int, start: int, pace: int, up: bool = False) -> np.ndarray:
    """The four-bit volume over time: one step every pace/64 s, 0 to hold."""
    if pace <= 0:
        return np.full(n, float(start), dtype=np.float32)
    steps = np.floor(np.arange(n) / sr * 64.0 / pace)
    vol = start + steps if up else start - steps
    return np.clip(vol, 0, 15).astype(np.float32)


@dataclass(frozen=True)
class Voice:
    """One channel, set up a particular way."""
    name: str
    about: str
    kind: str                         # pulse | wave | noise
    duty: int = 2
    table: tuple = ()
    level: int = 1                    # wave: 1 full, 2 half, 3 quarter
    volume: int = 15                  # envelope start
    pace: int = 0                     # envelope pace, 0 holds
    sweep: tuple = ()
    divider: int = 0
    shift: int = 4
    short: bool = False
    base_hz: float = 0.0
    gain: float = 1.0


def voice_floor(voice: Voice) -> float:
    return 0.0 if voice.base_hz else floor_hz(voice.kind)


def render_note(voice: Voice, hz: float, dur: float, sr: int = SR,
                level: float = 1.0) -> np.ndarray:
    """*voice* playing *hz* for *dur* seconds, through DAC and capacitor."""
    n = max(2, int(round(dur * sr)))
    hz = voice.base_hz or hz
    isr, m = sr * OVERSAMPLE, n * OVERSAMPLE

    # Velocity is the envelope's starting volume, four bits of it.
    start = int(np.clip(round(voice.volume * max(level, 1.0 / 15.0)), 1, 15))
    if voice.kind == "wave":
        shift = WAVE_SHIFT.get(voice.level)
        if shift is None:
            return np.zeros(n, dtype=np.float32)
        digital = np.floor(wave(hz, m, isr, voice.table) / 2 ** shift)
        # The wave channel has no envelope; velocity picks the coarse shift.
        if level < 0.6:
            digital = np.floor(digital / 2)
    else:
        raw = pulse(hz, m, isr, voice.duty, voice.sweep) if voice.kind == "pulse" \
            else noise(m, isr, voice.divider, voice.shift, voice.short)
        digital = raw * envelope(m, isr, start, voice.pace)

    # A DAC that is enabled but outputting silence sits at +1, and between
    # notes the capacitor has long since charged to it. Rendering each note
    # from an uncharged capacitor instead put a full-scale step at the front
    # of every one -- a thump whatever the volume, which is what made a soft
    # note measure nearly as loud as a hard one. So start from the idle
    # charge: take the +1 off first, and what the capacitor then chases is
    # only the note itself.
    out = capacitor(_down(dac(digital) - 1.0, sr), sr)
    out = out * voice.gain * 0.62 / 2.0              # a full-scale channel swings 2
    edge = min(n, int(0.003 * sr))
    if edge > 1:
        out[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
    return out.astype(np.float32)


def pcm(y: np.ndarray, sr: int = SR, rate: float = 8192.0) -> np.ndarray:
    """A recording as a Game Boy had to play one.

    No sample channel: games reloaded the wave table as fast as they could
    and played four bits a step through it. Four bits, a low rate, and the
    capacitor on top.
    """
    if len(y) == 0:
        return y.astype(np.float32)
    step = max(1, int(round(sr / rate)))
    held = np.repeat(y[::step], step)[: len(y)]
    peak = float(np.abs(held).max(initial=0.0)) or 1.0
    digital = np.clip(np.round((1.0 - held / peak) * 7.5), 0, 15)
    return (capacitor(dac(digital) - 1.0, sr) * peak).astype(np.float32)


# A handful of wave tables. The hardware gives no defaults worth using --
# every game wrote its own -- so these are chosen, not sourced.
TABLES = {
    "triangle": tuple(list(range(0, 16)) + list(range(15, -1, -1))),
    "saw": tuple(i // 2 for i in range(32)),
    "bass": (15, 15, 14, 13, 11, 9, 7, 5, 3, 2, 1, 0, 0, 0, 1, 2,
             3, 5, 7, 9, 11, 13, 14, 15, 15, 15, 15, 14, 12, 10, 8, 6),
}

VOICES: dict[str, Voice] = {
    "gb-pulse": Voice("gb-pulse", "a quarter-width square", "pulse", duty=1),
    "gb-pulse-thin": Voice("gb-pulse-thin", "an eighth-width square", "pulse", duty=0, volume=12),
    "gb-pulse-full": Voice("gb-pulse-full", "a half-width square", "pulse", duty=2, volume=11),
    "gb-pluck": Voice("gb-pluck", "a square that steps down as it plays", "pulse",
                      duty=1, pace=2),
    "gb-wave": Voice("gb-wave", "the wave channel with a round bass table", "wave",
                     table=TABLES["bass"]),
    "gb-kick": Voice("gb-kick", "a square swept down, the Game Boy kick", "pulse",
                     duty=2, pace=1, sweep=(1, 1, True), base_hz=180.0),
    "gb-snare": Voice("gb-snare", "the long register, stepped down fast", "noise",
                      divider=1, shift=3, pace=1),
    "gb-hat": Voice("gb-hat", "the long register, clocked fast and cut", "noise",
                    divider=0, shift=1, volume=9, pace=1),
    "gb-tom": Voice("gb-tom", "a square swept down less hard", "pulse",
                    duty=2, pace=2, sweep=(2, 2, True), base_hz=150.0),
    "gb-cymbal": Voice("gb-cymbal", "the long register, left to ring", "noise",
                       divider=0, shift=2, volume=10, pace=5),
    "gb-perc": Voice("gb-perc", "the short register, which has a pitch", "noise",
                     divider=2, shift=3, short=True, volume=11, pace=1),
}

_BY_ROLE = {"bass": "gb-wave", "lead": "gb-pulse", "chords": "gb-pulse-full",
            "rhythm": "gb-pulse-thin"}
DRUMS = {"kick": "gb-kick", "snare": "gb-snare", "hats": "gb-hat",
         "toms": "gb-tom", "cymbals": "gb-cymbal", "perc": "gb-perc"}


def voice_for(role: str, program: int | None = None) -> Voice:
    head = (role or "").split(":")[0]
    if head == "drums":
        return VOICES[DRUMS.get((role or "").split(":")[-1], "gb-perc")]
    return VOICES[_BY_ROLE.get(head, "gb-pulse")]
