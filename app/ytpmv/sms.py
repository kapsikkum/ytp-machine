"""The Master System's sound chip, the SN76489.

The simplest of the lot. Three square waves and a shift register, and that
is the entire instrument -- no duty control, no filter, nothing to shape a
tone with. A voice on this machine is a choice of pitch and a choice of
volume, sixteen steps of it, two decibels apart. Everything anybody ever did
on it was done with those.

  tone x3   a square at clock / (32n), n being ten bits. Hence 109 Hz at
            the bottom and far past hearing at the top
  noise     a sixteen-bit shift register tapped at bits 0 and 3, either
            run properly (white) or in a mode that repeats every sixteen
            steps, which is a buzz with a pitch rather than a hiss. Those
            taps are not a maximal-length polynomial, so "properly" means a
            cycle of 57337 rather than 65535 -- about eight seconds before
            it comes round, which nobody is going to hear
  volume    four bits of attenuation per channel, two decibels a step,
            and 15 is silence rather than quiet

Checked against SMSPower's chip notes rather than remembered: the divider,
the tap bits, the two-decibel step, and the noise counter reloads. The one
figure worth keeping to check the rest by is that register value $0FE gives
440.4 Hz.

There is one more thing this chip is good for. With no sample channel at
all, Master System games played speech by writing the volume register very
fast -- four bits of PCM straight out of the attenuator. That is what
"sms-voice" is, and it is as rough as it sounds.

A Mega Drive has one of these in it as well, for compatibility, and real
Mega Drive music used both: hats and counter-melodies came off this chip
while the FM did the rest.

numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ytpmv.pitch import SR

# NTSC. PAL machines run at 3546893 and are a different tuning.
CLOCK = 3579545.0

# The chip runs far above any rate we hand back and a square has harmonics
# all the way up, so it is made high and filtered on the way down.
OVERSAMPLE = 4

# The lowest note the divider can reach. Below this the chip has nothing,
# which is why Master System basslines sit where they do.
FLOOR_HZ = CLOCK / (32.0 * 1023.0)

# What the noise counter reloads with, by setting. The fourth takes its rate
# from tone channel 2 instead, which is how tuned noise was done.
NOISE_RELOADS = (0x10, 0x20, 0x40)

_LFSR_CACHE: dict[int, np.ndarray] = {}


def attenuation(step: int) -> float:
    """Volume register *step* as a factor. 0 is full, 15 is off, 2 dB apart."""
    step = int(np.clip(step, 0, 15))
    return 0.0 if step >= 15 else float(10.0 ** (-2.0 * step / 20.0))


def lfsr_bits(periodic: bool = False) -> np.ndarray:
    """The whole shift register sequence, which is short enough to keep.

    Sixteen bits, fed back from the parity of bits 0 and 3. The other mode
    does not tap anything: it just walks a single bit round, so it repeats
    every sixteen steps and reads as a pitch.
    """
    key = int(periodic)
    if key in _LFSR_CACHE:
        return _LFSR_CACHE[key]
    if periodic:
        out = [1] + [0] * 15                       # one in sixteen, a buzz
    else:
        reg, seen, out = 0x8000, set(), []
        while reg not in seen:
            seen.add(reg)
            out.append(reg & 1)
            parity = bin(reg & 0x0009).count("1") & 1   # bits 0 and 3
            reg = (reg >> 1) | (parity << 15)
    _LFSR_CACHE[key] = np.array(out, dtype=np.float32)
    return _LFSR_CACHE[key]


def _lowpass(y: np.ndarray, hz: float, sr: float) -> np.ndarray:
    if len(y) < 32 or hz >= sr / 2:
        return y
    n = 1 << int(np.ceil(np.log2(len(y) + 4096)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(y, n) / (1.0 + (f / hz) ** 8), n)[: len(y)]


def _down(y: np.ndarray, sr: int) -> np.ndarray:
    if OVERSAMPLE <= 1:
        return y.astype(np.float32)
    return _lowpass(y, 0.45 * sr, sr * OVERSAMPLE)[::OVERSAMPLE].astype(np.float32)


def divider(hz: float) -> int:
    """The ten-bit value the chip would be set to for *hz*.

    Kept because it is audible: the steps are coarse at the top, so high
    notes land wherever the divider lets them and the tuning drifts.
    """
    if hz <= 0:
        return 1023
    return int(np.clip(round(CLOCK / (32.0 * hz)), 1, 1023))


def square(hz: float, n: int, sr: int, bend: float = 0.0,
           bend_time: float = 0.01) -> np.ndarray:
    """A square wave, at the nearest pitch the divider can hold."""
    t = np.arange(n) / sr
    f = hz * 2.0 ** (bend * np.exp(-t / bend_time) / 12.0) if bend else np.full(n, float(hz))
    div = np.clip(np.round(CLOCK / (32.0 * np.maximum(f, 1e-6))), 1, 1023)
    real = CLOCK / (32.0 * div)
    phase = np.cumsum(real) / sr
    return np.where((phase % 1.0) < 0.5, 1.0, -1.0).astype(np.float32)


def noise(setting: int, n: int, sr: int, periodic: bool = False,
          tone_hz: float = 0.0) -> np.ndarray:
    """The shift register, clocked at one of its four rates."""
    bits = lfsr_bits(periodic)
    if setting >= 3 and tone_hz > 0:
        rate = CLOCK / (32.0 * divider(tone_hz))
    else:
        rate = CLOCK / 16.0 / (2.0 * NOISE_RELOADS[int(np.clip(setting, 0, 2))])
    idx = (np.arange(n) / sr * rate).astype(np.int64) % len(bits)
    return (bits[idx] * 2.0 - 1.0).astype(np.float32)


@dataclass(frozen=True)
class Voice:
    """One channel, set up a particular way.

    There is no timbre to set. All this can say is which generator, how loud
    it starts, how fast it fades and where the pitch goes -- which is the
    whole vocabulary the chip has.
    """
    name: str
    about: str
    kind: str                    # tone | noise
    volume: int = 0              # the attenuator, 0 loudest
    setting: int = 1             # noise: which rate
    periodic: bool = False       # noise: the sixteen-step buzz
    decay: float = 0.0           # seconds to fade to nothing, 0 to hold
    bend: float = 0.0
    bend_time: float = 0.01
    base_hz: float = 0.0
    transpose: int = 0
    gain: float = 1.0


def floor_hz(voice: Voice) -> float:
    """The lowest note *voice* can play. 0 for anything with no pitch."""
    return 0.0 if voice.base_hz or voice.kind == "noise" else FLOOR_HZ


def render_note(voice: Voice, hz: float, dur: float, sr: int = SR,
                level: float = 1.0) -> np.ndarray:
    """*voice* playing *hz* for *dur* seconds, as the chip would."""
    n = max(2, int(round(dur * sr)))
    hz = (voice.base_hz or hz) * 2.0 ** (voice.transpose / 12.0)
    if voice.kind == "tone" and not voice.base_hz:
        # A last resort only. Whether a part needs moving up is decided for
        # the whole part, before any note is played (see render) -- because
        # raising the notes under 109 Hz one at a time and leaving the rest
        # takes a line apart: D2 D3 A2 comes out D3 D3 A2, the octave leap
        # gone and the A now under its neighbours. If a note still arrives
        # down here, nobody decided for its part, and an octave up is at
        # least in tune with itself.
        while hz < FLOOR_HZ:
            hz *= 2.0
    isr, m = sr * OVERSAMPLE, n * OVERSAMPLE

    if voice.kind == "noise":
        raw = noise(voice.setting, m, isr, voice.periodic, hz)
    else:
        raw = square(hz, m, isr, voice.bend, voice.bend_time)
    # Levelled now, before the attenuator, the fade or the velocity touch it.
    # Levelling at the end set every note to the same peak whatever it had
    # been asked for, which threw away velocity and made psg-soft exactly as
    # loud as psg.
    out = _down(raw, sr)
    peak = float(np.abs(out).max(initial=0.0))
    if peak > 0:
        out = out / peak * 0.62 * voice.gain
    out = out * attenuation(voice.volume)

    t = np.arange(n) / sr
    if voice.decay > 0:
        out = out * np.maximum(0.0, 1.0 - t / voice.decay) ** 1.5
    # Velocity is the attenuator, so it moves in two-decibel steps and not
    # smoothly: sixteen of them is every volume this machine has.
    if level < 1.0:
        step = int(np.clip(round(-20.0 * np.log10(max(level, 1e-3)) / 2.0), 0, 15))
        out = out * attenuation(step)
    edge = min(n, int(0.002 * sr))
    if edge > 1:
        out[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
    return out.astype(np.float32)


def pcm(y: np.ndarray, sr: int = SR, rate: float = 11025.0) -> np.ndarray:
    """A recording played the way this chip had to play one.

    No sample channel exists, so games wrote the volume register as fast as
    the processor could manage it. Four bits, and those four bits are two
    decibels apart rather than evenly spaced -- so quiet passages land on a
    handful of widely spaced levels and come out as a kind of gravel.
    """
    if len(y) == 0:
        return y.astype(np.float32)
    step = max(1, int(round(sr / rate)))
    held = np.repeat(y[::step], step)[: len(y)]
    levels = np.array([attenuation(s) for s in range(16)], dtype=np.float32)
    mag = np.abs(held)
    idx = np.argmin(np.abs(mag[:, None] - levels[None, :]), axis=1)
    return (np.sign(held) * levels[idx]).astype(np.float32)


VOICES: dict[str, Voice] = {
    "psg": Voice("psg", "a square, which is all this chip has", "tone"),
    "psg-soft": Voice("psg-soft", "the same square, further back", "tone", volume=3),
    "psg-bass": Voice("psg-bass", "a square down where the bass goes", "tone", volume=1),
    "psg-pluck": Voice("psg-pluck", "a square that fades as it goes", "tone", decay=0.45),
    "sms-kick": Voice("sms-kick", "a square dropped hard", "tone", decay=0.09,
                      bend=26.0, bend_time=0.013, base_hz=62.0),
    "sms-snare": Voice("sms-snare", "white noise, cut short", "noise", setting=2, decay=0.12),
    "sms-hat": Voice("sms-hat", "white noise, cut very short", "noise", setting=0, decay=0.035),
    "sms-tom": Voice("sms-tom", "a square dropped less hard", "tone", decay=0.16,
                     bend=14.0, bend_time=0.03, base_hz=110.0),
    "sms-cymbal": Voice("sms-cymbal", "white noise, left to ring", "noise",
                        setting=2, decay=0.45),
    "sms-perc": Voice("sms-perc", "the register's short mode, which has a pitch",
                      "noise", setting=1, periodic=True, decay=0.06),
}

_BY_ROLE = {"bass": "psg-bass", "lead": "psg", "chords": "psg-soft", "rhythm": "psg-soft"}
DRUMS = {"kick": "sms-kick", "snare": "sms-snare", "hats": "sms-hat",
         "toms": "sms-tom", "cymbals": "sms-cymbal", "perc": "sms-perc"}


def voice_for(role: str, program: int | None = None) -> Voice:
    """The channel to play a part with. There is not much to choose from."""
    head = (role or "").split(":")[0]
    if head == "drums":
        return VOICES[DRUMS.get((role or "").split(":")[-1], "sms-perc")]
    return VOICES[_BY_ROLE.get(head, "psg")]
