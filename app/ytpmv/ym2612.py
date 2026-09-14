"""The Mega Drive's sound chip, near enough to be worth the name.

The YM2612 is six channels of four-operator FM. An operator is a sine
oscillator with an envelope; what makes FM is that an operator's output is
added to another's *phase* rather than to its output, so one wobbles the
other's pitch thousands of times a second and the ear hears a new timbre
instead of vibrato. Eight fixed wirings (the algorithms) say which operator
feeds which, and operator 1 can be wired back into itself.

This is a model of the chip, not a cycle-exact emulator like Nuked-OPN2. It
keeps the parts that are audible:

  * the log-domain sine, quantised the way the chip's table is, and 14-bit
    operator outputs -- the graininess is the chip's, not rounding error
  * the real envelope generator: rates 0-31 rather than seconds, ticking at
    a third of the sample rate, key scaling (high notes decay faster),
    attack curved and decay straight in the attenuation domain
  * all eight algorithms, feedback on operator 1, detune, and the
    0.5x-15x frequency multipliers
  * 53267 samples a second, which is the NTSC master clock over 144
  * the DAC channel: eight-bit PCM at whatever rate the driver managed,
    with the dead zone around silence that gives the console its dirty low
    end -- games played their drums and their speech through this, and so
    does a corpus

and drops what is not: the timers, the SSG-EG modes, the LFO, and the fact
that turning the DAC on costs you channel 6.

Two things deliberately sit outside the chip because they sat outside it on
real hardware too. Slap bass's pitch envelope is a driver writing new
frequencies every few milliseconds, and velocity is a driver rewriting an
operator's level; the chip itself has neither.

numpy only. Everything is worked out for a whole note at once rather than a
sample at a time, which is the only way this is fast enough in Python; the
one place that genuinely cannot be, operator 1's feedback, is solved for its
steady state instead (see _feedback).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.ytpmv.pitch import SR

# NTSC master clock over 7, then over 144: what one FM sample costs.
MASTER_CLOCK = 53693175.0 / 7.0
FM_RATE = MASTER_CLOCK / 144.0                  # 53267.4 Hz
EG_RATE = FM_RATE / 3.0                         # the envelope ticks every third sample

# -log2(sin) in 1/256ths, over the first quarter of a cycle; the other three
# quarters are this mirrored and negated, as on the chip.
_SIN = np.minimum(4095.0, np.round(
    -np.log2(np.sin((np.arange(256) + 0.5) * np.pi / 512.0)) * 256.0)).astype(np.int32)

# Average attenuation the envelope moves per tick, per rate. The chip spreads
# a repeating 8-step pattern of 0s and 1s over a shift, which comes to these.
_AVG = (0.5, 0.625, 0.75, 0.875)

# Detune, in cents, for DT 1-3 (4-7 are the same sizes downwards). The real
# table widens with the note; this follows that without reproducing it.
_DETUNE = (0.0, 3.4, 6.8, 10.2)

# The dead band around zero in the output DAC, as a fraction of full scale:
# about one least-significant bit of nine, which is what the hardware's is.
LADDER_GAP = 0.003

# Which operators reach the output, per algorithm: what velocity has to touch.
_CARRIERS = {0: (3,), 1: (3,), 2: (3,), 3: (3,), 4: (1, 3),
             5: (1, 2, 3), 6: (1, 2, 3), 7: (0, 1, 2, 3)}


def _rate(r: int, keycode: int, ks: int) -> int:
    """A rate as the envelope generator sees it: doubled, plus key scaling.

    *ks* decides how much the note matters. At 3 the top of the keyboard
    decays eight times faster than the bottom, which is how one patch stays
    a bass at the bottom and turns into a pluck at the top.
    """
    if r <= 0:
        return 0
    return min(63, 2 * int(r) + (keycode >> (3 - ks)))


def _per_tick(r: int) -> float:
    """Attenuation units the envelope moves per tick at effective rate *r*."""
    return 0.0 if r <= 0 else min(14.0, _AVG[r & 3] * 2.0 ** ((r >> 2) - 11))


@dataclass(frozen=True)
class Op:
    """One operator. Rates are 0-31 (rr 0-15), levels are the chip's own.

    tl  total level, 0 loudest, 127 silent, 0.75 dB a step
    sl  where the first decay stops, 0-15 (15 is silence)
    mul frequency multiplier, 0 meaning a half
    dt  detune, 0-7 (4-7 detune downwards)
    ks  key scaling of the rates, 0-3
    """
    mul: float = 1.0
    dt: int = 0
    tl: int = 0
    ar: int = 31
    d1r: int = 0
    sl: int = 0
    d2r: int = 0
    rr: int = 7
    ks: int = 0

    def envelope(self, ticks: np.ndarray, keycode: int, off: float) -> np.ndarray:
        """Attenuation over *ticks*, in the chip's units (0 loud, 1023 silent).

        Attack is a curve because the chip multiplies away what is left of
        the attenuation; decay, sustain and release are straight lines in
        attenuation, which is an exponential fade to the ear. *off* is the
        tick the key is let go, after which the release rate takes over.
        """
        a = _per_tick(_rate(self.ar, keycode, self.ks)) / 16.0
        d1 = _per_tick(_rate(self.d1r, keycode, self.ks))
        d2 = _per_tick(_rate(self.d2r, keycode, self.ks))
        rr = _per_tick(_rate(self.rr * 2 + 1, keycode, self.ks))
        sl = 1023.0 if self.sl >= 15 else self.sl * 32.0

        def held(t: np.ndarray) -> np.ndarray:
            """Attenuation at tick *t* with the key still down."""
            t = np.asarray(t, dtype=float)
            if a <= 0:                               # a rate of 0 never arrives
                return np.full_like(t, 1023.0)
            rise = np.maximum(0.0, 1024.0 * (1.0 - a) ** t - 1.0)
            t_a = math.log(1.0 / 1024.0) / math.log(1.0 - a) if a < 1.0 else 0.0
            since = np.maximum(0.0, t - t_a)
            fall = np.minimum(sl, d1 * since)
            later = np.maximum(0.0, since - sl / d1) if d1 > 0 else np.zeros_like(since)
            return np.where(t < t_a, rise, np.minimum(1023.0, fall + d2 * later))

        env = held(ticks)
        if len(ticks) and off < ticks[-1]:
            at_off = float(held(np.array([off]))[0])
            env = np.where(ticks < off, env,
                           np.minimum(1023.0, at_off + rr * (ticks - off)))
        return np.minimum(1023.0, env + self.tl * 8.0)


@dataclass(frozen=True)
class Patch:
    """A voice: four operators, one of the eight wirings, and feedback.

    *bend* and *bend_time* are the driver's, not the chip's: a slide onto
    the note over a few milliseconds, which is how Mega Drive drivers played
    slap bass. The chip itself has no such thing.
    """
    name: str
    about: str
    alg: int
    fb: int
    ops: tuple[Op, Op, Op, Op]
    bend: float = 0.0
    bend_time: float = 0.008
    gain: float = 1.0
    # A drum is not playing a note, so it ignores the one it is handed and
    # sounds at its own pitch instead.
    base_hz: float = 0.0
    # Semitones to shift the written note by so the patch sounds where it is
    # meant to. A patch whose carrier runs at half the note sounds an octave
    # below it, and the driver is expected to have written the note an octave
    # up to suit; this does that instead, so a part stays in tune whatever it
    # is played through.
    transpose: int = 0


def _op(phase: np.ndarray, env: np.ndarray, mod: np.ndarray | None = None) -> np.ndarray:
    """One operator's output, 14-bit signed, given its phase in cycles.

    *mod* is another operator's output added to the phase. Full scale there
    is four whole cycles, which is the chip's deepest modulation.
    """
    idx = np.floor(phase * 1024.0).astype(np.int64)
    if mod is not None:
        idx = idx + (mod * 0.5).astype(np.int64)
    idx &= 0x3FF
    quarter = idx & 0xFF
    attenuation = _SIN[np.where(idx & 0x100, 0xFF - quarter, quarter)] + env * 4.0
    amp = np.exp2(-np.minimum(attenuation, 8191.0) / 256.0) * 8192.0
    return np.trunc(np.where(idx & 0x200, -amp, amp))


def _feedback(phase: np.ndarray, env: np.ndarray, fb: int) -> np.ndarray:
    """Operator 1, which on this chip can modulate itself.

    On the hardware each sample's phase is bent by the average of the two
    before it, which is a recurrence and so cannot be done for a whole note
    at once. What that recurrence settles at is the fixed point of
    y = sin(t + b*y), and it settles within a few samples, so this solves
    for it directly instead. The waveform is the one feedback is wanted
    for: a sine leaning further and further over towards a sawtooth as *fb*
    rises.
    """
    if fb <= 0:
        return _op(phase, env)
    beta = math.pi * 2.0 ** (fb - 7)
    depth = beta * np.exp2(-env * 4.0 / 256.0)       # its own envelope scales the feedback
    theta = 2.0 * np.pi * phase
    y = np.sin(theta)
    for _ in range(8):
        y = np.sin(theta + depth * y)
    return _op(phase + depth * y / (2.0 * np.pi), env)


def _voice(phase: list[np.ndarray], env: list[np.ndarray], alg: int, fb: int) -> np.ndarray:
    """The four operators wired up the way algorithm *alg* says."""
    o1 = _feedback(phase[0], env[0], fb)
    p, e = phase, env
    if alg == 0:                                     # 1-2-3-4 in a line
        return _op(p[3], e[3], _op(p[2], e[2], _op(p[1], e[1], o1)))
    if alg == 1:                                     # 1 and 2 together into 3, into 4
        return _op(p[3], e[3], _op(p[2], e[2], o1 + _op(p[1], e[1])))
    if alg == 2:                                     # 2 into 3; 1 and 3 into 4
        return _op(p[3], e[3], o1 + _op(p[2], e[2], _op(p[1], e[1])))
    if alg == 3:                                     # 1 into 2; 2 and 3 into 4
        return _op(p[3], e[3], _op(p[1], e[1], o1) + _op(p[2], e[2]))
    if alg == 4:                                     # two pairs, side by side
        return _op(p[1], e[1], o1) + _op(p[3], e[3], _op(p[2], e[2]))
    if alg == 5:                                     # 1 into all three others
        return _op(p[1], e[1], o1) + _op(p[2], e[2], o1) + _op(p[3], e[3], o1)
    if alg == 6:                                     # 1 into 2, and two spare
        return _op(p[1], e[1], o1) + _op(p[2], e[2]) + _op(p[3], e[3])
    return o1 + _op(p[1], e[1]) + _op(p[2], e[2]) + _op(p[3], e[3])      # 7: no FM at all


def _lowpass(y: np.ndarray, hz: float, sr: float) -> np.ndarray:
    if len(y) < 32 or hz >= sr / 2:
        return y
    n = 1 << int(np.ceil(np.log2(len(y) * 2)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(y, n) / (1.0 + (f / hz) ** 8), n)[: len(y)]


def _highpass(y: np.ndarray, hz: float, sr: float) -> np.ndarray:
    if len(y) < 32 or hz <= 0:
        return y
    n = 1 << int(np.ceil(np.log2(len(y) * 2)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(y, n) * (1.0 - 1.0 / (1.0 + (f / hz) ** 4)), n)[: len(y)]


def _to_sr(x: np.ndarray, src: float, dst: float) -> np.ndarray:
    if abs(src - dst) < 1e-6 or len(x) < 2:
        return x.astype(np.float32)
    if dst < src:
        x = _lowpass(x, 0.45 * dst, src)
    n = max(1, int(round(len(x) * dst / src)))
    return np.interp(np.arange(n) * src / dst, np.arange(len(x)), x).astype(np.float32)


# ── Drums off the sample channel ─────────────────────────────────────────────
#
# A snare, a hi-hat and a cymbal are noise, and FM cannot make noise. The
# patches that tried were measured, and none of them was noise at all: the
# snare's spectrum was a cluster of pure tones round 2.6 kHz, flat to three
# decimal places of nothing, and it played as a shrill ping with no body. Turning
# feedback up does not get past that on this chip -- which is why Mega Drive
# games did not try either. Sonic's snares came off the DAC as eight-bit
# samples, and so do these: a drum built the way a drum machine builds one,
# then played through the same eight-bit, thirteen-kilohertz channel as the
# voice, grain and aliasing included.
#
# Built once each, from a fixed seed, because a sample is the same recording
# every time it is played.

DAC_RATE = 13300.0
_SAMPLED: dict[tuple[str, int], np.ndarray] = {}


def _band_noise(n: int, sr: int, lo: float, hi: float, seed: int) -> np.ndarray:
    """White noise with everything outside lo..hi Hz taken out, gently."""
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    shape = 1.0 / (1.0 + (lo / np.maximum(fr, 1.0)) ** 4) / (1.0 + (fr / hi) ** 4)
    y = np.fft.irfft(spec * shape, n)
    return y / (np.abs(y).max() + 1e-12)


def _drum_sample(name: str, sr: int) -> np.ndarray:
    key = (name, sr)
    if key in _SAMPLED:
        return _SAMPLED[key]
    if name == "md-snare":
        n = int(0.30 * sr)
        t = np.arange(n) / sr
        # The shell: a tone dropping onto about 180 Hz, gone in a few tens of
        # milliseconds. It is what makes it a drum rather than a hiss.
        f = 180.0 + 110.0 * np.exp(-t / 0.010)
        body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.045)
        # The wires: bright noise with a crack at the front and a short tail.
        wires = _band_noise(n, sr, 1400.0, 9000.0, 7)
        wires *= 0.55 * np.exp(-t / 0.012) + 0.75 * np.exp(-t / 0.085)
        y = 0.62 * body + wires
    elif name == "md-hat":
        n = int(0.12 * sr)
        t = np.arange(n) / sr
        y = _band_noise(n, sr, 6500.0, 16000.0, 11) * np.exp(-t / 0.022)
    else:                                            # md-cymbal
        n = int(1.1 * sr)
        t = np.arange(n) / sr
        noise = _band_noise(n, sr, 3200.0, 15000.0, 13)
        # A little of the metal: a few inharmonic partials under the wash.
        ring = sum(np.sin(2 * np.pi * hz * t) for hz in (3150.0, 4730.0, 6410.0)) / 3.0
        y = (noise + 0.18 * ring) * (0.35 * np.exp(-t / 0.03) + 0.65 * np.exp(-t / 0.38))
    attack = min(n, int(0.0015 * sr))
    y[:attack] *= np.linspace(0.0, 1.0, attack)
    # Not quite full scale: the output stage's coupling overshoots a hard
    # transient by a few percent, and a sample at 1.0 came out clipping.
    y = (0.9 * y / (np.abs(y).max() + 1e-12)).astype(np.float32)
    _SAMPLED[key] = y
    return y


SAMPLED_DRUMS = frozenset({"md-snare", "md-hat", "md-cymbal"})


def _play_sample(name: str, dur: float, sr: int, level: float, tail: float) -> np.ndarray:
    """A drum off the DAC: the sample, cut short if the next hit comes first."""
    y = _drum_sample(name, sr)
    n = min(len(y), max(2, int(round((dur + max(0.0, tail)) * sr))))
    # Velocity scales the sample before it reaches eight bits, which is what a
    # driver's volume did -- so a quiet hit is a coarser one, as it was.
    out = dac(y[:n] * float(min(1.0, max(0.0, level))), DAC_RATE, sr).astype(np.float32)
    edge = min(len(out), int(0.004 * sr))
    if n < len(y) and edge > 1:
        out[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
    return out


def render_note(patch: Patch, hz: float, dur: float, sr: int = SR,
                level: float = 1.0, tail: float = 0.18) -> np.ndarray:
    """*patch* playing *hz*, held down for *dur* seconds, as the chip would.

    The key comes back up at *dur* and the release rates take over, so the
    sound runs on past that for up to *tail* longer. Without it every note
    would be cut off partway down its own decay, and a few hundred of those
    a minute is the rumble you hear rather than the notes.

    *level* is velocity, which the chip has no notion of: drivers did it by
    rewriting the carriers' levels, and so does this.

    The noise drums are not FM at all; see SAMPLED_DRUMS.
    """
    if patch.name in SAMPLED_DRUMS:
        return _play_sample(patch.name, dur, sr, level, tail)
    n = max(2, int(round((dur + max(0.0, tail)) * FM_RATE)))
    t = np.arange(n) / FM_RATE
    ticks = np.arange(n) / 3.0
    hz = float(patch.base_hz or hz) * 2.0 ** (patch.transpose / 12.0)
    midi = 12.0 * math.log2(max(hz, 1e-6) / 440.0) + 69.0
    keycode = int(np.clip(round(midi) // 3, 0, 31))

    # The note's own frequency, plus the driver's slide onto it if it has one.
    f = np.full(n, hz)
    if patch.bend:
        f = f * 2.0 ** (patch.bend * np.exp(-t / patch.bend_time) / 12.0)

    # Velocity, as a driver did it: quieter carriers, in the chip's 0.75 dB steps.
    quieter = 0.0 if level >= 1.0 else min(96.0, -20.0 * math.log10(max(level, 1e-3)) / 0.75)
    carriers = _CARRIERS[patch.alg]

    phase, env = [], []
    for i, op in enumerate(patch.ops):
        cents = _DETUNE[op.dt & 3] * (1.0 if op.dt < 4 else -1.0) * (1.0 + keycode / 32.0)
        mul = 0.5 if op.mul == 0 else float(op.mul)
        phase.append(np.cumsum(f * mul * 2.0 ** (cents / 1200.0)) / FM_RATE)
        e = op.envelope(ticks, keycode, dur * EG_RATE)
        env.append(np.minimum(1023.0, e + quieter * 8.0) if i in carriers else e)

    y = _to_sr(_voice(phase, env, patch.alg, patch.fb) / 8192.0 / len(carriers) * patch.gain,
               FM_RATE, sr)
    # A release rate of 0 never reaches silence, and the buffer has to end
    # somewhere; a couple of milliseconds of fade is cheaper than a click.
    # After the resampling, not before, or its filter smears the fade back
    # out again and the buffer ends on a step after all.
    y = output(y, sr)
    edge = min(len(y), int(0.003 * sr))
    if edge > 1:
        y[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
    return y


def hold(y: np.ndarray, rate: float, sr: int = SR) -> np.ndarray:
    """Output at *rate* samples a second, held flat between them.

    No filter first, on purpose: the aliasing that produces is half of why
    old hardware sounds like old hardware.
    """
    step = max(1, int(round(sr / rate)))
    if step == 1:
        return y.astype(np.float32)
    return np.repeat(y[::step], step)[: len(y)].astype(np.float32)


def output(y: np.ndarray, sr: int = SR, gap: float = LADDER_GAP) -> np.ndarray:
    """What comes out of the console, rather than out of the chip.

    Two things happen between the one and the other. The ladder, below; and
    a coupling capacitor, which is why the high-pass is here and is not
    optional. FM puts sidebands at the carrier minus a multiple of the
    modulator, so a modulator at the note's own frequency lands one of them
    on nothing at all -- real DC, and a heap of sub-bass either side of it,
    on notes written nowhere near that low. On hardware the capacitor eats
    it. Without one, eight parts of it pile up as a rumble under the song.
    """
    return _highpass(ladder(y, gap), 25.0, sr).astype(np.float32)


def ladder(y: np.ndarray, gap: float = LADDER_GAP) -> np.ndarray:
    """The dead zone the chip's output never quite crosses.

    Its DAC cannot reach silence: quiet samples sit either side of zero
    rather than on it, which is crossover distortion, and it is why this
    console's low end sounds dragged along the floor. It is a fault of the
    output stage, so it marks the FM channels as surely as the PCM one --
    the "ladder effect" people bought hardware revisions over.

    The size of the gap is the whole character of it, and it is easy to set
    far too wide: a gap is a fixed amount, so it is nothing much on a loud
    passage and everything on a quiet one. At four times life size this was
    lifting anything below -24 dB by six decibels and crunching every soft
    part of every song.
    """
    if gap <= 0 or len(y) == 0:
        return y.astype(np.float32)
    return np.where(np.abs(y) < 1e-6, y, y + np.sign(y) * gap).astype(np.float32)


def dac(y: np.ndarray, rate: float = 13300.0, sr: int = SR, gap: float = LADDER_GAP) -> np.ndarray:
    """The chip's PCM channel: a sample played the way the Mega Drive played one.

    Drums and speech on this console were not synthesised. The driver fed
    eight-bit samples to the DAC as fast as it could spare the time for --
    rarely better than about 13 kHz, and often a good deal worse -- and what
    came out was band-limited by nothing at all. The aliasing that produces
    is half of why old hardware sounds like old hardware.
    """
    if len(y) == 0:
        return y.astype(np.float32)
    return output(np.round(hold(y, rate, sr) * 128.0) / 128.0, sr, gap)


# ── Patches ───────────────────────────────────────────────────────────────────
#
# Written for this chip rather than ported from a DX7, because the two are not
# the same instrument. The advice for FM slap bass everywhere is that the
# metallic tick wants a modulator twenty-six to forty times the note -- but the
# YM2612's multiplier stops at fifteen, so a Mega Drive slap bass cannot be
# built that way and never was. Fifteen, decaying to nothing in a few
# milliseconds, is what the console had, and it is the sound people mean.

_MUTE = Op(tl=127, ar=31)

PATCHES: dict[str, Patch] = {
    # Sonic 2's own. Not a reconstruction by ear: these are the bytes the
    # game sets, read out of the disassembly's voice bank -- Hill Top's FM2
    # is the bass part and plays voice $00 and nothing else.
    #
    # Everything guessed about this patch beforehand was wrong, and wrong in
    # the same direction: it is not two operators in parallel with a very
    # high modulator for the tick, it is all four in a line (algorithm 0),
    # feedback of 1 rather than 7, and the only high ratio is at the *top*
    # of the chain, where its fast decay sweeps the brightness of everything
    # below it instead of just adding a click. The carrier runs at half the
    # note, so the sound is an octave under what is written -- hence the
    # transpose, which puts it back.
    #
    # Operators 2 and 3 are the easy ones to swap, and swapping them is not
    # subtle: in a chain it decides whether the *carrier* is modulated hard
    # (operator 3 at total level 19, an index of about 5 radians -- bright)
    # or barely at all (level 48, about 0.4 -- a near sine). SMPS numbers its
    # operators backwards from the hardware, so the macro's op1 is operator
    # 4; see _smps2asm_inc.asm, which emits DT4, DT2, DT3, DT1 into the
    # registers for slots 1, 3, 2, 4.
    "bass": Patch(
        "bass", "slap bass, Sonic 2's own", alg=0, fb=1,
        bend=0.0, gain=0.9, transpose=12,
        ops=(
            Op(mul=9, dt=0, tl=37, ar=31, d1r=18, sl=2, d2r=0, rr=15, ks=0),
            Op(mul=0, dt=7, tl=48, ar=31, d1r=14, sl=2, d2r=4, rr=15, ks=0),
            Op(mul=0, dt=3, tl=19, ar=31, d1r=10, sl=2, d2r=4, rr=15, ks=1),
            Op(mul=0, dt=0, tl=4, ar=31, d1r=10, sl=2, d2r=3, rr=15, ks=1),
        )),
    "lead": Patch(
        "lead", "a hard synth lead", alg=2, fb=6, gain=1.24,
        ops=(
            Op(mul=1, tl=22, ar=31, d1r=10, sl=2, d2r=3, rr=10, ks=1),
            Op(mul=2, tl=34, ar=31, d1r=14, sl=3, d2r=4, rr=10, ks=2),
            Op(mul=1, dt=2, tl=28, ar=31, d1r=12, sl=2, d2r=3, rr=10, ks=1),
            Op(mul=1, tl=8, ar=31, d1r=8, sl=1, d2r=2, rr=8, ks=1),
        )),
    "organ": Patch(
        # Four sines at octaves and fifths, no FM at all: drawbars, which is
        # what algorithm 7 is for and what most of Hill Top is made of.
        "organ", "drawbar organ, no FM at all", alg=7, fb=0, gain=4.43,
        ops=(
            Op(mul=1, tl=10, ar=31, d1r=0, rr=8),
            Op(mul=2, tl=16, ar=31, d1r=0, rr=8),
            Op(mul=4, tl=24, ar=31, d1r=0, rr=8, ks=1),
            Op(mul=8, tl=34, ar=31, d1r=0, rr=8, ks=2),
        )),
    "brass": Patch(
        "brass", "synth brass, with the swell", alg=2, fb=5, gain=1.48,
        ops=(
            Op(mul=1, tl=30, ar=24, d1r=8, sl=2, d2r=2, rr=8, ks=1),
            Op(mul=1, tl=36, ar=22, d1r=10, sl=3, d2r=2, rr=8, ks=1),
            Op(mul=1, dt=1, tl=32, ar=26, d1r=9, sl=2, d2r=2, rr=8, ks=1),
            Op(mul=1, tl=10, ar=28, d1r=6, sl=1, d2r=1, rr=8, ks=1),
        )),
    "bell": Patch(
        "bell", "a struck bell, ringing on", alg=5, fb=2, gain=3.95,
        ops=(
            Op(mul=3, tl=28, ar=31, d1r=16, sl=6, d2r=8, rr=10, ks=2),
            Op(mul=1, tl=12, ar=31, d1r=9, sl=8, d2r=5, rr=6, ks=1),
            Op(mul=5, tl=24, ar=31, d1r=13, sl=9, d2r=7, rr=6, ks=2),
            Op(mul=7, tl=32, ar=31, d1r=15, sl=10, d2r=9, rr=6, ks=3),
        )),
    "piano": Patch(
        "piano", "the electric piano everyone had", alg=5, fb=4, gain=3.12,
        ops=(
            Op(mul=14, tl=36, ar=31, d1r=26, sl=12, d2r=18, rr=12, ks=3),
            Op(mul=1, tl=12, ar=31, d1r=12, sl=6, d2r=6, rr=9, ks=2),
            Op(mul=1, dt=1, tl=18, ar=31, d1r=13, sl=6, d2r=6, rr=9, ks=2),
            Op(mul=2, tl=30, ar=31, d1r=18, sl=8, d2r=9, rr=9, ks=2),
        )),
    "strings": Patch(
        "strings", "a soft pad to sit behind things", alg=4, fb=2, gain=2.28,
        ops=(
            Op(mul=1, tl=38, ar=18, d1r=6, sl=2, d2r=1, rr=5, ks=1),
            Op(mul=1, tl=14, ar=20, d1r=5, sl=1, d2r=1, rr=4, ks=1),
            Op(mul=2, tl=42, ar=17, d1r=6, sl=2, d2r=1, rr=5, ks=1),
            Op(mul=1, dt=1, tl=16, ar=19, d1r=5, sl=1, d2r=1, rr=4, ks=1),
        )),
}


# Drums. The console normally sampled these and played them off the DAC, and
# that is still here as the "dac" tone -- but the chip can synthesise a kit
# and plenty of games did, so this is what the song-wide switch uses: with it
# on, everything goes through the chip and nothing is left as a recording.
#
# A kick is a sine dropped hard and fast. Everything above it is the same
# trick: maximum feedback on a high multiplier is as close to noise as four
# operators get, and then it is only a question of how long it rings.
PATCHES.update({
    "md-kick": Patch(
        "md-kick", "a sine dropped hard, the FM kick", alg=0, fb=0,
        base_hz=54.0, bend=30.0, bend_time=0.012, gain=0.77,
        ops=(Op(mul=1, tl=40, ar=31, d1r=26, sl=4, d2r=20, rr=12),
             Op(mul=1, tl=64, ar=31, d1r=31, sl=15, d2r=31, rr=15),
             Op(mul=1, tl=64, ar=31, d1r=31, sl=15, d2r=31, rr=15),
             Op(mul=1, tl=2, ar=31, d1r=18, sl=15, d2r=0, rr=10))),
    "md-snare": Patch(
        "md-snare", "a snare sample off the DAC, as Sonic played its drums", alg=4, fb=7,
        base_hz=196.0, gain=1.9,
        ops=(Op(mul=15, tl=22, ar=31, d1r=24, sl=15, d2r=0, rr=12),
             Op(mul=14, tl=10, ar=31, d1r=17, sl=15, d2r=0, rr=12),
             Op(mul=11, tl=26, ar=31, d1r=26, sl=15, d2r=0, rr=12),
             Op(mul=13, tl=14, ar=31, d1r=17, sl=15, d2r=0, rr=12))),
    "md-hat": Patch(
        "md-hat", "a hi-hat sample off the DAC, over almost before it starts", alg=4, fb=7,
        base_hz=880.0, gain=3.03,
        ops=(Op(mul=15, tl=20, ar=31, d1r=24, sl=15, d2r=0, rr=15),
             Op(mul=13, tl=16, ar=31, d1r=20, sl=15, d2r=0, rr=15),
             Op(mul=14, tl=24, ar=31, d1r=24, sl=15, d2r=0, rr=15),
             Op(mul=15, tl=18, ar=31, d1r=20, sl=15, d2r=0, rr=15))),
    "md-tom": Patch(
        "md-tom", "a kick with less of a drop, and tuned", alg=0, fb=1,
        base_hz=130.0, bend=14.0, bend_time=0.03, gain=1.06,
        ops=(Op(mul=2, tl=38, ar=31, d1r=22, sl=5, d2r=14, rr=12),
             Op(mul=1, tl=60, ar=31, d1r=28, sl=15, d2r=31, rr=15),
             Op(mul=1, tl=44, ar=31, d1r=20, sl=6, d2r=12, rr=12),
             Op(mul=1, tl=6, ar=31, d1r=16, sl=15, d2r=0, rr=10))),
    "md-cymbal": Patch(
        "md-cymbal", "a cymbal sample off the DAC, left to ring", alg=4, fb=7,
        base_hz=1100.0, gain=2.43,
        ops=(Op(mul=15, tl=24, ar=31, d1r=16, sl=15, d2r=0, rr=6),
             Op(mul=14, tl=14, ar=31, d1r=13, sl=15, d2r=0, rr=5),
             Op(mul=13, tl=28, ar=31, d1r=16, sl=15, d2r=0, rr=6),
             Op(mul=11, tl=18, ar=31, d1r=13, sl=15, d2r=0, rr=5))),
    "md-perc": Patch(
        "md-perc", "a tick and nothing after it", alg=4, fb=6,
        base_hz=420.0, gain=2.79,
        ops=(Op(mul=15, tl=26, ar=31, d1r=24, sl=15, d2r=0, rr=15),
             Op(mul=9, tl=16, ar=31, d1r=20, sl=15, d2r=0, rr=15),
             Op(mul=12, tl=30, ar=31, d1r=24, sl=15, d2r=0, rr=15),
             Op(mul=7, tl=20, ar=31, d1r=20, sl=15, d2r=0, rr=15))),
})

DRUMS = {"kick": "md-kick", "snare": "md-snare", "hats": "md-hat",
         "toms": "md-tom", "cymbals": "md-cymbal", "perc": "md-perc"}


# What each General MIDI family sounds most like on this chip. The role the
# song analysis worked out wins where there is one, since it knows what the
# part is *doing*; the program number only says what it was called.
_BY_ROLE = {"bass": "bass", "lead": "lead", "chords": "organ", "rhythm": "brass"}
_BY_PROGRAM = (
    (0, 7, "piano"), (8, 15, "bell"), (16, 23, "organ"), (24, 31, "lead"),
    (32, 39, "bass"), (40, 51, "strings"), (52, 55, "organ"), (56, 63, "brass"),
    (64, 79, "lead"), (80, 87, "lead"), (88, 95, "strings"), (96, 103, "bell"),
    (104, 111, "piano"), (112, 119, "bell"), (120, 127, "lead"),
)


def patch_for(role: str, program: int | None = None) -> Patch:
    """The chip voice to play a part with, from what the song analysis found."""
    head = (role or "").split(":")[0]
    if head == "drums":
        return PATCHES[DRUMS.get((role or "").split(":")[-1], "md-perc")]
    name = _BY_ROLE.get(head)
    if name is None and program is not None:
        for lo, hi, guess in _BY_PROGRAM:
            if lo <= program <= hi:
                name = guess
                break
    return PATCHES[name or "lead"]
