"""The PC sound card's FM chips: the OPL2 (YM3812) and the OPL3 (YMF262).

Two-operator FM -- half the operators of the Mega Drive's chip -- which
Yamaha made up for by letting each operator be something other than a sine.
It is the sound of the AdLib and the Sound Blaster, and of Doom: that game's
music was a MIDI file played through this chip with a bank of patches called
GENMIDI, and that is exactly how this plays a song too.

  operators   a log-sine table and an exponent table, both reproduced here
              by formula and checked against Nuked-OPL3's ROM dumps: they
              match all 512 entries
  waveforms   the OPL2 has four -- sine, half sine, absolute sine, and
              "pulse sine" -- and the OPL3 adds four more, among them a
              square and a sawtooth-ish ramp. Same patches, sharper edges
  envelope    four-bit rates whose timing is not derived here but measured:
              Nuked-OPL3's counter logic run a sample at a time for every
              rate. Attack rate 1 comes out at 2.84 s and decay rate 1 at
              about 42 s, against published figures of 2.83 and 39
  key scaling KSL from the chip's own 16-entry ROM, KSR from the block
  the DAC     where the two chips really part company. The OPL2 fed a
              YM3014, which takes samples as a ten-bit mantissa and a
              three-bit exponent -- fine detail on quiet sounds, coarse steps
              on loud ones. The OPL3's DAC is a plain sixteen bits

The patches are Freedoom's GENMIDI (BSD-3-Clause; see data/), built with
Freedoom's own tool: 128 General MIDI instruments and 47 drums, in the
format Doom read. A real Doom GENMIDI lump dropped in its place works too.

Liberties, declared: feedback, which the chip computes from the previous two
samples, is solved over the whole note by iteration rather than one sample
at a time; tremolo and vibrato use the chip's rates (3.7 Hz and 6.1 Hz) and
shallow depths; and each note is rendered on its own, so there is no voice
stealing when a song asks for more than nine at once.

numpy only.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass

import numpy as np

from app.ytpmv.pitch import SR

RATE = 49716.0                          # 14.31818 MHz / 288: what one sample costs

# The two ROMs, from their formulas -- which reproduce Nuked-OPL3's dumps exactly.
LOGSIN = np.array([round(-math.log2(math.sin((i + 0.5) * math.pi / 512)) * 256)
                   for i in range(256)], dtype=np.int64)
EXP = np.array([round(2 ** ((255 - i) / 256) * 1024) for i in range(256)], dtype=np.int64)

MULT = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 20, 24, 24, 30, 30)   # in halves
KSL_ROM = (0, 32, 40, 45, 48, 51, 53, 55, 56, 58, 59, 60, 61, 62, 63, 64)
KSL_SHIFT = (8, 1, 2, 0)

# Units the envelope falls per sample at each effective rate (0-63), and the
# samples an attack takes to arrive. Measured by running Nuked-OPL3's
# envelope counter literally; see the module notes.
DECAY_PER_SAMPLE = (
    0.0, 0.0, 0.0, 0.0, 0.000244, 0.000305, 0.000366, 0.000427, 0.000488, 0.00061,
    0.000732, 0.000854, 0.000977, 0.001221, 0.001465, 0.001709, 0.001953, 0.002441,
    0.00293, 0.003418, 0.003906, 0.004883, 0.005859, 0.006836, 0.007812, 0.009766,
    0.011719, 0.013672, 0.015625, 0.019531, 0.023438, 0.027344, 0.03125, 0.039062,
    0.046875, 0.054688, 0.0625, 0.078125, 0.09375, 0.109375, 0.125, 0.15625, 0.1875,
    0.21875, 0.249992, 0.312492, 0.374992, 0.437492, 0.5, 0.625008, 0.750008,
    0.875008, 1.0, 1.250015, 1.500015, 1.750015, 2.0, 2.500031, 3.000031, 3.500031,
    4.0, 4.0, 4.0, 4.0)
ATTACK_SAMPLES = (
    None, None, None, None, 141316, 112644, 94212, 79876, 70660, 56324, 47108, 39940,
    35332, 28164, 23556, 19972, 17668, 14084, 11780, 9988, 8836, 7044, 5892, 4996,
    4420, 3524, 2948, 2500, 2212, 1764, 1476, 1252, 1108, 884, 740, 628, 556, 444,
    372, 316, 280, 224, 188, 160, 142, 114, 96, 82, 70, 54, 46, 39, 35, 27, 22, 21,
    19, 12, 11, 9, 0, 0, 0, 0)

GENMIDI_PATH = os.path.join(os.path.dirname(__file__), "data", "genmidi.lmp")


@dataclass(frozen=True)
class Operator:
    """One operator's registers, as GENMIDI stores them."""
    am: bool
    vib: bool
    sustained: bool                    # EG type: hold at the sustain level
    ksr: bool
    mult: int
    ksl: int
    tl: int                            # total level, 0.75 dB a step
    ar: int
    dr: int
    sl: int
    rr: int
    wave: int

    @classmethod
    def from_bytes(cls, avek: int, ad: int, sr_: int, wave: int, ksl: int, tl: int):
        return cls(bool(avek & 0x80), bool(avek & 0x40), bool(avek & 0x20), bool(avek & 0x10),
                   avek & 0x0F, (ksl >> 6) & 3, tl & 0x3F, ad >> 4, ad & 0x0F,
                   sr_ >> 4, sr_ & 0x0F, wave & 0x07)


@dataclass(frozen=True)
class Voice:
    modulator: Operator
    carrier: Operator
    feedback: int
    additive: bool                     # connection bit: add the two, not FM
    note_offset: int


@dataclass(frozen=True)
class Instrument:
    name: str
    voices: tuple                      # one, or two when the patch doubles itself
    fixed_note: int | None             # drums play one pitch whatever they are sent
    fine_tune: int


_BANK: list[Instrument] | None = None


def bank() -> list[Instrument]:
    """The 175 GENMIDI instruments: GM programs 0-127, then drum keys 35-81."""
    global _BANK
    if _BANK is not None:
        return _BANK
    with open(GENMIDI_PATH, "rb") as fh:
        data = fh.read()
    if data[:8] != b"#OPL_II#" or len(data) < 8 + 175 * 68:
        raise ValueError("not a GENMIDI lump")
    out = []
    for i in range(175):
        raw = data[8 + i * 36: 8 + (i + 1) * 36]
        flags, fine, fixed = struct.unpack("<HBB", raw[:4])
        voices = []
        for v in range(2 if flags & 0x0004 else 1):
            f = struct.unpack("<BBBBBBBBBBBBBBh", raw[4 + v * 16: 4 + (v + 1) * 16])
            voices.append(Voice(Operator.from_bytes(*f[0:6]), Operator.from_bytes(*f[7:13]),
                                (f[6] >> 1) & 7, bool(f[6] & 1), f[14]))
        name = data[8 + 175 * 36 + i * 32: 8 + 175 * 36 + (i + 1) * 32].split(b"\0")[0]
        out.append(Instrument(name.decode("ascii", "replace"), tuple(voices),
                              fixed if flags & 0x0001 else None, fine))
    _BANK = out
    return out


def instrument_for(program: int | None, drum_key: int | None = None) -> Instrument:
    """The patch Doom would have used: by GM program, or by GM drum key."""
    b = bank()
    if drum_key is not None:
        return b[128 + int(np.clip(drum_key, 35, 81)) - 35]
    return b[int(np.clip(program or 0, 0, 127))]


def _wave(phase: np.ndarray, env: np.ndarray, kind: int) -> np.ndarray:
    """One of the eight waveforms, exactly as Nuked-OPL3 builds each from the ROMs."""
    p = phase & 0x3FF
    lo = p & 0xFF
    mirrored = np.where(p & 0x100, LOGSIN[lo ^ 0xFF], LOGSIN[lo])
    doubled = np.where(p & 0x80, LOGSIN[((p ^ 0xFF) << 1) & 0xFF], LOGSIN[(p << 1) & 0xFF])
    neg = np.zeros_like(p, dtype=bool)
    if kind == 0:
        out, neg = mirrored, (p & 0x200) != 0
    elif kind == 1:
        out = np.where(p & 0x200, 0x1000, mirrored)
    elif kind == 2:
        out = mirrored
    elif kind == 3:
        out = np.where(p & 0x100, 0x1000, LOGSIN[lo])
    elif kind == 4:
        out, neg = np.where(p & 0x200, 0x1000, doubled), (p & 0x300) == 0x100
    elif kind == 5:
        out = np.where(p & 0x200, 0x1000, doubled)
    elif kind == 6:
        out, neg = np.zeros_like(p), (p & 0x200) != 0
    else:
        neg = (p & 0x200) != 0
        out = np.where(neg, (p & 0x1FF) ^ 0x1FF, p) << 3
    level = out + (env.astype(np.int64) << 3)
    amp = (EXP[level & 0xFF] << 1) >> np.minimum(level >> 8, 31)
    return np.where(neg, -amp - 1, amp)


def _fnum_block(hz: float) -> tuple[int, int]:
    """The F-number and block the chip would be set to: most precise that fits."""
    for block in range(8):
        f = round(hz * 2 ** 20 / (RATE * 2 ** block))
        if f < 1024:
            return max(1, f), block
    return 1023, 7


def _envelope(op: Operator, n: int, off: int, ksv: int, ksl_units: int,
              extra_tl: float) -> np.ndarray:
    """Attenuation over the note, in the chip's nine-bit units."""
    ks = ksv >> (0 if op.ksr else 2)
    rate = lambda reg: 0 if reg == 0 else min(63, ks + (reg << 2))
    sl = (0x1F if op.sl == 15 else op.sl) << 4
    t = np.arange(n, dtype=np.float64)
    atk = ATTACK_SAMPLES[rate(op.ar)] if op.ar else None
    if atk is None:
        env = np.full(n, 511.0)
        t_on = math.inf
    elif atk == 0:
        env = np.zeros(n)
        t_on = 0.0
    else:
        f = 1.0 - 512.0 ** (-1.0 / atk)
        env = np.maximum(0.0, 512.0 * (1.0 - f) ** t - 1.0)
        t_on = float(atk)
    if math.isfinite(t_on):
        d = DECAY_PER_SAMPLE[rate(op.dr)]
        since = np.maximum(0.0, t - t_on)
        fall = np.minimum(float(sl), d * since)
        # Past the sustain level: held, unless the patch is percussive, in
        # which case it keeps falling at the release rate even with the key down.
        if not op.sustained and d > 0:
            later = np.maximum(0.0, since - sl / d)
            fall = fall + DECAY_PER_SAMPLE[rate(op.rr)] * later
        env = np.where(t < t_on, env, np.minimum(511.0, fall))
    if off < n:
        at_off = float(env[min(off, n - 1)])
        env = np.where(t < off, env,
                       np.minimum(511.0, at_off + DECAY_PER_SAMPLE[rate(op.rr)] * (t - off)))
    total = env + (op.tl << 2) + (ksl_units >> KSL_SHIFT[op.ksl]) + extra_tl
    return np.minimum(total, 511.0)


def _tremolo(n: int) -> np.ndarray:
    """The chip's AM: a 210-step triangle every 64 samples, shallow depth."""
    pos = (np.arange(n) // 64) % 210
    return np.where(pos < 105, pos, 210 - pos) >> 4


def _render_voice(voice: Voice, note: float, n: int, off: int, level_units: float,
                  opl3: bool) -> np.ndarray:
    hz = 440.0 * 2 ** ((note - 69) / 12.0)                 # by now a real MIDI note
    fnum, block = _fnum_block(hz)
    ksv = (block << 1) | ((fnum >> 9) & 1)
    ksl_units = max(0, (KSL_ROM[fnum >> 6] << 2) - ((8 - block) << 5))
    # Vibrato: the chip nudges the F-number in an eight-step pattern at 6.1 Hz.
    vibpos = (np.arange(n) // 1024) & 7
    rng = ((fnum >> 7) & 7) >> 1
    vib = np.select([vibpos % 4 == 0, vibpos % 2 == 1, vibpos % 4 == 2],
                    [0, rng >> 1, rng], 0) * np.where(vibpos >= 4, -1, 1)

    def phase(op: Operator) -> np.ndarray:
        f = fnum + (vib if op.vib else 0)
        inc = ((np.asarray(f, dtype=np.int64) << block) >> 1) * MULT[op.mult] >> 1
        acc = np.cumsum(np.broadcast_to(inc, (n,)).astype(np.int64))
        return (acc >> 9).astype(np.int64)

    wave = (lambda w: w) if opl3 else (lambda w: w & 3)
    trem = _tremolo(n)
    m, c = voice.modulator, voice.carrier
    m_env = _envelope(m, n, off, ksv, ksl_units, level_units if voice.additive else 0.0)
    c_env = _envelope(c, n, off, ksv, ksl_units, level_units)
    if m.am:
        m_env = np.minimum(511.0, m_env + trem)
    if c.am:
        c_env = np.minimum(511.0, c_env + trem)

    pm = phase(m)
    mod = _wave(pm, m_env, wave(m.wave))
    if voice.feedback:
        # The chip bends each sample by the two before it. Solved over the whole
        # note by repeated passes, each reading the last pass's previous samples.
        for _ in range(6):
            prev = np.concatenate(([0], mod[:-1]))
            prev2 = np.concatenate(([0, 0], mod[:-2]))
            mod = _wave(pm + ((prev + prev2) >> (9 - voice.feedback)), m_env, wave(m.wave))
    pc = phase(c)
    if voice.additive:
        out = mod + _wave(pc, c_env, wave(c.wave))
    else:
        out = _wave(pc + mod, c_env, wave(c.wave))
    return out.astype(np.float64)


def ym3014(y: np.ndarray) -> np.ndarray:
    """The OPL2's DAC: a ten-bit mantissa and a three-bit exponent.

    Its resolution grows with the level, so loud passages move in coarse
    steps and quiet ones in fine -- the opposite of a plain DAC, and the grain
    you hear on an AdLib and not on a later card.
    """
    s = np.round(np.clip(y, -1.0, 1.0) * 32767.0)
    mag = np.abs(s)
    e = np.clip(np.floor(np.log2(np.maximum(mag, 1.0))) - 8, 0, 7)
    step = 2.0 ** e
    return (np.round(s / step) * step / 32767.0).astype(np.float32)


def _lowpass(y: np.ndarray, hz: float, sr: float) -> np.ndarray:
    if len(y) < 32 or hz >= sr / 2:
        return y
    n = 1 << int(np.ceil(np.log2(len(y) + 4096)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(y, n) / (1.0 + (f / hz) ** 8), n)[: len(y)]


def _highpass(y: np.ndarray, hz: float, sr: float) -> np.ndarray:
    if len(y) < 32:
        return y
    n = 1 << int(np.ceil(np.log2(len(y) + 8192)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(y, n) * (1j * f / hz) / (1.0 + 1j * f / hz), n)[: len(y)]


def render_note(instrument: Instrument, midi_note: float, dur: float, sr: int = SR,
                level: float = 1.0, opl3: bool = False, tail: float = 0.25) -> np.ndarray:
    """*instrument* playing *midi_note* held for *dur* seconds, as the chip would."""
    n_key = max(2, int(round(dur * RATE)))
    n = n_key + int(tail * RATE)
    # Velocity as a driver does it: more attenuation on the carrier.
    level_units = min(511.0, -20.0 * math.log10(max(level, 1e-3)) / 0.1875)
    out = np.zeros(n)
    for i, voice in enumerate(instrument.voices):
        if instrument.fixed_note is not None:
            note = instrument.fixed_note
        else:
            # Doom's music driver numbers its notes an octave below MIDI: its
            # note 57 is 440 Hz, where MIDI's is 69 (worked out from DMX's own
            # frequency table, as chocolate-doom reproduces it). GENMIDI's note
            # offsets were written against that, which is why its bass patches
            # carry a -12. Treating the notes as plain MIDI instead put every
            # part an octave low and E1M1's bass line at 20 Hz, under hearing.
            note = midi_note + 12 + voice.note_offset
            while note < 12:                       # DMX's own wrap, into range
                note += 12
            while note > 107:
                note -= 12
        if i == 1:                                   # the doubled voice, detuned as DMX does
            note += ((instrument.fine_tune / 2.0) - 64.0) / 32.0
        out += _render_voice(voice, note, n, n_key, level_units, opl3)
    y = out / 4084.0 / len(instrument.voices)
    # The card's output is AC-coupled. Without that, the waveforms that never
    # go negative -- half and absolute sine, which the OPL2 falls back on when
    # a patch asks for a waveform it lacks -- leave a DC offset in the mix,
    # and a song full of them rumbles.
    y = _highpass(y, 20.0, RATE)
    y = _lowpass(y, 0.45 * sr, RATE)
    m = max(2, int(round(n * sr / RATE)))
    y = np.interp(np.arange(m) * RATE / sr, np.arange(n), y).astype(np.float32)
    if not opl3:
        y = ym3014(y)
    edge = min(m, int(0.003 * sr))
    if edge > 1:
        y[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
    return y * np.float32(0.9)


def pcm(y: np.ndarray, sr: int = SR, sixteen: bool = False) -> np.ndarray:
    """The Sound Blaster's own DAC, which is where a game's speech went.

    The original card: eight bits at 11025 Hz, which is what Doom's sound
    effects were. The OPL3's era had sixteen-bit cards, played here at 22050.
    """
    if len(y) == 0:
        return y.astype(np.float32)
    rate = 22050.0 if sixteen else 11025.0
    step = max(1, int(round(sr / rate)))
    held = np.repeat(y[::step], step)[: len(y)]
    bits = 32768.0 if sixteen else 128.0
    return (np.round(held * bits) / bits).astype(np.float32)
