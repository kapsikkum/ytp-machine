"""The Commodore 64's sound chip, the SID: the 6581, and the 8580 that replaced it.

Three voices, and everything about them is more than the other chips here
have. Each can be a triangle, a sawtooth, a pulse of any width or noise; each
has a proper ADSR envelope; and all three go through a real analog filter,
low-, band- or high-pass with resonance. That filter is what makes a SID
sound like a SID -- and it is analog, so two chips of the same model do not
sound the same, and the two models sound very different.

  oscillator  a 24-bit accumulator per voice, stepped by a 16-bit frequency
              register at the chip's clock (985248 Hz on a PAL machine), so
              f = Fn x 985248 / 2^24. Sawtooth is its top twelve bits,
              triangle the same folded at the midpoint, pulse a comparison
              against a twelve-bit width
  noise       a 23-bit shift register, bit 0 = bit 22 XOR bit 17, clocked
              every time accumulator bit 19 rises, with eight of its bits
              wired out as the waveform
  envelope    attack climbs 0 to 255 in a straight line at one of sixteen
              rates, 2 ms to 8 s; decay and release fall in an
              approximately exponential curve made by stretching the step
              period at levels 0x5d, 0x36, 0x1a, 0x0e and 0x06; sustain is
              0x11 times a four-bit level
  DAC         a twelve-bit R-2R ladder. The 6581's is missing a
              termination resistor and its ratios are off (2R/R = 2.20), so
              it is not linear; the 8580's is correct (2.00)
  filter      cutoff from reSID's measured curves. The 6581 floors at
              220 Hz, climbs to 6 kHz, then drops back to 4.6 kHz as the
              register passes 0x7f -- a real discontinuity, not a mistake --
              and runs on to 18 kHz. The 8580 is close to linear to 12.5 kHz.
              Resonance is 1/Q = (15 - res)/8 on the 6581 and
              2^((4 - res)/8) on the 8580, from the same source
  the board   a 16 kHz low-pass and a 16 Hz high-pass on the C64's output

All from reSID (Dag Lem): the envelope rate periods and sustain levels, the
oscillator and shift register, the DAC ladder, and both cutoff curves. The
palette is Pepto's Colodore, as VICE ships it (see screen.py).

Liberties, declared: combined waveforms (reSID uses tables sampled off real
chips) are not modelled; the 6581's filter distortion and its per-chip
variation are not modelled, one curve standing for every 6581; the filter
is a clean state-variable one between those measured points; each part gets
its own filter rather than the three voices sharing one; and the envelope
bugs (the ADSR delay) are left out.

numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.ytpmv.pitch import SR

CLOCK = 985248.0                              # PAL
OVERSAMPLE = 4

# Clock cycles per envelope step, by four-bit rate. reSID.
RATE_PERIOD = (9, 32, 63, 95, 149, 220, 267, 313, 392, 977, 1954, 3126, 3907,
               11720, 19532, 31251)

# Where decay and release slow down, and by how much. reSID.
_EXP_STEPS = ((0x5D, 1), (0x36, 2), (0x1A, 4), (0x0E, 8), (0x06, 16), (0x00, 30))

# Measured cutoff curves, (register, Hz). reSID 0.16.
F0_6581 = ((0, 220), (128, 230), (256, 250), (384, 300), (512, 420), (640, 780),
           (768, 1600), (832, 2300), (896, 3200), (960, 4300), (992, 5000),
           (1008, 5400), (1016, 5700), (1023, 6000), (1024, 4600), (1032, 4800),
           (1056, 5300), (1088, 6000), (1120, 6600), (1152, 7200), (1280, 9500),
           (1408, 12000), (1536, 14500), (1664, 16000), (1792, 17100),
           (1920, 17700), (2047, 18000))
F0_8580 = ((0, 0), (128, 800), (256, 1600), (384, 2500), (512, 3300), (640, 4100),
           (768, 4800), (896, 5600), (1024, 6500), (1152, 7500), (1280, 8400),
           (1408, 9200), (1536, 9800), (1664, 10500), (1792, 11000),
           (1920, 11700), (2047, 12500))

_NOISE_BITS: np.ndarray | None = None
_DACS: dict[str, np.ndarray] = {}


def dac_table(model: str) -> np.ndarray:
    """The twelve-bit R-2R ladder, as reSID computes it for each chip."""
    if model in _DACS:
        return _DACS[model]
    bits = 12
    ratio, term = (2.20, False) if model == "6581" else (2.00, True)
    vbit = []
    for set_bit in range(bits):
        vn, r, r2 = 1.0, 1.0, ratio
        rn = r2 if term else math.inf
        for _ in range(set_bit):
            rn = r + r2 if rn == math.inf else r + r2 * rn / (r2 + rn)
        if rn == math.inf:
            rn = r2
        else:
            rn = r2 * rn / (r2 + rn)
            vn = vn * rn / r2
        for _ in range(set_bit + 1, bits):
            rn += r
            current = vn / rn
            rn = r2 * rn / (r2 + rn)
            vn = rn * current
        vbit.append(vn)
    codes = np.arange(1 << bits)
    table = sum(((codes >> j) & 1) * vbit[j] for j in range(bits))
    _DACS[model] = (table / table.max()).astype(np.float64)
    return _DACS[model]


def cutoff_hz(fc: float, model: str) -> float:
    """The filter's cutoff for an eleven-bit register value, off the measured curve."""
    pts = F0_6581 if model == "6581" else F0_8580
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    fc = float(np.clip(fc, 0, 2047))
    # The 6581's curve steps back down between 1023 and 1024; interp takes
    # the first match at 1023, so the step lands where the chip has it.
    return float(np.interp(fc, xs, ys))


def inv_q(res: int, model: str) -> float:
    res = int(np.clip(res, 0, 15))
    if model == "6581":
        # Its op-amps do not have the gain to go all the way; keep it finite.
        return max(0.12, (15 - res) / 8.0)
    return 2.0 ** ((4 - res) / 8.0)


def noise_bits() -> np.ndarray:
    """A long run of the shift register's bit 0 (bit 22 XOR bit 17).

    Generated seventeen bits at a time, since no bit depends on anything
    fewer than eighteen behind it -- which lets the whole run be made in a
    few thousand array operations instead of a million loop turns.
    """
    global _NOISE_BITS
    if _NOISE_BITS is not None:
        return _NOISE_BITS
    n = 1 << 20
    b = np.zeros(n + 23, dtype=np.uint8)
    b[:23] = [(0x7FFFF8 >> (22 - i)) & 1 for i in range(23)]   # reSID's reset value
    i = 23
    while i < len(b):
        j = min(i + 17, len(b))
        b[i:j] = b[i - 23:j - 23] ^ b[i - 18:j - 18]
        i = j
    _NOISE_BITS = b
    return b


def _noise_wave(steps: np.ndarray) -> np.ndarray:
    """Eight register bits wired out as a twelve-bit waveform, at each step."""
    b = noise_bits()
    k = (steps % (len(b) - 23)).astype(np.int64) + 23
    # Register bit n at step k is the bit generated n steps before it.
    taps = ((20, 11), (18, 10), (14, 9), (11, 8), (9, 7), (5, 6), (2, 5), (0, 4))
    out = np.zeros(len(k), dtype=np.int64)
    for reg_bit, out_bit in taps:
        out |= b[k - reg_bit].astype(np.int64) << out_bit
    return out


def envelope(n: int, sr: int, attack: int, decay: int, sustain: int, release: int,
             gate_samples: int, clock: float = CLOCK) -> np.ndarray:
    """The ADSR counter over time, 0..255, as reSID steps it."""
    t = np.arange(n) / sr * clock                                   # in clock cycles
    a_len = 255 * RATE_PERIOD[attack & 15]
    level = np.minimum(255.0, t / RATE_PERIOD[attack & 15])
    sus = 0x11 * (sustain & 15)

    def fall_times(start: int, stop: int, rate: int) -> np.ndarray:
        """Cumulative clock cycles to fall from *start* to each lower level."""
        per = []
        for lvl in range(start, stop, -1):
            mult = next(m for thresh, m in _EXP_STEPS if lvl > thresh) if lvl > 0 else 30
            per.append(RATE_PERIOD[rate & 15] * mult)
        return np.concatenate(([0.0], np.cumsum(per))) if per else np.array([0.0])

    dt = fall_times(255, sus, decay)
    since = np.maximum(0.0, t - a_len)
    held = 255 - np.searchsorted(dt, since, side="right") + 1
    held = np.maximum(held, sus)
    env = np.where(t < a_len, level, held).astype(np.float64)

    off = gate_samples / sr * clock
    if gate_samples < n:
        at = float(env[min(gate_samples, n - 1)])
        rt = fall_times(int(at), 0, release)
        since_off = np.maximum(0.0, t - off)
        rel = np.maximum(0, int(at) - np.searchsorted(rt, since_off, side="right") + 1)
        env = np.where(t < off, env, rel)
    return env


def _svf_response(freqs: np.ndarray, sr: float, fc: float, k: float, mode: str) -> np.ndarray:
    """The state-variable filter's frequency response, bilinear-transformed."""
    g = math.tan(math.pi * min(fc, 0.49 * sr) / sr)
    z = np.exp(1j * 2 * np.pi * freqs / sr)
    s = (z - 1) / (z + 1) / g
    den = s * s + k * s + 1
    parts = {"lp": 1 / den, "bp": s / den, "hp": s * s / den}
    return sum(parts[m] for m in mode.split("+"))


def _filter_static(y: np.ndarray, sr: float, fc: float, k: float, mode: str) -> np.ndarray:
    n = 1 << int(np.ceil(np.log2(len(y) + 8192)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(y, n) * _svf_response(f, sr, fc, k, mode), n)[: len(y)]


def _filter_sweep(y: np.ndarray, sr: float, fcs: np.ndarray, k: float, mode: str) -> np.ndarray:
    """A moving cutoff: a trapezoidal state-variable filter, a sample at a time.

    Stable at any cutoff below Nyquist, which matters -- the 6581 runs to
    18 kHz and the obvious digital filter goes unstable far below that.
    """
    out = np.empty(len(y))
    ic1 = ic2 = 0.0
    lp_on, bp_on, hp_on = "lp" in mode, "bp" in mode, "hp" in mode
    for i in range(len(y)):
        g = math.tan(math.pi * min(fcs[i], 0.49 * sr) / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = y[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        out[i] = (v2 if lp_on else 0.0) + (v1 if bp_on else 0.0) + \
                 ((y[i] - k * v1 - v2) if hp_on else 0.0)
    return out


def _board(y: np.ndarray, sr: int) -> np.ndarray:
    """The C64's output: 16 kHz low-pass, 16 Hz high-pass."""
    n = 1 << int(np.ceil(np.log2(len(y) + 8192)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    h = 1.0 / (1.0 + 1j * f / 16000.0) * (1j * f / 16.0) / (1.0 + 1j * f / 16.0)
    return np.fft.irfft(np.fft.rfft(y, n) * h, n)[: len(y)]


@dataclass(frozen=True)
class Voice:
    """One voice set up the way a C64 composer would: a waveform, an ADSR, a filter."""
    name: str
    about: str
    wave: str = "pulse"                 # tri | saw | pulse | noise
    pw: int = 0x800                     # pulse width, twelve bits
    pwm: tuple = ()                     # (rate Hz, depth) -- the pulse width swept, a SID staple
    adsr: tuple = (0, 9, 10, 6)
    filt: str = ""                      # "" for none, or lp / bp / hp / lp+bp ...
    fc: int = 1024
    fc_end: int | None = None           # a cutoff sweep over the note
    sweep_time: float = 0.3
    res: int = 8
    ring: float = 0.0                   # ring-modulate against a partner at this ratio
    drop: tuple = ()                    # (semitones, seconds): the pitch fall of a SID drum
    base_hz: float = 0.0
    gain: float = 1.0


def render_note(voice: Voice, hz: float, dur: float, sr: int = SR, level: float = 1.0,
                model: str = "6581", tail: float = 0.3) -> np.ndarray:
    """*voice* playing *hz* with the gate held for *dur* seconds."""
    hz = voice.base_hz or hz
    isr = sr * OVERSAMPLE
    gate = max(2, int(round(dur * isr)))
    n = gate + int(tail * isr)
    t = np.arange(n) / isr

    f = np.full(n, float(hz))
    if voice.drop:
        semis, secs = voice.drop
        f = f * 2.0 ** (semis * np.exp(-t / max(secs, 1e-4)) / 12.0)
    fn = np.clip(np.round(f * 2 ** 24 / CLOCK), 0, 65535)            # the 16-bit register
    inc = fn * CLOCK / isr                                           # accumulator per sample
    acc = np.cumsum(inc) % (1 << 24)

    top = (acc.astype(np.int64) >> 12) & 0xFFF
    if voice.wave == "saw":
        wave = top
    elif voice.wave == "tri":
        msb = (acc.astype(np.int64) >> 23) & 1
        folded = np.where(msb, ~acc.astype(np.int64), acc.astype(np.int64))
        if voice.ring:
            partner = np.cumsum(inc * voice.ring) % (1 << 24)
            flip = (partner.astype(np.int64) >> 23) & 1
            folded = np.where(flip, ~folded, folded)
        wave = (folded >> 11) & 0xFFF
    elif voice.wave == "noise":
        steps = np.floor(np.cumsum(inc) / (1 << 20)).astype(np.int64)   # bit 19 rising
        wave = _noise_wave(steps)
    else:
        pw = np.full(n, float(voice.pw))
        if voice.pwm:
            rate, depth = voice.pwm
            pw = pw + depth * np.sin(2 * np.pi * rate * t)
        wave = np.where(top >= np.clip(pw, 0, 4095), 0xFFF, 0)

    dac = dac_table(model)
    osc = dac[np.clip(wave, 0, 4095).astype(np.int64)] - 0.5
    env = envelope(n, isr, *voice.adsr, gate_samples=gate) / 255.0
    y = osc * env

    # Down to the output rate before the filter: the chip's filter is analog
    # and the oversampling was for the oscillators' edges.
    lp = np.fft.rfftfreq(1 << int(np.ceil(np.log2(n + 8192))), 1.0 / isr)
    y = np.fft.irfft(np.fft.rfft(y, len(lp) * 2 - 2) / (1.0 + (lp / (0.45 * sr)) ** 8),
                     len(lp) * 2 - 2)[:n][::OVERSAMPLE]
    m = len(y)

    if voice.filt:
        k = inv_q(voice.res, model)
        if voice.fc_end is None:
            y = _filter_static(y, sr, cutoff_hz(voice.fc, model), k, voice.filt)
        else:
            tt = np.arange(m) / sr
            reg = voice.fc_end + (voice.fc - voice.fc_end) * np.exp(-tt / max(voice.sweep_time, 1e-3))
            curve = np.array([cutoff_hz(r, model) for r in reg[::64]])
            fcs = np.repeat(curve, 64)[:m]
            y = _filter_sweep(y, sr, np.maximum(fcs, 20.0), k, voice.filt)

    y = _board(y, sr)
    y = y * level
    edge = min(m, int(0.003 * sr))
    if edge > 1:
        y[-edge:] *= np.linspace(1.0, 0.0, edge)
    return (y * 1.2 * voice.gain).astype(np.float32)


def digi(y: np.ndarray, sr: int = SR, rate: float = 8000.0, model: str = "6581") -> np.ndarray:
    """A recording played the way C64 games played speech.

    There was no sample channel. Writing the master volume register very
    fast moves the 6581's output, because a DC offset in its mixer leaks
    through the volume control -- so four bits of volume became four bits of
    PCM. The 8580 fixed the offset, and on it the same trick is barely
    audible, which is modelled: it comes out far quieter.
    """
    if len(y) == 0:
        return y.astype(np.float32)
    step = max(1, int(round(sr / rate)))
    held = np.repeat(y[::step], step)[: len(y)]
    peak = float(np.abs(held).max(initial=0.0)) or 1.0
    four = np.round((held / peak + 1.0) * 7.5) / 7.5 - 1.0
    out = _board(four * peak, sr)
    return (out * (1.0 if model == "6581" else 0.12)).astype(np.float32)


VOICES: dict[str, Voice] = {
    "sid-lead": Voice("sid-lead", "a pulse with its width swept, the C64 lead",
                      "pulse", pw=0x600, pwm=(4.5, 900), adsr=(0, 8, 11, 6)),
    "sid-bass": Voice("sid-bass", "a sawtooth through the filter as it closes",
                      "saw", adsr=(0, 7, 10, 5), filt="lp", fc=1500, fc_end=380,
                      sweep_time=0.18, res=11),
    "sid-pad": Voice("sid-pad", "a narrow pulse, softly attacked", "pulse", pw=0x300,
                     pwm=(0.8, 400), adsr=(6, 9, 12, 8), gain=0.9),
    "sid-rhythm": Voice("sid-rhythm", "a pulse cut short", "pulse", pw=0x400,
                        adsr=(0, 6, 4, 4)),
    # Ring-modulated against twice its own pitch, which keeps the note: the
    # chip's ring mod is a square-wave one, so the sidebands land on the
    # fundamental and its third. An inharmonic ratio sounds more like a bell
    # and is also out of tune with everything else in the song.
    "sid-bell": Voice("sid-bell", "a triangle ring-modulated, bell-like and in tune", "tri",
                      ring=2.0, adsr=(0, 10, 0, 9)),
    "sid-kick": Voice("sid-kick", "a triangle dropped fast", "tri", adsr=(0, 5, 0, 5),
                      drop=(28, 0.02), base_hz=55.0, gain=1.4),
    # Filter settings are register values, and one register means a very
    # different cutoff on each model -- 1600 is 16.8 kHz on a 6581 and 10.9
    # kHz on an 8580. Tunes written for one sounding wrong on the other is
    # real; these sit where both chips give something usable.
    "sid-snare": Voice("sid-snare", "noise, high-passed a little and cut short", "noise",
                       adsr=(0, 6, 0, 6), filt="hp+bp", fc=780, res=4, base_hz=2800.0),
    "sid-hat": Voice("sid-hat", "noise, high-passed hard", "noise", adsr=(0, 3, 0, 3),
                     filt="hp", fc=1060, res=2, base_hz=6000.0, gain=0.9),
    "sid-tom": Voice("sid-tom", "a triangle dropped less hard", "tri", adsr=(0, 7, 0, 7),
                     drop=(12, 0.04), base_hz=120.0, gain=1.2),
    "sid-cymbal": Voice("sid-cymbal", "noise, left to ring", "noise", adsr=(0, 10, 0, 10),
                        filt="hp", fc=1000, res=3, base_hz=5000.0, gain=0.9),
    "sid-perc": Voice("sid-perc", "a pulse ticked", "pulse", pw=0x200, adsr=(0, 2, 0, 2),
                      base_hz=900.0, gain=0.8),
}

_BY_ROLE = {"bass": "sid-bass", "lead": "sid-lead", "chords": "sid-pad", "rhythm": "sid-rhythm"}
DRUMS = {"kick": "sid-kick", "snare": "sid-snare", "hats": "sid-hat",
         "toms": "sid-tom", "cymbals": "sid-cymbal", "perc": "sid-perc"}


def voice_for(role: str, program: int | None = None) -> Voice:
    head = (role or "").split(":")[0]
    if head == "drums":
        return VOICES[DRUMS.get((role or "").split(":")[-1], "sid-perc")]
    if program is not None and 8 <= program <= 15 and head not in ("bass", "lead"):
        return VOICES["sid-bell"]
    return VOICES[_BY_ROLE.get(head, "sid-lead")]
