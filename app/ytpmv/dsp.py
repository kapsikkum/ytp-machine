"""Digital Signal Processing (DSP) routines for audio, filtering, and analysis.

Includes:
- ITU-R BS.1770 K-weighting using cascaded 2nd-order Direct Form II biquad filters.
- Zero-crossing snapping with slope matching.
- IIR low-pass and high-pass filters.
- 4x downsampling with anti-aliasing.
- Edge fading.
- Haas stereo widening.
- Spectral envelope liftering for formant preservation.
"""

from __future__ import annotations

import math
import numpy as np

try:
    import numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


if _HAVE_NUMBA:
    @numba.njit(fastmath=True)
    def _biquad_df2_core(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
        y = np.empty_like(x)
        w1 = 0.0
        w2 = 0.0
        for i in range(len(x)):
            w = x[i] - a1 * w1 - a2 * w2
            y[i] = b0 * w + b1 * w1 + b2 * w2
            # a decaying tail would sink into denormals, ~10x slower per sample
            if -1e-30 < w < 1e-30:
                w = 0.0
            w2 = w1
            w1 = w
        return y
else:
    def _biquad_df2_core(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
        y = np.empty_like(x)
        w1 = 0.0
        w2 = 0.0
        for i in range(len(x)):
            w = x[i] - a1 * w1 - a2 * w2
            y[i] = b0 * w + b1 * w1 + b2 * w2
            w2 = w1
            w1 = w
        return y


def biquad_df2(x: np.ndarray, b: tuple[float, float, float] | list[float] | np.ndarray,
                a: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    """Filter 1D or 2D signal *x* using a Direct Form II 2nd-order IIR biquad filter."""
    if len(x) == 0:
        return x.copy()
    a0 = float(a[0])
    b0, b1, b2 = float(b[0]) / a0, float(b[1]) / a0, float(b[2]) / a0
    a1, a2 = float(a[1]) / a0, float(a[2]) / a0

    if x.ndim == 1:
        x_contig = np.ascontiguousarray(x, dtype=np.float64)
        return _biquad_df2_core(x_contig, b0, b1, b2, a1, a2).astype(x.dtype)
    elif x.ndim == 2:
        out = np.empty_like(x, dtype=x.dtype)
        for ch in range(x.shape[1]):
            x_ch = np.ascontiguousarray(x[:, ch], dtype=np.float64)
            out[:, ch] = _biquad_df2_core(x_ch, b0, b1, b2, a1, a2).astype(x.dtype)
        return out
    else:
        raise ValueError("Audio array must be 1D or 2D")


def biquad_k_weight(x: np.ndarray, sr: int = 44100) -> np.ndarray:
    """ITU-R BS.1770 K-weighting using cascaded 2nd-order Direct Form II biquad IIR filters in the time domain.

    Stage 1: High shelf filter at 1681.97 Hz (+4 dB).
    Stage 2: High pass filter at 38.14 Hz.
    """
    if len(x) < 2:
        return x.copy()

    # Stage 1: High shelf filter (~ +4 dB above ~1.7 kHz)
    f0, gain, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    a_ = 10.0 ** (gain / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw, rt = math.cos(w0), 2.0 * math.sqrt(a_) * alpha
    b1 = (a_ * ((a_ + 1) + (a_ - 1) * cw + rt),
          -2 * a_ * ((a_ - 1) + (a_ + 1) * cw),
          a_ * ((a_ + 1) + (a_ - 1) * cw - rt))
    a1 = ((a_ + 1) - (a_ - 1) * cw + rt,
          2 * ((a_ - 1) - (a_ + 1) * cw),
          (a_ + 1) - (a_ - 1) * cw - rt)

    # Stage 2: High-pass filter (~38.1 Hz)
    f0, q = 38.13547087602444, 0.5003270373238773
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b2 = ((1 + cw) / 2.0, -(1 + cw), (1 + cw) / 2.0)
    a2 = (1 + alpha, -2 * cw, 1 - alpha)

    y1 = biquad_df2(x, b1, a1)
    y2 = biquad_df2(y1, b2, a2)
    return y2


def snap_to_zero_crossing(audio: np.ndarray, sample_idx: int, window: int = 64,
                          direction: str = "nearest") -> int:
    """Find closest zero-crossing sample with matching slope around sample_idx.

    Parameters:
    - audio: 1D or 2D audio array.
    - sample_idx: target sample index to snap from.
    - window: search radius in samples.
    - direction: "nearest", "left" ("before", "backward"), or "right" ("after", "forward").
    """
    if audio is None or len(audio) < 2:
        return sample_idx
    if audio.ndim > 1:
        audio = audio[:, 0]

    n = len(audio)
    target = int(max(0, min(n - 1, sample_idx)))

    # Determine local slope at target
    if target < n - 1 and target > 0:
        local_slope = float(audio[target + 1] - audio[target - 1])
    elif target < n - 1:
        local_slope = float(audio[target + 1] - audio[target])
    else:
        local_slope = float(audio[target] - audio[target - 1])

    target_slope_sign = 1 if local_slope > 0 else (-1 if local_slope < 0 else 0)

    # Determine search range
    dir_lower = direction.lower()
    if dir_lower in ("left", "before", "backward"):
        lo = max(0, target - window)
        hi = target
    elif dir_lower in ("right", "after", "forward"):
        lo = target
        hi = min(n - 1, target + window)
    else:  # "nearest"
        lo = max(0, target - window)
        hi = min(n - 1, target + window)

    if hi <= lo:
        return target

    # Find zero crossings in [lo, hi)
    # A zero crossing occurs between i and i+1 where audio[i] * audio[i+1] <= 0
    # and not both zero.
    best_matching_cand = None
    best_matching_dist = float("inf")
    best_any_cand = None
    best_any_dist = float("inf")

    for i in range(lo, hi):
        s0 = float(audio[i])
        s1 = float(audio[i + 1])
        if s0 == 0.0 and s1 == 0.0:
            continue
        if (s0 <= 0.0 and s1 >= 0.0) or (s0 >= 0.0 and s1 <= 0.0):
            # Choose sample with absolute value closer to zero
            cand = i if abs(s0) <= abs(s1) else i + 1
            # Check direction constraint on candidate
            if dir_lower in ("left", "before", "backward") and cand > target:
                continue
            if dir_lower in ("right", "after", "forward") and cand < target:
                continue

            dist = abs(cand - target)
            crossing_slope = s1 - s0
            crossing_slope_sign = 1 if crossing_slope > 0 else (-1 if crossing_slope < 0 else 0)

            if dist < best_any_dist:
                best_any_dist = dist
                best_any_cand = cand

            if target_slope_sign == 0 or crossing_slope_sign == target_slope_sign:
                if dist < best_matching_dist:
                    best_matching_dist = dist
                    best_matching_cand = cand

    if best_matching_cand is not None:
        return best_matching_cand
    if best_any_cand is not None:
        return best_any_cand
    return target


def lowpass_iir(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """2nd-order Butterworth low-pass IIR filter via Direct Form II biquad."""
    if len(x) < 2 or cutoff_hz <= 0 or cutoff_hz >= sr / 2:
        return x.copy()
    w0 = 2.0 * math.pi * cutoff_hz / sr
    alpha = math.sin(w0) / (2.0 * math.sqrt(2.0))
    cw = math.cos(w0)
    b0 = (1.0 - cw) / 2.0
    b1 = 1.0 - cw
    b2 = (1.0 - cw) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cw
    a2 = 1.0 - alpha
    return biquad_df2(x, (b0, b1, b2), (a0, a1, a2))


def highpass_iir(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """2nd-order Butterworth high-pass IIR filter via Direct Form II biquad."""
    if len(x) < 2 or cutoff_hz <= 0 or cutoff_hz >= sr / 2:
        return x.copy()
    w0 = 2.0 * math.pi * cutoff_hz / sr
    alpha = math.sin(w0) / (2.0 * math.sqrt(2.0))
    cw = math.cos(w0)
    b0 = (1.0 + cw) / 2.0
    b1 = -(1.0 + cw)
    b2 = (1.0 + cw) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cw
    a2 = 1.0 - alpha
    return biquad_df2(x, (b0, b1, b2), (a0, a1, a2))


def downsample_4x(x: np.ndarray) -> np.ndarray:
    """Downsample signal by a factor of 4 with anti-aliasing low-pass filtering."""
    if len(x) < 8:
        return x[::4].copy()
    # Normalized Nyquist cutoff at 0.2 of current sample rate (below downsampled Nyquist of 0.25)
    filtered = lowpass_iir(x, cutoff_hz=0.2 * 44100, sr=44100)
    return filtered[::4].copy()


def fade_edge(x: np.ndarray, fade_ms: float, sr: int) -> np.ndarray:
    """Apply linear fade-in and fade-out ramps to the edges of signal x."""
    if len(x) < 2 or fade_ms <= 0:
        return x.copy()
    fade_len = int(round(fade_ms * 0.001 * sr))
    fade_len = min(fade_len, len(x) // 2)
    if fade_len <= 0:
        return x.copy()
    out = x.copy()
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=out.dtype)
    if out.ndim == 1:
        out[:fade_len] *= ramp
        out[-fade_len:] *= ramp[::-1]
    elif out.ndim == 2:
        out[:fade_len, :] *= ramp[:, None]
        out[-fade_len:, :] *= ramp[::-1, None]
    return out


def haas_widen(audio: np.ndarray, delay_ms: float = 18.0, sr: int = 44100,
               side: str = "right") -> np.ndarray:
    """Apply Haas effect delay (10-30 ms) to widen stereo soundstage."""
    delay = int(round(delay_ms * 0.001 * sr))
    if delay <= 0:
        return audio.copy()
    if audio.ndim == 1:
        stereo = np.zeros((len(audio), 2), dtype=audio.dtype)
        stereo[:, 0] = audio
        if len(audio) > delay:
            stereo[delay:, 1] = audio[:-delay]
        return stereo
    out = audio.copy()
    if side == "right":
        if len(out) > delay:
            out[delay:, 1] = audio[:-delay, 1]
            out[:delay, 1] = 0.0
    else:
        if len(out) > delay:
            out[delay:, 0] = audio[:-delay, 0]
            out[:delay, 0] = 0.0
    return out


def spectral_envelope_lifter(g: np.ndarray, n_fft: int = 512, cutoff: int = 24) -> np.ndarray:
    """Extract smooth spectral envelope via cepstral liftering."""
    n = max(n_fft, 1 << int(math.ceil(math.log2(max(len(g), 64)))))
    spec = np.fft.rfft(g, n=n)
    mag = np.maximum(np.abs(spec), 1e-7)
    log_mag = np.log(mag)
    cep = np.fft.irfft(log_mag, n=n)
    lifter = np.zeros_like(cep)
    lifter[0] = 1.0
    cut = min(cutoff, len(lifter) // 2)
    lifter[1:cut] = 2.0
    env = np.exp(np.fft.rfft(cep * lifter, n=n).real)
    return env
