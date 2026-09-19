"""Pitch perfect: measure what note a clip is sung on, then sing it on another.

Speech does not sit on a note. It glides through several in one syllable, so
shifting a word by its *average* pitch leaves it wandering either side of the
note it is meant to be -- the word is roughly in key and audibly out of tune.
What a YTPMV wants is every voiced moment of the clip landing exactly on the
note, which is what this does:

  track_f0     YIN, frame by frame, so the wobble is measured and not averaged
  pitch_marks  one mark per glottal pulse, following that wobble
  Voice.render TD-PSOLA: re-lays those pulses at exactly the note's period

PSOLA moves pulses rather than resampling, so the vocal tract -- the formants
that make it sound like *this* speaker saying *this* vowel -- stays where it
was. Resampling (the `tape` mode) is kept too: it is the classic chipmunk
YTPMV sound, and some parts want it.

numpy only. Nothing here knows about the corpus or about MIDI.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass

import numpy as np

SR = 44100

_FRAME = 0.040          # YIN window: two periods of the lowest pitch tracked
_HOP = 0.005
_FMIN, _FMAX = 60.0, 600.0
_YIN_THRESH = 0.15      # first dip below this is the period (de Cheveigné's 0.1-0.15)
_APERIODIC = 0.35       # above this at the chosen period, the frame is not voiced
_SILENCE = 0.05         # frames quieter than this fraction of the loudest are silence
_UNVOICED_STEP = 0.010  # hop for copying consonants through untouched
_MAX_STRETCH = 4.0      # a voiced stretch is never drawn out further than this
_FADE_IN = 0.002
_FADE_OUT = 0.010
_MIN_NOTE = 0.030

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * math.log2(f / 440.0)


def note_name(m: float) -> str:
    n = int(round(m))
    return f"{_NOTE_NAMES[n % 12]}{n // 12 - 1}"


def decode_audio(path: str, start: float | None = None,
                 duration: float | None = None, sr: int = SR,
                 pcm: str = "f32le") -> np.ndarray:
    """Mono float32 audio from anything ffmpeg can read.

    pcm="s16le" decodes through 16-bit, as a wav would: ffmpeg downmixes
    stereo 3 dB quieter that way than to float, and levels measured against
    a fixed threshold were tuned on it.
    """
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{max(start, 0.0):.4f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.4f}"]
    cmd += ["-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", pcm, "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg failed: " + proc.stderr.decode(errors="replace")[-400:])
    if pcm == "s16le":
        raw = proc.stdout[: len(proc.stdout) // 2 * 2]
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def loudness(path: str, start: float, duration: float, win: int,
             sr: int = 16000) -> np.ndarray:
    """RMS of each whole *win*-sample window of a slice, on the int16 scale.

    Piped rather than written to a temp file: the readers this replaced named
    theirs by process id, and the generation worker and the editor's request
    threads share one process, so they read each other's audio.
    """
    return rms_windows(decode_audio(path, start, duration, sr, "s16le"), win)


def rms_windows(x: np.ndarray, win: int) -> np.ndarray:
    """RMS of each whole *win*-sample window of *x*, on the int16 scale."""
    x = x.astype(np.float64) * 32768.0
    n = max(0, (len(x) - 1) // win)          # the windows range(0, len-win, win) gave
    return np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(axis=1))


# ── Tracking ──────────────────────────────────────────────────────────────────

@dataclass
class F0Track:
    times: np.ndarray     # frame centres, seconds
    f0: np.ndarray        # Hz; meaningful only where voiced
    voiced: np.ndarray    # bool
    rms: np.ndarray
    hop: float

    def frame_at(self, t: float) -> int:
        if len(self.times) == 0:
            return 0
        i = int(round((t - self.times[0]) / self.hop))
        return min(max(i, 0), len(self.times) - 1)


def _median_filter(a: np.ndarray, width: int) -> np.ndarray:
    if len(a) < width:
        return a.copy()
    pad = width // 2
    p = np.pad(a, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, width), axis=1)


def _yin_block(frames: np.ndarray, W: int, tmin: int, tmax: int, sr: int):
    n, L = frames.shape
    nfft = 1 << int(math.ceil(math.log2(L + W)))
    A = np.fft.rfft(frames[:, :W], nfft)
    B = np.fft.rfft(frames, nfft)
    # acf[k] = sum_j a[j] * s[j+k]; nfft >= L+W so nothing wraps for k <= tmax
    acf = np.fft.irfft(np.conj(A) * B, nfft)[:, :tmax + 1]
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
    taus = np.arange(tmax + 1)
    e0 = cs[:, W]
    etau = cs[:, taus + W] - cs[:, taus]
    d = np.maximum(e0[:, None] + etau - 2.0 * acf, 0.0)
    d[:, 0] = 0.0
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(np.cumsum(d[:, 1:], axis=1), 1e-12)

    sub = cmnd[:, tmin:tmax]
    below = sub < _YIN_THRESH
    has = below.any(axis=1)
    first = np.where(has, below.argmax(axis=1), sub.argmin(axis=1))
    tau = np.empty(n, dtype=np.int64)
    for i in range(n):
        t = int(first[i])
        if has[i]:                       # walk down to the bottom of the dip
            while t + 1 < sub.shape[1] and sub[i, t + 1] < sub[i, t]:
                t += 1
        tau[i] = t + tmin
    r = np.arange(n)
    # YIN picks the period; the fine position within a sample comes from a
    # normalised correlation instead. Speech is always getting louder or
    # quieter, and a fading frame makes the plain difference smaller at longer
    # lags -- so YIN's own minimum read every decaying note ~10 cents flat,
    # even a note whose pulses were spaced exactly. acf/sqrt(e0*etau) is 1 at
    # the true period whatever the envelope does, so its peak is not dragged.
    ncc = acf / np.sqrt(np.maximum(e0[:, None] * etau, 1e-20))
    aperiodicity = cmnd[r, tau]
    # The difference's minimum can sit a sample or two off the correlation's
    # peak; step to the peak before refining.
    near = np.clip(tau[:, None] + np.arange(-2, 3)[None, :], tmin, tmax - 1)
    tau = near[r, np.argmax(ncc[r[:, None], near], axis=1)]
    a = ncc[r, tau - 1]
    b = ncc[r, tau]
    c = ncc[r, np.minimum(tau + 1, tmax)]
    den = a - 2.0 * b + c
    shift = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    shift = np.clip(shift, -1.0, 1.0)
    return sr / (tau + shift), aperiodicity, np.sqrt(e0 / W)


def track_f0(x: np.ndarray, sr: int = SR, fmin: float = _FMIN,
             fmax: float = _FMAX, aperiodic: float = _APERIODIC) -> F0Track:
    """Frame-by-frame pitch by YIN, with a voiced/unvoiced decision."""
    W = int(_FRAME * sr)
    hop = int(_HOP * sr)
    tmin = max(2, int(sr / fmax))
    tmax = int(math.ceil(sr / fmin))
    L = W + tmax
    x = np.asarray(x, dtype=np.float64)
    if len(x) < L:
        x = np.pad(x, (0, L - len(x)))
    n = 1 + (len(x) - L) // hop

    f0s, aps, rmss = [], [], []
    block = 400                       # frames at a time keeps memory flat on long input
    for s in range(0, n, block):
        k = min(block, n - s)
        idx = np.arange(L)[None, :] + hop * (s + np.arange(k))[:, None]
        f, ap, rms = _yin_block(x[idx], W, tmin, tmax, sr)
        f0s.append(f); aps.append(ap); rmss.append(rms)
    f0 = np.concatenate(f0s)
    ap = np.concatenate(aps)
    rms = np.concatenate(rmss)

    loud = rms > max(1e-4, _SILENCE * float(rms.max(initial=0.0)))
    voiced = loud & (ap < aperiodic)

    # A voiced run shorter than 3 frames is a click YIN happened to like.
    v = voiced.copy()
    i = 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            if j - i < 3:
                v[i:j] = False
            i = j
        else:
            i += 1
    voiced = v

    if voiced.any():
        # Octave errors are YIN's one habit: fold them back towards the median,
        # then smooth what is left so a single bad frame cannot jerk the marks.
        med = float(np.median(f0[voiced]))
        f0 = np.where(f0 > 1.8 * med, f0 / 2.0, f0)
        f0 = np.where(f0 < 0.55 * med, f0 * 2.0, f0)
        sm = f0.copy()
        sm[voiced] = _median_filter(f0[voiced], 5)
        f0 = sm

    times = (np.arange(n) * hop + W / 2.0) / sr
    return F0Track(times=times, f0=f0, voiced=voiced, rms=rms, hop=_HOP)


@dataclass
class PitchInfo:
    f0: float | None          # median pitch of the voiced frames, Hz
    midi: float | None
    note: str | None
    voiced_ratio: float       # voiced frames / audible frames
    stability: float          # spread of the pitch, cents (std); lower is steadier
    duration: float

    def as_dict(self) -> dict:
        return {
            "f0_hz": round(self.f0, 2) if self.f0 else None,
            "midi": round(self.midi, 2) if self.midi is not None else None,
            "note": self.note,
            "voiced_ratio": round(self.voiced_ratio, 3),
            # None, not inf, for a sound with no pitch: JSON has no infinity,
            # and every hiss-cut drum hit 500'd on the way out.
            "stability_cents": round(self.stability, 1) if math.isfinite(self.stability) else None,
            "duration": round(self.duration, 3),
        }


def describe(track: F0Track, duration: float) -> PitchInfo:
    loud = track.rms > max(1e-4, _SILENCE * float(track.rms.max(initial=0.0)))
    nv = int(track.voiced.sum())
    ratio = nv / max(int(loud.sum()), 1)
    if nv == 0:
        return PitchInfo(None, None, None, 0.0, float("inf"), duration)
    f = track.f0[track.voiced]
    med = float(np.median(f))
    cents = 1200.0 * np.log2(f / med)
    m = hz_to_midi(med)
    return PitchInfo(med, m, note_name(m), float(min(ratio, 1.0)),
                     float(np.std(cents)), duration)


def detect_pitch(x: np.ndarray, sr: int = SR) -> PitchInfo:
    return describe(track_f0(x, sr), len(x) / sr)


# ── Marks and re-synthesis ────────────────────────────────────────────────────

def _match(x: np.ndarray, prev: float, P: float) -> float:
    """Where the period after the one at *prev* lines up best with it."""
    n = len(x)
    h = max(2, int(P / 2))
    r = max(1, int(0.2 * P))
    p0 = int(round(prev))
    guess = int(round(prev + P))
    if p0 - h < 0 or guess + r + h > n:
        return prev + P
    ref = x[p0 - h:p0 + h].astype(np.float64)
    seg = x[guess - r - h:guess + r + h].astype(np.float64)
    dots = np.correlate(seg, ref, mode="valid")          # one per candidate lag
    e = np.concatenate([[0.0], np.cumsum(seg ** 2)])
    energy = e[2 * h:] - e[:-2 * h]
    ncc = dots / np.sqrt(np.maximum(energy * float(ref @ ref), 1e-20))
    k = int(np.argmax(ncc))
    shift = 0.0
    if 0 < k < len(ncc) - 1:
        a, b, c = ncc[k - 1], ncc[k], ncc[k + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            shift = float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))
    return float(guess - r + k) + shift


def pitch_marks(x: np.ndarray, track: F0Track, sr: int = SR):
    """One mark per pitch period through voiced stretches, a fixed hop elsewhere.

    Returns (marks, periods, voiced) as arrays: sample positions, the local
    period in samples, and whether each mark sits in voiced sound.
    """
    n = len(x)
    marks: list[float] = []
    periods: list[float] = []
    voiced: list[bool] = []
    unv = int(_UNVOICED_STEP * sr)
    pos = 0.0
    while pos < n:
        i = track.frame_at(pos / sr)
        if track.voiced[i] and track.f0[i] > 0:
            P = sr / float(track.f0[i])
            if voiced and voiced[-1] and pos - marks[-1] < 1.5 * P:
                # Put the mark where this period looks most like the last one.
                #
                # Snapping every mark to the biggest peak is the textbook
                # shortcut, and it put real speech 10-15 cents flat: as the
                # formants move through a vowel, the biggest peak slides
                # against the pulse that made it, a sample or two a period.
                # Laid down again at exactly the note's spacing, that slide
                # *is* a pitch -- each grain carries its offset with it.
                # Matching the waveform instead keeps every mark on the same
                # point of its pulse, so neighbouring grains are alike and the
                # only periodicity left in the output is the spacing it is
                # given. (WSOLA's trick, used here to place PSOLA's marks.)
                m = _match(x, marks[-1], P)
            else:
                a, b = int(pos), min(int(pos + P) + 1, n)
                if b <= a:
                    break
                m = float(a + int(np.argmax(x[a:b])))
            if marks and m <= marks[-1]:
                m = marks[-1] + 1.0
            marks.append(m); periods.append(P); voiced.append(True)
            pos = m + P
        else:
            marks.append(pos); periods.append(float(unv)); voiced.append(False)
            pos += unv
    return (np.rint(np.asarray(marks)).astype(np.int64), np.asarray(periods),
            np.asarray(voiced, dtype=bool))


@dataclass
class Warp:
    """Where in the sample a moment of the rendered note comes from.

    Shared by audio and picture, so a stretched vowel holds the mouth open for
    exactly as long as it is heard.
    """
    v0: float = 0.0           # voiced stretch in the sample, seconds
    v1: float = 0.0
    s: float = 1.0            # how much that stretch is drawn out
    rate: float = 1.0         # tape speed; when not 1, it replaces the rest

    def src(self, t: float) -> float:
        if self.rate != 1.0:
            return t * self.rate
        if self.s == 1.0 or t < self.v0:
            return t
        span = (self.v1 - self.v0) * self.s
        if t < self.v0 + span:
            return self.v0 + (t - self.v0) / self.s
        return self.v1 + (t - self.v0 - span)


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _grain(x: np.ndarray, centre: int, half: int) -> np.ndarray:
    a, b = centre - half, centre + half
    if a >= 0 and b <= len(x):
        g = x[a:b]
    else:
        g = np.zeros(2 * half, dtype=x.dtype)
        lo, hi = max(a, 0), min(b, len(x))
        if hi > lo:
            g[lo - a:hi - a] = x[lo:hi]
    return g * _hann(2 * half)


def _fade(y: np.ndarray, sr: int, end: int | None = None) -> np.ndarray:
    fi = min(len(y), int(_FADE_IN * sr))
    if fi:
        y[:fi] *= np.linspace(0.0, 1.0, fi)
    end = len(y) if end is None else min(end, len(y))
    fo = min(end, int(_FADE_OUT * sr))
    if fo:
        y[end - fo:end] *= np.linspace(1.0, 0.0, fo)
        y[end:] = 0.0
    return y


def _resample(x: np.ndarray, ratio: float) -> np.ndarray:
    """Play *x* at *ratio* times the speed (and pitch)."""
    if ratio == 1.0 or len(x) < 2:
        return x.copy()
    pos = np.arange(0.0, len(x) - 1, ratio)
    return np.interp(pos, np.arange(len(x)), x).astype(np.float32)


class Voice:
    """A sample, measured once, that can be sung on any note."""

    def __init__(self, x: np.ndarray, sr: int = SR, aperiodic: float = _APERIODIC):
        self.x = np.asarray(x, dtype=np.float32)
        self.sr = sr
        self.track = track_f0(self.x, sr, aperiodic=aperiodic)
        self.info = describe(self.track, len(self.x) / sr)
        self.marks, self.periods, self.mvoiced = pitch_marks(self.x, self.track, sr)
        vt = self.track.times[self.track.voiced]
        self.v0 = float(vt[0]) if len(vt) else 0.0
        self.v1 = float(vt[-1]) if len(vt) else 0.0
        self._cache: dict = {}

    @property
    def duration(self) -> float:
        return len(self.x) / self.sr

    def plan(self, dur: float, stretch: bool) -> Warp:
        L = self.duration
        vlen = self.v1 - self.v0
        if not stretch or dur <= L or vlen < 0.02:
            return Warp(self.v0, self.v1, 1.0)
        s = (dur - (L - vlen)) / vlen
        return Warp(self.v0, self.v1, float(min(max(s, 1.0), _MAX_STRETCH)))

    def render(self, target_hz: float | None, dur: float, mode: str = "perfect",
               semitones: float = 0.0, stretch: bool = True) -> tuple[np.ndarray, Warp]:
        """The sample sung for *dur* seconds on *target_hz*.

        mode    perfect  every voiced pulse at exactly target_hz (PSOLA)
                tape     sped up or slowed down until its median pitch is target_hz
                raw      untouched, bar *semitones* applied tape-style (drums)
        stretch whether a note longer than the sample draws its vowel out;
                off, the note simply ends when the sample does
        """
        dur = max(float(dur), _MIN_NOTE)
        key = (None if target_hz is None else round(target_hz, 3),
               round(dur, 2), mode, round(semitones, 3), stretch)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        sr = self.sr
        n_out = int(round(dur * sr))

        if mode == "perfect" and target_hz and self.info.f0 and len(self.marks):
            warp = self.plan(dur, stretch)
            y = self._psola(float(target_hz), n_out, warp)
            end = n_out if stretch else min(n_out, int(round(self.duration * sr)))
        else:
            if mode == "tape" and target_hz and self.info.f0:
                ratio = float(target_hz) / self.info.f0
            else:
                ratio = 2.0 ** (semitones / 12.0)
            warp = Warp(rate=ratio)
            z = _resample(self.x, ratio)
            y = np.zeros(n_out, dtype=np.float32)
            k = min(n_out, len(z))
            y[:k] = z[:k]
            end = k
        y = _fade(y, sr, end)
        self._cache[key] = (y, warp)
        return y, warp

    def _psola(self, target_hz: float, n_out: int, warp: Warp) -> np.ndarray:
        sr = self.sr
        x = self.x
        marks, periods, mv = self.marks, self.periods, self.mvoiced
        Pt = sr / target_hz
        pad = int(max(periods.max(initial=0.0), Pt, _UNVOICED_STEP * sr)) + 2
        out = np.zeros(n_out + 2 * pad, dtype=np.float64)
        unv_half = int(_UNVOICED_STEP * sr)
        t = 0.0
        while t < n_out:
            src = warp.src(t / sr) * sr
            if src >= len(x):
                break
            k = int(np.searchsorted(marks, src))
            if k >= len(marks) or (k > 0 and src - marks[k - 1] < marks[k] - src):
                k -= 1
            c = int(round(t)) + pad
            if mv[k]:
                half = max(2, int(round(periods[k])))
                g = _grain(x, int(marks[k]), half) * min(1.0, Pt / periods[k])
                out[c - half:c + half] += g
                t += Pt
            else:
                g = _grain(x, int(round(src)), unv_half)
                out[c - unv_half:c + unv_half] += g
                t += unv_half
        return out[pad:pad + n_out].astype(np.float32)
