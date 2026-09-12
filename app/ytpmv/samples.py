"""A word or phrase, said by the corpus, ready to be played as an instrument.

The saying is done by the sentence machinery itself -- resolve_text and
_build_video -- so a sample is exactly what typing that word would have
produced: runs, splices, ~reversal~ and all. What is added is only what an
instrument needs: the dead air trimmed off the front so a note lands on the
beat, the pitch measured, and the picture decoded to frames for the grid.

A single word has *takes* -- every usable recording of it -- and take N is
always the same recording, so a choice made in the page survives to the render.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

import app.generate as g
from app.database import active
from app.ytpmv import drums
from app.ytpmv.pitch import SR, Voice, decode_audio

FPS = g._FPS
SRC_W, SRC_H = 480, 270

_LEAD_FLOOR = 0.04     # a sample starts where it first reaches this share of its peak
_TAIL_FLOOR = 0.02
_LEAD_KEEP = 0.004     # ... less a few ms, so the attack itself is not shaved
_MAX_SAMPLE = 3.0      # a phrase longer than this is not an instrument, it is a speech

_cache: "OrderedDict[tuple, Sample]" = OrderedDict()
_CACHE_MAX = 48
_lock = threading.Lock()
_building: dict[str, threading.Lock] = {}

log = logging.getLogger(__name__)


def _build_lock(name: str) -> threading.Lock:
    with _lock:
        return _building.setdefault(name, threading.Lock())


class SampleError(ValueError):
    """The voice cannot say what was asked."""


@dataclass
class Sample:
    text: str
    take: int
    takes: int                    # how many takes the word has (1 for phrases)
    hit: str | None               # drum group it was cut into a hit for, if any
    mp4: str
    offset: float                 # seconds of lead-in trimmed from the mp4
    voice: Voice
    source: str = ""
    _frames: dict = field(default_factory=dict, repr=False)

    @property
    def url(self) -> str:
        return "/output/" + os.path.basename(self.mp4)

    @property
    def duration(self) -> float:
        return self.voice.duration

    def frames(self, w: int, h: int) -> np.ndarray:
        """The picture at w x h, one frame per 1/FPS s from the trimmed start."""
        got = self._frames.get((w, h))
        if got is not None:
            return got
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{self.offset:.3f}",
               "-i", self.mp4, "-t", f"{self.duration + 0.1:.3f}",
               "-vf", f"scale={w}:{h}:force_original_aspect_ratio=disable,fps={FPS}",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError("FFmpeg failed: " + proc.stderr.decode(errors="replace")[-400:])
        arr = np.frombuffer(proc.stdout, dtype=np.uint8)
        n = len(arr) // (w * h * 3)
        if n == 0:
            arr = np.zeros((1, h, w, 3), dtype=np.uint8)
        else:
            arr = arr[: n * w * h * 3].reshape(n, h, w, 3)
        self._frames[(w, h)] = arr
        return arr

    def describe(self) -> dict:
        return {"text": self.text, "take": self.take, "takes": self.takes, "hit": self.hit,
                "sample_url": self.url, "source": self.source,
                **self.voice.info.as_dict()}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


# /ah/, /s/, /ow/ -- one sound out of the middle of words, wherever the corpus
# has it. The aligner already wrote down where every phoneme of every clip
# falls, which is what makes a word spliceable; the same timings make each
# sound playable on its own. A vowel cut out of a word is a far better
# instrument than the word: no consonants top and tail, so it holds a note.
_UNIT = re.compile(r"^/([a-z]{1,3}\d?)/$")

# ARPAbet's vowels. They carry the pitch, so they are the ones worth singing.
VOWELS = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey",
          "ih", "iy", "ow", "oy", "uh", "uw"}

# How long each kind of sound ought to be: (shortest, longest, the length to
# prefer). Sorting purely by length looked sensible and was not -- the longest
# "ah" in the corpus is 1.4 seconds, inside the word "but", which is the
# aligner having slipped rather than a sound anybody made. Ranking by nearness
# to a plausible length throws those out and keeps the ones worth playing.
_SHAPE = {
    "vowel":     (0.07, 0.60, 0.20),
    "fricative": (0.05, 0.40, 0.14),
    "plosive":   (0.03, 0.20, 0.07),
    "other":     (0.05, 0.35, 0.12),
}
FRICATIVES = {"s", "sh", "z", "zh", "f", "v", "th", "dh", "hh"}
PLOSIVES = {"p", "b", "t", "d", "k", "g", "ch", "jh"}
_UNITS_KEPT = 60         # instances of each sound to keep
_unit_cache: dict[str, dict[str, list[dict]]] = {}


def _kind(name: str) -> str:
    return ("vowel" if name in VOWELS else "fricative" if name in FRICATIVES
            else "plosive" if name in PLOSIVES else "other")


def units() -> dict[str, list[dict]]:
    """Every phoneme this voice has, as clips of its own: {"ah": [clip, ...]}.

    Each is a slice of a real clip, marked *edited* so it is cut exactly where
    the aligner put the boundary rather than padded like a word.
    """
    corpus = active()["slug"]
    got = _unit_cache.get(corpus)
    if got is not None:
        return got
    g._ensure_cache()
    found: dict[str, list[dict]] = {}
    for rows in (g._clips_by_word_cache or {}).values():
        for clip in rows:
            phones = clip.get("phones")
            if not phones:
                continue
            if isinstance(phones, str):
                try:
                    phones = json.loads(phones)
                except ValueError:
                    continue
            for i, (ph, a, b) in enumerate(phones):
                name = str(ph).lower().rstrip("0123456789")
                lo, hi, _want = _SHAPE[_kind(name)]
                dur = b - a
                if not lo <= dur <= hi:
                    continue
                start = clip["start_time"] + max(a, 0.0)
                end = clip["start_time"] + b
                found.setdefault(name, []).append({
                    "id": f"{clip.get('id')}#{i}",
                    "word": f"/{name}/",
                    "source_id": clip.get("source_id"),
                    "source_file": clip["source_file"],
                    "start_time": start,
                    "end_time": end,
                    "prev_end": start,
                    "next_start": end,
                    "edited": 1,            # cut where the aligner said, not padded
                    # A word's fades would swallow a 60ms sound whole.
                    "fade_in": 0.004,
                    "fade_out": 0.006,
                    "unit_of": clip.get("word"),
                })
    for name, rows in found.items():
        want = _SHAPE[_kind(name)][2]
        rows.sort(key=lambda r: abs((r["end_time"] - r["start_time"]) - want))
        found[name] = rows[:_UNITS_KEPT]
    _unit_cache[corpus] = found
    return found


def unit_takes(name: str) -> list[dict]:
    return units().get(name.lower(), [])


# *spew*, *click* -- a non-verbal clip rather than a word. The corpus keeps
# these apart from speech precisely because they are not words, and they make
# far better drums than anything anybody says.
_NOISE = re.compile(r"^\*([a-z0-9-]+)\*$")


def noises() -> dict[str, int]:
    """Every non-verbal noise the voice has, and how many recordings of each."""
    g._ensure_cache()
    found = {k: len(v) for k, v in (g._noise_by_word or {}).items() if v}
    if len(g._all_noises or []) > 1 and len(found) > 1:
        found["noise"] = len(g._all_noises)      # the "any noise" pool
    return dict(sorted(found.items(), key=lambda kv: -kv[1]))


def noise_takes(kind: str) -> list[dict]:
    """Every recording of one noise, in a fixed order."""
    g._ensure_cache()
    rows = (g._all_noises if kind == "noise" else (g._noise_by_word or {}).get(kind)) or []
    return sorted(rows, key=lambda r: (r.get("id") or 0, r["start_time"]))


def takes_for(text: str) -> list[dict]:
    """The takes behind *text*: a phoneme, a noise, or a word."""
    text = _norm(text)
    u = _UNIT.match(text)
    if u:
        return unit_takes(u.group(1))
    m = _NOISE.match(text)
    return noise_takes(m.group(1)) if m else takes(text)


def takes(word: str) -> list[dict]:
    """Every recording of *word* worth playing, in a fixed order.

    The same band pick_clip uses -- a clip far shorter or longer than the
    word's median is a bad timestamp, not a take -- and the same vetoes.
    """
    g._ensure_cache()
    rows = [r for r in (g._clips_by_word_cache or {}).get(word, []) if not g._vetoed(word, r)]
    if len(rows) > 1:
        durs = sorted(r["end_time"] - r["start_time"] for r in rows)
        median = durs[len(durs) // 2]
        floor = max(g._duration_floor(word), 0.5 * median)
        ceiling = max(2.5 * median, 0.45)
        good = [r for r in rows if floor <= r["end_time"] - r["start_time"] <= ceiling]
        rows = good or rows
    return sorted(rows, key=lambda r: (r.get("id") or 0, r["start_time"]))


def _trim(x: np.ndarray) -> tuple[int, int]:
    if len(x) == 0:
        return 0, 0
    hop = int(0.005 * SR)
    n = max(1, len(x) // hop)
    env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    peak = float(env.max(initial=0.0))
    if peak <= 0:
        return 0, len(x)
    lead = np.nonzero(env >= _LEAD_FLOOR * peak)[0]
    tail = np.nonzero(env >= _TAIL_FLOOR * peak)[0]
    a = max(0, int(lead[0]) * hop - int(_LEAD_KEEP * SR))
    b = min(len(x), (int(tail[-1]) + 2) * hop)
    return a, max(b, a + hop)


def build(text: str, take: int = 0, hit: str | None = None) -> Sample:
    """The corpus saying *text* (take *take*, for a single word) as a Sample.

    *hit* names a drum (kick, snare, hats...) to cut the word down into, the
    way a beatboxer would -- see drums.py. The picture is cut to match.

    Raises SampleError when the voice cannot say it.
    """
    text = _norm(text)
    if not text:
        raise SampleError("Nothing to say.")
    if hit is not None and hit not in drums.GROUPS:
        hit = "perc"
    corpus = active()["slug"]
    key = (corpus, text, int(take), hit)
    with _lock:
        cached = _cache.get(key)
        # The output sweep may have taken the mp4 since; then build it again.
        if cached is not None and os.path.exists(cached.mp4):
            _cache.move_to_end(key)
            return cached

    g._ensure_cache()
    unit = _UNIT.match(text)
    noise = None if unit else _NOISE.match(text)
    if unit:
        word_takes = unit_takes(unit.group(1))
        if not word_takes:
            raise SampleError(f"This voice has no {text} sound the aligner could find.")
    elif noise:
        word_takes = noise_takes(noise.group(1))
        if not word_takes:
            raise SampleError(f"This voice has no {noise.group(1)} noise.")
    elif " " not in text and "~" not in text and "*" not in text:
        word_takes = takes(text)
    else:
        word_takes = []
    n_takes = max(1, len(word_takes))
    if word_takes:
        clip = word_takes[int(take) % len(word_takes)]
        segments = [dict(clip)]
        source = f"{os.path.basename(clip['source_file'])} @ {clip['start_time']:.2f}s"
    else:
        # A phrase, a splice or a marked-up word: the sentence machinery picks
        # at random, so seed it -- the same text and take must give the same
        # sample every time, or a preview would not be what gets rendered.
        state = random.getstate()
        random.seed(f"{corpus}|{text}|{take}")
        try:
            segments, _report = g.resolve_text(text)
        finally:
            random.setstate(state)
        source = "spliced" if segments and len(segments) > 1 else ""
    if not segments:
        raise SampleError(f"This voice can't say \"{text}\".")

    # The clip itself is in the name, not just its index. Take 3 of a word is
    # whatever sorts third, and that moves when the corpus is edited or the
    # ranking changes -- so a file cached under (voice, text, take) alone came
    # back as the sound that used to be there. A phoneme unit made that
    # obvious: every one of them returned a 1.4s sample it no longer used.
    ident = "|".join(str(clip.get(k)) for k in ("id", "start_time", "end_time")) if word_takes         else "|".join(f"{sg.get('id')}:{sg['start_time']:.3f}" for sg in segments)
    digest = hashlib.sha1(f"{corpus}|{text}|{take}|{ident}".encode()).hexdigest()[:12]
    os.makedirs("output", exist_ok=True)
    mp4 = os.path.join("output", f"ytpmv_sample_{digest}.mp4")
    # One build per file at a time, written aside and moved into place. The
    # page asks for every part's sample at once, and two parts suggested the
    # same word and take: both requests saw no file, both encoded into the
    # same path, and the interleaved result was a file ffmpeg could not
    # decode -- which then sat there answering every later request with 500.
    with _build_lock(digest):
        x = None
        if os.path.exists(mp4):
            try:
                x = decode_audio(mp4)
            except RuntimeError:
                log.warning("ytpmv: %s would not decode; building it again", mp4)
                x = None
        if x is None:
            tmp = f"{mp4[:-4]}.{os.getpid()}.{threading.get_ident()}.tmp.mp4"
            try:
                g._build_video(segments, tmp)
                os.replace(tmp, mp4)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            x = decode_audio(mp4)

    a, b = _trim(x)
    b = min(b, a + int(_MAX_SAMPLE * SR))
    audio = x[a:b]
    if hit:
        audio, start = drums.shape(audio, hit)
        a += start
    sample = Sample(text=text, take=int(take) % n_takes, takes=n_takes, hit=hit, mp4=mp4,
                    offset=a / SR, voice=Voice(audio), source=source)
    with _lock:
        _cache[key] = sample
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return sample


def clear_cache() -> None:
    with _lock:
        _cache.clear()
    _unit_cache.clear()
