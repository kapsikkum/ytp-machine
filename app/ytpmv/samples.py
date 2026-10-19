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
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

import app.generate as g
from app.database import active, list_corpora, using_corpus
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
_building: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_frame_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
_frame_bytes = 0
_FRAME_CACHE_BYTES = 128 * 1024 * 1024

log = logging.getLogger(__name__)


def _build_lock(name: str) -> threading.Lock:
    with _lock:
        return _building.setdefault(name, threading.Lock())


def _store_frames(key: tuple, arr: np.ndarray) -> None:
    global _frame_bytes
    with _lock:
        old = _frame_cache.pop(key, None)
        if old is not None:
            _frame_bytes -= old.nbytes
        # A single full-size three-second sample can be almost 500 MB. The
        # render owns its live frames; retaining them afterwards is optional.
        if arr.nbytes > _FRAME_CACHE_BYTES:
            return
        while _frame_cache and _frame_bytes + arr.nbytes > _FRAME_CACHE_BYTES:
            _, gone = _frame_cache.popitem(last=False)
            _frame_bytes -= gone.nbytes
        _frame_cache[key] = arr
        _frame_bytes += arr.nbytes


def _source_stamp(path: str) -> tuple:
    if not os.path.isabs(path):
        from app.database import resolve_path
        path = resolve_path(path)
    path = os.path.normcase(os.path.realpath(path))
    try:
        info = os.stat(path)
        # Inode/creation metadata also distinguishes an imported replacement
        # whose archive preserves the original modification time and size.
        return (path, info.st_mtime_ns, info.st_size, info.st_ctime_ns, info.st_ino)
    except OSError:
        return (path, None, None, None, None)


class SampleError(ValueError):
    """The voice cannot say what was asked."""


@contextmanager
def corpus_context(corpus: str | None = None):
    """Use an installed voice for this operation, preserving the user's selection."""
    if corpus is not None and (not isinstance(corpus, str) or not corpus.strip()):
        raise SampleError("Choose an installed corpus for this voice.")
    entry = (next((c for c in list_corpora() if c["slug"] == corpus), None)
             if corpus is not None else active())
    if entry is None:
        raise SampleError(f"The corpus '{corpus}' is not installed. Choose another voice.")
    with using_corpus(entry):
        yield entry


def corpus_pool() -> dict:
    # A local reference never swaps the generator's process-global sentence
    # voice. Parallel auditions can consequently use different corpora.
    return g._pool(active())


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
    corpus: str = ""
    _sources: tuple = field(default_factory=tuple, repr=False)
    _segments: tuple = field(default_factory=tuple, repr=False)

    def ensure_video(self, check_cancel=None) -> None:
        """Materialize a single-clip picture only when a caller needs it."""
        if check_cancel:
            check_cancel()
        if os.path.exists(self.mp4):
            return
        if self._segments:
            if not all(_source_stamp(stamp[0]) == stamp for stamp in self._sources):
                raise SampleError("The recording changed; select the sound again.")
            with corpus_context(self.corpus):
                _ensure_video(self.mp4, self._segments, check_cancel)

    @property
    def url(self) -> str:
        self.ensure_video()
        return "/output/" + os.path.basename(self.mp4)

    @property
    def duration(self) -> float:
        return self.voice.duration

    def frames(self, w: int, h: int, check_cancel=None, *, fit: str = "contain", pixel_aspect: float = 1) -> np.ndarray:
        """The picture at w x h, one frame per 1/FPS s from the trimmed start."""
        if check_cancel:
            check_cancel()
        self.ensure_video(check_cancel)
        fit = "cover" if fit == "cover" else "contain"
        key = (self.mp4, self.offset, self.duration, w, h, fit, pixel_aspect)
        with _lock:
            got = _frame_cache.get(key)
            if got is not None:
                _frame_cache.move_to_end(key)
                return got
        from app.media_decode import video
        arr = video(self.mp4, self.offset, self.duration, w, h, fit, pixel_aspect, FPS,
                    check_cancel, lambda: self._frames_ffmpeg(w, h, check_cancel, fit, pixel_aspect))
        _store_frames(key, arr)
        return arr

    def _frames_ffmpeg(self, w, h, check_cancel, fit, pixel_aspect):
        # Retro canvases can have non-square display pixels. Fit against the
        # final picture's aspect, including any source sample-aspect metadata,
        # so the eventual canvas upscale does not stretch the video again.
        bound = "max" if fit == "cover" else "min"
        framing = (f"scale=w='max(1,{bound}({w},{h}*dar/{pixel_aspect:.12g}))'"
                   f":h='max(1,{bound}({h},{w}*{pixel_aspect:.12g}/dar))'")
        framing += (f",crop={w}:{h}:exact=1" if fit == "cover" else
                    f",pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black")
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-filter_threads", "2",
               "-ss", f"{self.offset:.3f}",
               "-i", self.mp4, "-t", f"{self.duration + 0.1:.3f}",
               "-vf", f"format=rgb24,{framing},setsar=1,fps={FPS}",
               "-threads", "2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        from app.media_process import run
        proc = run(cmd, capture_output=True, timeout=30, check_cancel=check_cancel)
        if proc.returncode != 0:
            raise RuntimeError("FFmpeg failed: " + proc.stderr.decode(errors="replace")[-400:])
        arr = np.frombuffer(proc.stdout, dtype=np.uint8)
        n = len(arr) // (w * h * 3)
        if n == 0:
            arr = np.zeros((1, h, w, 3), dtype=np.uint8)
        else:
            arr = arr[: n * w * h * 3].reshape(n, h, w, 3)
        return arr

    def describe(self, *, include_video: bool = True) -> dict:
        return {"text": self.text, "take": self.take, "takes": self.takes, "hit": self.hit,
                "corpus": self.corpus,
                "sample_url": self.url if include_video else None, "source": self.source,
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
    pool = corpus_pool()
    found: dict[str, list[dict]] = {}
    for rows in pool["clips_by_word"].values():
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
    pool = corpus_pool()
    found = {k: len(v) for k, v in pool["noise_by_word"].items() if v}
    if len(pool["all_noises"]) > 1 and len(found) > 1:
        found["noise"] = len(pool["all_noises"])      # the "any noise" pool
    return dict(sorted(found.items(), key=lambda kv: -kv[1]))


def noise_takes(kind: str) -> list[dict]:
    """Every recording of one noise, in a fixed order."""
    pool = corpus_pool()
    rows = (pool["all_noises"] if kind == "noise" else pool["noise_by_word"].get(kind)) or []
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
    pool = corpus_pool()
    scores = pool["splice_scores"]
    rows = [r for r in pool["clips_by_word"].get(word, [])
            if r.get("id") is None or scores.get((word.lower(), int(r["id"])), 0) > g._VETO_SCORE]
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


def build(text: str, take: int = 0, hit: str | None = None, check_cancel=None,
          corpus: str | None = None, *, audio_only: bool = False) -> Sample:
    """Build a playable word in an installed corpus, or the operation's voice."""
    with corpus_context(corpus):
        return _build(text, take, hit, check_cancel, audio_only=audio_only)


def _build(text: str, take: int = 0, hit: str | None = None, check_cancel=None,
           *, audio_only: bool = False) -> Sample:
    """The corpus saying *text* (take *take*, for a single word) as a Sample.

    *hit* names a drum (kick, snare, hats...) to cut the word down into, the
    way a beatboxer would -- see drums.py. The picture is cut to match.

    Raises SampleError when the voice cannot say it.
    """
    if check_cancel:
        check_cancel()
    decode_options = {"check_cancel": check_cancel} if check_cancel else {}
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
        if (cached is not None and (cached._segments or os.path.exists(cached.mp4))
                and cached._sources and all(_source_stamp(stamp[0]) == stamp for stamp in cached._sources)):
            _cache.move_to_end(key)
        else:
            cached = None
    if cached is not None:
        if not audio_only:
            cached.ensure_video(check_cancel)
        return cached

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
        with g._voice_lock:
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
    source_stamps = tuple(_source_stamp(path) for path in dict.fromkeys(sg["source_file"] for sg in segments))
    ident = tuple(tuple(sg.get(k) for k in ("id", "source_file", "start_time", "end_time",
                                           "prev_end", "next_start", "edited")) for sg in segments)
    digest = hashlib.sha1(repr((corpus, text, take, ident, source_stamps)).encode()).hexdigest()[:12]
    os.makedirs("output", exist_ok=True)
    mp4 = os.path.join("output", f"ytpmv_sample_{digest}.mp4")
    x = None
    if word_takes and audio_only:
        try:
            s, e = g.extract_window(clip)
            x = decode_audio(clip["source_file"], s, e - s, **decode_options)
            if not len(x):
                x = None
        except RuntimeError:
            if check_cancel:
                check_cancel()
    if x is not None:
        # The video recipe is retained below; frames()/url will build it if
        # this same sample is subsequently used for a video preview/export.
        return _finish_sample(text, take, n_takes, hit, mp4, x, source, corpus,
                              source_stamps, key, segments, check_cancel)
    # One build per file at a time, written aside and moved into place. The
    # page asks for every part's sample at once, and two parts suggested the
    # same word and take: both requests saw no file, both encoded into the
    # same path, and the interleaved result was a file ffmpeg could not
    # decode -- which then sat there answering every later request with 500.
    with _build_lock(mp4):
        x = None
        if os.path.exists(mp4):
            try:
                x = decode_audio(mp4, **decode_options)
            except RuntimeError:
                if check_cancel:
                    check_cancel()
                log.warning("ytpmv: %s would not decode; building it again", mp4)
                x = None
        if x is None:
            tmp = f"{mp4[:-4]}.{os.getpid()}.{threading.get_ident()}.tmp.mp4"
            try:
                g._build_video(segments, tmp,
                               progress=(lambda *_: check_cancel()) if check_cancel else None)
                os.replace(tmp, mp4)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            try:
                x = decode_audio(mp4, **decode_options)
            except RuntimeError as exc:
                if check_cancel:
                    check_cancel()
                # Built fine and still no sound. For one clip that is the mp4
                # cutting a 40-60ms consonant to no AAC frames at all; its
                # picture is still good and its sound is read from the
                # recording below. Otherwise the recording it was cut from has
                # none: that word is unsayable, not a server fault.
                if not word_takes:
                    os.remove(mp4)
                    raise SampleError(f'"{text}" has no sound in its recording') from exc

    if word_takes:
        # One clip: its sound straight from the recording, the same window
        # the recommender measured, which the mp4's picture also starts at.
        # The mp4's own sound is cut to whole AAC frames of 23ms, which a
        # word never notices -- and which left an 80ms phoneme of Morshu's
        # as one frame, too short to hold a pitch, so every part that was
        # recommended a pitched vowel was handed one with no pitch at all.
        try:
            s, e = g.extract_window(clip)
            src = decode_audio(clip["source_file"], s, e - s, **decode_options)
            if len(src):
                x = src
        except RuntimeError:
            if check_cancel:
                check_cancel()
            pass                              # the mp4's sound will do
        if x is None:
            os.remove(mp4)
            raise SampleError(f'"{text}" has no sound in its recording')
    if check_cancel:
        check_cancel()
    return _finish_sample(text, take, n_takes, hit, mp4, x, source, corpus,
                          source_stamps, key, segments if word_takes else (), check_cancel)


def _ensure_video(mp4, segments, check_cancel):
    with _build_lock(mp4):
        if check_cancel:
            check_cancel()
        if os.path.exists(mp4):
            return
        tmp = f"{mp4[:-4]}.{os.getpid()}.{threading.get_ident()}.tmp.mp4"
        try:
            g._build_video(list(segments), tmp,
                           progress=(lambda *_: check_cancel()) if check_cancel else None)
            if check_cancel:
                check_cancel()
            os.replace(tmp, mp4)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def _finish_sample(text, take, n_takes, hit, mp4, x, source, corpus,
                   source_stamps, key, segments, check_cancel):
    if check_cancel:
        check_cancel()
    a, b = _trim(x)
    b = min(b, a + int(_MAX_SAMPLE * SR))
    audio = x[a:b]
    if hit:
        audio, start = drums.shape(audio, hit)
        a += start
    sample = Sample(text=text, take=int(take) % n_takes, takes=n_takes, hit=hit, mp4=mp4,
                    offset=a / SR, voice=Voice(audio), source=source, corpus=corpus,
                    _sources=source_stamps, _segments=tuple(dict(sg) for sg in segments))
    with _lock:
        _cache[key] = sample
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return sample


def clear_cache() -> None:
    global _frame_bytes
    with _lock:
        _cache.clear()
        _frame_cache.clear()
        _frame_bytes = 0
    _unit_cache.clear()
