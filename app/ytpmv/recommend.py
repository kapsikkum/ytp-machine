"""Which word should play which part.

Nobody wants to audition 1,400 words to find a kick drum. Each role has words
that tend to make the right *kind* of sound -- a plosive for a kick, a hiss for
a hat, a long open vowel for a bass line -- and every recording of those words
the corpus actually has is measured and scored for the job:

  pitched parts   voiced for most of its length, steady in pitch, long enough
                  to hold a note
  drums           a sharp attack, short, and bright or dark to suit the drum

Measurements are cached per corpus on disk, so only the first song on a new
voice pays for them. The preferred words are only a starting point: a voice
that never said "boom" still gets a kick from whatever it did say that hits
hardest.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import app.generate as g
from app.database import DATA_DIR, active
from app.ytpmv import drums, samples
from app.ytpmv.music import Song, auto_octave
from app.ytpmv.pitch import SR, decode_audio, describe, track_f0

log = logging.getLogger(__name__)

PREFERRED = {
    # Drums are cut down to a hit (drums.py), so what matters is the sound a
    # word *starts* with, or the hiss it has in it -- not the word.
    "drums:kick":    ["boom", "bum", "big", "bob", "but", "bad", "bed", "bag", "dad", "dog", "ball", "box",
                      "back", "done", "bang"],
    "drums:snare":   ["kiss", "kids", "cats", "kicks", "chips", "cups", "cakes", "keys", "cheese", "chess",
                      "cat", "kit", "check", "chip", "tea", "top", "get", "got"],
    "drums:hats":    ["see", "sit", "this", "yes", "so", "is", "its", "says", "she", "sss", "shh", "cheese",
                      "kiss", "us"],
    "drums:toms":    ["dum", "done", "dog", "bum", "doom", "bong", "down", "dad", "bed", "boom", "good"],
    "drums:cymbals": ["crash", "splash", "wish", "fish", "shush", "she", "shoes", "sea", "yes", "chips"],
    "drums:perc":    ["tick", "tock", "cup", "pop", "top", "tap", "clap", "click", "kit"],
    "bass":          ["oh", "no", "go", "all", "more", "now", "low", "you", "moo", "boo"],
    "lead":          ["yeah", "me", "oh", "you", "no", "la", "ah", "so", "hi", "my", "why", "wow"],
    "chords":        ["ah", "oh", "all", "you", "more", "now", "me", "no"],
    "rhythm":        ["no", "go", "oh", "yeah", "you", "me", "ah", "my", "all"],
}
# Bumped whenever measure() learns something new, so a cache written by an
# older version is re-measured instead of quietly scoring on missing fields.
_MEASURE_VERSION = 2
_FALLBACK_WORDS = 30     # most frequent words tried when too few preferred exist
_TAKES_PER_WORD = 4
_ALTERNATIVES = 5

_lock = threading.Lock()
_mem: dict[str, dict] = {}      # corpus -> {clip key: measurement}


def _cache_path(corpus: str) -> str:
    return os.path.join(DATA_DIR, "ytpmv", f"analysis-{corpus}.json")


def _load(corpus: str) -> dict:
    with _lock:
        if corpus not in _mem:
            try:
                with open(_cache_path(corpus), encoding="utf-8") as fh:
                    _mem[corpus] = json.load(fh)
            except (OSError, ValueError):
                _mem[corpus] = {}
        return _mem[corpus]


def _save(corpus: str) -> None:
    path = _cache_path(corpus)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _lock:
        data = dict(_mem.get(corpus, {}))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def _clip_key(clip: dict) -> str:
    return f"{clip.get('id')}:{clip['start_time']:.3f}:{clip['end_time']:.3f}"


def measure(x: np.ndarray) -> dict:
    """What a sound is like, in the terms the scores below care about."""
    a, b = samples._trim(x)
    x = x[a:b]
    dur = len(x) / SR
    info = describe(track_f0(x), dur)
    hop = int(0.005 * SR)
    n = max(1, len(x) // hop)
    env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) if len(x) >= hop else np.zeros(1)
    peak = float(env.max(initial=0.0))
    attack = float(np.argmax(env >= 0.8 * peak) * 0.005) if peak > 0 else 1.0
    head = x[: int(0.06 * SR)]
    if len(head) >= 256:
        spec = np.abs(np.fft.rfft(head * np.hanning(len(head))))
        freqs = np.fft.rfftfreq(len(head), 1.0 / SR)
        centroid = float((spec * freqs).sum() / max(spec.sum(), 1e-9))
    else:
        centroid = 0.0
    if len(x) >= 256:
        power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
        total = max(power.sum(), 1e-12)
        hf = float(power[freqs >= 4000].sum() / total)
        lf = float(power[freqs < 250].sum() / total)
    else:
        hf = lf = 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    crest = peak / rms if rms > 0 else 0.0
    return {
        "dur": round(dur, 4),
        "f0": round(info.f0, 2) if info.f0 else None,
        "midi": round(info.midi, 2) if info.midi is not None else None,
        "voiced": round(info.voiced_ratio, 3),
        "stability": round(min(info.stability, 1200.0), 1),
        "attack": round(attack, 4),
        "centroid": round(centroid, 1),
        "hf": round(hf, 3),                  # share of the energy above 4 kHz: hiss
        "lf": round(lf, 3),                  # ... below 250 Hz: body, what a kick is
        "crest": round(crest, 2),            # peak over average: how much it punches
    }


def _measure_clip(clip: dict, groups: tuple[str, ...] = ()) -> dict | None:
    """The clip measured whole, and cut into each drum hit in *groups*."""
    try:
        start, end = g.extract_window(clip)
        x = decode_audio(clip["source_file"], start, end - start)
        out = {"v": _MEASURE_VERSION, "word": measure(x)}
        a, b = samples._trim(x)
        for grp in groups:
            out[grp] = measure(drums.shape(x[a:b], grp)[0])
        return out
    except Exception as exc:  # noqa: BLE001 -- one bad file must not sink the lot
        log.warning("ytpmv: could not measure clip %s: %s", clip.get("id"), exc)
        return None


def _band(v: float, lo: float, hi: float, soft: float) -> float:
    """1 inside [lo, hi], falling to 0 over *soft* either side."""
    if lo <= v <= hi:
        return 1.0
    d = lo - v if v < lo else v - hi
    return max(0.0, 1.0 - d / soft)


def score(role: str, m: dict) -> float:
    """How well a sound suits a role. For drums, *m* measures the cut hit."""
    dur, voiced, stab = m["dur"], m["voiced"], m["stability"]
    attack, cen, hf = m["attack"], m["centroid"], m.get("hf", 0.0)
    sharp = _band(attack, 0.0, 0.015, 0.05)
    # Shaped hits land in a narrow range, so the band is narrow too.
    punch = _band(m.get("crest", 0.0), 2.4, 8.0, 0.8)
    if role == "drums:kick":
        # A thump is *low*, and a low centroid is not the same thing: a spew
        # low-passed has a low centroid and no bass in it at all. What makes a
        # kick is energy under 250 Hz, arriving all at once.
        return (2.5 * _band(m.get("lf", 0.0), 0.45, 1.0, 0.35) + 2 * sharp + punch
                + 0.5 * _band(cen, 0, 400, 500))
    if role == "drums:snare":
        # A crack then a hiss: sharp, noisy rather than sung, and it punches.
        return 2 * sharp + 2 * _band(hf, 0.35, 1.0, 0.3) + (1 - voiced) + punch
    if role == "drums:hats":
        # Graded on the hiss itself rather than a band that everything clears:
        # scored in bands, eight different words tied on 4.50 and the pick
        # among them was whichever happened to sort first.
        return 3 * hf + _band(cen, 6000, 16000, 4000) + 0.5 * (1 - voiced) + 0.5 * punch
    if role == "drums:toms":
        # A tom is a kick with a note in it: body, a clean hit, and a pitch.
        return (1.5 * _band(m.get("lf", 0.0), 0.25, 0.9, 0.3) + 1.5 * sharp + voiced
                + 0.5 * punch + _band(cen, 0, 1200, 1000))
    if role == "drums:cymbals":
        # A cymbal is a hiss that rings: bright, and long enough to hear it go.
        return 3 * hf + 1.5 * _band(dur, 0.25, 0.6, 0.3) + 0.5 * (1 - voiced)
    if role.startswith("drums"):
        return 2.5 * sharp + (1 - voiced) + punch
    # Pitched: the whole point is a note, so no pitch at all is disqualifying.
    if not m["f0"]:
        return -5.0
    s = 2.5 * voiced + 1.5 * max(0.0, 1.0 - stab / 150.0) + _band(dur, 0.25, 0.8, 0.3)
    if role == "bass":
        s += 0.5 * _band(m["f0"], 60, 150, 80)
    return s


def _words_for(role: str, available: dict[str, list[dict]], frequent: list[str],
               noises: tuple[str, ...] = ()) -> list[str]:
    words = [w for w in PREFERRED.get(role, PREFERRED["rhythm"]) if available.get(w)]
    # A corpus with non-verbal noises has better drums in it than any word: a
    # spew or a click is already a percussive sound, so they are auditioned for
    # the kit alongside the words and win on their own merits.
    if role.startswith("drums"):
        words = [f"*{k}*" for k in noises] + words
    if len(words) < 3:
        words += [w for w in frequent if w not in words]
    return words


def recommend(song: Song, progress=None) -> dict[str, dict]:
    """For each part: the suggested sound, alternatives, and an octave to play it in.

    Returns {part_id: {"text", "take", "octave", "mode", "score", "pitch",
    "alternatives": [{"text", "take", "score"}]}}.
    """
    corpus = active()["slug"]
    g._ensure_cache()
    cbw = g._clips_by_word_cache or {}
    frequent = [w for w, rows in sorted(cbw.items(), key=lambda kv: -len(kv[1]))
                if w.isalpha()][:_FALLBACK_WORDS]
    roles = sorted({p.role for p in song.parts})
    # "noise" is the any-of pool: every take is a different kind of sound, so
    # it is no use as one instrument. It stays available to type by hand.
    noise_kinds = tuple(k for k in samples.noises() if k != "noise")

    # Every (word, take) worth measuring for any role, measured once.
    wanted: dict[str, list[tuple[int, dict]]] = {}
    for role in roles:
        for w in _words_for(role, cbw, frequent, noise_kinds):
            if w in wanted:
                continue
            tk = samples.takes_for(w)
            if not tk:
                continue
            step = max(1, len(tk) // _TAKES_PER_WORD)
            wanted[w] = [(i, tk[i]) for i in range(0, len(tk), step)][:_TAKES_PER_WORD]

    cache = _load(corpus)
    groups = tuple(sorted({r.split(":", 1)[1] for r in roles if r.startswith("drums:")}))

    def needs(clip) -> bool:
        m = cache.get(_clip_key(clip))
        return (not isinstance(m, dict) or m.get("v") != _MEASURE_VERSION
                or "word" not in m or any(grp not in m for grp in groups))

    todo = list({_clip_key(c): c for pairs in wanted.values() for _i, c in pairs if needs(c)}.values())
    if todo:
        log.info("ytpmv: measuring %d takes for %s", len(todo), corpus)
        done = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            for clip, m in zip(todo, pool.map(lambda c: _measure_clip(c, groups), todo)):
                done += 1
                if progress:
                    progress("measuring", done, len(todo))
                if m is not None:
                    with _lock:
                        old = cache.get(_clip_key(clip))
                        keep = (old if isinstance(old, dict) and old.get("v") == _MEASURE_VERSION
                                else {})
                        cache[_clip_key(clip)] = {**keep, **m}
        _save(corpus)

    out: dict[str, dict] = {}
    used: dict[str, int] = {}
    for part in song.parts:
        ranked = []
        for w in _words_for(part.role, cbw, frequent, noise_kinds):
            for i, clip in wanted.get(w, []):
                entry = cache.get(_clip_key(clip)) or {}
                m = entry.get(part.drum_group) if part.is_drums else entry.get("word")
                if m is None:
                    continue
                # Nothing gets a head start -- not the words at the top of the
                # list above, and not the noises, which were given a bonus for
                # being "real" percussion and won places they did not deserve.
                # The measurements decide. The one thumb on the scale is
                # against repetition: the same sound on every tile is a worse
                # video than a slightly less ideal second choice.
                ranked.append((score(part.role, m) - 0.6 * used.get(w, 0), w, i, m))
        ranked.sort(key=lambda r: -r[0])
        # One entry per word in the alternatives, best take of each.
        seen, uniq = set(), []
        for r in ranked:
            if r[1] not in seen:
                seen.add(r[1])
                uniq.append(r)
        if not uniq:
            out[part.id] = {"text": None, "take": 0, "octave": 0,
                            "mode": "raw" if part.is_drums else "perfect",
                            "score": None, "pitch": None, "alternatives": []}
            continue
        best = uniq[0]
        used[best[1]] = used.get(best[1], 0) + 1
        m = best[3]
        out[part.id] = {
            "text": best[1],
            "take": best[2],
            "octave": 0 if part.is_drums else auto_octave(part.median_pitch(), m["midi"]),
            # A voice with no pitch (a whisper) cannot be put *on* a note, but
            # tape speed still follows the tune up and down.
            "mode": "raw" if part.is_drums else ("perfect" if m["f0"] else "tape"),
            "score": round(best[0], 2),
            "pitch": m,
            "alternatives": [{"text": w, "take": i, "score": round(s, 2)}
                             for s, w, i, _m in uniq[1:1 + _ALTERNATIVES]],
        }
    return out
