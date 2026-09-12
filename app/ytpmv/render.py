"""Play a MIDI file with the corpus: every note sung, every part a tile.

The sound is built in numpy -- each note is the part's sample put on that
note by pitch.Voice, summed into one stereo mix. The picture is built the
same way: each tile re-starts its clip on every note and follows the note's
own time-warp, so a stretched vowel holds the mouth open for as long as it is
heard. Frames go straight into one ffmpeg as raw video.

That is deliberately not a filter graph. A song is thousands of notes, and
the sentence renderer already found where "one ffmpeg input per clip" ends:
out of file handles well before a single verse.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import math
import os
import re
import subprocess
import tempfile
import uuid
import wave
from dataclasses import dataclass

import numpy as np

import app.generate as g
from app.database import DATA_DIR, active
from app.ytpmv import music, recommend, samples
from app.ytpmv.pitch import SR, Warp, midi_to_hz

log = logging.getLogger(__name__)

FPS = samples.FPS
MIDI_DIR = os.path.join(DATA_DIR, "ytpmv", "midi")
MAX_SECONDS = float(os.environ.get("YTPMV_MAX_SECONDS", "300"))
MAX_MIDI_BYTES = 2 * 1024 * 1024
_OUT_W, _OUT_H = 1920, 1080
_TAIL = 0.8                    # let the last notes ring out after the final note-off
_MODES = ("perfect", "tape", "raw")

# How loud and where each kind of part sits unless told otherwise.
_VOLUME = {"lead": 1.0, "bass": 1.0, "rhythm": 0.75, "chords": 0.7,
           "drums:kick": 1.0, "drums:snare": 0.9, "drums:hats": 0.55,
           "drums:toms": 0.8, "drums:cymbals": 0.5, "drums:perc": 0.7}
_PAN = {"lead": 0.0, "bass": 0.0, "rhythm": -0.35, "chords": 0.35,
        "drums:kick": 0.0, "drums:snare": 0.05, "drums:hats": 0.3,
        "drums:toms": -0.2, "drums:cymbals": -0.3, "drums:perc": 0.25}

_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class YtpmvError(ValueError):
    """Something about the request that the person making it can fix."""


# ── MIDI files ────────────────────────────────────────────────────────────────

def save_midi(data: bytes, filename: str = "") -> str:
    if not data:
        raise YtpmvError("That file is empty.")
    if len(data) > MAX_MIDI_BYTES:
        raise YtpmvError("That MIDI file is too big (2 MB is the limit).")
    if not data.startswith(b"MThd"):
        raise YtpmvError("That doesn't look like a MIDI file.")
    mid = hashlib.sha1(data).hexdigest()[:12]
    os.makedirs(MIDI_DIR, exist_ok=True)
    path = os.path.join(MIDI_DIR, f"{mid}.mid")
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(data)
    title = os.path.splitext(os.path.basename(filename or ""))[0][:80]
    if title:
        with open(path + ".title", "w", encoding="utf-8") as fh:
            fh.write(title)
    return mid


def load_song(midi_id: str) -> music.Song:
    if not _ID_RE.match(midi_id or ""):
        raise YtpmvError("No such MIDI file.")
    path = os.path.join(MIDI_DIR, f"{midi_id}.mid")
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        raise YtpmvError("That MIDI file has gone -- upload it again.") from None
    try:
        with open(path + ".title", encoding="utf-8") as fh:
            title = fh.read().strip()
    except OSError:
        title = ""
    try:
        song = music.parse(data, title=title)
    except Exception as exc:  # noqa: BLE001 -- mido raises a zoo of types on bad files
        raise YtpmvError(f"Couldn't read that MIDI file: {exc}") from exc
    if not song.parts:
        raise YtpmvError("That MIDI file has no notes in it.")
    return song


def analyse(midi_id: str, progress=None) -> dict:
    """The song in plain words, plus a suggested sound for every part."""
    song = load_song(midi_id)
    recs = recommend.recommend(song, progress=progress)
    out = song.summary()
    out["midi_id"] = midi_id
    # What this voice has besides words. A spew or a click is already a
    # percussive sound, so the page offers them for any part.
    out["noises"] = samples.noises()
    out["voice"] = active()["name"]
    for p in out["parts"]:
        rec = recs.get(p["id"], {})
        p["recommended"] = rec
        p["settings"] = _default_settings_for(song.part(p["id"]), rec)
    return out


# ── Settings ──────────────────────────────────────────────────────────────────

def _default_settings_for(part: music.Part, rec: dict) -> dict:
    return {
        "id": part.id,
        "text": rec.get("text"),
        "take": rec.get("take", 0),
        "octave": rec.get("octave", 0),
        "transpose": 0,
        "volume": _VOLUME.get(part.role, 0.8),
        "pan": _PAN.get(part.role, 0.0),
        "mode": rec.get("mode") or ("raw" if part.is_drums else "perfect"),
        "sustain": "ring" if part.is_drums else "note",
        # Drums are cut down to a hit (the "b" of a word, not the word) --
        # a kick that says "bum" on every beat is a person, not a drum.
        "hit": part.is_drums,
        "mute": rec.get("text") is None,
        "visible": rec.get("text") is not None,
    }


def _num(v, default, lo, hi, kind=float):
    try:
        x = kind(v)
    except (TypeError, ValueError):
        return default
    if isinstance(x, float) and not math.isfinite(x):
        return default
    return max(lo, min(hi, x))


def _merge(defaults: dict, given: dict | None) -> dict:
    s = dict(defaults)
    if not given:
        return s
    if isinstance(given.get("text"), str) and given["text"].strip():
        s["text"] = given["text"].strip()[:60]
        if given["text"].strip().lower() != (defaults.get("text") or "") and "take" not in given:
            s["take"] = 0
    s["take"] = _num(given.get("take", s["take"]), s["take"], 0, 10_000, int)
    s["octave"] = _num(given.get("octave", s["octave"]), s["octave"], -4, 4, int)
    s["transpose"] = _num(given.get("transpose", s["transpose"]), 0, -24, 24, int)
    s["volume"] = _num(given.get("volume", s["volume"]), s["volume"], 0.0, 2.0)
    s["pan"] = _num(given.get("pan", s["pan"]), s["pan"], -1.0, 1.0)
    if given.get("mode") in _MODES:
        s["mode"] = given["mode"]
    if given.get("sustain") in ("note", "ring"):
        s["sustain"] = given["sustain"]
    if "hit" in given:
        s["hit"] = bool(given["hit"])
    if s["text"] and defaults.get("text") is None:
        # Given a sound where none was suggested: it is meant to be heard.
        s["mute"], s["visible"] = False, True
    for k in ("mute", "visible"):
        if k in given:
            s[k] = bool(given[k])
    return s


# ── Layout ────────────────────────────────────────────────────────────────────

def grid(n: int) -> tuple[int, int, int, int]:
    """(cols, rows, tile_w, tile_h) for n tiles, as near 16:9 overall as it gets."""
    n = max(n, 1)
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        aspect = (cols * 16) / (rows * 9)
        cost = abs(math.log(aspect / (16 / 9))) + 0.3 * (cols * rows - n)
        if best is None or cost < best[0]:
            best = (cost, cols, rows)
    _c, cols, rows = best
    tw = min(samples.SRC_W, _OUT_W // cols)
    th = tw * 9 // 16
    if th * rows > _OUT_H:
        th = _OUT_H // rows
        tw = th * 16 // 9
    return cols, rows, tw - tw % 2, th - th % 2


# ── Rendering ─────────────────────────────────────────────────────────────────

@dataclass
class _Hit:
    start: float
    dur: float                 # how long the note sounds
    warp: Warp
    flip: bool


def _write_wav(path: str, stereo: np.ndarray) -> None:
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _finish_mix(mix: np.ndarray) -> np.ndarray:
    """Loud, never clipped: level the busy parts, then round off the peaks."""
    ref = float(np.percentile(np.abs(mix), 99.7)) if mix.size else 0.0
    if ref <= 1e-6:
        return mix
    mix = np.tanh(mix * (0.8 / ref))
    peak = float(np.abs(mix).max())
    return mix * (0.89 / peak) if peak > 0 else mix


def render_ytpmv(params: dict, progress=None) -> dict:
    """Render a song. *params* is what POST /api/ytpmv/render was given:

        midi_id   from save_midi
        parts     [{id, text, take, octave, transpose, volume, pan, mode,
                    sustain, mute, visible}, ...] -- anything left out is
                  filled from the recommendation
        options   {max_seconds, speed, transpose, flip, flash, dim, labels}
    """
    def say(stage, done, total):
        if progress:
            progress(stage, done, total)

    song = load_song(params.get("midi_id", ""))
    opts = params.get("options") or {}
    max_s = _num(opts.get("max_seconds", MAX_SECONDS), MAX_SECONDS, 1.0, MAX_SECONDS)
    speed = _num(opts.get("speed", 1.0), 1.0, 0.25, 4.0)
    g_transpose = _num(opts.get("transpose", 0), 0, -24, 24, int)
    flip = bool(opts.get("flip", True))
    flash = bool(opts.get("flash", True))
    dim = bool(opts.get("dim", True))
    labels = bool(opts.get("labels", False))

    given = {p.get("id"): p for p in (params.get("parts") or []) if isinstance(p, dict)}
    # Recommendations only for parts the request did not choose a sound for.
    need = [p for p in song.parts if not (given.get(p.id) or {}).get("text")]
    recs = recommend.recommend(music.Song(need, song.duration, song.bpm, song.time_signature,
                                          song.key, song.key_confidence)) if need else {}
    settings = {p.id: _merge(_default_settings_for(p, recs.get(p.id, {})), given.get(p.id))
                for p in song.parts}

    T = min(song.duration / speed + _TAIL, max_s)
    n_frames = max(1, int(round(T * FPS)))
    spf = SR // FPS                                # 1764: whole samples per frame
    N = n_frames * spf
    mix = np.zeros((N, 2), dtype=np.float32)

    playing = [p for p in song.parts if settings[p.id]["text"] and not settings[p.id]["mute"]]
    shown = [p for p in song.parts if settings[p.id]["text"] and settings[p.id]["visible"]]
    wanted = {p.id: p for p in playing + shown}
    playing_ids = {p.id for p in playing}

    # Samples first: the cheapest place to find out a word cannot be said.
    sample_of: dict[str, samples.Sample] = {}
    problems: list[dict] = []
    for i, part in enumerate(wanted.values()):
        say("samples", i, len(wanted))
        s = settings[part.id]
        try:
            sample_of[part.id] = samples.build(s["text"], s["take"], _hit(part, s))
        except samples.SampleError as exc:
            problems.append({"part": part.id, "name": part.name, "text": s["text"],
                             "error": str(exc)})
    say("samples", len(wanted), len(wanted))
    if not sample_of:
        raise YtpmvError("None of the parts has a sound this voice can make.")

    hits: dict[str, list[_Hit]] = {}
    total = sum(len(p.notes) for p in wanted.values() if p.id in sample_of)
    done = [0]

    def tick():
        done[0] += 1
        if done[0] % 200 == 0:
            say("notes", done[0], total)

    for part in wanted.values():
        if part.id in sample_of:
            hits[part.id] = _play(part, settings[part.id], sample_of[part.id].voice,
                                  mix if part.id in playing_ids else None,
                                  0.0, T, speed, g_transpose, flip, tick)
    say("notes", total, total)

    tmpdir = tempfile.mkdtemp(prefix="ytpmv_")
    wav = os.path.join(tmpdir, "mix.wav")
    _write_wav(wav, _finish_mix(mix))
    del mix

    tiles = [p for p in shown if p.id in sample_of]
    if not tiles:
        tiles = [p for p in playing if p.id in sample_of][:1]
    cols, rows, tw, th = grid(len(tiles))
    W, H = cols * tw, rows * th
    frames_of = {p.id: sample_of[p.id].frames(tw, th) for p in tiles}
    starts_of = {p.id: [h.start for h in hits.get(p.id, [])] for p in tiles}

    run_id = uuid.uuid4().hex[:10]
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"ytpmv_{run_id}.mp4")
    vf = ["format=yuv420p"]
    font = g.subtitle_font() if labels else None
    if font:
        for i, p in enumerate(tiles):
            x, y = (i % cols) * tw, (i // cols) * th
            text = settings[p.id]["text"].replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
            f = font.replace("\\", "/").replace(":", "\\:")
            vf.insert(0, f"drawtext=fontfile='{f}':text='{text}':expansion=none:fontcolor=white"
                         f":fontsize={max(12, th // 12)}:borderw=2:bordercolor=black"
                         f":x={x + 8}:y={y + th - th // 12 - 10}")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
           "-map", "0:v", "-map", "1:a", "-vf", ",".join(vf),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2",
           "-shortest", "-movflags", "+faststart", out_path]
    errlog = open(os.path.join(tmpdir, "ffmpeg.log"), "w+b")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errlog)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    try:
        for f in range(n_frames):
            if f % 25 == 0:
                say("frames", f, n_frames)
            t = f / FPS
            for i, p in enumerate(tiles):
                x0, y0 = (i % cols) * tw, (i // cols) * th
                canvas[y0:y0 + th, x0:x0 + tw] = _tile(frames_of[p.id], hits.get(p.id, []),
                                                       starts_of[p.id], t, flash, dim)
            try:
                proc.stdin.write(canvas.tobytes())
            except (BrokenPipeError, OSError):
                break
        proc.stdin.close()
        say("encoding", 0, 1)
        rc = proc.wait()
        if rc != 0:
            errlog.seek(0)
            raise RuntimeError("FFmpeg failed: " + errlog.read().decode(errors="replace")[-500:])
        say("encoding", 1, 1)
    finally:
        if proc.poll() is None:
            proc.kill()
        errlog.close()
        for name in os.listdir(tmpdir):
            try:
                os.remove(os.path.join(tmpdir, name))
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    log.info("YTPMV %s  %d parts  %.1fs  %dx%d", out_path, len(tiles), T, W, H)
    return {
        "video_url": f"/output/{os.path.basename(out_path)}",
        "duration": round(T, 2),
        "size": [W, H],
        "song": {k: v for k, v in song.summary().items() if k != "parts"},
        "parts": [{"id": p.id, "name": p.name, "role": p.role,
                   **{k: settings[p.id][k] for k in ("text", "take", "octave", "transpose", "mode",
                                                   "mute", "visible")},
                   "pitch": sample_of[p.id].voice.info.as_dict() if p.id in sample_of else None}
                  for p in song.parts],
        "problems": problems,
    }


def _hit(part: music.Part, s: dict) -> str | None:
    return part.drum_group if part.is_drums and s.get("hit", True) else None


def _play(part: music.Part, s: dict, voice, mix: np.ndarray | None, t_from: float,
          T: float, speed: float, g_transpose: int, flip: bool, tick=None) -> list[_Hit]:
    """Sing every note of *part* between t_from and T into *mix*; return its hits.

    Times are in the rendered song's clock (after *speed*), and the mix starts
    at t_from. *mix* None still works out the hits, for a tile that is shown
    but muted.
    """
    N = mix.shape[0] if mix is not None else 0
    gl = math.cos((s["pan"] + 1) * math.pi / 4)
    gr = math.sin((s["pan"] + 1) * math.pi / 4)
    shift = 12 * s["octave"] + s["transpose"] + (0 if part.is_drums else g_transpose)
    ref = part.median_pitch() + 12 * s["octave"]           # for a voice with no pitch
    mode = s["mode"]
    if mode == "perfect" and not voice.info.f0:
        mode = "tape"
    notes = sorted(part.notes, key=lambda n: (n.start, n.pitch))
    starts = sorted({n.start for n in notes})
    hits: list[_Hit] = []
    last_start = None
    for n in notes:
        if tick:
            tick()
        t0 = n.start / speed
        if t0 >= T:
            break
        if t0 < t_from:
            continue
        j = bisect.bisect_right(starts, n.start)
        nxt = starts[j] / speed if j < len(starts) else T
        if s["sustain"] == "ring":
            dur, stretch = min(voice.duration, nxt - t0), False
        else:
            dur, stretch = n.dur / speed, True
        dur = min(dur, T - t0)
        if dur <= 0:
            continue
        midi = n.pitch + shift
        if part.is_drums or mode == "raw":
            semis = s["transpose"]
            if part.drum_group == "toms":
                # Toms are tuned: GM gives each its own note, low to high.
                semis += n.pitch - round(part.median_pitch())
            y, warp = voice.render(None, dur, "raw", semitones=semis, stretch=stretch)
        elif mode == "tape" and not voice.info.f0:
            y, warp = voice.render(None, dur, "raw", semitones=midi - ref, stretch=stretch)
        else:
            y, warp = voice.render(midi_to_hz(midi), dur, mode, stretch=stretch)
        if mix is not None and s["volume"] > 0:
            a = int(round((t0 - t_from) * SR))
            k = min(len(y), N - a)
            if k > 0:
                gain = s["volume"] * (0.35 + 0.65 * n.velocity / 127.0)
                mix[a:a + k, 0] += y[:k] * (gain * gl)
                mix[a:a + k, 1] += y[:k] * (gain * gr)
        if n.start != last_start:              # a chord is one hit on screen
            hits.append(_Hit(t0, dur, warp, flip and len(hits) % 2 == 1))
            last_start = n.start
    return hits


def preview_part(midi_id: str, part_settings: dict, seconds: float = 6.0) -> dict:
    """A few seconds of one part, alone, as it will sound -- audio only.

    Starts at the part's first note rather than the top of the song, since a
    part that comes in at bar 17 would otherwise preview as silence.
    """
    song = load_song(midi_id)
    part = song.part(part_settings.get("id", ""))
    if part is None:
        raise YtpmvError("No such part.")
    s = _merge(_default_settings_for(part, {}), part_settings)
    if not s["text"]:
        raise YtpmvError("Give the part a word first.")
    try:
        sample = samples.build(s["text"], s["take"], _hit(part, s))
    except samples.SampleError as exc:
        raise YtpmvError(str(exc)) from None
    s["volume"], s["pan"] = 1.0, 0.0
    t_from = max(0.0, (part.notes[0].start if part.notes else 0.0) - 0.05)
    T = min(t_from + seconds, song.duration + _TAIL)
    mix = np.zeros((int((T - t_from) * SR), 2), dtype=np.float32)
    _play(part, s, sample.voice, mix, t_from, T, 1.0, 0, False)
    name = hashlib.sha1(repr((midi_id, sorted(s.items()))).encode()).hexdigest()[:12]
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", f"ytpmv_sample_{name}.wav")
    _write_wav(path, _finish_mix(mix))
    return {"audio_url": "/output/" + os.path.basename(path),
            "from": round(t_from, 2), "settings": s, "sample": sample.describe()}


def _scale(img: np.ndarray, k: float) -> np.ndarray:
    if k == 1.0:
        return img
    # uint32: 255 * 320 does not fit in 16 bits, and the flash wrapped to cyan.
    return np.minimum(img.astype(np.uint32) * int(k * 256) >> 8, 255).astype(np.uint8)


def _tile(frames: np.ndarray, hits: list[_Hit], starts: list[float], t: float,
          flash: bool, dim: bool) -> np.ndarray:
    i = bisect.bisect_right(starts, t) - 1
    if i < 0:
        return _scale(frames[0], 0.35) if dim else frames[0]
    h = hits[i]
    since = t - h.start
    src = h.warp.src(min(since, h.dur))
    img = frames[min(max(int(src * FPS), 0), len(frames) - 1)]
    if h.flip:
        img = img[:, ::-1]
    if flash and since < 2.0 / FPS:
        return _scale(img, 1.25)
    if dim and since > h.dur:
        return _scale(img, max(0.45, 1.0 - (since - h.dur) / 0.3 * 0.55))
    return img
