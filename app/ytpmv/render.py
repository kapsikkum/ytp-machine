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
import random
import re
import subprocess
import tempfile
import uuid
import wave
from dataclasses import dataclass

import numpy as np

import app.generate as g
from app.database import DATA_DIR, active
from app.ytpmv import music, nes, recommend, samples, tone as tones, ym2612
from app.ytpmv.pitch import SR, Warp, midi_to_hz

log = logging.getLogger(__name__)

FPS = samples.FPS
MIDI_DIR = os.path.join(DATA_DIR, "ytpmv", "midi")
MAX_SECONDS = float(os.environ.get("YTPMV_MAX_SECONDS", "300"))
MAX_MIDI_BYTES = 2 * 1024 * 1024
_OUT_W, _OUT_H = 1920, 1080
_TAIL = 0.8                    # let the last notes ring out after the final note-off
_MODES = ("perfect", "tape", "raw")
# How a part avoids sounding like one recording on a loop.
VARY = {
    "off": "the one take, every time",
    "rotate": "the best few takes in turn",
    "random": "those takes in no order",
    "ultra": "a different word every hit, each sung on the note",
}
_ULTRA_POOL = 5          # words a part throws about in the ultra mode

# How loud and where each kind of part sits unless told otherwise.
_VOLUME = {"lead": 1.0, "bass": 1.0, "rhythm": 0.75, "chords": 0.7,
           # The kick sits a little above the rest: it is the only part whose
           # job is to be felt, and peak-matching a saturated low-passed hit
           # leaves it quieter than it sounds like it should be.
           "drums:kick": 1.2, "drums:snare": 0.9, "drums:hats": 0.55,
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
    # Single sounds, cut out of words wherever the aligner found them.
    out["units"] = {k: len(v) for k, v in sorted(samples.units().items(),
                                                 key=lambda kv: -len(kv[1]))}
    out["voice"] = active()["name"]
    for p in out["parts"]:
        rec = recs.get(p["id"], {})
        p["recommended"] = rec
        p["settings"] = _default_settings_for(song.part(p["id"]), rec)
    return out


# ── Settings ──────────────────────────────────────────────────────────────────

# The machines a song can be played on, and what to call them. Each has two
# ways of going about it, because each machine had two: a synthesiser, and a
# channel for playing back a recording.
#
# The synthesised ones throw the voice away -- the chip makes the sound from
# nothing and only the timing and loudness of the note survive. Which is a
# thing worth wanting, and also a strange default for a machine whose whole
# job is a man saying words, so the sampled ones are here as well: those keep
# every bit of the voice, pitch-perfect as ever, and play it out through the
# channel the console actually used for speech.
CHIPS = {
    "md": "the Mega Drive's YM2612, synthesising: four-operator FM",
    "md-voice": "the voice, played off the Mega Drive's 8-bit sample channel",
    "nes": "the NES's 2A03, synthesising: two pulses, a triangle and noise",
    "nes-voice": "the voice, played off the NES's delta-modulation channel",
}

# What people typed before the names got shorter, and before there was
# anything to choose between.
CHIP_ALIASES = {"megadrive": "md", "megadrive-voice": "md-voice",
                "genesis": "md", "true": "md", "on": "md", "yes": "md"}


def which_chip(asked) -> str | None:
    """Which machine *asked* means, or None for none.

    True still means the Mega Drive, which is all there was when this was a
    checkbox rather than a choice.
    """
    if asked is True:
        return "md"
    if not asked or asked is False:
        return None
    name = str(asked).strip().lower()
    name = CHIP_ALIASES.get(name, name)
    return name if name in CHIPS else None


def chip_tone(part: music.Part, chip: str = "megadrive") -> str:
    """What *part* is played by when the whole song goes through *chip*.

    Every part, drums included. A console usually sampled its kit rather than
    synthesising it, and that is still available per part -- but "through the
    chip" ought to mean through the chip, so the synthesised settings use the
    synthesised kit.
    """
    machine, _, mode = chip.partition("-")
    if mode == "voice":
        return tones.SAMPLED[machine]
    if machine == "nes":
        return nes.voice_for(part.role, part.program).name
    return ym2612.patch_for(part.role, part.program).name


def _default_settings_for(part: music.Part, rec: dict) -> dict:
    return {
        "id": part.id,
        "text": rec.get("text"),
        "take": rec.get("take", 0),
        # The best few takes of this word, for the variety modes to rotate
        # through when one is turned on, and the other words that would have
        # done, for the ultra mode to throw about.
        "takes": rec.get("takes") or [rec.get("take", 0)],
        "pool": rec.get("pool") or [{"text": rec.get("text"), "take": rec.get("take", 0)}],
        "octave": rec.get("octave", 0),
        "transpose": 0,
        "volume": _VOLUME.get(part.role, 0.8),
        "pan": _PAN.get(part.role, 0.0),
        "mode": rec.get("mode") or ("raw" if part.is_drums else "perfect"),
        "sustain": "ring" if part.is_drums else "note",
        # What the part is played through, and always nothing until asked.
        # The file naming General MIDI 36 (Slap Bass) is a hint, not an
        # instruction: the point of this thing is a corpus speaking, and
        # swapping the voice for a synthesiser is a decision to be made per
        # song rather than one made on your behalf.
        "tone": "clean",
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
    if isinstance(given.get("takes"), list):
        picked = [_num(t, 0, 0, 10_000, int) for t in given["takes"][:6]]
        s["takes"] = picked or [s["take"]]
    elif "take" in given or (given.get("text") or "").strip():
        # A word or take chosen by hand is the one meant; nothing rotates away
        # from it, and the ultra mode throws that one word about instead.
        s["takes"] = [s["take"]]
        s["pool"] = [{"text": s["text"], "take": s["take"]}]
    if isinstance(given.get("pool"), list):
        chosen = [{"text": str(e.get("text"))[:60], "take": _num(e.get("take", 0), 0, 0, 10_000, int)}
                  for e in given["pool"][:_ULTRA_POOL] if isinstance(e, dict) and e.get("text")]
        s["pool"] = chosen or s["pool"]
    s["octave"] = _num(given.get("octave", s["octave"]), s["octave"], -4, 4, int)
    s["transpose"] = _num(given.get("transpose", s["transpose"]), 0, -24, 24, int)
    s["volume"] = _num(given.get("volume", s["volume"]), s["volume"], 0.0, 2.0)
    s["pan"] = _num(given.get("pan", s["pan"]), s["pan"], -1.0, 1.0)
    if given.get("mode") in _MODES:
        s["mode"] = given["mode"]
    if given.get("sustain") in ("note", "ring"):
        s["sustain"] = given["sustain"]
    asked_tone = tones.resolve(given.get("tone"))
    if asked_tone:
        s["tone"] = asked_tone
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
    take: int = 0              # which of the part's takes played it


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
    # Both off by default. Rotating takes does stop a part sounding like a
    # loop, but it also changes the part: takes differ in length and
    # inflection as well as in the exact sound, and across a whole song that
    # reads as the song wandering rather than as a player. Available, not
    # assumed -- a repeated sample is what the form sounds like.
    vary = opts.get("vary", "off")
    if isinstance(vary, bool):
        vary = "rotate" if vary else "off"
    if vary not in VARY:
        vary = "off"
    jitter = bool(opts.get("jitter", False))
    # One switch for the lot: every part out through an emulated sound chip.
    # Parts that were given a tone of their own keep it.
    chip = which_chip(opts.get("chip", False))

    given = {p.get("id"): p for p in (params.get("parts") or []) if isinstance(p, dict)}
    # Recommendations only for parts the request did not choose a sound for.
    need = [p for p in song.parts if not (given.get(p.id) or {}).get("text")]
    recs = recommend.recommend(music.Song(need, song.duration, song.bpm, song.time_signature,
                                          song.key, song.key_confidence)) if need else {}
    settings = {p.id: _merge(_default_settings_for(p, recs.get(p.id, {})), given.get(p.id))
                for p in song.parts}
    if chip:
        for p in song.parts:
            asked = given.get(p.id) or {}
            if tones.overrides_switch(asked.get("tone")):
                continue
            settings[p.id]["tone"] = chip_tone(p, chip)
            # The octave a part was moved by is there because a mouth cannot
            # go where the score does: a bass line at 49 Hz comes back up an
            # octave, chords come down three. The chip has no such limit, and
            # those shifts are not musical decisions -- applying them to it
            # spreads parts written in one register across five octaves and
            # takes the arrangement apart. So a part the chip is playing goes
            # back to the octave it was written in, unless one was asked for.
            if "octave" not in asked and settings[p.id]["tone"] in tones.SYNTH:
                settings[p.id]["octave"] = 0

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
    sample_of: dict[str, list[samples.Sample]] = {}
    problems: list[dict] = []
    for i, part in enumerate(wanted.values()):
        say("samples", i, len(wanted))
        s = settings[part.id]
        if vary == "ultra":
            want = [(e["text"], e["take"]) for e in s["pool"][:_ULTRA_POOL]]
        elif vary == "off":
            want = [(s["text"], s["take"])]
        else:
            want = [(s["text"], t) for t in s["takes"]]
        got, failed = [], None
        for text, take in dict.fromkeys(want):  # in order, no repeats
            try:
                got.append(samples.build(text, take, _hit(part, s)))
            except samples.SampleError as exc:
                failed = exc
        if got:
            sample_of[part.id] = got
        else:
            problems.append({"part": part.id, "name": part.name, "text": s["text"],
                             "error": str(failed)})
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
            hits[part.id] = _play(part, settings[part.id], sample_of[part.id],
                                  mix if part.id in playing_ids else None,
                                  0.0, T, speed, g_transpose, flip, tick, jitter, vary)
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
    frames_of = {p.id: [smp.frames(tw, th) for smp in sample_of[p.id]] for p in tiles}
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
        "options": {"vary": vary, "jitter": jitter, "speed": speed,
                    "transpose": g_transpose, "max_seconds": max_s},
        "parts": [{"id": p.id, "name": p.name, "role": p.role,
                   **{k: settings[p.id][k] for k in ("text", "take", "octave", "transpose", "mode",
                                                   "tone", "mute", "visible")},
                   "played": [smp.text for smp in sample_of.get(p.id, [])],
                   "takes_played": len(sample_of.get(p.id, [])),
                   "pitch": sample_of[p.id][0].voice.info.as_dict() if p.id in sample_of else None}
                  for p in song.parts],
        "problems": problems,
    }


def _hit(part: music.Part, s: dict) -> str | None:
    return part.drum_group if part.is_drums and s.get("hit", True) else None


def _play(part: music.Part, s: dict, takes: list, mix: np.ndarray | None, t_from: float,
          T: float, speed: float, g_transpose: int, flip: bool, tick=None,
          jitter: bool = True, vary: str = "rotate") -> list[_Hit]:
    """Sing every note of *part* between t_from and T into *mix*; return its hits.

    *takes* are the Samples to play, rotated between hits (see VARY). Times are
    in the rendered song's clock (after *speed*), and the mix starts at t_from.
    *mix* None still works out the hits, for a tile that is shown but muted.
    """
    N = mix.shape[0] if mix is not None else 0
    gl = math.cos((s["pan"] + 1) * math.pi / 4)
    gr = math.sin((s["pan"] + 1) * math.pi / 4)
    shift = 12 * s["octave"] + s["transpose"] + (0 if part.is_drums else g_transpose)
    ref = part.median_pitch() + 12 * s["octave"]           # for a voice with no pitch
    # Seeded on the part, so a render is the same twice: "random" here means
    # unpatterned, not different every time you press the button.
    rng = random.Random(f"{part.id}|{len(takes)}|{s['text']}")
    notes = sorted(part.notes, key=lambda n: (n.start, n.pitch))
    starts = sorted({n.start for n in notes})
    hits: list[_Hit] = []
    last_start = None
    pick = 0
    for n in notes:
        if tick:
            tick()
        t0 = n.start / speed
        if t0 >= T:
            break
        if t0 < t_from:
            continue
        if n.start != last_start:              # a chord is one hit: one take
            pick = (0 if vary == "off" or len(takes) == 1 else
                    rng.randrange(len(takes)) if vary in ("random", "ultra") else
                    len(hits) % len(takes))
        sample = takes[pick]
        voice = sample.voice
        mode = s["mode"]
        if mode == "perfect" and not voice.info.f0:
            mode = "tape"
        j = bisect.bisect_right(starts, n.start)
        nxt = starts[j] / speed if j < len(starts) else T
        if s["sustain"] == "ring":
            dur, stretch = min(voice.duration, nxt - t0), False
        else:
            dur, stretch = n.dur / speed, True
        dur = min(dur, T - t0)
        if dur <= 0:
            continue
        # A drum hit that is identical every beat is a machine gun. A hair of
        # detune (drums only -- a pitched part is meant to be exactly in tune)
        # and a decibel either way is what a person hitting something sounds
        # like. Quantised, so the note cache still gets hits.
        detune = round(rng.uniform(-0.25, 0.25), 2) if jitter and part.is_drums else 0.0
        loud = 10 ** (rng.uniform(-1.2, 1.2) / 20.0) if jitter else 1.0
        midi = n.pitch + shift
        if part.is_drums or mode == "raw":
            semis = s["transpose"] + detune
            if part.drum_group == "toms":
                # Toms are tuned: GM gives each its own note, low to high.
                semis += n.pitch - round(part.median_pitch())
            y, warp = voice.render(None, dur, "raw", semitones=semis, stretch=stretch)
        elif mode == "tape" and not voice.info.f0:
            y, warp = voice.render(None, dur, "raw", semitones=midi - ref, stretch=stretch)
        else:
            y, warp = voice.render(midi_to_hz(midi), dur, mode, stretch=stretch)
        if s.get("tone", "clean") != "clean":
            # A patch synthesises the note, so it has to be told which note --
            # and how hard it was struck, since it is not playing a recording
            # of that any more. Drums have no note, and say so.
            y = tones.apply(y, s["tone"], hz=None if part.is_drums else midi_to_hz(midi),
                            level=max(0.2, n.velocity / 127.0))
        if mix is not None and s["volume"] > 0:
            a = int(round((t0 - t_from) * SR))
            k = min(len(y), N - a)
            if k > 0:
                gain = s["volume"] * loud * (0.35 + 0.65 * n.velocity / 127.0)
                mix[a:a + k, 0] += y[:k] * (gain * gl)
                mix[a:a + k, 1] += y[:k] * (gain * gr)
        if n.start != last_start:
            hits.append(_Hit(t0, dur, warp, flip and len(hits) % 2 == 1, pick))
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
    _play(part, s, [sample], mix, t_from, T, 1.0, 0, False)
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


def _tile(takes: list, hits: list[_Hit], starts: list[float], t: float,
          flash: bool, dim: bool) -> np.ndarray:
    i = bisect.bisect_right(starts, t) - 1
    if i < 0:
        return _scale(takes[0][0], 0.35) if dim else takes[0][0]
    h = hits[i]
    frames = takes[h.take % len(takes)]
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
