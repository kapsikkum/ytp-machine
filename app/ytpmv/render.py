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
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
import wave
from dataclasses import dataclass, replace
from contextlib import ExitStack

import numpy as np

import app.generate as g
from app.database import DATA_DIR, active
from app.ytpmv import (gb, master as mastering, music, nes, recommend, samples,
                        scope, screen as screens, sid, sms, tone as tones, ym2612)
from app.ytpmv.pitch import SR, Warp, midi_to_hz

log = logging.getLogger(__name__)

FPS = samples.FPS
MIDI_DIR = os.path.join(DATA_DIR, "ytpmv", "midi")
MAX_SECONDS = float(os.environ.get("YTPMV_MAX_SECONDS", "900"))
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
_RING_MAX = 1.5          # how long a synthesised note left to ring may go on for

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


_songs: dict = {}


def _midi_path(midi_id: str) -> str:
    if not _ID_RE.fullmatch(midi_id or ""):
        raise YtpmvError("No such MIDI file.")
    path = os.path.join(MIDI_DIR, f"{midi_id}.mid")
    if not os.path.isfile(path):
        raise YtpmvError("That MIDI file has gone -- upload it again.")
    return path


def _forget_midi(midi_id: str) -> None:
    for key in list(_songs):
        if key[0] == midi_id:
            _songs.pop(key, None)


def rename_midi(midi_id: str, title: str) -> dict:
    path = _midi_path(midi_id)
    title = title.strip()
    if not title or len(title) > 80 or any(ord(c) < 32 for c in title):
        raise YtpmvError("Use a MIDI title of 1–80 characters on one line.")
    temporary = path + ".title." + uuid.uuid4().hex + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as fh:
            fh.write(title)
        os.replace(temporary, path + ".title")
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    try:
        os.remove(path + ".info.json")
    except FileNotFoundError:
        pass
    _forget_midi(midi_id)
    return {"midi_id": midi_id, "title": title}


def delete_midi(midi_id: str) -> dict:
    path = _midi_path(midi_id)
    os.remove(path)
    for suffix in (".title", ".info.json"):
        try:
            os.remove(path + suffix)
        except FileNotFoundError:
            pass
    _forget_midi(midi_id)
    return {"midi_id": midi_id, "deleted": True}


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
    # One render asks for the song three times (analyse, recommend, render),
    # and a big file is seconds of parsing each time. Nothing changes a Song
    # once it is parsed, so the same one is handed out while the file is.
    key = (midi_id, hashlib.sha1(data).hexdigest(), title)
    song = _songs.get(key)
    if song is None:
        try:
            song = music.parse(data, title=title)
        except Exception as exc:  # noqa: BLE001 -- mido raises a zoo of types on bad files
            raise YtpmvError(f"Couldn't read that MIDI file: {exc}") from exc
        if len(_songs) >= 4:
            _songs.pop(next(iter(_songs)))
        _songs[key] = song
    if not song.parts:
        raise YtpmvError("That MIDI file has no notes in it.")
    return song


def library() -> list[dict]:
    """Every MIDI file ever uploaded, newest first, each described.

    The description is cached beside the file, because a big arrangement
    takes a second or two to read and the list is read every time the page
    opens. It is redone when the file or its title is newer than the cache.
    """
    try:
        names = os.listdir(MIDI_DIR)
    except FileNotFoundError:
        return []
    out = []
    for name in names:
        mid = name[:-4]
        if not name.endswith(".mid") or not _ID_RE.match(mid):
            continue
        path = os.path.join(MIDI_DIR, name)
        cache = path + ".info.json"
        newest = max(os.path.getmtime(p) for p in (path, path + ".title") if os.path.exists(p))
        info = None
        if os.path.exists(cache) and os.path.getmtime(cache) >= newest:
            try:
                with open(cache, encoding="utf-8") as fh:
                    info = json.load(fh)
            except (OSError, ValueError):
                info = None
        if info is None:
            try:
                song = load_song(mid)
            except YtpmvError:
                continue                        # unreadable: nothing to offer
            drums = [p for p in song.parts if p.is_drums]
            info = {"title": song.title, "duration": round(song.duration, 1),
                    "bpm": round(song.bpm), "key": song.key,
                    "time_signature": f"{song.time_signature[0]}/{song.time_signature[1]}",
                    "tempo_changes": song.tempo_changes,
                    "parts": len(song.parts) - len(drums), "drums": len(drums),
                    "notes": sum(len(p.notes) for p in song.parts),
                    "description": song.describe()}
            try:
                with open(cache, "w", encoding="utf-8") as fh:
                    json.dump(info, fh)
            except OSError:
                pass
        out.append({"midi_id": mid, **info, "bytes": os.path.getsize(path),
                    "added": round(os.path.getmtime(path))})
    out.sort(key=lambda m: -m["added"])
    return out


def analyse(midi_id: str, progress=None) -> dict:
    """The song in plain words, plus a suggested sound for every part."""
    song = load_song(midi_id)
    recs = recommend.recommend(song, progress=progress, pool=samples.corpus_pool())
    out = song.summary()
    out["midi_id"] = midi_id
    # What this voice has besides words. A spew or a click is already a
    # percussive sound, so the page offers them for any part.
    out["noises"] = samples.noises()
    # Single sounds, cut out of words wherever the aligner found them.
    out["units"] = {k: len(v) for k, v in sorted(samples.units().items(),
                                                 key=lambda kv: -len(kv[1]))}
    out["voice"] = active()["name"]
    out["corpus"] = active()["slug"]
    # What each part can be played by, and what every chip would pick for it
    # if left alone, so the page can say so before anything is rendered.
    out["instruments"] = {**tones.catalogue(), "programs": music.GM_PROGRAMS, "chips": CHIPS}
    low = bass_line(song)
    for p in out["parts"]:
        rec = recs.get(p["id"], {})
        part = song.part(p["id"])
        p["recommended"] = rec
        p["settings"] = _default_settings_for(part, rec)
        p["chip_tones"] = {c: chip_tone(part, c, as_bass=(part.id == low)) for c in CHIPS}
        # And with no voice, where a drum has nothing to play but the kit.
        p["chip_tones_synth"] = {c: chip_tone(part, c, as_bass=(part.id == low), voice_drums=False)
                                 for c in CHIPS}
        # A bounded overview of real MIDI notes for the track workbench.
        stride = max(1, math.ceil(len(part.notes) / 1024))
        p["activity"] = [[round(n.start, 4), round(n.dur, 4), n.pitch]
                         for n in part.notes[::stride]]
        p["program_tones"] = {
            c: [chip_tone(part, c, as_bass=(part.id == low), program=program)
                for program in range(len(music.GM_PROGRAMS))]
            for c in CHIPS if not part.is_drums and not c.endswith("-voice")}
    return out


def timeline_notes(midi_id: str, start: float, end: float | None, part_ids: list[str]) -> dict:
    """Read-only viewport detail, independent of voice recommendations or ML."""
    song = load_song(midi_id)
    end = song.duration if end is None else end
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise YtpmvError("Choose a finite timeline range with its end after its start.")
    if not part_ids or len(part_ids) > 64 or any(song.part(pid) is None for pid in part_ids):
        raise YtpmvError("Choose between one and 64 installed MIDI parts.")
    limit = max(1, 24000 // len(part_ids))
    out, counts = {}, {}
    for pid in dict.fromkeys(part_ids):
        notes = [n for n in song.part(pid).notes if n.start < end and n.start + n.dur >= start]
        counts[pid] = len(notes)
        stride = max(1, math.ceil(len(notes) / limit))
        out[pid] = [[round(n.start, 5), round(n.dur, 5), n.pitch] for n in notes[::stride]]
    return {"parts": out, "counts": counts, "sampled": any(counts[p] > len(out[p]) for p in out)}


def recommend_part(midi_id: str, part_id: str, corpus: str | None = None) -> dict:
    """The sound choices for one part in one installed voice."""
    song = load_song(midi_id)
    part = song.part(part_id)
    if part is None:
        raise YtpmvError("No such part.")
    with samples.corpus_context(corpus) as voice:
        one = music.Song([part], song.duration, song.bpm, song.time_signature,
                         song.key, song.key_confidence)
        rec = recommend.recommend(one, pool=samples.corpus_pool()).get(part_id, {})
        return {"id": part_id, "corpus": voice["slug"], "voice": voice["name"],
                "recommended": rec, "settings": _default_settings_for(part, rec),
                "noises": samples.noises(),
                "units": {k: len(v) for k, v in samples.units().items()}}


def prepare_parts(parts: list[dict], synth: bool = False) -> list[dict]:
    """Validate voice choices before queueing, and pin omitted choices now."""
    result = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        chosen = dict(part)
        if not synth:
            with samples.corpus_context(chosen.get("corpus")) as corpus:
                chosen["corpus"] = corpus["slug"]
        result.append(chosen)
    return result


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
    "sms": "the Master System's SN76489: three squares and a shift register",
    "sms-voice": "the voice, hammered out of the Master System's volume register",
    "snes": "the SNES's S-DSP playing a General MIDI bank cut down to fit it",
    "snes-voice": "the voice through the SNES's sampler, grain and echo and all",
    "gb": "the Game Boy's chip: two pulses, a wave table and noise; green screen",
    "gb-voice": "the voice through a Game Boy's wave table; green screen",
    "gbc": "the same chip as the Game Boy, on the Color's screen",
    "gbc-voice": "the voice through the same wave table, on the Color's screen",
    "opl2": "an AdLib's OPL2 playing Doom's instrument bank; EGA screen",
    "opl2-voice": "the voice off an original Sound Blaster; EGA screen",
    "opl3": "a Sound Blaster's OPL3, same bank, eight waveforms; VGA screen",
    "opl3-voice": "the voice off a sixteen-bit Sound Blaster; VGA screen",
    "sid": "the C64's 6581 SID, filter and all; C64 screen",
    "sid-voice": "the voice hammered out of the 6581's volume register; C64 screen",
    "sid8580": "the later 8580 SID, cleaner filter; C64 screen",
    "sid8580-voice": "the same trick on an 8580, where it barely works; C64 screen",
}

# Machines whose picture is not named after them.
SCREEN_OF = {"opl2": "ega", "opl3": "vga", "sid": "c64", "sid8580": "c64"}

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


def synth_chip(chip: str | None) -> str:
    """The synthesising setting to play a song on when there is no voice.

    No chip means the Mega Drive, and a -voice setting means the same machine
    synthesising, since there is nothing to play off its sample channel.
    """
    return (chip or "md").partition("-")[0]


def screen_for(chip: str | None) -> str:
    """Which console's picture goes with *chip*: the same machine's, or none."""
    if not chip:
        return "none"
    machine = chip.partition("-")[0]
    machine = SCREEN_OF.get(machine, machine)
    return machine if machine in screens.SIZES else "none"


def bass_line(song: music.Song) -> str | None:
    """The part to treat as the bass when nothing in the song is called one.

    A MIDI file does not have to say which part is the bass, and plenty do
    not: Megalovania's bass line comes out of the analysis as "rhythm",
    because it is played on a piano patch. A synthesised part is then given
    whatever "rhythm" gets -- a thin pulse on the NES rather than the
    triangle that is the machine's actual bass voice -- and the song loses
    its bottom end. So the lowest melodic line is taken as the bass, as long
    as it really is low and is not the tune.
    """
    melodic = [p for p in song.parts if not p.is_drums and p.notes]
    if not melodic or any(p.role == "bass" for p in melodic):
        return None
    low = min(melodic, key=lambda p: p.median_pitch())
    return low.id if low.role != "lead" and low.median_pitch() < 53 else None


def chip_tone(part: music.Part, chip: str = "md", as_bass: bool = False,
              program: int | None = None, voice_drums: bool = True) -> str:
    """What *part* is played by when the whole song goes through *chip*.

    Tunes are synthesised by the chip. Drums, with *voice_drums*, are the
    voice's own drum hits played off that machine's sample channel -- which
    is how these consoles did their kits, Sonic's snares being samples off
    the DAC, and is also the point of the thing: the melody on the chip, the
    drums still him. It is what every chip used to do by accident, until a
    fix made the kits synthesise, and it was missed. The synthesised kits are
    still there to choose per part, and they are what a song with no voice
    gets, having nothing else to play its drums with.

    *as_bass* plays the part as the bass whatever the analysis called it; see
    bass_line. *program* stands in for the General MIDI instrument the file
    gave the part, when another was asked for.
    """
    machine, _, mode = chip.partition("-")
    program = part.program if program is None or part.is_drums else program
    if mode == "voice" or (part.is_drums and voice_drums):
        return tones.SAMPLED[machine]
    role = "bass" if as_bass and not part.is_drums else part.role
    if machine == "nes":
        return nes.voice_for(role, program).name
    if machine == "sms":
        return sms.voice_for(role, program).name
    if machine in ("gb", "gbc"):              # one chip; the two differ in their screens
        return gb.voice_for(role, program).name
    if machine in ("opl2", "opl3", "snes"):   # the patch or sample is chosen per note
        return machine
    if machine in ("sid", "sid8580"):
        return sid.voice_for(role, program).name
    return ym2612.patch_for(role, program).name


def floor_hz(tone: str) -> float:
    """The lowest note *tone* can play, or 0 when it has no such limit."""
    if tone in nes.VOICES:
        return nes.floor_hz(nes.VOICES[tone])
    if tone in sms.VOICES:
        return sms.floor_hz(sms.VOICES[tone])
    if tone in gb.VOICES:
        return gb.voice_floor(gb.VOICES[tone])
    return 0.0


def octaves_to_fit(notes: list, floor: float, shift: int) -> int:
    """Whole octaves to raise a part by so its lowest note is not under *floor*.

    For the whole part at once, never per note: raising only the notes that
    fall under the limit and leaving the rest keeps each note in tune and
    wrecks the line, since a note can end up above one it was written below.
    """
    if floor <= 0 or not notes:
        return 0
    lowest = midi_to_hz(min(n.pitch for n in notes) + shift)
    up = 0
    while lowest * 2.0 ** up < floor - 1e-6 and up < 6:
        up += 1
    return up


def _default_settings_for(part: music.Part, rec: dict) -> dict:
    return {
        "id": part.id,
        "corpus": active()["slug"],
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
        "pan": _num(rec.get("pan"), part.pan if part.pan is not None else _PAN.get(part.role, 0.0), -1.0, 1.0),
        "haas": bool(rec.get("haas", False)),
        "mode": rec.get("mode") or ("raw" if part.is_drums else "perfect"),
        "sustain": "ring" if part.is_drums else "note",
        # What the part is played through, and always nothing until asked.
        # The file naming General MIDI 36 (Slap Bass) is a hint, not an
        # instruction: the point of this thing is a corpus speaking, and
        # swapping the voice for a synthesiser is a decision to be made per
        # song rather than one made on your behalf.
        "tone": "clean",
        # The General MIDI instrument, which is what picks an OPL patch and
        # nudges which voice the other chips choose. The file's, until changed.
        "program": part.program,
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
    if "corpus" in given:
        s["corpus"] = given["corpus"]
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
    if defaults.get("program") is not None and given.get("program") is not None:
        s["program"] = _num(given["program"], defaults["program"], 0, 127, int)
    if "hit" in given:
        s["hit"] = bool(given["hit"])
    if "haas" in given:
        s["haas"] = bool(given["haas"])
    if s["text"] and defaults.get("text") is None:
        # Given a sound where none was suggested: it is meant to be heard.
        s["mute"], s["visible"] = False, True
    for k in ("mute", "visible"):
        if k in given:
            s[k] = bool(given[k])
    return s


# ── Layout ────────────────────────────────────────────────────────────────────

def grid(n: int) -> tuple[int, int, int, int]:
    """Nominal grid on a fixed widescreen canvas; rows may contain fewer cells."""
    n = max(n, 1)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows, _OUT_W // cols, _OUT_H // rows


GRID_LAYOUTS = ("balanced", "classic", "two-rows", "two-columns", "strips", "featured")


def classic_grid_shape(n: int) -> tuple[int, int]:
    """The original grid's row/column choice, with equal widescreen tiles."""
    n = max(1, n)
    return min(((cols, math.ceil(n / cols)) for cols in range(1, n + 1)),
               key=lambda shape: abs(math.log(shape[0] / shape[1])) + .3 * (shape[0] * shape[1] - n))


def grid_cells(n: int, width: int = _OUT_W, height: int = _OUT_H,
               layout: str = "balanced", centered: bool = True) -> list[tuple[int, int, int, int]]:
    """(x, y, width, height) in the widescreen canvas.

    Balance the row counts, putting the extra cells in the upper rows. Three
    parts are two on top and one centred across the full lower row. Fill layouts
    partition every pixel, including console-sized canvases. Classic keeps equal
    16:9 cells, leaving unused space black and optionally centering incomplete rows.
    """
    n = max(1, n)
    if layout == "classic":
        cols, rows = classic_grid_shape(n)
        scale = min(width / (cols * 16), height / (rows * 9))
        tw, th = max(1, math.floor(16 * scale)), max(1, math.floor(9 * scale))
        top = (height - rows * th) // 2 if centered else 0
        cells = []
        for r in range(rows):
            count = min(cols, n - r * cols)
            left = (width - count * tw) // 2 if centered else 0
            cells.extend((left + c * tw, top + r * th, tw, th) for c in range(count))
        return cells
    if layout == "featured" and n > 1:
        split = width * 2 // 3
        return [(0, 0, split, height)] + [(x + split, y, w, h)
            for x, y, w, h in grid_cells(n - 1, width - split, height, "sidebar")]
    rows = {"two-rows": min(n, 2), "two-columns": math.ceil(n / 2),
            "strips": 1, "sidebar": min(n, max(1, round(math.sqrt(n * height / width))))}.get(layout, grid(n)[1])
    cells = []
    for row_index in range(rows):
        count = n // rows + (row_index < n % rows)
        y0, y1 = row_index * height // rows, (row_index + 1) * height // rows
        for col in range(count):
            x0, x1 = col * width // count, (col + 1) * width // count
            cells.append((x0, y0, x1 - x0, y1 - y0))
    return cells


def ordered_tiles(tiles: list, order) -> list:
    """Ignore stale/duplicate IDs; unspecified visible tracks keep MIDI order."""
    if not isinstance(order, list):
        return tiles
    ranks = {pid: i for i, pid in reversed(list(enumerate(order))) if isinstance(pid, str)}
    return sorted(tiles, key=lambda p: ranks.get(p.id, len(order)))


# ── Rendering ─────────────────────────────────────────────────────────────────

@dataclass
class _Hit:
    start: float
    dur: float                 # how long the note sounds
    warp: Warp
    flip: bool
    take: int = 0              # which of the part's takes played it


def _write_wav(path: str, stereo: np.ndarray) -> None:
    # A minute at a time: a fifteen minute song converted in one go is three
    # more copies of it held at once, a gigabyte for the sake of a file write.
    step = 60 * SR
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        for i in range(0, len(stereo), step):
            w.writeframes((np.clip(stereo[i:i + step], -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def _finish_mix(mix: np.ndarray) -> np.ndarray:
    """Mastered, in place: see master.finish."""
    return mastering.finish(mix)


def _through_chip(song: music.Song, settings: dict, given: dict, chip: str | None,
                  g_transpose: int = 0, synth: bool = False) -> None:
    """Put the parts in *settings* through *chip*, in place.

    A part that was given a tone of its own keeps it; every other part gets
    whatever the chip plays that part with. With *synth* there is no voice:
    only a synthesised tone can be kept, every part plays whether or not a
    word was found for it, and its label is what it is played by.
    """
    low = bass_line(song) if chip else None
    for p in song.parts:
        if p.id not in settings:
            continue
        s_ = settings[p.id]
        asked = given.get(p.id) or {}
        # Every part knows the chip, chosen tone or not: a SID voice asked for
        # by name on a song through the 8580 is played by the 8580.
        s_["chip"] = chip
        keep = tones.overrides_switch(asked.get("tone"))
        if synth:
            keep = keep and tones.resolve(asked.get("tone")) in tones.SYNTH
        if chip and not keep:
            s_["tone"] = chip_tone(p, chip, as_bass=(p.id == low), program=s_.get("program"),
                                   voice_drums=not synth)
        if synth or s_["tone"] in tones.SYNTH:
            # Muted for want of a word is not muted: there are no words.
            s_["text"] = s_["tone"]
            s_["mute"] = bool(asked.get("mute", False))
            s_["visible"] = bool(asked.get("visible", True))
        # The octave a part was moved by is there because a mouth cannot
        # go where the score does: a bass line at 49 Hz comes back up an
        # octave, chords come down three. A chip has no such limit, and
        # those shifts are not musical decisions -- applying them to it
        # spreads parts written in one register across five octaves and
        # takes the arrangement apart. So a part a chip is playing goes
        # back to the octave it was written in, unless one was asked for --
        # whether the switch put the chip there or the part asked for it.
        if "octave" not in asked and s_["tone"] in tones.SYNTH:
            s_["octave"] = 0

    # Some machines cannot go as low as a part is written: the Master System
    # stops at 109 Hz, an NES pulse at 55. Such a part moves up by whole
    # octaves, all of it together, until its lowest note is playable. Unless
    # an octave was asked for, in which case the asker has heard it.
    for p in song.parts:
        if p.id not in settings:
            continue
        s_ = settings[p.id]
        floor = floor_hz(s_["tone"])
        if floor <= 0 or p.is_drums or "octave" in (given.get(p.id) or {}):
            continue
        shift = 12 * s_["octave"] + s_["transpose"] + g_transpose
        s_["octave"] += octaves_to_fit(p.notes, floor, shift)


def _close_stem(stem) -> None:
    mapping = getattr(stem, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def render_ytpmv(params: dict, progress=None) -> dict:
    with ExitStack() as resources:
        return _render_ytpmv(params, progress, resources)


def preview_mix(params: dict, progress=None) -> dict:
    """Play a selection or the remaining song through the video render's mixer."""
    options = dict(params.get("options") or {})
    length = params.get("seconds", 10)
    options["max_seconds"] = MAX_SECONDS if length is None else _num(length, 10, 1, MAX_SECONDS)
    options["flip"] = False
    keyboard = params.get("keyboard_notes")
    if keyboard is not None:
        if (not isinstance(keyboard, list) or not 1 <= len(keyboard) <= 24
                or any(type(note) is not int or not 0 <= note <= 127 for note in keyboard)
                or len(set(keyboard)) != len(keyboard)):
            raise YtpmvError("Choose 1–24 distinct MIDI notes for the keyboard.")
        options["max_seconds"] = len(keyboard) * 2
    with ExitStack() as resources:
        result = _render_ytpmv({**params, "options": options}, progress, resources,
                            audio_only=True, start_seconds=params.get("start", 0),
                            audition_note=params.get("note"), audition_part=params.get("part_id"),
                            audition_notes=keyboard)
    if keyboard is not None:
        result["keyboard"] = [{"note": note, "offset": i * 2,
                               "duration": min(2, result["duration"] - i * 2)}
                              for i, note in enumerate(keyboard)]
    return result


def _render_ytpmv(params: dict, progress, resources: ExitStack,
                  audio_only: bool = False, start_seconds: float = 0,
                  audition_note: int | None = None, audition_part: str | None = None,
                  audition_notes: list[int] | None = None) -> dict:
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
    source_song = song
    opts = params.get("options") or {}
    max_s = _num(opts.get("max_seconds", MAX_SECONDS), MAX_SECONDS, 1.0, MAX_SECONDS)
    speed = _num(opts.get("speed", 1.0), 1.0, 0.25, 4.0)
    if audition_note is not None or audition_notes is not None:
        part = song.part(audition_part)
        if part is None:
            raise YtpmvError("No such instrument to audition.")
        # A private one-note score; never mutate the cached MIDI or its parts.
        pitches = audition_notes if audition_notes is not None else [audition_note]
        part = replace(part, notes=[music.Note(i * 2 * speed, .65 * speed, int(note), 100)
                                    for i, note in enumerate(pitches)])
        song = replace(song, parts=[part], duration=((len(pitches) - 1) * 2 + .65) * speed)
        start_seconds = 0
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
    # Measure every part and set it where its job says it should sit, rather
    # than trusting a table that cannot tell a whisper from a brass patch.
    balance = bool(opts.get("balance", True))
    # Only as many notes at once as the chip had channels; off by default.
    accurate = bool(opts.get("accurate", False))
    # One switch for the lot: every part out through an emulated sound chip.
    # Parts that were given a tone of their own keep it.
    chip = which_chip(opts.get("chip", False))
    # And the picture goes through the same machine: its resolution, and its
    # colours and no others. Not a separate choice -- a Mega Drive soundtrack
    # over a picture the Mega Drive could never have drawn is two machines,
    # and nobody asking for one of them meant that.
    # No voice at all: the corpus is left out entirely, every part is played
    # by the chip, and every tile draws its part's waveform instead of a clip.
    synth = bool(opts.get("synth", False))
    if synth:
        chip = synth_chip(chip)
    screen = screen_for(chip)

    given = {p.get("id"): p for p in prepare_parts(params.get("parts") or [], synth)}
    # Recommendations only for parts the request did not choose a sound for,
    # and none at all when nothing is going to be said.
    def needs_voice(p):
        asked = given.get(p.id) or {}
        tone = (tones.resolve(asked.get("tone")) if tones.overrides_switch(asked.get("tone"))
                else chip_tone(p, chip) if chip else "clean")
        return tone not in tones.SYNTH

    need = [] if synth else [p for p in song.parts if needs_voice(p) and not (given.get(p.id) or {}).get("text")
                             and not ((given.get(p.id) or {}).get("mute")
                                      and not (given.get(p.id) or {}).get("visible", True))]
    recs = {}
    groups = {}
    for part in need:
        slug = (given.get(part.id) or {}).get("corpus", active()["slug"])
        groups.setdefault(slug, []).append(part)
    for slug, parts in groups.items():
        with samples.corpus_context(slug):
            recs.update(recommend.recommend(
                music.Song(parts, song.duration, song.bpm, song.time_signature,
                           song.key, song.key_confidence), progress=progress,
                pool=samples.corpus_pool()))
    settings = {p.id: _merge(_default_settings_for(p, recs.get(p.id, {})), given.get(p.id))
                for p in song.parts}
    _through_chip(source_song, settings, given, chip, g_transpose, synth)
    chip_parts = {p.id for p in song.parts if settings[p.id]["tone"] in tones.SYNTH}

    t_from = _num(start_seconds, 0, 0, max(0, song.duration / speed)) if audio_only else 0.0
    T = min(song.duration / speed + _TAIL, t_from + max_s)
    n_frames = max(1, int(round((T - t_from) * FPS)))
    spf = SR // FPS                                # 1764: whole samples per frame
    N = n_frames * spf
    mix = np.zeros((N, 2), dtype=np.float32)

    playing = [p for p in song.parts if settings[p.id]["text"] and not settings[p.id]["mute"]
               and p.notes]
    # No more notes at once than the machine had channels for; see tones.allocate.
    cuts = (tones.allocate([(p, settings[p.id].get("tone")) for p in playing], speed)
            if accurate else None)
    shown = [] if audio_only else [p for p in song.parts if settings[p.id]["text"] and settings[p.id]["visible"]
             and p.notes]
    wanted = {p.id: p for p in playing + shown}
    playing_ids = {p.id for p in playing}

    # Samples first: the cheapest place to find out a word cannot be said.
    sample_of: dict[str, list[samples.Sample]] = {}
    problems: list[dict] = []
    for i, part in enumerate(wanted.values()):
        say("samples", i, len(wanted))
        s = settings[part.id]
        if part.id in chip_parts:
            sample_of[part.id] = []             # the chip needs nothing to play from
            continue
        if vary == "ultra":
            want = [(e["text"], e["take"]) for e in s["pool"][:_ULTRA_POOL]]
        elif vary == "off":
            want = [(s["text"], s["take"])]
        else:
            want = [(s["text"], t) for t in s["takes"]]
        got, failed = [], None
        for text, take in dict.fromkeys(want):  # in order, no repeats
            try:
                got.append(samples.build(text, take, _hit(part, s),
                                         corpus=s["corpus"],
                                         check_cancel=lambda: say("samples", i, len(wanted))))
            except samples.SampleError as exc:
                failed = exc
        if got:
            sample_of[part.id] = got
        else:
            problems.append({"part": part.id, "name": part.name, "text": s["text"],
                             "error": str(failed)})
    say("samples", len(wanted), len(wanted))
    if not sample_of:
        raise YtpmvError("There is nothing to play." if synth else
                         "None of the parts has a sound this voice can make.")

    hits: dict[str, list[_Hit]] = {}
    total = sum(len(p.notes) for p in wanted.values() if p.id in sample_of)
    done = [0]

    def tick():
        done[0] += 1
        if done[0] % 200 == 0:
            say("notes", done[0], total)

    # Each part is played into a buffer of its own, measured, and only then
    # added to the mix at whatever level the measurement says it wants. One
    # buffer, cleared and used again, rather than one per part: a three minute
    # song is 60 MB of it and there is no need to hold eight at once.
    #
    # Without a voice every shown part is played into it, muted or not,
    # because its tile is a picture of the sound it makes.
    tiles = [p for p in shown if p.id in sample_of]
    if not tiles and not synth:
        tiles = [p for p in playing if p.id in sample_of][:1]
    tiles = ordered_tiles(tiles, opts.get("tile_order"))
    layout = opts.get("layout", "balanced")
    if layout not in GRID_LAYOUTS:
        layout = "balanced"
    media_fit = "cover" if opts.get("media_fit") == "cover" else "contain"
    center_tiles = bool(opts.get("center_tiles", True))
    W, H = _OUT_W, _OUT_H
    draft = bool(opts.get("draft", False))
    is_retro = screen in screens.SIZES
    if is_retro:
        cw, ch = screens.SIZES[screen]
    else:
        # Samples are 480p-wide sources. Decode/composite near their native
        # size and upscale the finished canvas once, rather than retaining
        # hundreds of megabytes of already-upscaled frames per take.
        cw = min(W, math.ceil(math.sqrt(len(tiles) or 1)) * samples.SRC_W)
        ch = max(2, int(cw * H / W) // 2 * 2)
        if draft:
            cw, ch = max(2, cw // 4 * 2), max(2, ch // 4 * 2)
    output_cells = grid_cells(len(tiles), W, H, layout, center_tiles)
    cells = {p.id: (x*cw//W, y*ch//H, (x+w)*cw//W-x*cw//W, (y+h)*ch//H-y*ch//H)
             for p, (x,y,w,h) in zip(tiles, output_cells)}
    traces: dict[str, np.ndarray] = {}
    scratch = np.zeros_like(mix) if balance or chip_parts else None
    levels: dict[str, dict] = {}
    # With balance, every part is kept as a stem on disk until all of them are
    # measured, because where one belongs depends on the rest; see
    # master.balance. Mono float16: the pan is a constant pair of gains, put
    # back when it is mixed. A three minute song of thirty parts is about half
    # a gigabyte of these, gone as soon as the mix is made.
    stems: list[dict] = []
    stem_temp = tempfile.TemporaryDirectory(prefix="ytpmv_stems_") if balance else None
    stemdir = resources.enter_context(stem_temp) if stem_temp else None
    chunk = 60 * SR
    # Where the last part wrote into scratch. Everything outside it is still
    # zero, so only that much needs clearing, and only a part's own stretch
    # needs turning into a stem and mixing: a song of three hundred parts,
    # most of them playing for seconds, was spending most of its time adding
    # up silence.
    span = [0, 0]
    for part in wanted.values():
        say("notes", done[0], total)
        if part.id not in sample_of:
            continue
        into = mix if part.id in playing_ids else None
        if (balance and into is not None) or part.id in chip_parts:
            scratch[span[0]:span[1]] = 0.0
            into = scratch
        span = [N, 0]
        hits[part.id] = _play(part, settings[part.id], sample_of[part.id], into,
                              t_from, T, speed, g_transpose, flip, tick, jitter, vary, cuts,
                              span)
        lo, hi = span[0], max(span)
        if part.id in chip_parts and part.id in cells and not audio_only:
            traces[part.id] = scope.traces(
                scratch.mean(axis=1), n_frames, spf, max(1, cells[part.id][2] - 1),
                hz=None if part.is_drums else midi_to_hz(
                    part.median_pitch() + 12 * settings[part.id]["octave"]
                    + settings[part.id]["transpose"] + g_transpose))
        if part.id in chip_parts and not balance and part.id in playing_ids:
            mix[lo:hi] += scratch[lo:hi]
        if balance and into is scratch and part.id in playing_ids:
            pan = (settings[part.id]["pan"] + 1) * math.pi / 4
            gl, gr = math.cos(pan), math.sin(pan)
            stem = np.lib.format.open_memmap(os.path.join(stemdir, f"{len(stems)}.npy"),
                                             mode="w+", dtype=np.float16, shape=(N,))
            resources.callback(_close_stem, stem)
            denom = gl + gr if abs(gl + gr) > 1e-6 else 1.0
            for i in range(lo, hi, chunk):        # a new stem is zeros already
                j = min(hi, i + chunk)
                stem[i:j] = scratch[i:j].sum(axis=1) / denom
            # What the person asked for over and above the old table, kept:
            # the measurement decides where a part sits on its own, and
            # somebody who has moved a slider still outranks it.
            vol = settings[part.id]["volume"]
            ref = _VOLUME.get(part.role, 0.8)
            trim = 20.0 * math.log10(max(vol, 1e-4) / ref) if ref > 0 else 0.0
            stems.append({"id": part.id, "role": part.role, "trim_db": trim, "stem": stem,
                          "pan": (gl, gr), "power": mastering.block_power(stem),
                          "span": (lo, hi)})
    del scratch
    if stemdir:
        placed = mastering.balance(stems) if stems else {}
        for st in stems:
            was, gain, rode = placed[st["id"]]
            pair = np.array(st["pan"], dtype=np.float32) * np.float32(gain)
            lo, hi = st["span"]
            for i in range(lo, hi, chunk):
                j = min(hi, i + chunk)
                mono = st["stem"][i:j].astype(np.float32) * mastering.envelope(rode, i, j)
                mix[i:j] += mono[:, None] * pair
            levels[st["id"]] = {"lufs": round(was, 1) if math.isfinite(was) else None,
                                "gain_db": round(20.0 * math.log10(max(gain, 1e-6)), 1)}
        for st in stems:
            _close_stem(st["stem"])
        stems.clear()
        stem_temp.cleanup()
    say("notes", total, total)

    tmpdir = resources.enter_context(tempfile.TemporaryDirectory(prefix="ytpmv_"))
    wav = os.path.join(tmpdir, "mix.wav")
    _write_wav(wav, _finish_mix(mix))
    del mix

    if audio_only:
        os.makedirs("output", exist_ok=True)
        output = os.path.join("output", f"ytpmv_sample_mix_{uuid.uuid4().hex[:12]}.wav")
        # The preview is already complete before its public path appears.
        staged = output + ".tmp"
        try:
            shutil.copyfile(wav, staged)
            os.replace(staged, output)
        finally:
            if os.path.exists(staged):
                os.remove(staged)
        return {"audio_url": "/output/" + os.path.basename(output),
                "from": round(t_from, 2), "duration": round(T - t_from, 2),
                "problems": problems, "options": {"speed": speed, "transpose": g_transpose,
                    "balance": balance, "chip": chip, "synth": synth},
                "parts": [{"id": p.id, "corpus": settings[p.id]["corpus"],
                           "volume": settings[p.id]["volume"], "pan": settings[p.id]["pan"]}
                          for p in playing if p.id in sample_of]}

    frames_of = {p.id: [smp.frames(cells[p.id][2], cells[p.id][3], fit=media_fit,
                                 pixel_aspect=(output_cells[i][2]/cells[p.id][2])/(output_cells[i][3]/cells[p.id][3]),
                                                 check_cancel=lambda: say("frames", 0, n_frames))
                                       for smp in sample_of[p.id]]
                                  for i, p in enumerate(tiles) if p.id not in chip_parts}
    starts_of = {p.id: [h.start for h in hits.get(p.id, [])] for p in tiles}

    run_id = uuid.uuid4().hex[:10]
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"ytpmv_{run_id}.mp4")
    completed = [False]

    def discard_partial():
        if not completed[0]:
            try:
                os.remove(out_path)
            except OSError:
                pass

    resources.callback(discard_partial)
    vf = ["format=yuv420p"]
    src_w, src_h = cw, ch
    font = g.subtitle_font() if labels else None
    if font:
        for p, (x, y, label_tw, label_th) in zip(tiles, output_cells):
            text = settings[p.id]["text"].replace("\\", "\\\\").replace("'", "’").replace(":", "\\:")
            f = font.replace("\\", "/").replace(":", "\\:")
            vf.insert(0, f"drawtext=fontfile='{f}':text='{text}':expansion=none:fontcolor=white"
                         f":fontsize={max(12, label_th // 12)}:borderw=2:bordercolor=black"
                         f":x={x + 8}:y={y + label_th - label_th // 12 - 10}")
    if not is_retro and (src_w, src_h) != (W, H):
        vf.insert(0, f"scale={W}:{H}:flags=bilinear")
    if is_retro:
        # Blown back up first, so the labels below are still placed in the
        # full-size picture and stay readable rather than becoming eight
        # chunky squares.
        vf.insert(0, f"scale={W}:{H}:flags=neighbor")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{src_w}x{src_h}", "-r", str(FPS), "-i", "-", "-i", wav,
           "-map", "0:v", "-map", "1:a", "-vf", ",".join(vf),
           *_encoder_for_ytpmv(),
           "-crf", "16" if is_retro else "23",
           "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2",
           "-shortest", "-movflags", "+faststart", out_path]
    errlog = resources.enter_context(open(os.path.join(tmpdir, "ffmpeg.log"), "w+b"))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errlog)
    from app.media_process import cancellation_watch, stop, wait
    resources.callback(stop, proc)
    stage = ["frames", 0, n_frames]
    watchers = resources.enter_context(ExitStack())
    watchers.enter_context(cancellation_watch(proc, (lambda: say(*stage)) if progress else None))
    canvas = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    drawn: dict[int, tuple] = {}              # tile -> the state last drawn into canvas
    last_frame = None                         # the last retro frame, while nothing moves
    try:
        for f in range(n_frames):
            stage[1] = f
            if f % 25 == 0:
                say("frames", f, n_frames)
            t = f / FPS
            moved = bool(chip_parts)
            for i, p in enumerate(tiles):
                x0, y0, tile_w, tile_h = cells[p.id]
                if p.id in chip_parts:
                    # A pixel of black between tiles, so the traces do not run together.
                    canvas[y0:y0 + tile_h - 1, x0:x0 + tile_w - 1] = _scope_tile(
                        traces[p.id][f], hits.get(p.id, []), starts_of[p.id], t,
                        tile_w - 1, tile_h - 1, flash, dim, screen)
                    continue
                state = _tile_state(frames_of[p.id], hits.get(p.id, []), starts_of[p.id],
                                    t, flash, dim)
                if drawn.get(i) == state:
                    continue                  # the canvas already shows it
                drawn[i] = state
                moved = True
                canvas[y0:y0 + tile_h, x0:x0 + tile_w] = _draw_tile(frames_of[p.id], state)
            try:
                if is_retro:
                    if moved or last_frame is None:
                        last_frame = screens.apply(canvas, screen).tobytes()
                    proc.stdin.write(last_frame)
                else:
                    proc.stdin.write(memoryview(canvas))
            except (BrokenPipeError, OSError):
                break
        proc.stdin.close()
        stage[:] = ["encoding", 0, 1]
        say("encoding", 0, 1)
        rc = wait(proc, check_cancel=lambda: say("encoding", 0, 1))
        if rc != 0:
            errlog.seek(0)
            raise RuntimeError("FFmpeg failed: " + errlog.read().decode(errors="replace")[-500:])
        say("encoding", 1, 1)
    finally:
        stop(proc)

    watchers.close()
    completed[0] = True
    log.info("YTPMV %s  %d parts  %.1fs  %dx%d", out_path, len(tiles), T, W, H)
    return {
        "video_url": f"/output/{os.path.basename(out_path)}",
        "duration": round(T, 2),
        "size": [W, H],
        "song": {k: v for k, v in song.summary().items() if k != "parts"},
        "options": {"vary": vary, "jitter": jitter, "speed": speed, "balance": balance,
                    "accurate": accurate,
                    "screen": screen, "synth": synth, "chip": chip,
                    "transpose": g_transpose, "max_seconds": max_s,
                    "layout": layout, "media_fit": media_fit, "center_tiles": center_tiles,
                    "tile_order": [p.id for p in tiles]},
        "parts": [{"id": p.id, "name": p.name, "role": p.role,
                   **{k: settings[p.id][k] for k in ("corpus", "text", "take", "octave", "transpose", "mode",
                                                   "tone", "program", "mute", "visible", "pan", "haas")},
                   "polyphony": p.polyphony()[0],
                   "typical_polyphony": round(p.polyphony()[1], 2),
                   # How loud the part turned out, and how far it had to be
                   # moved to sit where its job wants it.
                   **levels.get(p.id, {}),
                   "played": [smp.text for smp in sample_of.get(p.id, [])],
                   "takes_played": len(sample_of.get(p.id, [])),
                   "pitch": sample_of[p.id][0].voice.info.as_dict() if sample_of.get(p.id) else None}
                  for p in song.parts],
        "problems": problems,
    }


def _hit(part: music.Part, s: dict) -> str | None:
    return part.drum_group if part.is_drums and s.get("hit", True) else None


def _play(part: music.Part, s: dict, takes: list, mix: np.ndarray | None, t_from: float,
          T: float, speed: float, g_transpose: int, flip: bool, tick=None,
          jitter: bool = True, vary: str = "rotate", cuts: dict | None = None,
          span: list | None = None) -> list[_Hit]:
    """Sing every note of *part* between t_from and T into *mix*; return its hits.

    *span*, when given as [lo, hi], is widened to cover every sample written.

    *takes* are the Samples to play, rotated between hits (see VARY). Times are
    in the rendered song's clock (after *speed*), and the mix starts at t_from.
    *mix* None still works out the hits, for a tile that is shown but muted.
    No takes at all is a part with no voice: the chip plays every note from
    nothing, which it would have done anyway, and there is no clip to warp.
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
    beat_index = {start: i for i, start in enumerate(starts)}
    hits: list[_Hit] = []
    last_start = None
    pick = 0
    for n in notes:
        if tick:
            tick()
        t0 = n.start / speed
        if t0 >= T:
            break
        if n.start != last_start:              # a chord is one hit: one take
            pick = (0 if vary == "off" or len(takes) <= 1 else
                    rng.randrange(len(takes)) if vary in ("random", "ultra") else
                    beat_index[n.start] % len(takes))
        j = bisect.bisect_right(starts, n.start)
        nxt = starts[j] / speed if j < len(starts) else T
        if not takes:
            dur = min(_RING_MAX, nxt - t0) if s["sustain"] == "ring" else n.dur / speed
            dur = min(dur, T - t0)
            if dur <= 0:
                continue
        else:
            sample = takes[pick]
            voice = sample.voice
            mode = s["mode"]
            if mode == "perfect" and not voice.info.f0:
                mode = "tape"
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
        # Keep the same take/jitter sequence as the full song when seeking,
        # and retain notes whose attack was before the preview cursor.
        if t0 + dur + (0.018 if s.get("haas") else 0) <= t_from:
            last_start = n.start
            continue
        midi = n.pitch + shift
        if not takes:
            # Only its length and level reach the chip; see tones.apply.
            y, warp = np.full(max(1, int(round(dur * SR))), 0.5, dtype=np.float32), None
        elif part.is_drums or mode == "raw":
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
            # OPL picks its patch from the GM instrument, or for drums the key;
            # the SID needs to know which of its two chips it is.
            y = tones.apply(y, s["tone"], hz=None if part.is_drums else midi_to_hz(midi),
                            level=max(0.2, n.velocity / 127.0), program=s.get("program", part.program),
                            key=n.pitch if part.is_drums else None,
                            model="8580" if str(s.get("chip") or "").startswith("sid8580")
                            else "6581")
        cut = cuts.get((part.id, n.start, n.pitch)) if cuts else None
        if cut is not None:
            keep = int(round((cut - t0) * SR))
            if keep <= 0:
                continue                      # never got a channel
            y = y[:keep].copy()
            fade = min(len(y), int(0.004 * SR))
            y[len(y) - fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        if mix is not None and s["volume"] > 0:
            a = int(round((t0 - t_from) * SR))
            k = min(len(y), N - a)
            if a < 0:
                gain = s["volume"] * loud * (0.35 + 0.65 * n.velocity / 127.0)
                delay = int(round(.018 * SR)) if s.get("haas") else 0
                for channel, offset, pan_gain in ((0, a + (delay if s["pan"] > 0 else 0), gl),
                                                  (1, a + (delay if s["pan"] <= 0 else 0), gr)):
                    dest, source = max(0, offset), max(0, -offset)
                    count = min(len(y) - source, N - dest)
                    if count > 0:
                        mix[dest:dest + count, channel] += y[source:source + count] * (gain * pan_gain)
                        if span is not None:
                            span[0] = min(span[0], dest); span[1] = max(span[1], dest + count)
            elif k > 0:
                gain = s["volume"] * loud * (0.35 + 0.65 * n.velocity / 127.0)
                if s.get("haas"):
                    delay = int(round(0.018 * SR))  # 18ms Haas delay
                    if s["pan"] <= 0:
                        mix[a:a + k, 0] += y[:k] * (gain * gl)
                        k_d = min(k, max(0, N - (a + delay)))
                        if k_d > 0:
                            mix[a + delay:a + delay + k_d, 1] += y[:k_d] * (gain * gr)
                    else:
                        k_d = min(k, max(0, N - (a + delay)))
                        if k_d > 0:
                            mix[a + delay:a + delay + k_d, 0] += y[:k_d] * (gain * gl)
                        mix[a:a + k, 1] += y[:k] * (gain * gr)
                else:
                    mix[a:a + k, 0] += y[:k] * (gain * gl)
                    mix[a:a + k, 1] += y[:k] * (gain * gr)
                if span is not None:
                    span[0] = min(span[0], a)
                    span[1] = min(N, max(span[1], a + k + (int(round(0.018 * SR))
                                                         if s.get("haas") else 0)))
        if n.start != last_start:
            hits.append(_Hit(t0, dur, warp, flip and len(hits) % 2 == 1, pick))
            last_start = n.start
    return hits


def preview_part(midi_id: str, part_settings: dict, seconds: float = 6.0,
                 chip=None, synth: bool = False) -> dict:
    """A few seconds of one part, alone, as it will sound -- audio only.

    Starts at the part's first note rather than the top of the song, since a
    part that comes in at bar 17 would otherwise preview as silence. *chip*
    is the song's switch, so the part is heard on the machine it will be.
    """
    song = load_song(midi_id)
    part = song.part(part_settings.get("id", ""))
    if part is None:
        raise YtpmvError("No such part.")
    s = _merge(_default_settings_for(part, {}), part_settings)
    chip = synth_chip(which_chip(chip)) if synth else which_chip(chip)
    _through_chip(song, {part.id: s}, {part.id: part_settings}, chip, synth=synth)
    sample = None
    if not synth:
        if not s["text"]:
            raise YtpmvError("Give the part a word first.")
        try:
            sample = samples.build(s["text"], s["take"], _hit(part, s), corpus=s["corpus"])
        except samples.SampleError as exc:
            raise YtpmvError(str(exc)) from None
    s["volume"], s["pan"] = 1.0, 0.0
    t_from = max(0.0, (part.notes[0].start if part.notes else 0.0) - 0.05)
    T = min(t_from + seconds, song.duration + _TAIL)
    mix = np.zeros((int((T - t_from) * SR), 2), dtype=np.float32)
    _play(part, s, [sample] if sample else [], mix, t_from, T, 1.0, 0, False)
    name = hashlib.sha1(repr((midi_id, sorted(s.items()), seconds,
                              sample.mp4 if sample else None)).encode()).hexdigest()[:12]
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", f"ytpmv_sample_{name}.wav")
    _write_wav(path, _finish_mix(mix))
    return {"audio_url": "/output/" + os.path.basename(path),
            "from": round(t_from, 2), "settings": s,
            "sample": sample.describe() if sample else None}


def _encoder_for_ytpmv() -> list[str]:
    """Encoder flags for YTPMV output. Uses the same selection as generate.py."""
    try:
        from app.generate import _encoder_settings
        settings = _encoder_settings()
        if len(settings) > 1 and settings[1] == "libx264":
            return ["-c:v", "libx264", "-preset", "veryfast"]
        return settings
    except (ImportError, Exception):
        return ["-c:v", "libx264", "-preset", "veryfast"]


def _scale(img: np.ndarray, k: float) -> np.ndarray:
    """Brighten or dim an image by factor *k*, staying in uint8."""
    if abs(k - 1.0) < 0.001:
        return img
    if k <= 1.0:
        # Dimming: direct uint8 multiply truncates naturally, no overflow possible
        out = np.empty_like(img)
        np.multiply(img, k, out=out, casting='unsafe')
        return out
    # Brightening: clip to 255
    return np.minimum(
        np.multiply(img, np.float32(k), dtype=np.float32),
        255.0
    ).astype(np.uint8)


def _scope_tile(trace: np.ndarray, hits: list[_Hit], starts: list[float], t: float,
                w: int, h: int, flash: bool, dim: bool, screen: str | None = None) -> np.ndarray:
    """A voiceless part's tile: its waveform, lit on the hit and dimmed after."""
    i = bisect.bisect_right(starts, t) - 1
    if i < 0:
        return scope.draw(trace, w, h, 0.35 if dim else 1.0, screen=screen)
    h_ = hits[i]
    since = t - h_.start
    level = max(0.45, 1.0 - (since - h_.dur) / 0.3 * 0.55) if dim and since > h_.dur else 1.0
    return scope.draw(trace, w, h, level, hot=flash and since < 2.0 / FPS, screen=screen)


def _tile_state(takes: list, hits: list[_Hit], starts: list[float], t: float,
                flash: bool, dim: bool) -> tuple[int, int, bool, float]:
    """What a tile shows at *t*: (take, frame, flipped, brightness).

    Equal states draw equal pictures, so a tile whose state has not changed
    since the last frame need not be drawn again -- which, in a song of three
    hundred parts, is most of them most of the time.
    """
    i = bisect.bisect_right(starts, t) - 1
    if i < 0:
        return 0, 0, False, 0.35 if dim else 1.0
    h = hits[i]
    take = h.take % len(takes)
    since = t - h.start
    src = h.warp.src(min(since, h.dur))
    idx = min(max(int(src * FPS), 0), len(takes[take]) - 1)
    k = (1.25 if flash and since < 2.0 / FPS else
         max(0.45, 1.0 - (since - h.dur) / 0.3 * 0.55) if dim and since > h.dur else 1.0)
    return take, idx, h.flip, k


def _draw_tile(takes: list, state: tuple[int, int, bool, float]) -> np.ndarray:
    take, idx, flip, k = state
    img = takes[take][idx]
    return _scale(img[:, ::-1] if flip else img, k)


def _tile(takes: list, hits: list[_Hit], starts: list[float], t: float,
          flash: bool, dim: bool) -> np.ndarray:
    return _draw_tile(takes, _tile_state(takes, hits, starts, t, flash, dim))
