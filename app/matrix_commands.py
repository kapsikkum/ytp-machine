"""What a message to the Matrix bot is asking for.

Kept apart from the bot itself so it can be tested without a homeserver, or
matrix-nio installed at all: this is plain string handling, and it is where
the mistakes would be.

    !ytp say nice chocolate cake            a sentence video, words on the picture
    !ytp say --no-subs nice chocolate cake  ... without them
    !ytp mv                               the last MIDI posted here, as a YTPMV
    !ytp mv lead=yeah kick=boom max=60    ... with some parts' sounds chosen
    !ytp mv octave:bass=+1 mode:lead=tape
    !ytp mv info                          what the song is and what it would use
    !ytp voices / !ytp voice <name>       list or switch the corpus (admins)
    !ytp queue                            how busy it is
    !ytp help

In a direct message the prefix is optional, and a message that is not a
command at all is taken as something to say -- one person talking to a bot
does not need to address it.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

DEFAULT_PREFIX = "!ytp"

# Per-part settings the mv command accepts, and how to read each value.
_PART_KEYS = {
    "octave":    int,
    "take":      int,
    "transpose": int,
    "mode":      str,
    "volume":    float,
    "pan":       float,
    "sustain":   str,
    "tone":      str,
    "program":   int,
}
_MODES = {"perfect", "tape", "raw"}
_SUSTAINS = {"note", "ring"}
# What a part can be played through. "megadrive" and "slap" are what the one
# hand-built bass patch was called before the chip itself existed.
_TONES = {"clean", "voice", "dac", "dpcm",
          # the Mega Drive's
          "bass", "lead", "organ", "brass", "bell", "piano", "strings",
          "md-kick", "md-snare", "md-hat", "md-tom", "md-cymbal", "md-perc",
          # the NES's
          "pulse", "pulse-thin", "pulse-full", "triangle",
          "nes-kick", "nes-snare", "nes-hat", "nes-tom", "nes-cymbal", "nes-perc",
          # the Master System's and the SNES's
          "psg", "psg-soft", "psg-bass", "psg-pluck", "psg-pcm", "brr",
          "sms-kick", "sms-snare", "sms-hat", "sms-tom", "sms-cymbal", "sms-perc",
          # the Game Boy's
          "gb-pulse", "gb-pulse-thin", "gb-pulse-full", "gb-pluck", "gb-wave", "gb-pcm",
          "gb-kick", "gb-snare", "gb-hat", "gb-tom", "gb-cymbal", "gb-perc",
          # the PC's and the C64's
          "opl2", "opl3", "snes", "sb-pcm", "sb16-pcm", "sid-digi",
          "sid-lead", "sid-bass", "sid-pad", "sid-rhythm", "sid-bell",
          "sid-kick", "sid-snare", "sid-hat", "sid-tom", "sid-cymbal", "sid-perc",
          "megadrive", "slap"}
_CHIPS = {"off", "no", "md", "md-voice", "nes", "nes-voice",
          "sms", "sms-voice", "snes", "snes-voice", "gb", "gb-voice", "gbc", "gbc-voice",
          "opl2", "opl2-voice", "opl3", "opl3-voice",
          "sid", "sid-voice", "sid8580", "sid8580-voice",
          "megadrive", "megadrive-voice"}      # the longer names still work
# Whole-song options.
_VARY = {"off", "rotate", "random", "ultra"}
_OPTIONS = {
    "max":       ("max_seconds", float),
    "vary":      ("vary", "vary"),
    "jitter":    ("jitter", "bool"),
    "speed":     ("speed", float),
    "transpose": ("transpose", int),
    "key":       ("transpose", int),
    "flip":      ("flip", "bool"),
    "flash":     ("flash", "bool"),
    "dim":       ("dim", "bool"),
    "labels":    ("labels", "bool"),
    "chip":      ("chip", "chip"),
    "balance":   ("balance", "bool"),
}
_TRUE = {"1", "yes", "on", "true", "y"}
_FALSE = {"0", "no", "off", "false", "n"}

HELP = """\
{p} say <words>        — a video of the voice saying it (--no-subs to drop the captions)
{p} mv                 — play the last MIDI file posted here, one tile per instrument
{p} mv info            — what the song is, and which word each part would use
{p} mv chip=md         — the whole song on an emulated sound chip, every part of it.
{p} mv chip=nes          md is four-operator FM, nes is two pulses and a noise
{p} mv chip=sms          register, sms is three squares and nothing else. All three
{p} mv chip=snes         throw the voice away -- add -voice to keep it instead,
{p} mv chip=gb           (gb and gbc share a chip and differ in their screens)
{p} mv chip=opl2         Doom's instrument bank on an AdLib; opl3 on a Sound Blaster
{p} mv chip=sid          the C64: sid is the 6581, sid8580 the later chip
                         played off that machine's own sample channel. snes plays
                         a General MIDI bank cut down to what a SNES could carry.
                         The picture goes through the same console's resolution
                         and colours
{p} mv lead=yeah kick=boom max=60 octave:bass=+1 tone:bass=bass
                         vary=rotate|random (several takes of a part's word rather
                         than one) or vary=ultra (a different word every hit, still
                         on the note), jitter=on (a hair of level and tuning per hit),
                         choose a part's word, or tweak it; parts are named by role
                         (lead, bass, rhythm, chords, kick, snare, hats, toms, cymbals)
                         or by instrument; tone:<part>= picks what one part is played
                         by, whatever the chip (sid-bell, pulse, triangle, dac...;
                         voice leaves the part as recorded); program:<part>= swaps its
                         General MIDI instrument, which picks the OPL patch
                         (program:lead=29 is overdriven guitar); a sound can be a word, a noise (*spew*)
                         or one phoneme cut out of words (/ah/, /s/); max= caps the
                         length in seconds
{p} mv balance=off     — leave the parts at their table levels instead of measuring
                         each one and moving it to where its job wants it
{p} voices             — the voices available
{p} voice <name>       — switch voice (admins only)
{p} queue              — how many videos are waiting
Reply to a MIDI file with {p} mv to use that one instead of the latest."""


@dataclass
class Command:
    name: str                                   # say | mv | info | voices | voice | queue | help | unknown
    text: str = ""                              # say: the words; voice: the slug; unknown: what was typed
    subtitles: bool = True                      # captions are the default; --no-subs turns them off
    parts: dict[str, dict] = field(default_factory=dict)   # mv: part name -> settings
    options: dict = field(default_factory=dict)            # mv: song options
    errors: list[str] = field(default_factory=list)


def _bool(v: str) -> bool | None:
    v = v.lower()
    return True if v in _TRUE else False if v in _FALSE else None


def _tokens(s: str) -> list[str]:
    try:
        return shlex.split(s)
    except ValueError:                          # an unbalanced quote: split plainly
        return s.split()


def _mv(args: list[str]) -> Command:
    cmd = Command("mv")
    if args and args[0].lower() in ("info", "describe", "what"):
        return Command("info")
    for tok in args:
        if "=" not in tok:
            cmd.errors.append(f"don't know what to do with {tok!r} — expected part=word or key=value")
            continue
        key, value = tok.split("=", 1)
        key, value = key.strip().lower(), value.strip()
        if not key or not value:
            cmd.errors.append(f"{tok!r} is missing a side")
            continue
        if ":" in key:                          # setting:part=value
            setting, part = key.split(":", 1)
            conv = _PART_KEYS.get(setting)
            if conv is None or not part:
                cmd.errors.append(f"unknown part setting {setting!r} (try "
                                  + ", ".join(sorted(_PART_KEYS)) + ")")
                continue
            try:
                v = conv(value.lstrip("+")) if conv is not str else value.lower()
            except ValueError:
                cmd.errors.append(f"{setting} for {part} should be a number, not {value!r}")
                continue
            if setting == "mode" and v not in _MODES:
                cmd.errors.append(f"mode is one of {', '.join(sorted(_MODES))}")
                continue
            if setting == "sustain" and v not in _SUSTAINS:
                cmd.errors.append(f"sustain is one of {', '.join(sorted(_SUSTAINS))}")
                continue
            if setting == "tone" and v not in _TONES:
                cmd.errors.append(f"tone is one of {', '.join(sorted(_TONES))}")
                continue
            if setting == "program" and not 0 <= v <= 127:
                cmd.errors.append("program is a General MIDI instrument, 0 to 127")
                continue
            cmd.parts.setdefault(part, {})[setting] = v
        elif key in _OPTIONS:
            name, conv = _OPTIONS[key]
            if conv == "chip":
                v = value.lower()
                if v not in _CHIPS:
                    cmd.errors.append("chip is one of off, md, md-voice, nes, nes-voice, "
                                      "sms, sms-voice, snes, snes-voice, gb, gb-voice, gbc, gbc-voice, "
                                      "opl2, opl2-voice, opl3, opl3-voice, "
                                      "sid, sid-voice, sid8580, sid8580-voice")
                    continue
                cmd.options[name] = False if v in ("off", "no") else v
            elif conv == "vary":
                if value.lower() not in _VARY:
                    cmd.errors.append(f"vary is one of {', '.join(sorted(_VARY))}")
                    continue
                cmd.options[name] = value.lower()
            elif conv == "bool":
                b = _bool(value)
                if b is None:
                    cmd.errors.append(f"{key} should be on or off")
                    continue
                cmd.options[name] = b
            else:
                try:
                    cmd.options[name] = conv(value.lstrip("+"))
                except ValueError:
                    cmd.errors.append(f"{key} should be a number, not {value!r}")
        else:                                   # part=word(s)
            if value.lower() in ("mute", "off", "none", "-"):
                cmd.parts.setdefault(key, {}).update(mute=True, visible=False)
            else:
                cmd.parts.setdefault(key, {})["text"] = value.replace("_", " ")
    return cmd


def parse(body: str, prefix: str = DEFAULT_PREFIX, direct: bool = False) -> Command | None:
    """The command in *body*, or None when the message is not for the bot.

    *direct* is a one-to-one room: the prefix may be left off, and plain text
    means "say this".
    """
    text = (body or "").strip()
    if not text:
        return None
    # A reply quotes the message it answers in "> " lines ahead of the reply
    # itself (the plain-text fallback); those are not what was asked.
    lines = [ln for ln in text.splitlines() if not ln.startswith(">")]
    text = "\n".join(lines).strip()
    if not text:
        return None

    low = text.lower()
    p = prefix.lower()
    if low == p or low.startswith(p + " ") or low.startswith(p + "\n"):
        rest = text[len(prefix):].strip()
    elif direct:
        rest = text
        if not re.match(r"^(say|mv|ytpmv|voices?|queue|help)\b", rest, re.I):
            return Command("say", text=rest)
    else:
        return None

    if not rest:
        return Command("help")
    verb, _, args = rest.partition(" ")
    verb = verb.lower().strip()
    args = args.strip()
    if verb in ("say", "s"):
        subs = True                      # the words on screen, unless asked otherwise
        words = args
        for flag, on in (("--subs", True), ("--subtitles", True), ("-s", True),
                         ("--no-subs", False), ("--nosubs", False), ("--no-subtitles", False)):
            if words.startswith(flag + " ") or words == flag:
                subs, words = on, words[len(flag):].strip()
                break
        if not words:
            return Command("help", errors=["say what?"])
        return Command("say", text=words, subtitles=subs)
    if verb in ("mv", "ytpmv", "midi", "song"):
        return _mv(_tokens(args))
    if verb == "voices":
        return Command("voices")
    if verb == "voice":
        return Command("voice", text=args) if args else Command("voices")
    if verb in ("queue", "status"):
        return Command("queue")
    if verb in ("help", "?"):
        return Command("help")
    return Command("unknown", text=verb)


def is_midi(filename: str | None, mimetype: str | None = None) -> bool:
    name = (filename or "").lower()
    return name.endswith((".mid", ".midi", ".kar", ".smf")) or (mimetype or "").lower() in (
        "audio/midi", "audio/x-midi", "audio/mid")


def match_part(name: str, parts: list[dict]) -> dict | None:
    """Which part of a song *name* means: an id, a role, or a word of its name.

    *parts* are the analysis's part summaries (id, name, role, instrument).
    """
    n = name.lower().strip()
    if not n:
        return None
    for p in parts:
        if p["id"].lower() == n:
            return p
    for p in parts:
        role = p["role"].lower()
        if role == n or role.split(":")[-1] == n:
            return p
    aliases = {"melody": "lead", "tune": "lead", "drums": "kick", "hat": "hats", "hihat": "hats",
               "hihats": "hats", "cymbal": "cymbals", "tom": "toms", "backing": "rhythm",
               "guitar": "rhythm", "pad": "chords"}
    if n in aliases:
        return match_part(aliases[n], parts)
    for p in parts:
        if n in p["instrument"].lower() or n in p["name"].lower():
            return p
    return None
