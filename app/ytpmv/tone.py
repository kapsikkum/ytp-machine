"""What a part is played through, once its notes are already right.

Pitch and timing are settled before this; a tone only decides what makes the
sound. There are two kinds, and the difference decides whether you still hear
the voice at all:

  clean     the voice, untouched
  dac       the voice through the Mega Drive's PCM channel -- eight bits at
            about 13 kHz, with the dead zone around silence
  dpcm      the voice through the NES's delta-modulation channel, which can
            only move two steps at a time and so blunts whatever it plays
  psg-pcm   the voice off the Master System, which had no sample channel
            and did speech by hammering the volume register instead
  brr       the voice through the SNES, which is all a SNES does: four-bit
            compression, the chip's own interpolation, and its echo
  a patch   not the voice: one of the emulated chips synthesising the note.
            The YM2612's (bass, lead, organ, brass, bell, piano, strings,
            and an FM kit), the NES's (pulse, triangle, and a kit off the
            noise register), or the Master System's (squares and noise, and
            nothing else, because that is all it has)

That middle one is how the console did its drums and its speech, so it is
what a corpus becomes when a whole song is routed through the chip -- the
melodies go to FM and the drums stay Michael Rosen, eight-bit. Which is the
arrangement a real Mega Drive track had.

The chip itself is app/ytpmv/ym2612.py; this is only the routing.
"""

from __future__ import annotations

import numpy as np

from app.ytpmv import gb, nes, opl, sid, sms, snes, ym2612
from app.ytpmv.pitch import SR

# Everything a part's tone may be set to, and what to call it. Two machines,
# and for each the same pair of choices: let its synthesiser play the note, or
# put the recorded voice through the channel that machine kept its samples on.
TONES = {
    "clean": "as recorded",
    "dac": "the voice off the Mega Drive's 8-bit sample channel",
    "dpcm": "the voice off the NES's delta-modulation channel",
    "psg-pcm": "the voice hammered out of the Master System's volume register",
    "brr": "the voice through the SNES's sampler, grain and echo and all",
    "gb-pcm": "the voice pushed through a Game Boy's wave table, four bits a step",
    "sb-pcm": "the voice off an original Sound Blaster's eight-bit DAC",
    "sb16-pcm": "the voice off a sixteen-bit Sound Blaster",
    "sid-digi": "the voice hammered out of the SID's volume register",
    "opl2": "the OPL2 playing Doom's GENMIDI patch for the part's instrument",
    "opl3": "the OPL3 playing the same patches, with all eight waveforms",
    "snes": "the SNES playing a sampled instrument, one short BRR sample of it",
    **{name: patch.about for name, patch in ym2612.PATCHES.items()},
    **{name: voice.about for name, voice in nes.VOICES.items()},
    **{name: voice.about for name, voice in sms.VOICES.items()},
    **{name: voice.about for name, voice in gb.VOICES.items()},
    **{name: voice.about for name, voice in sid.VOICES.items()},
}

# What these were called when there was one hand-built bass patch and no chip.
# And "voice", which is "clean" said on purpose: see overrides_switch.
ALIASES = {"slap": "bass", "megadrive": "bass", "voice": "clean"}

# The tones that replace the voice rather than colour it.
SYNTH = (frozenset(ym2612.PATCHES) | frozenset(nes.VOICES) | frozenset(sms.VOICES)
         | frozenset(gb.VOICES) | frozenset(sid.VOICES) | {"opl2", "opl3", "snes"})

# What each machine calls "the voice, played off this machine". The SNES has
# no other kind: it is a sampler and nothing else, so its only entry is here.
SAMPLED = {"md": "dac", "nes": "dpcm", "sms": "psg-pcm", "snes": "brr",
           "gb": "gb-pcm", "gbc": "gb-pcm", "opl2": "sb-pcm", "opl3": "sb16-pcm",
           "sid": "sid-digi", "sid8580": "sid-digi"}

# How fast the driver managed to feed the DAC. Mega Drive games rarely did
# better than this, and the graininess is the point.
DAC_RATE = 13300.0


# ── Channels ──────────────────────────────────────────────────────────────────
#
# How many notes each machine could actually sound at once, and on what. With
# "accurate" on, a song gets no more than that: a note that finds every
# channel of its kind busy takes one over and cuts off what was on it, which
# is exactly what a driver did when a song asked for more than the chip had.
# The Mega Drive's sixth FM channel is the DAC's when anything uses the DAC.
CHANNELS = {
    ("md", "fm"): 6, ("md", "dac"): 1,
    ("nes", "pulse"): 2, ("nes", "triangle"): 1, ("nes", "noise"): 1, ("nes", "dpcm"): 1,
    ("sms", "tone"): 3, ("sms", "noise"): 1, ("sms", "pcm"): 1,
    ("gb", "pulse"): 2, ("gb", "wave"): 1, ("gb", "noise"): 1,
    ("sid", "voice"): 3, ("sid", "digi"): 1,
    ("opl2", "fm"): 9, ("opl2", "pcm"): 1, ("opl3", "fm"): 18, ("opl3", "pcm"): 1,
    ("snes", "voice"): 8,
}
# Who keeps a channel when there are not enough: a note only takes one from a
# note that matters no more than it does, and otherwise is not heard.
_PRIORITY = {"lead": 4, "bass": 3, "drums": 3}


def channel(tone: str | None) -> tuple[str, str] | None:
    """The (machine, kind of channel) *tone* is played on; None for the voice."""
    t = resolve(tone)
    if t in (None, "clean"):
        return None
    if t == "dac" or t in ym2612.SAMPLED_DRUMS:
        return "md", "dac"
    if t in ym2612.PATCHES:
        return "md", "fm"
    if t == "dpcm":
        return "nes", "dpcm"
    if t in nes.VOICES:
        return "nes", nes.VOICES[t].kind
    if t == "psg-pcm":
        return "sms", "pcm"
    if t in sms.VOICES:
        return "sms", sms.VOICES[t].kind
    if t == "gb-pcm":
        return "gb", "wave"                  # the samples went out through the wave table
    if t in gb.VOICES:
        return "gb", gb.VOICES[t].kind
    if t == "sid-digi":
        return "sid", "digi"
    if t in sid.VOICES:
        return "sid", "voice"
    if t in ("opl2", "opl3"):
        return t, "fm"
    if t in ("sb-pcm", "sb16-pcm"):
        return ("opl2" if t == "sb-pcm" else "opl3"), "pcm"
    if t in ("snes", "brr"):
        return "snes", "voice"
    return None


def allocate(parts: list, speed: float = 1.0) -> dict[tuple, float]:
    """When each note is cut off for want of a channel, in the rendered clock.

    *parts* is [(Part, tone)]. Returns {(part id, note start, pitch): seconds}
    for the notes that lose their channel -- at or before their own start for
    a note that never got one. Notes not in it play out as written.
    """
    wanted = {channel(t) for _, t in parts} - {None}
    size = dict(CHANNELS)
    if ("md", "dac") in wanted:
        size[("md", "fm")] = 5
    events = []
    for part, tone in parts:
        ch = channel(tone)
        if ch is None:
            continue
        rank = _PRIORITY.get(part.role.split(":")[0], 1)
        for n in part.notes:
            events.append((n.start / speed, -rank, n.pitch, part.id, (n.start + n.dur) / speed, ch, n.start))
    events.sort()
    busy: dict[tuple, list] = {}              # channel kind -> [[end, rank, key], ...]
    cuts: dict[tuple, float] = {}
    for t0, neg_rank, pitch, pid, t1, ch, start in events:
        rank, key = -neg_rank, (pid, start, pitch)
        slots = busy.setdefault(ch, [])
        free = [s for s in slots if s[0] <= t0]
        if len(slots) < size.get(ch, 1):
            slots.append([t1, rank, key])
            continue
        if free:
            slot = min(free, key=lambda s: s[0])
        else:
            # Everything is still sounding: take the least important, and of
            # those the one that has been going longest.
            slot = min(slots, key=lambda s: (s[1], s[0]))
            if slot[1] > rank:
                cuts[key] = t0                # nowhere to go: not heard
                continue
            cuts[slot[2]] = t0
        slot[:] = [t1, rank, key]
    return cuts


def resolve(name: str | None) -> str | None:
    """The tone *name* means, or None when it is not one."""
    if not name:
        return None
    name = str(name).strip().lower()
    name = ALIASES.get(name, name)
    return name if name in TONES else None


def overrides_switch(name: str | None) -> bool:
    """Whether a part's own tone outranks the song-wide chip switch.

    "clean" does not, and that is the whole point of this existing. It is
    the default every part carries, and the page echoes its parts back
    exactly as it was given them -- so counting it as a choice meant the
    switch did nothing at all from the web page, for every song, while the
    command line (which sends no parts) worked perfectly.

    Which left no way to keep one part as the voice, untouched, while the rest of
    the song goes through a chip. "voice" is that: the same sound as "clean",
    and never sent unless somebody picked it.
    """
    chosen = resolve(name)
    if chosen == "clean":
        return str(name).strip().lower() == "voice"
    return bool(chosen)


# Every synthesising machine, as the page lists them: its tunes and its kit.
# Machines sharing a chip share an entry -- the Game Boy and the Color, the
# OPL2 and OPL3 (whose patches come from the part's instrument instead), the
# two SIDs.
MACHINE_OF = {"md": "md", "nes": "nes", "sms": "sms", "snes": "snes", "gb": "gb", "gbc": "gb",
              "opl2": "opl", "opl3": "opl", "sid": "sid", "sid8580": "sid"}
_KITS = {"md": (ym2612.PATCHES, ym2612.DRUMS), "nes": (nes.VOICES, nes.DRUMS),
         "sms": (sms.VOICES, sms.DRUMS), "gb": (gb.VOICES, gb.DRUMS),
         "sid": (sid.VOICES, sid.DRUMS)}
_MACHINE_NAMES = {"md": "Mega Drive", "nes": "NES", "sms": "Master System", "snes": "SNES",
                  "gb": "Game Boy", "opl": "OPL2 / OPL3", "sid": "SID"}


def catalogue() -> dict:
    """What a part can be played by, grouped for choosing from."""
    machines = {}
    for m, name in _MACHINE_NAMES.items():
        if m in ("opl", "snes"):              # one setting; the instrument picks the sound
            both = [[t, TONES[t]] for t in (("opl2", "opl3") if m == "opl" else ("snes",))]
            machines[m] = {"name": name, "melodic": both, "drums": both}
            continue
        voices, drums = _KITS[m]
        kit = set(drums.values())
        machines[m] = {"name": name,
                       "melodic": [[t, TONES[t]] for t in voices if t not in kit],
                       "drums": [[t, TONES[t]] for t in voices if t in kit]}
    sampled = list(dict.fromkeys(SAMPLED.values()))
    return {"machines": machines, "machine_of": MACHINE_OF,
            "voice": [["voice", "the voice, as recorded, whatever the chip"]]
                     + [[t, TONES[t]] for t in sampled],
            "synth": sorted(SYNTH)}


def apply(y: np.ndarray, tone: str, sr: int = SR, hz: float | None = None,
          level: float = 1.0, program: int | None = None, key: int | None = None,
          model: str = "6581") -> np.ndarray:
    """*y* played through *tone*. "clean" and anything unknown pass through.

    *hz* is the note, where the part has one. *program* is the part's General
    MIDI instrument and *key* its drum key, which is how the OPL chips pick a
    patch. *model* chooses between the SID's two chips.
    """
    tone = resolve(tone) or "clean"
    if tone == "clean" or len(y) == 0:
        return y
    peak = float(np.abs(y).max(initial=0.0))
    if peak <= 0:
        return y

    if tone == "snes":
        inst = snes.instrument_for(program, key)
        note = key if key is not None else (69.0 + 12.0 * float(np.log2(hz / 440.0)) if hz else 60.0)
        x = snes.render_note(inst, note, len(y) / sr, sr, level)
        return (x * peak).astype(np.float32)
    if tone in ("opl2", "opl3"):
        inst = opl.instrument_for(program, key)
        note = 69.0 + 12.0 * float(np.log2(hz / 440.0)) if hz else 60.0
        x = opl.render_note(inst, note, len(y) / sr, sr, level, opl3=(tone == "opl3"))
        return (x * peak).astype(np.float32)
    if tone in SYNTH:
        # Drums have no note and do not need one: every drum voice carries its
        # own pitch. This used to demand a note before synthesising anything,
        # so every chip's kit quietly fell through to the voice on the Mega
        # Drive's sample channel below -- on every chip, ever since the kits
        # were added -- and nothing noticed, because the tests only ever
        # called the chips directly rather than through here.
        hz = hz or 220.0
        # The voice is gone here: all that is kept of it is how long it went on
        # for and how loud it was, so the part still sits where it sat before.
        # The Mega Drive's patches are rendered longer than the note on
        # purpose -- the release rings on past the key coming up, and the
        # mixer is happy to have that overlap whatever follows.
        if tone in nes.VOICES:
            x = nes.render_note(nes.VOICES[tone], hz, len(y) / sr, sr, level)
        elif tone in sms.VOICES:
            x = sms.render_note(sms.VOICES[tone], hz, len(y) / sr, sr, level)
        elif tone in gb.VOICES:
            x = gb.render_note(gb.VOICES[tone], hz, len(y) / sr, sr, level)
        elif tone in sid.VOICES:
            x = sid.render_note(sid.VOICES[tone], hz, len(y) / sr, sr, level, model=model)
        else:
            x = ym2612.render_note(ym2612.PATCHES[tone], hz, len(y) / sr, sr, level)
        return (x * peak).astype(np.float32)
    if tone == "dpcm":
        # The channel's level runs 0 to 127 and idles in the middle.
        return ((nes.dpcm(y / peak, 10, sr) / 64.0 - 1.0) * peak).astype(np.float32)
    if tone == "psg-pcm":
        return (sms.pcm(y / peak, sr) * peak).astype(np.float32)
    if tone in ("sb-pcm", "sb16-pcm"):
        return (opl.pcm(y / peak, sr, sixteen=(tone == "sb16-pcm")) * peak).astype(np.float32)
    if tone == "sid-digi":
        return (sid.digi(y / peak, sr, model=model) * peak).astype(np.float32)
    if tone == "gb-pcm":
        return (gb.pcm(y / peak, sr) * peak).astype(np.float32)
    if tone == "brr":
        # Longer than it went in: the echo rings on past the note, and the
        # mixer is happy to have that overlap whatever comes next.
        return (snes.play(y / peak, sr) * peak).astype(np.float32)
    return (ym2612.dac(y / peak, DAC_RATE, sr) * peak).astype(np.float32)
