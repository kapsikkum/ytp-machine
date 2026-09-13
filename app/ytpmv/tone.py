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

from app.ytpmv import nes, sms, snes, ym2612
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
    **{name: patch.about for name, patch in ym2612.PATCHES.items()},
    **{name: voice.about for name, voice in nes.VOICES.items()},
    **{name: voice.about for name, voice in sms.VOICES.items()},
}

# What these were called when there was one hand-built bass patch and no chip.
ALIASES = {"slap": "bass", "megadrive": "bass"}

# The tones that replace the voice rather than colour it.
SYNTH = frozenset(ym2612.PATCHES) | frozenset(nes.VOICES) | frozenset(sms.VOICES)

# What each machine calls "the voice, played off this machine". The SNES has
# no other kind: it is a sampler and nothing else, so its only entry is here.
SAMPLED = {"md": "dac", "nes": "dpcm", "sms": "psg-pcm", "snes": "brr"}

# How fast the driver managed to feed the DAC. Mega Drive games rarely did
# better than this, and the graininess is the point.
DAC_RATE = 13300.0


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
    """
    chosen = resolve(name)
    return bool(chosen) and chosen != "clean"


def apply(y: np.ndarray, tone: str, sr: int = SR, hz: float | None = None,
          level: float = 1.0) -> np.ndarray:
    """*y* played through *tone*. "clean" and anything unknown pass through.

    *hz* is the note, where the part has one. A patch needs it, because it is
    synthesising rather than processing; a drum has no note, so it goes to the
    sample channel instead -- which is where the console put its drums anyway.
    """
    tone = resolve(tone) or "clean"
    if tone == "clean" or len(y) == 0:
        return y
    peak = float(np.abs(y).max(initial=0.0))
    if peak <= 0:
        return y

    if tone in SYNTH and hz:
        # The voice is gone here: all that is kept of it is how long it went on
        # for and how loud it was, so the part still sits where it sat before.
        # The Mega Drive's patches are rendered longer than the note on
        # purpose -- the release rings on past the key coming up, and the
        # mixer is happy to have that overlap whatever follows.
        if tone in nes.VOICES:
            x = nes.render_note(nes.VOICES[tone], hz, len(y) / sr, sr, level)
        elif tone in sms.VOICES:
            x = sms.render_note(sms.VOICES[tone], hz, len(y) / sr, sr, level)
        else:
            x = ym2612.render_note(ym2612.PATCHES[tone], hz, len(y) / sr, sr, level)
        return (x * peak).astype(np.float32)
    if tone == "dpcm":
        # The channel's level runs 0 to 127 and idles in the middle.
        return ((nes.dpcm(y / peak, 10, sr) / 64.0 - 1.0) * peak).astype(np.float32)
    if tone == "psg-pcm":
        return (sms.pcm(y / peak, sr) * peak).astype(np.float32)
    if tone == "brr":
        # Longer than it went in: the echo rings on past the note, and the
        # mixer is happy to have that overlap whatever comes next.
        return (snes.play(y / peak, sr) * peak).astype(np.float32)
    return (ym2612.dac(y / peak, DAC_RATE, sr) * peak).astype(np.float32)
