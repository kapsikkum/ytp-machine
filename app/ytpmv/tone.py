"""What a part is played through, once its notes are already right.

Pitch and timing are settled before this; a tone only decides what makes the
sound. There are two kinds, and the difference decides whether you still hear
the voice at all:

  clean     the voice, untouched
  dac       the voice through the chip's PCM channel -- eight bits at about
            13 kHz, with the dead zone around silence. Still the voice, the
            way a Mega Drive played a sampled one
  a patch   not the voice: the emulated YM2612 synthesising the note
            (bass, lead, organ, brass, bell, piano, strings)

That middle one is how the console did its drums and its speech, so it is
what a corpus becomes when a whole song is routed through the chip -- the
melodies go to FM and the drums stay Michael Rosen, eight-bit. Which is the
arrangement a real Mega Drive track had.

The chip itself is app/ytpmv/ym2612.py; this is only the routing.
"""

from __future__ import annotations

import numpy as np

from app.ytpmv import ym2612
from app.ytpmv.pitch import SR

# Everything a part's tone may be set to, and what to call it.
TONES = {
    "clean": "as recorded",
    "dac": "the voice off the chip's 8-bit sample channel",
    **{name: patch.about for name, patch in ym2612.PATCHES.items()},
}

# What these were called when there was one hand-built bass patch and no chip.
ALIASES = {"slap": "bass", "megadrive": "bass"}

# The tones that replace the voice rather than colour it.
SYNTH = frozenset(ym2612.PATCHES)

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
        # Longer than the note on purpose: the release rings on past the key
        # coming up, and the mixer is happy to have it overlap what follows.
        x = ym2612.render_note(ym2612.PATCHES[tone], hz, len(y) / sr, sr, level)
        return (x * peak).astype(np.float32)
    return (ym2612.dac(y / peak, DAC_RATE, sr) * peak).astype(np.float32)
