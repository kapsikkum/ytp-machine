"""The picture when there is no voice: every part as its own oscilloscope.

With nothing recorded there is no clip to cut to, so each tile shows what its
part is actually putting out, the way chiptune videos draw a channel -- a
trace in neon red on black, held still by triggering on a rising zero
crossing. It is drawn at the console's own resolution and then goes through
that console's palette like any other picture, so on a Game Boy the red comes
out as whichever green it lands nearest.
"""

from __future__ import annotations

import numpy as np

from app.ytpmv.pitch import SR

WINDOW = 0.03          # seconds of sound across a tile, when the part has no pitch
PERIODS = 4            # otherwise this many cycles of its middle note
SEARCH = 0.02          # how far on to look for a trigger: a 50 Hz period

# (trace, trace on a hit, glow, axis). Chosen to land on a proper red in every
# palette: a pinker red is nearer the NES's magenta than its red, and pink on
# the EGA is grey. The Game Boy has no red to land on, and red comes out as a
# dark green against its darkest, so it draws in its lightest shade instead.
_COLOURS = {
    None: ((255, 40, 40), (255, 120, 110), (110, 0, 22), (34, 0, 8)),
    "gb": ((255, 255, 255), (255, 255, 255), (150, 150, 150), (70, 70, 70)),
}


def traces(y: np.ndarray, n_frames: int, spf: int, width: int, sr: int = SR,
           hz: float | None = None) -> np.ndarray:
    """(n_frames, width, 2): the lowest and highest point in each column.

    *y* is the part on its own, mono. Each column keeps its range rather than
    one sample, so a note too high to draw as a line comes out as the band it
    fills instead of an aliased zigzag, and columns are widened to meet their
    neighbours so the trace never breaks. *hz* is the part's middle note,
    which sets how much of the sound a tile holds: a few cycles of it, so a
    bass is not two wobbles and a lead not a solid block.
    """
    width = max(1, int(width))
    window = min(0.04, max(0.006, PERIODS / hz)) if hz else WINDOW
    span = max(width, int(window * sr) // width * width)
    search = int(SEARCH * sr)
    peak = float(np.percentile(np.abs(y), 99.9)) if len(y) else 0.0
    scale = 0.9 / peak if peak > 1e-6 else 0.0
    out = np.zeros((n_frames, width, 2), dtype=np.float16)
    padded = np.concatenate([y, np.zeros(span + search + 1, dtype=y.dtype)]) * scale
    for f in range(n_frames):
        c = f * spf
        head = padded[c:c + search + 1]
        rise = np.nonzero((head[:-1] <= 0.0) & (head[1:] > 0.0))[0]
        a = c + (int(rise[0]) + 1 if len(rise) else 0)
        cols = padded[a:a + span].reshape(width, -1)
        lo, hi = cols.min(axis=1), cols.max(axis=1)
        if width > 1:
            lo_next, hi_next = lo[1:].copy(), hi[1:].copy()
            lo[:-1] = np.minimum(lo[:-1], hi_next)
            hi[:-1] = np.maximum(hi[:-1], lo_next)
        out[f, :, 0], out[f, :, 1] = lo, hi
    return out


def draw(trace: np.ndarray, w: int, h: int, level: float = 1.0, hot: bool = False,
         screen: str | None = None) -> np.ndarray:
    """One tile, *h* x *w*: the trace, its glow, and the axis it sits on."""
    NEON, HOT, GLOW, AXIS = (np.array(c, dtype=np.float32)
                             for c in _COLOURS.get(screen, _COLOURS[None]))
    img = np.zeros((h, w, 3), dtype=np.float32)
    mid = (h - 1) / 2.0
    amp = max(1.0, h / 2.0 - 2.0)
    img[int(round(mid))] = AXIS
    cols = trace[:w].astype(np.float32)
    top = np.clip(np.round(mid - cols[:, 1] * amp), 0, h - 1)
    bot = np.clip(np.round(mid - cols[:, 0] * amp), 0, h - 1)
    rows = np.arange(h, dtype=np.float32)[:, None]
    glow = (rows >= top - 1) & (rows <= bot + 1)
    core = (rows >= top) & (rows <= bot)
    img[glow] = GLOW
    img[core] = HOT if hot else NEON
    if level != 1.0:
        img *= level
    return img.astype(np.uint8)
