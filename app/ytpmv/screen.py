"""Put the picture through the console as well as the sound.

A frame is cut down to what the machine could actually put on a television:
its resolution, and its colours and no others. It comes back at that size and
is blown up again during the encode, so what you see is chunky and banded the
way the hardware was.

  md    320x224, and nine bits of colour: three per channel, eight levels
        each, 512 in all. The levels are not evenly spaced. Measured off
        hardware they run 0, 52, 87, 116, 144, 172, 206, 255 -- squashed
        together in the middle and stretched at the ends -- so a linear
        eighth-steps ramp, which is what you would write if you assumed,
        gets the midtones visibly wrong
  nes   256x240, and no freedom at all: sixty-four fixed colours burned
        into the chip, of which a real frame could show about sixteen at
        once. There is no mixing and no choosing -- every pixel is one of
        these or it is not on the screen
  sms   256x192 and two bits a channel, so sixty-four colours, though
        unlike the NES they are yours to choose
  snes  256x224 and five bits a channel, which after the other three is
        an embarrassment of riches -- thirty-two thousand
  gb    160x144 and four shades of green, and that is all of it
  gbc   160x144, five bits a channel like the SNES, but shown on an LCD
        that bled its colours into each other and never reached full
        brightness -- which is the whole look, so it is modelled

The palette below is the one FCEUX ships, which is what most people picture
when they picture NES colours. There is no single correct answer: the 2C02
emits composite video and a television decodes it, so the palette depends on
the set, the cable and the decoder. Spot-check it against colours everybody
knows -- $0F black, $30 white, $21 the sky in Super Mario Bros.

numpy only. The work per frame is a shrink and a lookup.
"""

from __future__ import annotations

import numpy as np

# The eight levels a Mega Drive colour component can actually come out at,
# measured off hardware by TmEE. Not a linear ramp.
MD_LEVELS = np.array([0, 52, 87, 116, 144, 172, 206, 255], dtype=np.uint8)

# The 2C02's sixty-four, in index order $00-$3F. $0D is a blanking level that
# real hardware was never meant to show and $0E/$0F/$1E/$1F and friends are
# black; they are kept in place so the indices are the machine's own.
_NES_HEX = (
    "747474", "24188c", "0000a8", "44009c", "8c0074", "a80010", "a40000", "7c0800",
    "402c00", "004400", "005000", "003c14", "183c5c", "000000", "000000", "000000",
    "bcbcbc", "0070ec", "2038ec", "8000f0", "bc00bc", "e40058", "d82800", "c84c0c",
    "887000", "009400", "00a800", "009038", "008088", "000000", "000000", "000000",
    "fcfcfc", "3cbcfc", "5c94fc", "cc88fc", "f478fc", "fc74b4", "fc7460", "fc9838",
    "f0bc3c", "80d010", "4cdc48", "58f898", "00e8d8", "787878", "000000", "000000",
    "fcfcfc", "a8e4fc", "c4d4fc", "d4c8fc", "fcc4fc", "fcc4d8", "fcbcb0", "fcd8a8",
    "fce4a0", "e0fca0", "a8f0bc", "b0fccc", "9cfcf0", "c4c4c4", "000000", "000000",
)
NES_PALETTE = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in _NES_HEX],
                       dtype=np.uint8)

# Two bits a channel on the Master System -- sixty-four colours in all, of
# which a frame could hold thirty-two. Evenly spaced, unlike the Mega
# Drive's: these are the obvious quarters rather than anything measured, so
# if they ever matter enough to argue about, go and measure a console.
SMS_LEVELS = np.array([0, 85, 170, 255], dtype=np.uint8)

# Five bits a channel on the SNES, which is a straight ramp: the value is
# repeated into the low bits so that all-ones comes out as white.
SNES_LEVELS = np.array([(v << 3) | (v >> 2) for v in range(32)], dtype=np.uint8)

# The Game Boy's four shades, lightest first, as ares draws them. Nothing on
# the machine emits RGB -- it is a reflective green LCD -- so these are what
# one emulator settled on rather than a hardware figure, and they are the
# pea-soup ones everybody remembers.
GB_SHADES = np.array([[0xAE, 0xD9, 0x27], [0x58, 0xA0, 0x28],
                      [0x20, 0x62, 0x29], [0x1A, 0x45, 0x2A]], dtype=np.uint8)

# IBM's sixteen, as an EGA card powers up: the CGA's RGBI colours with dark
# yellow pulled down to brown, which is the one everybody remembers. This is
# what an AdLib-era PC game was drawn in.
EGA_PALETTE = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in (
    "000000", "0000aa", "00aa00", "00aaaa", "aa0000", "aa00aa", "aa5500", "aaaaaa",
    "555555", "5555ff", "55ff55", "55ffff", "ff5555", "ff55ff", "ffff55", "ffffff")],
    dtype=np.uint8)

# The C64's sixteen, Pepto's Colodore measurements as VICE ships them. Like
# the NES, the VIC-II emits composite and a television decides the colour,
# so no palette is the right one; this is the most carefully measured.
C64_PALETTE = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in (
    "000000", "ffffff", "96282e", "5bd6ce", "9f2dad", "41b936", "2724c4", "eff347",
    "9f4815", "5e3500", "da5f66", "474747", "787878", "91ff84", "6864ff", "aeaeae")],
    dtype=np.uint8)

# VGA's DAC: six bits a channel, repeated into the low bits.
VGA_LEVELS = np.array([(v << 2) | (v >> 4) for v in range(64)], dtype=np.uint8)

# What each machine could put on a television.
SCREENS = {
    "none": "as rendered",
    "md": "320x224, nine bits of colour",
    "nes": "256x240, and the 2C02's sixty-four colours",
    "sms": "256x192, six bits of colour",
    "snes": "256x224, fifteen bits of colour",
    "ega": "320x200, the EGA's sixteen colours",
    "vga": "320x200, eighteen bits of colour",
    "c64": "320x200, the VIC-II's sixteen colours",
    "gb": "160x144, four shades of green",
    "gbc": "160x144, fifteen bits through the Color's washed-out LCD",
}

_LEVEL_LUTS: dict[str, np.ndarray] = {}
_NES_LUT: np.ndarray | None = None

# Which machine quantises each channel on its own, and to what.
_LEVELS = {"md": MD_LEVELS, "sms": SMS_LEVELS, "snes": SNES_LEVELS, "vga": VGA_LEVELS}

# Machines with a fixed set of colours and nothing between them.
_FIXED = {"nes": NES_PALETTE, "ega": EGA_PALETTE, "c64": C64_PALETTE}
_FIXED_LUTS: dict[str, np.ndarray] = {}


def _level_lut(name: str) -> np.ndarray:
    """For every byte a channel might be, the level *name* comes out at."""
    if name not in _LEVEL_LUTS:
        levels = _LEVELS[name]
        v = np.arange(256)[:, None]
        _LEVEL_LUTS[name] = levels[np.argmin(np.abs(v - levels[None, :].astype(int)), axis=1)]
    return _LEVEL_LUTS[name]


def _fixed_lut(name: str) -> np.ndarray:
    """The nearest of a fixed palette, for every colour rounded to five bits."""
    if name not in _FIXED_LUTS:
        pal = _FIXED[name]
        step = np.arange(32) * 255.0 / 31.0
        grid = np.stack(np.meshgrid(step, step, step, indexing="ij"), axis=-1).reshape(-1, 3)
        w = np.array([0.30, 0.59, 0.11])
        d = ((grid[:, None, :] - pal[None, :, :].astype(float)) ** 2 * w).sum(axis=2)
        _FIXED_LUTS[name] = pal[np.argmin(d, axis=1)]
    return _FIXED_LUTS[name]


def _nes_lut() -> np.ndarray:
    """The nearest of the sixty-four, for every colour rounded to five bits.

    Thirty-two thousand entries worked out once, so a frame costs one lookup
    rather than sixty-four distances per pixel.
    """
    global _NES_LUT
    if _NES_LUT is None:
        step = np.arange(32) * 255.0 / 31.0
        grid = np.stack(np.meshgrid(step, step, step, indexing="ij"), axis=-1).reshape(-1, 3)
        # Weighted towards green, which is where the eye keeps most of its
        # judgement; matching on plain distance sends greys off into colour.
        w = np.array([0.30, 0.59, 0.11])
        d = ((grid[:, None, :] - NES_PALETTE[None, :, :].astype(float)) ** 2 * w).sum(axis=2)
        _NES_LUT = NES_PALETTE[np.argmin(d, axis=1)]
    return _NES_LUT


def gb_shade(img: np.ndarray) -> np.ndarray:
    """Each pixel as whichever of the four greens its brightness falls in."""
    lum = (img[:, :, 0].astype(np.int32) * 299 + img[:, :, 1].astype(np.int32) * 587
           + img[:, :, 2].astype(np.int32) * 114) // 1000
    return np.ascontiguousarray(GB_SHADES[3 - np.clip(lum // 64, 0, 3)])


def gbc_lcd(img: np.ndarray) -> np.ndarray:
    """Five bits a channel, then the Color's LCD bleeding them into each other.

    The screen was not a clean RGB panel: each colour leaked into its
    neighbours and nothing reached full brightness, which is why every Game
    Boy Color game looks faintly washed out and dim. These weights are ares's
    model of that; the cap at 960 of 1023 is the dimness.
    """
    r = (img[:, :, 0] >> 3).astype(np.int32)
    g = (img[:, :, 1] >> 3).astype(np.int32)
    b = (img[:, :, 2] >> 3).astype(np.int32)
    rr = np.minimum(960, r * 26 + g * 4 + b * 2)
    gg = np.minimum(960, g * 24 + b * 8)
    bb = np.minimum(960, r * 6 + g * 4 + b * 22)
    out = np.stack([rr, gg, bb], axis=-1) * 255 // 1023
    return np.ascontiguousarray(out.astype(np.uint8))


def _shrink(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Down to *w* x *h*, averaging what falls in each new pixel.

    Averaged rather than sampled: picking one pixel out of every eight makes
    the edges crawl about from frame to frame, which reads as a bad encode
    rather than as a console.
    """
    src_h, src_w = img.shape[:2]
    ys = np.linspace(0, src_h, h + 1).astype(np.int64)
    xs = np.linspace(0, src_w, w + 1).astype(np.int64)
    # Summed straight into int32. Casting the whole frame to float first is
    # six million conversions for the sake of an average, and the first pass
    # has already thrown away four fifths of the rows anyway.
    a = np.add.reduceat(img, ys[:-1], axis=0, dtype=np.int32)
    a = np.add.reduceat(a, xs[:-1], axis=1, dtype=np.int32)
    counts = np.diff(ys)[:, None, None] * np.diff(xs)[None, :, None]
    return a // np.maximum(counts, 1)


# What each machine put on the screen, in its own pixels.
SIZES = {"md": (320, 224), "nes": (256, 240), "sms": (256, 192), "snes": (256, 224),
         "gb": (160, 144), "gbc": (160, 144), "ega": (320, 200), "vga": (320, 200),
         "c64": (320, 200)}


def apply(frame: np.ndarray, screen: str) -> np.ndarray:
    """*frame* reduced to what the named machine could show.

    Comes back at the machine's own size, not the size it went in at: blowing
    it up again is somebody else's job, and ffmpeg will do it for nothing
    while it is encoding anyway.
    """
    if screen not in SIZES:
        return frame
    w, h = SIZES[screen]
    small = np.clip(_shrink(frame, w, h), 0, 255).astype(np.uint8)
    if screen in _LEVELS:
        return np.ascontiguousarray(_level_lut(screen)[small])
    if screen == "gb":
        return gb_shade(small)
    if screen == "gbc":
        return gbc_lcd(small)
    if screen in ("ega", "c64"):
        idx = ((small[:, :, 0] >> 3).astype(np.int32) * 1024
               + (small[:, :, 1] >> 3).astype(np.int32) * 32
               + (small[:, :, 2] >> 3).astype(np.int32))
        return np.ascontiguousarray(_fixed_lut(screen)[idx])
    idx = ((small[:, :, 0] >> 3).astype(np.int32) * 1024
           + (small[:, :, 1] >> 3).astype(np.int32) * 32
           + (small[:, :, 2] >> 3).astype(np.int32))
    return np.ascontiguousarray(_nes_lut()[idx])
