#!/usr/bin/env python3
"""The picture put through a console. Plain asserts.

    python tests/test_screen.py

Needs numpy. The thing worth checking is not that the filter does something
-- anything that throws colours away looks retro at a glance -- but that what
comes out is only ever a colour the machine could actually have produced.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ytpmv import screen

failures: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


rng = np.random.default_rng(1312)
frame = rng.integers(0, 256, (540, 960, 3), dtype=np.uint8)
photo = np.clip(np.add.outer(np.linspace(0, 255, 540), np.zeros(960))[:, :, None]
                + rng.integers(-40, 40, (540, 960, 3)), 0, 255).astype(np.uint8)

print("the NES palette is the hardware's")
check("sixty-four entries", screen.NES_PALETTE.shape, (64, 3))
# Colours anybody who has seen the machine can check by eye.
check("$0F is black", tuple(int(v) for v in screen.NES_PALETTE[0x0F]), (0, 0, 0))
check("$30 is its white, which is not quite white",
      tuple(int(v) for v in screen.NES_PALETTE[0x30]), (252, 252, 252))
check("$21 is the Super Mario Bros sky",
      tuple(int(v) for v in screen.NES_PALETTE[0x21]), (60, 188, 252))

print("the Mega Drive's levels are the hardware's")
check("eight of them", len(screen.MD_LEVELS), 8)
check("the Master System has four a channel", len(screen.SMS_LEVELS), 4)
check("the SNES has thirty-two", len(screen.SNES_LEVELS), 32)
check("and all-ones on the SNES is white", int(screen.SNES_LEVELS[31]), 255)
check("as all-zeroes is black", int(screen.SNES_LEVELS[0]), 0)
check("measured off a console, not a linear ramp",
      [int(v) for v in screen.MD_LEVELS], [0, 52, 87, 116, 144, 172, 206, 255])
# The point of the table: a linear ramp would put the middle steps elsewhere.
linear = [round(i * 255 / 7) for i in range(8)]
check("which is not what evenly spaced eighths would give",
      [int(v) for v in screen.MD_LEVELS] == linear, False)

print("what comes out")
for name, (w, h) in screen.SIZES.items():
    for src in (frame, photo):
        out = screen.apply(src, name)
        check(f"{name} comes back at the machine's own size", out.shape, (h, w, 3))
        check(f"{name} comes back as bytes", out.dtype, np.dtype(np.uint8))
    out = screen.apply(photo, name)
    if name == "nes":
        # The only machine with no say in the matter: its colours are burnt in.
        legal = {tuple(int(v) for v in p) for p in screen.NES_PALETTE}
        seen = {tuple(int(v) for v in c) for c in np.unique(out.reshape(-1, 3), axis=0)}
        check("every NES colour is one of the sixty-four", seen <= legal, True)
    else:
        allowed = {int(v) for v in screen._LEVELS[name]}
        check(f"every {name} channel value is a level the hardware has",
              set(int(v) for v in np.unique(out)) <= allowed, True)
        most = len(screen._LEVELS[name]) ** 3
        check(f"and no more than {most} colours are available",
              len(np.unique(out.reshape(-1, 3), axis=0)) <= most, True)

print("leaving it alone")
for name in ("none", "", "amiga", None):
    out = screen.apply(frame, name)
    check(f"{name!r} changes nothing", out.shape, frame.shape)
check("and nothing is copied when nothing is asked for",
      screen.apply(frame, "none") is frame, True)

print("the shrink averages rather than samples")
# Half the picture bright, half dark: sampling could return either, and an
# average has to land in the middle. Sampling is what makes edges crawl.
half = np.zeros((100, 100, 3), dtype=np.uint8)
half[:, :50] = 200
small = screen._shrink(half, 2, 1)
check("a block of light and dark averages to the middle",
      int(small[0, 0, 0]), 200)
check("and the dark half stays dark", int(small[0, 1, 0]), 0)
mixed = screen._shrink(half, 1, 1)
check("the whole picture averages to half of it", int(mixed[0, 0, 0]), 100)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
