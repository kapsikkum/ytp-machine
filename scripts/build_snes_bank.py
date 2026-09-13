#!/usr/bin/env python3
"""Cut a General MIDI SoundFont down to a bank a SNES could have carried.

    python scripts/build_snes_bank.py GeneralUser-GS.sf2

Writes app/ytpmv/data/snes_bank.npz, which is what the SNES setting plays.
Any General MIDI font works: programs from bank 0, drums from bank 128's
first kit. Which font a bank came from, and whatever it says about its
author and copyright, is kept inside it and printed when it is built.

A SNES soundtrack had 64 KB for every sample in it, so no game carried a
General MIDI bank. What they carried was one short sample per instrument,
kept well under the DSP's rate, looped on a boundary the compression could
hold. So that is what this makes of the font, rather than shipping 31 MB of
it:

  one sample   per program, the one the font plays at a middle note (or low
               down, for a bass), and one per drum key
  16 kHz       at most. Samples were routinely kept lower still
  16-sample    loops: BRR works in blocks of sixteen, so a loop has to be a
  loops        whole number of them, and converters stretched the sample to
               make it so. The small change in pitch that makes is folded
               into the rate, so the note stays in tune
  BRR          the sample goes through the compression once, here, the way
               it went through it once when a game was built

Its amplitude envelope is kept, as the font has it -- delay, attack, hold,
decay, sustain and release -- along with its tuning, root key and level.
Filters, modulators, LFOs, and every zone but the one chosen are dropped.

numpy only.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ytpmv import snes  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "app", "ytpmv", "data", "snes_bank.npz")
STORE_HZ = 16000.0
MAX_ONESHOT = 1.5        # seconds of a sample that does not loop
MAX_LOOPED = 2.5         # how far into a sample its loop may end
DRUM_KEYS = range(27, 88)

# Generator numbers, from the SoundFont 2.04 specification.
G = dict(startAddrsOffset=0, endAddrsOffset=1, startloopAddrsOffset=2, endloopAddrsOffset=3,
         startAddrsCoarseOffset=4, endAddrsCoarseOffset=12, delayVolEnv=33, attackVolEnv=34,
         holdVolEnv=35, decayVolEnv=36, sustainVolEnv=37, releaseVolEnv=38, instrument=41,
         keyRange=43, velRange=44, startloopAddrsCoarseOffset=45, initialAttenuation=48,
         endloopAddrsCoarseOffset=50, coarseTune=51, fineTune=52, sampleID=53, sampleModes=54,
         scaleTuning=56, overridingRootKey=58)
_RANGES = (G["keyRange"], G["velRange"])
# What a preset zone adds to its instrument's value rather than replacing it.
_ADDITIVE = {G[k] for k in ("delayVolEnv", "attackVolEnv", "holdVolEnv", "decayVolEnv",
                            "sustainVolEnv", "releaseVolEnv", "initialAttenuation",
                            "coarseTune", "fineTune")}
_DEFAULTS = {G["delayVolEnv"]: -12000, G["attackVolEnv"]: -12000, G["holdVolEnv"]: -12000,
             G["decayVolEnv"]: -12000, G["sustainVolEnv"]: 0, G["releaseVolEnv"]: -12000,
             G["scaleTuning"]: 100, G["overridingRootKey"]: -1}


def _chunks(data: bytes, start: int, end: int):
    while start + 8 <= end:
        cid, size = data[start:start + 4], struct.unpack_from("<I", data, start + 4)[0]
        yield cid, start + 8, size
        start += 8 + size + (size & 1)


class SoundFont:
    def __init__(self, path: str):
        data = open(path, "rb").read()
        if data[:4] != b"RIFF" or data[8:12] != b"sfbk":
            raise SystemExit(f"{path} is not a SoundFont")
        parts = {}
        for cid, off, size in _chunks(data, 12, len(data)):
            if cid == b"LIST":
                for sub, soff, ssize in _chunks(data, off + 4, off + size):
                    parts[sub] = (soff, ssize)
        off, size = parts[b"smpl"]
        self.pcm = np.frombuffer(data, dtype="<i2", count=size // 2, offset=off)

        def table(name, fmt):
            o, s = parts[name]
            n = struct.calcsize(fmt)
            return [struct.unpack_from(fmt, data, o + i) for i in range(0, s, n)]

        self.phdr = table(b"phdr", "<20sHHHIII")
        self.pbag = table(b"pbag", "<HH")
        self.pgen = table(b"pgen", "<HH")
        self.inst = table(b"inst", "<20sH")
        self.ibag = table(b"ibag", "<HH")
        self.igen = table(b"igen", "<HH")
        self.shdr = table(b"shdr", "<20sIIIIIBbHH")
        self.info = {}
        for cid, off, size in _chunks(data, 12, len(data)):
            if cid == b"LIST" and data[off:off + 4] == b"INFO":
                for sub, soff, ssize in _chunks(data, off + 4, off + size):
                    if sub in (b"INAM", b"IENG", b"ICOP", b"ICRD", b"ICMT"):
                        self.info[sub.decode()] = data[soff:soff + ssize].split(bytes(1))[0].decode("latin-1")

    @staticmethod
    def _zones(bags, gens, first, last):
        zones = []
        for b in range(first, last):
            z = {}
            for g in range(bags[b][0], bags[b + 1][0]):
                op, amount = gens[g]
                z[op] = amount if op in _RANGES else struct.unpack("<h", struct.pack("<H", amount))[0]
            zones.append(z)
        return zones

    @staticmethod
    def _covers(z, key, vel):
        for op, v in ((G["keyRange"], key), (G["velRange"], vel)):
            if op in z and not (z[op] & 0xFF) <= v <= (z[op] >> 8):
                return False
        return True

    def voice(self, bank: int, program: int, key: int, vel: int = 100):
        """The generators and sample the font plays for this note, or None."""
        for i, (name, prog, bnk, bag, *_rest) in enumerate(self.phdr[:-1]):
            if prog == program and bnk == bank:
                break
        else:
            return None
        pz = self._zones(self.pbag, self.pgen, bag, self.phdr[i + 1][3])
        pglobal = pz.pop(0) if pz and G["instrument"] not in pz[0] else {}
        for p in pz:
            if not self._covers(p, key, vel):
                continue
            ii = p[G["instrument"]]
            iz = self._zones(self.ibag, self.igen, self.inst[ii][1], self.inst[ii + 1][1])
            iglobal = iz.pop(0) if iz and G["sampleID"] not in iz[0] else {}
            for z in iz:
                if not self._covers(z, key, vel):
                    continue
                gens = {**_DEFAULTS, **iglobal, **z}
                for src in (pglobal, p):
                    for op, v in src.items():
                        if op in _ADDITIVE:
                            gens[op] = gens.get(op, _DEFAULTS.get(op, 0)) + v
                return name.split(b"\0")[0].decode("latin-1"), gens
        return None


def _fft_resample(y: np.ndarray, n: int) -> np.ndarray:
    """*y* stretched to *n* samples, band-limited so nothing folds back."""
    if n == len(y) or len(y) < 2:
        return y.astype(np.float64)
    spec = np.fft.rfft(y)
    keep = min(len(spec), n // 2 + 1)
    out = np.zeros(n // 2 + 1, dtype=complex)
    out[:keep] = spec[:keep]
    return np.fft.irfft(out, n) * (n / len(y))


def _seconds(timecents: int) -> float:
    return 0.0 if timecents <= -12000 else float(2.0 ** (timecents / 1200.0))


def loop_hz(x: np.ndarray, loop, rate: float) -> float | None:
    """The pitch a loop repeats at, from the harmonics it actually contains.

    A loop of whole cycles only has energy at multiples of rate / length; the
    fundamental is the largest step those multiples share. A loop that is not
    whole cycles has no such step, and gives nothing.
    """
    if not loop or loop[1] - loop[0] < 8:
        return None
    seg = x[loop[0]:loop[1]]
    c = np.abs(np.fft.rfft(seg - seg.mean()))[1:]
    if c.max(initial=0.0) <= 0:
        return None
    strong = np.nonzero(c > 0.08 * c.max())[0] + 1
    step = int(np.gcd.reduce(strong))
    # And the fundamental has to be there to hear, not only implied: an organ
    # whose sub-octave stop is a whisper under the note repeats at the lower
    # pitch, but nobody hears it as that note, and nor did whoever tuned it.
    if c[step - 1] < 0.25 * c.max():
        return None
    return step * rate / len(seg)


def cut(sf: SoundFont, bank: int, program: int, key: int, store_hz: float = STORE_HZ,
        encode: bool = True, fix_octaves: bool = False):
    got = sf.voice(bank, program, key)
    if got is None:
        return None
    name, g = got
    s = g[G["sampleID"]]
    _n, start, end, lstart, lend, rate, pitch, correction, _link, _type = sf.shdr[s]
    start += g.get(0, 0) + 32768 * g.get(4, 0)
    end += g.get(1, 0) + 32768 * g.get(12, 0)
    lstart += g.get(2, 0) + 32768 * g.get(45, 0)
    lend += g.get(3, 0) + 32768 * g.get(50, 0)
    y = sf.pcm[start:end].astype(np.float64) / 32768.0
    looped = g.get(G["sampleModes"], 0) in (1, 3) and start <= lstart < lend <= end
    ls, le = (lstart - start, lend - start) if looped else (0, 0)
    if looped and le / rate > MAX_LOOPED:
        looped = False
    if not looped:
        y = y[:int(MAX_ONESHOT * rate)]
        fade = min(len(y), int(0.01 * rate))
        if fade:
            y[-fade:] *= np.linspace(1.0, 0.0, fade)
    else:
        y = y[:le]

    # The rate it is kept at, nudged so the loop is a whole number of blocks.
    store = min(float(rate), store_hz)
    if looped:
        # Rounded down, so the stretch never lifts it over the cap -- unless
        # the loop is shorter than one block, which has to be stretched up.
        blocks = max(1, int((le - ls) * store / rate / 16.0))
        store = 16.0 * blocks * rate / (le - ls)
    n = max(16, int(round(len(y) * store / rate)))
    x = _fft_resample(y, n)
    lead = 0
    if looped:
        ls16 = int(round(ls * store / rate))
        lead = (-ls16) % 16                         # pad so the loop starts on a block
        x = np.concatenate([np.zeros(lead), x])
        ls, le = ls16 + lead, ls16 + lead + 16 * blocks
        x = x[:le]
    if encode:
        x = snes.brr(x.astype(np.float32)).astype(np.float64)
    peak = float(np.abs(x).max(initial=0.0))
    if peak <= 0:
        return None
    root = g[G["overridingRootKey"]] if g[G["overridingRootKey"]] >= 0 else pitch
    # Fonts ripped from games are often labelled an octave out: a loop that
    # is one cycle at 524 Hz marked as middle C, so every note plays an
    # octave over what the score says. Where the loop's own pitch is a whole
    # octave or two from its label -- and only then, since any font's samples
    # sit a few cents off -- the label is what gets corrected.
    cents = int(correction) + 100 * g.get(G["coarseTune"], 0) + g.get(G["fineTune"], 0)
    moved = 0
    hz = loop_hz(x, (ls, le) if looped else None, store) if bank == 0 and fix_octaves else None
    if hz:
        labelled = 440.0 * 2.0 ** ((root - cents / 100.0 - 69) / 12.0)
        off = 12.0 * np.log2(hz / labelled)
        octaves = int(round(off / 12.0))
        if octaves and abs(octaves) <= 2 and abs(off - 12 * octaves) < 0.6:
            root += 12 * octaves
            moved = octaves
    return {
        "name": name,
        "pcm": np.round(x / peak * 32767).astype(np.int16),
        "rate": store, "lead": lead,
        "loop": [int(ls), int(le)] if looped else None,
        "root": int(root), "cents": cents, "octaves_corrected": moved,
        "scale": g[G["scaleTuning"]] / 100.0,
        "attenuation_db": g.get(G["initialAttenuation"], 0) / 10.0,
        "env": [_seconds(g[G["delayVolEnv"]]), _seconds(g[G["attackVolEnv"]]),
                _seconds(g[G["holdVolEnv"]]), _seconds(g[G["decayVolEnv"]]),
                max(0.0, g[G["sustainVolEnv"]] / 10.0), _seconds(g[G["releaseVolEnv"]])],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sf2")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--store-hz", type=float, default=STORE_HZ,
                    help="the most a sample is kept at (16000; 32000 keeps a rip at the DSP's own rate)")
    ap.add_argument("--ripped", action="store_true",
                    help="the font's samples already came out of games' BRR: do not compress them twice")
    ap.add_argument("--fix-octaves", action="store_true",
                    help="correct samples whose loops play a whole octave from their label")
    args = ap.parse_args()
    sf = SoundFont(args.sf2)
    entries, pcm = {}, []
    at = 0
    wanted = [(f"p{p}", 0, p, 48 if 32 <= p <= 39 or p in (43, 58, 67, 70) else 60) for p in range(128)]
    wanted += [(f"d{k}", 128, 0, k) for k in DRUM_KEYS]
    for label, bank, program, key in wanted:
        e = cut(sf, bank, program, key, args.store_hz, not args.ripped, args.fix_octaves)
        if e is None:
            continue
        data = e.pop("pcm")
        e.update(offset=at, length=len(data))
        entries[label] = e
        pcm.append(data)
        at += len(data)
    allpcm = np.concatenate(pcm)
    entries["_source"] = {"file": os.path.basename(args.sf2), **sf.info,
                          "built": {"store_hz": args.store_hz, "ripped": args.ripped,
                                    "fix_octaves": args.fix_octaves}}
    np.savez_compressed(args.out, pcm=allpcm, index=np.frombuffer(json.dumps(entries).encode(), dtype=np.uint8))
    print("from", entries["_source"])
    fixed = {k: e["octaves_corrected"] for k, e in entries.items()
             if not k.startswith("_") and e.get("octaves_corrected")}
    print(f"{len(fixed)} labelled an octave or two out, corrected:",
          ", ".join(f"{entries[k]['name']} {v:+d}" for k, v in fixed.items()) or "none")
    print(f"{len(entries) - 1} instruments, {allpcm.nbytes / 1e6:.1f} MB of samples, "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB on disk -> {args.out}")


if __name__ == "__main__":
    main()
