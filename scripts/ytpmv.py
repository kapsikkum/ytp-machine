#!/usr/bin/env python3
"""Play a MIDI file with the corpus, without the server.

    python scripts/ytpmv.py song.mid
    python scripts/ytpmv.py song.mid --info
    python scripts/ytpmv.py song.mid --part lead=yeah --part kick=boom --max 30
    python scripts/ytpmv.py song.mid --only bass --check

A part is named by its id (t3c2), its role (lead, bass, kick, hats...) or any
word of its name. --part sets its sound (word or phrase), and the same
KEY=VALUE shape sets anything else a part has:

    --part lead=yeah  --octave lead=+1  --mode bass=tape  --take kick=2

--check re-measures the finished audio note by note and reports how far off
the notes it landed, in cents. It is only meaningful with --only on a single
pitched part, since a mix of parts has no one pitch to measure.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def _find_part(song, name: str):
    name = name.lower()
    for p in song.parts:
        if p.id.lower() == name:
            return p
    for p in song.parts:
        if p.role.lower() == name or p.role.split(":")[-1] == name:
            return p
    for p in song.parts:
        if name in p.name.lower():
            return p
    raise SystemExit(f"no part called {name!r}; parts are: "
                     + ", ".join(f"{p.id} ({p.name})" for p in song.parts))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("midi")
    ap.add_argument("--out", help="where to write the mp4 (default: output/)")
    ap.add_argument("--info", action="store_true", help="describe the song and stop")
    ap.add_argument("--max", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--transpose", type=int, default=0)
    ap.add_argument("--labels", action="store_true", help="write each part's word on its tile")
    ap.add_argument("--vary", choices=("off", "rotate", "random", "ultra"), default=None,
                    help="play more than one take of a part's word, in turn or in no order; "
                         "ultra plays a different word every hit, still on the note "
                         "(default off, the same take every hit)")
    ap.add_argument("--jitter", action=argparse.BooleanOptionalAction, default=None,
                    help="a hair of level on every hit, and of tuning on drums (default off)")
    ap.add_argument("--chip", nargs="?", const="md", default=None,
                    choices=("md", "md-voice", "nes", "nes-voice",
                             "sms", "sms-voice", "snes", "gb", "gb-voice", "gbc", "gbc-voice"),
                    help="play the whole song on an emulated sound chip. The plain "
                         "names synthesise it and the voice is gone; the -voice ones "
                         "keep him, played off that machine's own sample channel. "
                         "The picture goes through the same console")
    ap.add_argument("--balance", action=argparse.BooleanOptionalAction, default=None,
                    help="measure every part and move it to where its job wants it "
                         "(on by default; --no-balance for the old fixed levels)")
    ap.add_argument("--only", action="append", default=[], help="play just these parts")
    ap.add_argument("--part", action="append", default=[], metavar="PART=WORDS")
    for k in ("take", "octave", "mode", "volume", "pan", "sustain", "tone"):
        ap.add_argument(f"--{k}", action="append", default=[], metavar=f"PART={k.upper()}")
    ap.add_argument("--check", action="store_true", help="measure the result's tuning")
    args = ap.parse_args()

    from app.ytpmv import render

    with open(args.midi, "rb") as fh:
        mid = render.save_midi(fh.read(), os.path.basename(args.midi))
    info = render.analyse(mid)
    song = render.load_song(mid)

    print(info["description"])
    print(f"key {info['key']} (confidence {info['key_confidence']}), "
          f"{info['bpm']} BPM, {info['time_signature']}")
    for p in info["parts"]:
        r = p["recommended"]
        rng = f"{p['range'][0]}-{p['range'][1]}" if p["range"] else "kit"
        pitch = (r.get("pitch") or {}).get("midi")
        print(f"  {p['id']:14} {p['name']:30} {p['notes']:5} notes  {rng:8}  "
              f"-> {r.get('text')!s:10} take {r.get('take')}  octave {r.get('octave', 0):+d}  "
              f"{r.get('mode')}" + (f"  (sample at {pitch})" if pitch else ""))
    if args.info:
        return 0

    parts: dict[str, dict] = {}

    def setting(spec: str, key: str, conv=str):
        if "=" not in spec:
            raise SystemExit(f"expected PART=VALUE, got {spec!r}")
        name, value = spec.split("=", 1)
        parts.setdefault(_find_part(song, name).id, {})[key] = conv(value)

    for spec in args.part:
        setting(spec, "text")
    for spec in args.take:
        setting(spec, "take", int)
    for spec in args.octave:
        setting(spec, "octave", int)
    for spec in args.mode:
        setting(spec, "mode")
    for spec in args.volume:
        setting(spec, "volume", float)
    for spec in args.pan:
        setting(spec, "pan", float)
    for spec in args.sustain:
        setting(spec, "sustain")
    for spec in args.tone:
        setting(spec, "tone")
    if args.only:
        keep = {_find_part(song, n).id for n in args.only}
        for p in song.parts:
            if p.id not in keep:
                parts.setdefault(p.id, {}).update(mute=True, visible=False)

    options = {"speed": args.speed, "transpose": args.transpose, "labels": args.labels,
               "chip": args.chip}
    if args.balance is not None:
        options["balance"] = args.balance
    if args.vary:
        options["vary"] = args.vary
    if args.jitter is not None:
        options["jitter"] = args.jitter
    if args.max:
        options["max_seconds"] = args.max

    last = [None]

    def progress(stage, done, total):
        if stage != last[0]:
            print(f"\n{stage}", end="", flush=True)
            last[0] = stage
        print(f"\r{stage} {done}/{total}   ", end="", flush=True)

    result = render.render_ytpmv({"midi_id": mid, "parts": [dict(id=k, **v) for k, v in parts.items()],
                                  "options": options}, progress=progress)
    print()
    path = os.path.join("output", os.path.basename(result["video_url"]))
    if args.out:
        shutil.copyfile(path, args.out)
        path = args.out
    for pr in result["problems"]:
        print(f"  ! {pr['name']}: {pr['error']}")
    print(f"{path}  {result['duration']}s  {result['size'][0]}x{result['size'][1]}")

    if args.check:
        _check(path, song, result, options)
    return 0


def _check(path, song, result, options) -> None:
    """Measure every sounding note of the render against the note it should be."""
    import numpy as np
    from app.ytpmv import pitch as p

    heard = [r for r in result["parts"] if not r["mute"] and r["text"]]
    if len(heard) != 1:
        print("--check needs exactly one audible part (use --only)")
        return
    r = heard[0]
    part = song.part(r["id"])
    if part.is_drums:
        print("--check measures pitched parts; drums have no note to be on")
        return
    x = p.decode_audio(path)
    speed = options.get("speed", 1.0)
    errs = []
    notes = sorted(part.notes, key=lambda n: n.start)
    for i, n in enumerate(notes):
        a, b = n.start / speed, (n.start + n.dur) / speed
        if b - a < 0.06 or b * p.SR > len(x):
            continue
        # A note under a held one, or a chord, has more than one pitch to find.
        if any(o is not n and o.start < n.start + n.dur and n.start < o.start + o.dur
               for o in notes[max(0, i - 8):i + 8]):
            continue
        seg = x[int((a + 0.01) * p.SR):int((b - 0.005) * p.SR)]
        tr = p.track_f0(seg)
        f = tr.f0[tr.voiced]
        if len(f) < 3:
            continue
        want = p.midi_to_hz(n.pitch + 12 * r["octave"] + r["transpose"] + options.get("transpose", 0))
        errs.append(float(np.median(1200 * np.log2(f / want))))
    if not errs:
        print("no voiced notes long enough to measure")
        return
    e = np.abs(np.array(errs))
    # Octave slips by the tracker are not tuning errors; report them apart.
    octave = np.abs(e - 1200) < 60
    tuned = e[~octave]
    print(f"tuning over {len(errs)} solo notes: median {np.median(tuned):.1f} cents off, "
          f"90% within {np.percentile(tuned, 90):.1f} cents"
          + (f", {int(octave.sum())} measured an octave out (tracker, not tuning)" if octave.any() else ""))


if __name__ == "__main__":
    sys.exit(main())
