"""Isolated full-workflow profiling; never opens a user's corpus database.

Run with the app dependencies, optional PyAV and psutil installed. Every mode
runs in a fresh interpreter and a disposable corpus/output directory. A real
recording can be supplied read-only with --source (at least four seconds).
"""
from __future__ import annotations

import argparse
import cProfile
from collections import defaultdict
from contextlib import ExitStack, chdir
import hashlib
import io
import json
import os
import platform
import pstats
import random
import statistics
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def worker(args):
    import psutil
    import mido
    import numpy as np

    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / "app").rglob("*.py")}
    destination = Path(args.output).resolve()
    source = Path(args.source).resolve() if args.source else None
    with tempfile.TemporaryDirectory(prefix="ytp-workflow-") as directory, chdir(directory):
        work = Path(directory)
        os.environ["MRS_DATA_DIR"] = str(work)
        os.environ["MRS_AUDIO_DECODER"] = "pyav" if args.worker in {"pyav", "pyav-video"} else "ffmpeg"
        os.environ["MRS_VIDEO_DECODER"] = "pyav" if args.worker == "pyav-video" else "ffmpeg"
        from app import database as db, generate as g, media_process, media_decode
        from app.ytpmv import pitch, samples, recommend, render

        corpus_dir = work / "corpora" / "profile"
        (corpus_dir / "downloads").mkdir(parents=True)
        if source is None:
            source = corpus_dir / "downloads" / "source.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                "testsrc2=s=480x270:r=25:d=4", "-f", "lavfi", "-i",
                "aevalsrc=0.2*sin(2*PI*150*t)+0.1*sin(2*PI*300*t):s=44100:d=4",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)],
                check=True, capture_output=True)
        else:
            copied = corpus_dir / "downloads" / "source.mp4"
            shutil.copyfile(source, copied)
            source = copied
        entry = {"slug": "profile", "name": "Profile", "dir": str(corpus_dir),
                 "db": str(corpus_dir / "corpus.db")}
        db._active = entry
        db.init_db()
        with db.get_db() as conn:
            conn.execute("INSERT INTO sources (id, video_id, source_file) VALUES (1,'profile','downloads/source.mp4')")
            for i, word in enumerate(("ah", "oh", "me", "no", "all", "you", "more", "go")):
                start = .2 + i * .4
                conn.execute("INSERT INTO word_clips (id, source_id, word, start_time, end_time, source_file, edited, phones) "
                    "VALUES (?,1,?,?,?,?,1,?)", (i+1, word, start, start+.3, 'downloads/source.mp4',
                                               json.dumps([["AH", start+.03, start+.25]])))

        midi = mido.MidiFile()
        for track_index in range(args.tracks):
            channel = track_index % 15
            if channel >= 9:
                channel += 1
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.Message("program_change", program=(track_index*16)%128, channel=channel))
            for i in range(args.notes):
                note = 48 + (track_index % 8)*5 + i % 9
                track.append(mido.Message("note_on", note=note, velocity=90, time=0, channel=channel))
                track.append(mido.Message("note_off", note=note, velocity=0, time=60, channel=channel))
        buffer = io.BytesIO()
        midi.save(file=buffer)
        midi_id = render.save_midi(buffer.getvalue(), "Profile dense MIDI.mid")
        song = render.load_song(midi_id)
        parts = [{"id": p.id, "text": ("ah", "oh", "me", "no")[i % 4], "corpus": "profile"}
                 for i, p in enumerate(song.parts)]
        params = {"midi_id": midi_id, "parts": parts,
                  "options": {"balance": True, "max_seconds": args.seconds, "flip": False,
                              **({"chip":args.chip,"synth":True} if args.chip else {})}}

        totals = defaultdict(lambda: [0, 0.])
        lock = threading.Lock()
        def instrument(name, function):
            def wrapped(*a, **kw):
                begin = time.perf_counter()
                try:
                    return function(*a, **kw)
                finally:
                    with lock:
                        totals[name][0] += 1
                        totals[name][1] += time.perf_counter() - begin
            return wrapped

        real_build = samples.build
        def legacy_build(*a, **kw):
            kw["audio_only"] = False
            return real_build(*a, **kw)

        process = psutil.Process()
        measurements = []
        with ExitStack() as stack:
            if args.worker == "legacy":
                stack.enter_context(patch.object(samples, "build", side_effect=legacy_build))
            for module, attr, name in ((pitch, "decode_audio", "audio_decode"),
                    (samples, "decode_audio", "audio_decode"), (recommend, "decode_audio", "audio_decode"),
                    (g, "_build_video", "sample_video_encode"), (pitch.Voice, "render", "note_synthesis"),
                    (pitch, "track_f0", "pitch_tracking"), (recommend, "track_f0", "pitch_tracking"),
                    (pitch, "_decode_audio_ffmpeg", "cli_audio"),
                    (samples.Sample, "_frames_ffmpeg", "cli_video"),
                    (media_decode, "_audio_pyav", "native_audio"),
                    (media_decode, "_video_pyav", "native_video"),
                    (render.mastering, "block_power", "loudness_blocks"),
                    (render.mastering, "finish", "master_finish"),
                    (render.tones, "apply", "chip_processing"),
                    (samples.Sample, "frames", "frame_decode"), (render, "_play", "play_and_mix"),
                    (media_process, "run", "media_subprocess"), (subprocess, "Popen", "process_spawn")):
                stack.enter_context(patch.object(module, attr, instrument(name, getattr(module, attr))))

            def measure(name, function, cold):
                if cold:
                    samples.clear_cache()
                    recommend._mem.clear()
                    for path in (work / "output").glob("ytpmv_sample_*.mp4"):
                        path.unlink()
                    for path in work.rglob("*measure*.json"):
                        path.unlink()
                    # The recommender's persistent cache is explicitly cold.
                    cache = Path(recommend._cache_path("profile"))
                    cache.unlink(missing_ok=True)
                totals.clear()
                peak = {"rss_bytes": 0, "threads": 0, "processes": 0}
                stop = threading.Event()
                def monitor():
                    while not stop.is_set():
                        procs = [process] + process.children(recursive=True)
                        rss = threads = alive = 0
                        for proc in procs:
                            try:
                                rss += proc.memory_info().rss
                                threads += proc.num_threads()
                                alive += 1
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        peak["rss_bytes"] = max(peak["rss_bytes"], rss)
                        peak["threads"] = max(peak["threads"], threads)
                        peak["processes"] = max(peak["processes"], alive)
                        stop.wait(.01)
                thread = threading.Thread(target=monitor, daemon=True)
                if args.resources:
                    thread.start()
                begin = time.perf_counter()
                profiler = cProfile.Profile() if args.profile else None
                try:
                    if profiler:
                        profiler.enable()
                    result = function()
                finally:
                    if profiler:
                        profiler.disable()
                    elapsed = time.perf_counter() - begin
                    stop.set()
                    if args.resources:
                        thread.join()
                if profiler:
                    text = io.StringIO()
                    pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(35)
                    destination.with_name(destination.stem + "-" + name +
                        ("-cold" if cold else "-warm") + ".profile.txt").write_text(text.getvalue(), encoding="utf-8")
                digest = None
                output = result.get("audio_url") or result.get("video_url")
                if output:
                    path = work / output.lstrip("/")
                    if path.suffix == ".wav":
                        digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    else:
                        # Compare decoded audiovisual output, independent of
                        # container creation dates and encoder metadata.
                        media = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                            "-map", "0:v", "-map", "0:a", "-f", "framemd5", "-"],
                            capture_output=True, check=True).stdout
                        digest = hashlib.sha256(media).hexdigest()
                else:
                    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
                row = {"workload": name, "cache": "cold" if cold else "warm", "seconds": elapsed,
                       "stages": {k: {"calls": n, "seconds_inclusive": t} for k,(n,t) in totals.items()},
                       "sampled_peak_process_tree": peak if args.resources else None, "output_sha256": digest}
                measurements.append(row)
                print(f"{args.worker} {name} {row['cache']}: {elapsed:.3f}s", flush=True)

            tasks = [("recommendation", lambda: recommend.recommend(song, pool=samples.corpus_pool())),
                     ("part_audition", lambda: render.preview_part(midi_id, parts[0], seconds=args.seconds,
                            chip=args.chip, synth=bool(args.chip))),
                     ("keyboard_bank", lambda: render.preview_mix({**params, "part_id": parts[0]["id"],
                          "keyboard_notes": list(range(60,73))})),
                     ("dense_mix", lambda: render.preview_mix({**params, "seconds": args.seconds})),
                     ("video_render", lambda: render.render_ytpmv(params))]
            for name, function in tasks:
                if args.workload and name not in args.workload:
                    continue
                measure(name, function, True)
                measure(name, function, False)
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value
               for name, value in source_hashes.items()):
            raise RuntimeError("Application source changed during profiling; rerun against a fixed checkout")
        destination.write_text(json.dumps({"mode": args.worker, "notes": args.tracks*args.notes,
            "tracks":args.tracks, "seconds":args.seconds, "chip":args.chip, "source": str(source),
            "python": sys.version, "platform": platform.platform(), "app_source_sha256": source_hashes,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "measurements": measurements}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", choices=("legacy", "lazy", "pyav", "pyav-video"))
    parser.add_argument("--workload", action="append", choices=("recommendation", "part_audition",
        "keyboard_bank", "dense_mix", "video_render"))
    parser.add_argument("--profile", action="store_true", help="Extra cProfile reports; affects timing")
    parser.add_argument("--resources", action="store_true", help="Sample native RSS/threads; affects timing")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tracks", type=int, default=4)
    parser.add_argument("--notes", type=int, default=48, help="Notes per track")
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--chip", choices=("md", "nes", "opl3", "snes", "sid", "gb"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    modes = [(repeat, mode) for repeat in range(max(1, args.repeats))
             for mode in ("legacy", "lazy", "pyav", "pyav-video")]
    random.Random(0).shuffle(modes)
    for repeat, mode in modes:
        path = output.with_name(output.stem + f"-{repeat}-" + mode + ".json")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", mode, "--output", str(path)]
        cmd += ["--tracks",str(args.tracks),"--notes",str(args.notes),"--seconds",str(args.seconds)]
        if args.chip:
            cmd += ["--chip",args.chip]
        if args.source:
            cmd += ["--source", str(Path(args.source).resolve())]
        for workload in args.workload or []:
            cmd += ["--workload", workload]
        if args.profile:
            cmd += ["--profile"]
        if args.resources:
            cmd += ["--resources"]
        subprocess.run(cmd, check=True)
        report = json.loads(path.read_text(encoding="utf-8"))
        report["repeat"] = repeat
        reports.append(report)
    reference = next(r["measurements"] for r in reports if r["mode"] == "legacy")
    if any(r["app_source_sha256"] != reports[0]["app_source_sha256"] for r in reports):
        raise RuntimeError("Application source differs across modes; rerun against a fixed checkout")
    for report in reports:
        for expected, actual in zip(reference, report["measurements"]):
            actual["matches_legacy_output"] = actual["output_sha256"] == expected["output_sha256"]
    medians = {}
    for mode in ("legacy", "lazy", "pyav", "pyav-video"):
        for expected in reference:
            key = expected["workload"] + "-" + expected["cache"]
            times = [m["seconds"] for r in reports if r["mode"] == mode for m in r["measurements"]
                     if m["workload"] == expected["workload"] and m["cache"] == expected["cache"]]
            medians.setdefault(mode, {})[key] = statistics.median(times)
    output.write_text(json.dumps({"reports": reports, "median_seconds": medians,
        "limitations": ["Modes shuffled across repetitions; one cold/warm pair per fresh worker.",
                        "RSS/threads and cProfile are opt-in separate runs; both perturb timing.",
                        "RSS/threads sampled at 10ms when enabled, includes child processes and may miss short peaks.",
                        "Inclusive stages overlap and must not be summed.",
                        "OS file cache is not flushed; cold means app sample/measurement caches."]}, indent=2), encoding="utf-8")
    if any(not m["matches_legacy_output"] for report in reports for m in report["measurements"]):
        raise SystemExit("Workflow output parity failed; inspect the saved report")


if __name__ == "__main__":
    main()
