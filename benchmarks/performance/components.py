"""Measure pure request components; no production corpus/database writes."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, chdir
import hashlib
import json
import random
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-dictionary", action="store_true")
    parser.add_argument("--compare-ref", help="Interleave current functions with functions from this git revision")
    args = parser.parse_args()
    from app import phonemes, database as db
    from app.ytpmv import render, samples
    from app.ytpmv.music import Song, Part, Note

    results = []
    def reference(module,name):
        if not args.compare_ref:
            return None
        from benchmarks.performance.reference import original_function
        return original_function(module,name,args.compare_ref)[0]

    def measure(name, call, original=None):
        functions = {"current":call, **({"reference":original} if original else {})}
        elapsed = {name:[] for name in functions}
        hashes = {}
        for repeat in range(args.repeats):
            order = list(functions)
            random.Random(repeat).shuffle(order)
            for mode in order:
                begin = time.perf_counter()
                value = functions[mode]()
                elapsed[mode].append(time.perf_counter() - begin)
                raw = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
                digest = hashlib.sha256(raw.encode()).hexdigest()
                if mode in hashes:
                    assert hashes[mode] == digest, f"{name} is not deterministic"
                hashes[mode] = digest
        timings = elapsed["current"]
        raw = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        result = {"name": name, "seconds": timings, "median_seconds": statistics.median(timings),
                  "output_sha256": hashlib.sha256(raw.encode()).hexdigest(), "output_json_bytes":len(raw)}
        results.append(result)
        if original:
            result["reference_seconds"] = elapsed["reference"]
            result["reference_median_seconds"] = statistics.median(elapsed["reference"])
            result["reference_output_sha256"] = hashes["reference"]
            result["output_sha256"] = hashes["current"]
            result["exact_output_match"] = hashes["reference"] == hashes["current"]
            assert result["exact_output_match"], f"{name} changed output"
        print(f"{name}: median {result['median_seconds']:.4f}s; {len(raw)} JSON bytes", flush=True)

    if not args.skip_dictionary:
        # Uses the existing installed dictionary and its original load policy.
        # A missing installation is not a representative benchmark environment.
        measure("pronunciation_dictionary", phonemes._load_cmudict, reference(phonemes,"_load_cmudict"))
    parts = [Part(id=f"part-{i}", name=f"Track {i}", track=i, channel=i%16, program=i%128,
                  role=("lead", "chords", "rhythm")[i%3],
                  notes=[Note(j*.1,.08,48+(i+j)%36,90) for j in range(1000)]) for i in range(100)]
    song = Song(parts, 100, 120, (4,4), "C", 1)
    with patch.object(render, "load_song", return_value=song), \
            patch.object(render.recommend, "recommend", return_value={}), \
            patch.object(samples, "corpus_pool", return_value={}), \
            patch.object(samples, "noises", return_value={}), \
            patch.object(samples, "units", return_value={}), \
            patch.object(render, "active", return_value={"slug":"profile", "name":"Profile"}):
        old_analyse = reference(render,"analyse")
        measure("midi_metadata_100_tracks_100000_notes", lambda: render.analyse("profile"),
                (lambda: old_analyse("profile")) if old_analyse else None)
        old_timeline = reference(render,"timeline_notes")
        measure("timeline_late_64_tracks_100000_notes", lambda: render.timeline_notes(
            "profile", 95, 98, [p.id for p in parts[:64]]), (lambda: old_timeline(
            "profile",95,98,[p.id for p in parts[:64]])) if old_timeline else None)

    old_play = reference(render,"_play")
    chip_parts = [Part(str(i),str(i),i,i,i*16,
        [Note(j/16,.0625,48+i*5+j%9,90) for j in range(48)],role="rhythm") for i in range(4)]
    settings = []
    for part in chip_parts:
        settings.append({**render._default_settings_for(part,{}), "tone":"opl3", "text":"",
                         "sustain":"cut", "program":part.program})
    def play_chip(play):
        mix = np.zeros((3*44100,2),np.float32)
        hits = []
        for part,setting in zip(chip_parts,settings):
            got = play(part,setting,[],mix,0,3,1,0,False,jitter=False,vary="off")
            hits.append([(hit.start,hit.dur,hit.take,hit.flip) for hit in got])
        return {"audio_sha256":hashlib.sha256(mix.tobytes()).hexdigest(),"hits":hits}
    measure("opl3_repeated_notes_4_tracks",lambda:play_chip(render._play),
            (lambda:play_chip(old_play)) if old_play else None)

    # A request with many parts currently validates each voice by rescanning
    # all installed corpus directories. These are disposable empty fixtures.
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        for i in range(10):
            folder = root/f"voice-{i}"
            folder.mkdir()
            (folder/"corpus.db").touch()
        stack.enter_context(patch.object(db,"CORPORA_DIR",str(root)))
        stack.enter_context(patch.object(db,"LEGACY_DB",str(root/"missing.db")))
        stack.enter_context(patch.object(db,"_active",{"slug":"voice-0", "dir":str(root/"voice-0")}))
        payload = [{"id":str(i),"corpus":f"voice-{i%10}"} for i in range(300)]
        old_prepare = reference(render,"prepare_parts")
        measure("validate_300_parts_10_corpora", lambda: render.prepare_parts(payload),
                (lambda: old_prepare(payload)) if old_prepare else None)
    document = {"results":results, "reference_revision":args.compare_ref, "limitations":[
        "Component fixture isolates metadata and validation; recommendation/media work is excluded.",
        "Dictionary first call includes imports, later calls reuse NLTK imports and OS caches.",
        "Timing excludes output serialization/hash calculation; JSON byte count is recorded separately."],
        "app_source_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (ROOT/"app").rglob("*.py")}}
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(document,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
