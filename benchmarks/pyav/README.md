# PyAV decoding benchmark

This branch includes optional application audio/video decoders, lazy audio-only
sample preparation and a full-workflow profiler. Both decoders still default to
the FFmpeg CLI. PyAV is installed separately with `requirements-pyav.txt`; the
production dependency list and image remain unchanged. See
[integration status](INTEGRATION.md) for switches, compatibility limits and gates.

## Initial measurements

Each figure below is the median of the per-case speedup ratios. Each case uses
nine timed calls per implementation, after an initial call used for output
comparison. These are **uncached decoding measurements, not whole-render speedups**.

| Work | Windows local | data-server isolated Linux |
| --- | ---: | ---: |
| Audio decoding, 24 cases | 3.48x | 4.82x |
| Real-recording audio subset, 12 cases | 3.89x | 4.70x |
| Video-frame decoding, 9 cases | 3.91x | 4.07x |
| Byte-identical initial outputs | 33/33 | 33/33 |

Raw results, including individual timings, array sizes, checksums, versions,
first-call measurements and comparison errors:

- [Windows](results/windows.json)
- [Linux on data-server](results/linux.json)

The Linux run used a disposable container from the deployed immutable image,
limited to two CPUs, 2 GiB memory and 256 PIDs. Production also has a two-CPU
quota, but has a larger memory allowance. Corpus videos were mounted read-only;
the benchmark did not use or change any corpus database. The temporary container
was removed when the run ended. Both reports have the same benchmark source
hash, app decoding source hashes, and hashes for the two real input recordings.

## What was measured

The baseline calls the current application's `pitch.decode_audio()` and
`Sample.frames()`, including their FFmpeg process startup and pipe transfers.
The alternatives open media in-process with PyAV 19.0.1:

- Audio: seek, trim decoded source samples, downmix and resample to mono 44.1 kHz.
  Both float PCM and the application's distinct 16-bit PCM conversion path run.
- Video: decode, RGB conversion, scale, center pad/crop, sample-aspect correction,
  25 fps conversion and creation of the final NumPy frame array.

Cases cover synthetic AAC and PCM audio; landscape/portrait H.264 videos at
25/30 fps; source seeking; odd output sizes; Fit/Fill; retro display pixels;
and read-only Michael Rosen and Busy B ASMR source recordings. Real inputs are
identified by hashes in the reports and are not included in this branch.

Initial output arrays are compared for sample/frame count, numerical error,
finiteness, output size and byte identity. Acceptance gates allow at most
1e-5 audio MAE and 2/32768 peak error, or 1 video MAE and 8 peak channel error;
the recorded final runs were stricter in practice: every comparison had zero
error and identical bytes. Timing repetitions are not separate correctness
comparisons. Errors are retained as failed cases, and the command exits nonzero
if any case fails.

Imports, fixture encoding, cache clearing, output comparisons and reporting are
outside the timing regions. Audio container close and video graph-cycle cleanup
are included. PyAV video uses two decoder threads and two filter graph threads.
The first-call figures are not cold filesystem benchmarks: OS caching is
uncontrolled. Implementations are called in a deterministically shuffled order.

The existing decoded-frame cache is retained as a separate baseline: hits were
about 0.002 ms on both platforms. PyAV decoding cannot improve an existing cache
hit by the factors above. Output buffer size is recorded; peak native/child RSS
and end-to-end memory use were not measured.

## Important findings

The first audio prototype cropped after resampling and used an audio-specific
seek. It was fast but failed real clip-boundary comparisons. The final version
uses container-default seeking, rounds source sample boundaries and trims before
resampling. It then matched all tested audio samples exactly. This ordering is
important for phonemes and other very short clips.

An initial Linux run also hit the benchmark container's PID/thread limit during
repeated video decoding. Bounding PyAV's worker counts and explicitly collecting
filter graph cycles before returning fixed the repeated run; that cleanup is
included in the final timings. Long-running native-resource behavior still
needs a load test before integrating a backend.

PyAV wheels and the CLI have different FFmpeg library builds: Windows CLI 8.0.1,
Linux CLI 7.1.5, and the PyAV wheel's libavcodec 63/libavformat 63 libraries.
The reports capture all versions. The results compare available implementations;
they do not isolate subprocess startup as the sole cause of the speedup.

## Reproduce locally

Use a Python environment with the application's normal dependencies and FFmpeg
on PATH, then add the optional benchmark dependency:

```sh
python -m pip install -r benchmarks/pyav/requirements.txt
python benchmarks/pyav/benchmark.py --repeats 9 --output output/pyav-benchmark/results.json
```

This makes synthetic fixtures under `output/pyav-benchmark/`. To include real
recordings, add `--source /path/to/video.mp4` for each one. Real source cases
read windows at 5.123, 13.017 and 21.12 seconds; use files at least 23 seconds
long. Files are read only. `--case real-` selects just the real-input cases.

For a disposable Linux container, mount the benchmark directory at
`/bench/benchmarks/pyav`, a writable scratch directory at `/bench/output`, and
individual source files read-only under `/inputs`. Use the application's image
with `/bin/sh` as entrypoint and run `run-container.sh` from this directory,
passing these arguments:

```sh
--work-dir /bench/output/fixtures --output /bench/output/results.json --repeats 9 \
--baseline-commit 225df9dd8b5ca474905324d71cd043d10a87d3d6 \
--environment-label isolated-2cpu-2gib \
--source /inputs/first.mp4 --source /inputs/second.mp4
```

This installs PyAV into the disposable container's `/tmp`, leaving the image
and the application container unchanged. Use `--cpus=2 --memory=2g --pids-limit=256`
to reproduce the limits of the recorded server run.

## Original integration gates

The measurements justify trying an optional audio decoder first, followed by
sample-frame decoding. Before enabling either in the app, measure complete cold
and warm recommendations/previews/renders, verify cancellation and malformed-file
behavior, load-test native threads/RSS, retain an FFmpeg fallback, and preserve
the existing cache/context isolation. Pitch synthesis, chip emulation, composition
and final encoding have not been sped up by this experiment.

PyAV's upstream API references are in its [source documentation](https://github.com/PyAV-Org/PyAV/tree/main/docs).

## Integration workflow measurements

Median seconds from three randomized repetitions per mode, with resource
sampling disabled. Four tracks, 192 MIDI notes, three-second preview/video
and a 13-note bank. Eager CLI uses the current bounded worker settings.
All 120 workflow outputs per platform matched the eager reference:
recommendation JSON, WAV bytes, and decoded audiovisual frame hashes.

| Platform / cold workload | Eager CLI | Lazy CLI | PyAV audio | PyAV audio + video |
| --- | ---: | ---: | ---: | ---: |
| Windows / recommendation | 5.531 | 4.226 | 3.961 | 3.707 |
| Windows / part audition | 0.900 | 0.706 | 0.653 | 0.667 |
| Windows / keyboard bank | 0.583 | 0.380 | 0.322 | 0.322 |
| Windows / dense mix | 1.171 | 0.480 | 0.213 | 0.204 |
| Windows / video render | 2.512 | 2.382 | 2.514 | 2.031 |
| Linux (2 CPUs) / recommendation | 4.279 | 3.013 | 2.241 | 2.112 |
| Linux (2 CPUs) / part audition | 0.897 | 0.656 | 0.572 | 0.526 |
| Linux (2 CPUs) / keyboard bank | 0.668 | 0.455 | 0.373 | 0.317 |
| Linux (2 CPUs) / dense mix | 1.646 | 0.543 | 0.168 | 0.166 |
| Linux (2 CPUs) / video render | 3.753 | 3.771 | 3.705 | 3.816 |

Warm dense mixes stayed close to the existing cache-hit baseline. Full
video rendering did not improve consistently across platforms; its PyAV
backend remains opt-in. These fixture results are not universal speedups
for large songs or all corpora. CPU quotas constrain scheduling, not host
load; individual timings and outliers are retained in the raw reports.

- [Windows workflow results](results/workflow-windows.json)
- [Linux workflow results](results/workflow-linux.json)
- [Backend switches and remaining rollout gates](INTEGRATION.md)
