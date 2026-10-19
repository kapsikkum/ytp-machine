# Decoder integration

The application now has separately selectable audio and sample-frame backends.
Both default to `ffmpeg`; final video encoding always uses the existing encoder.
Install the optional backend alongside the normal app environment:

```sh
python -m pip install -r requirements-pyav.txt
MRS_AUDIO_DECODER=pyav uvicorn main:app --host 127.0.0.1 --port 8766
# Sample frames can be enabled independently:
MRS_VIDEO_DECODER=pyav uvicorn main:app --host 127.0.0.1 --port 8766
```

Use the deployment's normal launch command with these environment variables;
the examples above do not change its other settings. Removing a variable or
setting it to `ffmpeg` restores the CLI implementation. Missing PyAV and known
unsupported media fall back automatically; cancellation and deadline exceptions
are propagated instead of starting another decode.

## Changes

- Audio-only single-clip samples read the original recording without first
  encoding an MP4. The sample retains its corpus, source version and video
  recipe; a later video request materializes the picture under the same file
  build lock. Phrases/effects retain the existing splice pipeline.
- Audio decoding trims source samples before resampling, preserving short
  consonants/vowels and the existing float/int16 downmix behavior.
- Decoder and filter worker counts are bounded, including the CLI fallback.
  PyAV audio uses an explicit single-worker `abuffer`/`aformat` graph because
  `AudioResampler` internally creates an unbounded filter worker pool.
- Video preserves the existing RGB conversion, aspect-aware Fit/Fill, crop/pad,
  sample aspect and FPS chain. Native filter graph cycles are collected before
  returning, including when cancelled.

## Compatibility boundary

Audio-only compressed containers, unbounded AAC reads, AAC windows touching the
encoded tail, nonzero container start times and missing timestamps retain the
CLI's timing policy. Video fractional seeks, variable frame-rate input,
rotation/display matrices and nonzero origins likewise retain the CLI path.
These are measured compatibility cases, not equivalent native implementations.

Cancellation is checked at entry and between decoded/filter output frames.
The 30-second deadline is cooperative; a blocked native codec call cannot be
forcibly killed as a CLI subprocess can. Open/read I/O timeouts are also set.
Hard cancellation of pathological native decoders remains a production rollout
gate; the backend is optional and off by default.

## Reproduce complete workflows

```sh
python -m pip install -r benchmarks/pyav/requirements.txt
python tests/test_media_decode.py
python benchmarks/pyav/workflow.py --repeats 3 --output output/workflows.json
```

The profiler builds a disposable corpus and a four-part, 192-note MIDI. It
compares eager CLI, lazy CLI, lazy PyAV audio, and lazy PyAV audio+video in
randomized fresh interpreters. Each runs cold/warm recommendation, part audition,
13-note keyboard bank, dense preview mix and full video output. Audio WAV bytes
and decoded audiovisual frame hashes are compared with the eager CLI result.
No existing corpus database is opened; an optional `--source` recording is
copied into the disposable corpus and never modified.
The eager reference uses the current bounded CLI worker settings; it is a
comparison of sample preparation/backend choices, not an untouched production
binary. This fixture does not represent every large MIDI or corpus.

`--resources` samples process-tree RSS/threads, including children. Run this
separately: on Windows process enumeration materially distorts timing.
`--profile` creates cProfile text reports separately; it also perturbs timings
and records the main thread rather than complete worker-thread CPU attribution.
`--worker pyav --workload recommendation` selects one mode/workflow.

Stage times are inclusive and overlap. A cold run clears app sample/measurement
caches, not OS file caches or already imported modules. Recommendation is the
first workload and includes pronunciation dictionary initialization. Its cold
profile showed NLTK/CMU dictionary loading as a major cost; decoder-only speedups
must not be presented as whole-song speedups.

## Rollout gates

1. PCM/frame parity, cancellation and repeated native RSS/thread tests on both
   Windows and a constrained Linux container.
2. Full output parity and repeated complete-workflow timings on both platforms.
3. Long-running/concurrent native resource and hard-cancellation validation,
   then opt-in audio rollout before wider video support.

The main requirement list, Docker image and production services are unchanged
on this benchmark branch. Benchmark/prototype results in README predate these
resource and compatibility guards; use workflow results for integration claims.
