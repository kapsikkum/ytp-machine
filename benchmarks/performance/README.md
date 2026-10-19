# Performance audit: preserve the result

This audit follows the optional [PyAV integration](../pyav/INTEGRATION.md).
The additional application changes target repeated work without changing the
UI, recommendations, random choices, note timing, DSP, media dimensions or
encoding settings. The comparison revision is `c40b2c7` on this branch.

## Measured changes

Median seconds from five interleaved reference/current repetitions:

| Component fixture | Windows before → after | Linux before → after |
| --- | ---: | ---: |
| Build installed pronunciation dictionary | 1.617 → 0.682 | 2.085 → 0.855 |
| MIDI metadata, 100 tracks / 100,000 notes | 0.486 → 0.361 | 0.542 → 0.437 |
| Repeating OPL3 notes, four tracks | 3.273 → 0.615 | 2.642 → 0.520 |
| Validate 300 parts / ten installed corpora | 0.106 → 0.002 | 0.041 → 0.002 |

All component output hashes match the reference on both platforms. The
unchanged late timeline query is included as a control in the raw reports;
its small timing difference is noise, not an optimization.

These are component fixtures, not whole-request or production latency claims.
The metadata fixture excludes recommendation/media work. Dictionary loading
includes imports on its first invocation; later repetitions reuse imports and
OS caches, so its median describes rebuilding the dictionary, not cold process
startup or a normal cached dictionary lookup. Outer result serialization and
hashing are untimed, except the chip fixture's internal audio-buffer checksum.
No statistical confidence interval or peak-memory improvement is claimed.

Linux ran in a disposable container from production image revision `225df9d`,
with two CPUs, 2 GiB RAM and 256 PIDs. The container mounted benchmark code and
disposable fixtures; it did not mount production corpus data. Both production
services remained unchanged. Raw timings, hashes and application source
manifests are in [Windows](results/windows.json) and [Linux](results/linux.json).

The four changes are:

- Normalize each distinct pronunciation phone once while building the final
  dictionary directly from CMU entries. Keep alternative pronunciations,
  insertion order, Unicode digit handling and independent mutable lists.
- Reuse the 128-program chip tables for identical part roles within one MIDI
  analysis request. Return independent lists for every part.
- Snapshot installed corpora once per part-validation request. Keep defaults,
  validation errors and corpus scoping; introduce no cross-request registry cache.
- Reuse exact synthesized notes within one part. The LRU holds at most 8 MiB
  and 256 notes and includes exact duration, peak, frequency, velocity, program,
  percussion key, sample rate and SID model in its key. Hits return copies.
  Sample-dependent effects, empty, silent and nonfinite input bypass it.
  The repeating OPL3 fixture reduces synthesis calls from 192 to 36. Unique-note
  workloads can see little benefit and retain bounded cache/miss overhead.

## Output and regression gates

Seven focused parity tests cover every synthesized tone, exact cache keys,
mutation isolation, signed zero/nonfinite samples, cache limits, dictionary
alternatives, program tables and corpus validation. All 48 Python suites and
nine browser suites passed locally. Optional native decoder tests and the
new parity tests also passed in the isolated Linux environment; native decoder,
parity and cross-corpus tests passed again in the Windows PyAV environment.

The [workflow comparisons](results/workflow-parity.json) record 18 matching
before/after output hashes: ordinary cold/warm recommendation, audition,
13-note keyboard bank, mix and video; a 32-track / 16,384-note mix and video;
and OPL3 mix/video. JSON and WAV hashes compare complete bytes; video compares
the decoded audio/video framemd5 output. This is fixture coverage, not proof
for every possible source or platform. Profiled workflow wall times are not
used for the speedup table because profiling and concurrent machine load
distort them.

## Remaining bottlenecks and priority

| Area | Evidence / present behavior | Next work that preserves output |
| --- | --- | --- |
| Large MIDI browser rendering | Measured 300-track / 153,600-overview-note fixture: 36,328 DOM elements, including 300 part editors although only ten timeline rows are mounted. One run spent 668 ms on response/hydration and recorded a 340 ms long task. | Mount the active part editor on demand and batch overview construction. Preserve control state, keyboard focus, undo, saved settings and every existing note. |
| Video tile composition | In the profiled 32-track warm fixture, scaling within 1,760 tile draws occupied about 1.13 s of a 2.34 s profiled render. | Bounded reuse of scaled frames keyed by every crop/fit/size/source option. Compare decoded frames exactly before enabling it. |
| Retro video processing | In the post-cache chip video profile, screen processing took about 0.46 s of 1.17 s; shrinking accounted for about 0.34 s. Native-size canvases are still passed through the shrink routine. | Verify an identity-size fast path and reuse immutable geometry calculations, with all palettes and unusual dimensions tested. No palette or interpolation change. |
| Loudness and temporary stems | The large mix profile attributes about 0.21 s to loudness blocks and 0.09 s to stem memmap opens out of 0.80 s. Long songs retain full logical stem lengths even though span-aware writes/mixing already exist. | Measure actual I/O and memory pressure on long songs; reuse exact block work or handles. Preserve floating-point order, loudness gates and storage semantics. |
| Shared job queue | Code review: sentence rendering, video and audio previews share one worker. Fast decode still waits behind an active render. | Instrument queue delay separately from execution, then evaluate bounded scheduling with cancellation/resource tests. Do not silently reorder jobs or raise concurrency on a two-CPU host. |
| Startup / pronunciation | Normalization is improved, but NLTK import and first dictionary initialization remain. | Profile fresh process startup independently. Avoid changing fallback/download policy or alternative pronunciations. |
| Metadata transfer | The 100-track fixture still returns about 2.95 MB of JSON; serialization and browser hydration remain. | Measure serialization separately; eliminate duplicate client construction before considering compatible payload changes. |
| Corpus editor / registry | Code review: relevant word/source indexes already exist; source clips avoid loading all phone tables, gap queries use bisect, envelope/spectrum results are bounded and cached. First source decode and per-corpus count queries can still cost time. | Benchmark real large corpora and cache misses before changing queries. Cache invalidation must cover corpus edits, installs and switches. Preserve RMS/STFT and clip boundaries. |
| Sentence / splice rendering | Code review: versioned clip cache and ffprobe cache already exist; dynamic modifiers correctly bypass cached clips. FFmpeg filtering/encoding remains real work. | Measure source extraction, cache misses and encoding separately using representative sentences. No FPS, codec-quality or resolution reductions. |
| Ingest / transcription / alignment | Code review only this pass; model inference and device selection are separate from render latency. No fresh GPU/model profile was collected. | Profile actual ingestion before changing model batching or execution. Precision, beam search and model changes require separate output gates. |
| Optional PyAV | Earlier decoding gains are documented separately; backends still default to FFmpeg. Native hard cancellation and longer concurrent soaks remain rollout gates. | Finish those gates before changing defaults or deploying optional dependencies. |

The browser measurements are [one exploratory run](results/browser.json), not
production network timings. Scroll measurements include two animation frames
and are not pure handler CPU time. Timeline row virtualization already works;
further note simplification would change the visible result and is excluded.

Full-length temporary-stem sizes are logical sizes, not measured physical disk
writes. The audit did not collect a production traffic latency distribution,
all-user concurrency trace, full ingestion profile or exhaustive memory profile.
These remaining issues are findings and follow-up candidates, not fixes claimed
by this change. The three `*.profile.txt` files under `results/` contain the
post-change warm mix, video and chip-video profiles cited above.

## Reproduce

Use the normal application dependencies, FFmpeg, the existing installed NLTK
CMU dictionary and a checkout containing reference revision `c40b2c7`:

```sh
python benchmarks/performance/components.py --compare-ref c40b2c7 --repeats 5 --output output/performance-audit/components.json
python tests/test_performance_parity.py
python benchmarks/pyav/workflow.py --worker lazy --profile --output output/performance-audit/workflows.json
python benchmarks/pyav/workflow.py --worker lazy --profile --tracks 32 --notes 512 --workload dense_mix --workload video_render --output output/performance-audit/large.json
python benchmarks/pyav/workflow.py --worker lazy --profile --chip opl3 --tracks 4 --notes 96 --workload dense_mix --workload video_render --output output/performance-audit/chip.json
```

The reference helper extracts only original function definitions using
`git show`; current dependencies are shared so the comparison isolates these
function changes. For an exported checkout, `MRS_PERF_REFERENCE_DIR` can point
at the old source tree instead. Full workflow hashes require independent
before/after source checkouts, the same dependency/FFmpeg versions, and the same
generated fixture settings. The profiler records application source hashes.

For the browser probe, start `python tests/frontend_fixture.py --port 8891`,
install Playwright with Chromium, then run:

```sh
node -e "require('./benchmarks/performance/browser.js')(require('playwright').chromium).catch(e=>{console.error(e);process.exitCode=1})"
```

All benchmark fixtures are disposable. Do not point them at production databases.
