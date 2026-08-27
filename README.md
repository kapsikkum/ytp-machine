# YTP Machine

Type a sentence, get a video of it being said, cut word by word from recorded
speech. The shipped corpus is Michael Rosen reading his own poetry. Nothing in
the code is specific to him — point the ingest scripts at any speaker.

Words the corpus has are spliced in directly. Words it doesn't are built from
phonemes taken out of other words, so `rice` can come from the `r` of one word
and the `ice` of another.

## Run it

```bash
docker compose up --build
```

<http://localhost:8765>. On a fresh volume it needs a corpus — see below.

No GPU needed. Torch comes from the CPU wheel index; the default index pulls
gigabytes of CUDA payload a headless server never touches.

Without Docker:

```bash
pip install -r requirements.txt
python -m uvicorn main:app --port 8765
```

`ffmpeg` must be on `PATH` either way.

## How it works

- **Ingest** downloads a video, transcribes with Whisper, refines timings with
  stable-ts.
- **Alignment** (`app/forced_align.py`) runs torchaudio's CTC aligner (Wav2Vec2
  BASE 960h) for character-level timings. That's what makes sub-word cutting
  work.
- **Generation** (`app/generate.py`) resolves a sentence into clips, preferring
  the longest real spoken runs it can find, and falls back to phoneme splicing
  per word. ffmpeg cuts and concatenates.

Votes on a clip are stored in `splice_ratings` and weight it up or down for
that word next time.

---

# The corpus

One voice: a directory holding one database and the videos it indexes.

```
corpora/michael-rosen/corpus.db      timings: this word, this file, 4.12s-4.61s
corpora/michael-rosen/downloads/     the videos those timings point into
corpora/michael-rosen/transcripts/   true text, where there is any
```

The database stores only timings, so the two halves are useless apart and a
corpus always moves as a whole. Paths inside are relative to the corpus
directory, so copying it anywhere works. Michael Rosen's is ~110 MB: 48 videos,
~9,500 clips, ~1,400 distinct words.

Corpora are **not in git** — binary that never delta-compresses, and derived
data besides. They travel as a bundle: one tarball of database + videos +
transcripts.

## The shipped corpus

Already here. `corpora/michael-rosen/` is committed, so a clone is a working
install; compose mounts it read-only at `/corpus` and the entrypoint copies it
into the volume on first start.

It's committed as loose files rather than a bundle only because the bundle is
103 MB and GitHub rejects files over 100 MB. As 49 files, none over 9 MB, it
goes in fine and avoids Git LFS.

To move one somewhere else:

```bash
python scripts/corpus.py pack                       # -> corpus-YYYY-MM-DD.tar.zst
python scripts/corpus.py unpack corpus-*.tar.zst
CORPUS_URL=https://example.invalid/corpus.tar.gz docker compose up
```

The entrypoint takes any of three: loose files at `/corpus`, a bundle at
`/corpus`, or `CORPUS_URL`. With none and no database it says what's missing
and stops.

`output/` stays out of git — about a gigabyte, all reproducible in seconds.

## Building a new corpus

Any speaker with a decent amount of clear, single-voice footage works. More
distinct words means fewer splices and better output.

One command does the lot — download, transcribe, align, pack:

```bash
python scripts/build_corpus.py @GarbageTime420 --limit 40
python scripts/build_corpus.py https://youtube.com/@Name --name my-corpus --model medium
```

It names the corpus from the channel handle unless you pass `--name`, and
defaults to 40 videos (`--limit 0` for the whole channel). Videos already
ingested are skipped, so an interrupted run resumes by re-running it. Use
`--skip-ingest` / `--skip-refine` / `--skip-pack` to redo part of it, and
`--data-dir` to build somewhere other than the project root.

It checks the dependencies, ffmpeg and the GPU before starting, because the
alternative is finding out four hours in.

**Use a GPU.** All the cost is transcription and alignment. Every script takes
`--device auto|cuda|cpu` (or `$MRS_DEVICE`), default `auto`. On an RTX 3070 Ti
transcription runs ~10x realtime against ~1x on CPU.

`requirements.txt` installs CPU torch on purpose, so a GPU ingest wants its own
environment:

```bash
python -m venv .venv-cuda
.venv-cuda/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
.venv-cuda/bin/pip install -r requirements.txt
```

On Windows keep that venv, and the corpus, out of OneDrive. The sync engine
locks DLLs mid-install and torch then fails with `WinError 32` naming an
innocent-looking file.

### The stages by hand

`build_corpus.py` runs 1, 2 and 5. Steps 3 and 4 need human input and are
optional.

**1. Ingest.** One video, a local file, or a channel:

```bash
python scripts/ingest.py https://www.youtube.com/watch?v=XXXXXXXXXXX
python scripts/ingest.py /path/to/video.mp4 --model medium
python scripts/ingest_channel.py https://www.youtube.com/@SomeChannel/videos --limit 40 --skip-errors
```

`--model` trades speed for accuracy. Every later step inherits these labels, so
`small` or `medium` is worth it. `--normalise` re-encodes to the corpus format
(480x270), which makes bundles far smaller and generation cheaper.

**2. Sharpen the timings.** Whisper's word boundaries run early and wander by
100–350 ms, so short clips grab the tail of the previous word:

```bash
python scripts/realign.py --model small                 # redo timestamps, e.g. after a model change
python scripts/refine_boundaries.py --source-id 3       # CTC alignment, dry run on one
python scripts/refine_boundaries.py --all --apply       # frame-accurate, everything
```

`refine_boundaries` matters most for splice quality. It needs `--all` or a
`--source-id` to pick targets and `--apply` to write.

**3. Fix misheard labels** (optional, needs a real transcript). Relabels only
high-confidence phonetically-close swaps, never timestamps. Put the true text
in `transcripts/<video_id>.txt`:

```bash
python scripts/correct.py --all            # dry run
python scripts/correct.py --all --apply
```

**4. Catch non-verbal noises** — clicks, pops, spews — which Whisper skips:

```bash
python scripts/find_noises.py              # dry run
python scripts/find_noises.py --apply
```

Not generic: it carries a hardcoded list of two Michael Rosen videos, because
scanning everything turned up mostly breaths. For another speaker edit
`CURATED` at the top of `main()`, or skip the step.

**5. Pack.**

```bash
python scripts/corpus.py pack --corpus <name>
python scripts/corpus.py info corpus-*.tar.zst
```

`pack` checkpoints the SQLite WAL first; tarring a live database with an
unmerged WAL captures a torn state. Bundles are `.tar.zst`, or `.tar.gz` if
`zstandard` isn't installed. `unpack` reads either.

## How hard it tries

A corpus can only say what it has heard. When a word isn't there, the splicer
builds it from phonemes taken out of other words — and when even those aren't
there, what happens next is a per-corpus setting, stored in the corpus database
so it travels inside the bundle:

| mode | behaviour |
| --- | --- |
| `strict` | real clips and clean splices only; anything else is reported missing (default) |
| `loose` | substitutes a near-enough phoneme, and guesses a pronunciation for words the dictionary doesn't have |
| `desperate` | as loose, and drops sounds nothing can cover — always produces something |

Set it from the **effort** dropdown next to the voice selector, or:

```bash
curl -X POST localhost:8765/api/splice-mode -H 'Content-Type: application/json' -d '{"mode":"loose"}'
```

The modes are additive: substitutions and dropped phonemes are priced far above
any achievable saving, so wherever `strict` finds a splice, all three modes give
the identical result. Turning it up only changes words that would otherwise come
back missing.

A word that needed a substitution, a dropped sound, or a guessed pronunciation
is marked `approx` in the API and shown with a dashed border, so an approximation
is never passed off as a faithful splice.

Which you want depends on the corpus. With 7,000 distinct words, `strict` rarely
gives up on anything and the other modes barely fire. With 30 words, `strict`
can say almost nothing, and `desperate` is the difference between a corpus that
works and one that only quotes itself.

## Installing a corpus on a running server

Drop a bundle in `packs/` and restart — the entrypoint installs anything it
finds there. Or do it directly:

```bash
python scripts/corpus.py install my-corpus.tar.zst --name my-corpus
curl -X POST localhost:8765/api/reload
```

`MRS_CORPUS` pins which one is served. Without it the first alphabetically
wins, so adding a corpus can silently switch voices.

To add to a corpus that's already running, the container carries the ingest
dependencies too:

```bash
docker compose run --rm app python scripts/ingest.py <youtube-url>
docker compose run --rm app python scripts/refine_boundaries.py --all --apply
curl -X POST localhost:8765/api/reload
```

## API

Generation is queued, not done in the request: a long sentence takes minutes
and any proxy in front will 504 first. Submitting returns a job id; poll it.

```bash
id=$(curl -s -X POST localhost:8765/api/generate -H 'Content-Type: application/json' -d '{"text":"nice chocolate cake"}' | jq -r .id)
curl -s localhost:8765/api/jobs/$id | jq
```

`?wait=1` blocks instead, which is easier from a script.

One job runs at a time. Generation is ffmpeg-bound on a shared box, and two at
once exhausted the container's thread ceiling before the queue existed.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/generate` | `{"text": "..."}` → `202` with a job id. `?wait=1` to block |
| `GET /api/jobs/{id}` | status, stage, progress, result |
| `GET /api/queue` | queued and running counts |
| `GET /api/words` | vocabulary with clip counts; the frontend fetches it once |
| `GET /api/suggest` | autocomplete from real spoken runs |
| `POST /api/rate` | vote a clip up or down for that word |
| `GET`/`POST /api/splice-mode` | read or set how hard the splicer tries |
| `POST /api/reload` | drop the clip cache after an ingest or correction |
| `GET /api/stats` | corpus totals |

## Layout

```
main.py                       FastAPI app; API, frontend, output/
app/generate.py               sentence -> clips -> ffmpeg
app/phonemes.py               CMU pronunciations, phoneme-level splice planning
app/forced_align.py           torchaudio CTC alignment for sub-word timings
app/database.py               SQLite schema, corpus selection, portable paths
app/device.py                 CPU or CUDA, chosen once for every model
scripts/build_corpus.py       channel link -> packed corpus, one command
scripts/ingest.py             download, transcribe, store word clips
scripts/ingest_channel.py     the same, for a whole channel
scripts/realign.py            better word timestamps via stable-ts
scripts/refine_boundaries.py  frame-accurate boundaries via CTC alignment
scripts/correct.py            relabel misheard words from a real transcript
scripts/find_noises.py        pull non-verbal noises out of the gaps
scripts/corpus.py             pack / unpack / migrate / inspect
corpora/<name>/               one voice: corpus.db + downloads/ + transcripts/
packs/                        drop bundles here to install on next start
```

## Upgrading a pre-`corpora/` install

Older installs kept the database loose in the data directory with `downloads/`
beside it. Still read, but it can't sit next to another corpus — only one
`downloads/` to go round.

The container migrates itself on start, so a redeploy is enough. By hand:

```bash
python scripts/corpus.py migrate            # report only
python scripts/corpus.py migrate --apply    # -> corpora/michael-rosen/
```

It moves rather than copies, and is a no-op once done. If you pinned
`MRS_CORPUS=default`, change it to the migrated name.

## Source material

The shipped corpus comes from publicly posted recordings of Michael Rosen
reading his own children's poetry. Nothing in it is adult or offensive. By
default nothing is synthesised either: a word with no clips and no viable splice
is reported missing rather than approximated, and the modes above have to be
turned up deliberately, per corpus.

It's a toy for making silly videos out of poetry readings. Keep it there, and
don't use it to put words in anyone's mouth in a way that misrepresents them.
