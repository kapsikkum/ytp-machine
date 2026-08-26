# YTP Machine

Type a sentence, get a video of it being said — cut word by word from a corpus
of recorded speech. The shipped corpus is Michael Rosen performing his own
children's poetry, but nothing in the code is specific to him: point the ingest
scripts at any speaker and you get a machine that talks in their voice.

Words the corpus has are spliced in directly. Words it doesn't are built from
phonemes taken out of other words, which is where most of the work is: finding
where inside a clip a sound actually begins and ends, so `rice` can be made from
the `r` of one word and the `ice` of another without the seam being audible.

## How it works

**Ingest** downloads a video, transcribes it with Whisper, then refines the word
timings with stable-ts. **Alignment** (`app/forced_align.py`) runs torchaudio's
CTC forced aligner, Wav2Vec2 BASE 960h, over each clip to get character-level
timings — that is what makes sub-word cutting possible. **Generation**
(`app/generate.py`) resolves a sentence into clips, preferring the longest real
spoken runs it can find so a phrase that was actually said comes out sounding
like one, and falls back to phoneme splicing per word. `ffmpeg` does the cutting
and concatenation.

Feedback matters: a down-vote on a splice is recorded in `splice_ratings`, and
the splicer avoids those clips for that word next time unless it has nothing
else.

## Running it

```bash
docker compose up --build
```

Then open <http://localhost:8765>. On a fresh volume it needs a corpus — see
below.

No GPU required. Torch comes from the CPU wheel index; the default index pulls
the CUDA build, gigabytes of driver payload a headless server never uses.

Without Docker:

```bash
pip install -r requirements.txt
python -m uvicorn main:app --port 8765
```

`ffmpeg` must be on `PATH` either way.

---

# The corpus

A corpus is one voice: a directory holding one database and the videos that
database indexes.

```
corpora/michael-rosen/corpus.db      timings: this word, this file, 4.12s–4.61s
corpora/michael-rosen/downloads/     the videos those timings point into
corpora/michael-rosen/transcripts/   true text, where there is any
```

The two halves are one unit — the database stores only timings, so on its own
it points at files that are not there, and the videos alone are just videos.
That is why a corpus always moves as a whole. Every corpus has the same shape
and its own `downloads/`, so any number of them sit side by side without
colliding. The Michael Rosen corpus is about 110 MB — 48 videos, ~9,500 word
clips, ~1,400 distinct words.

Paths inside are stored relative to the corpus directory, never absolute, which
is what makes a corpus portable: copy the directory anywhere and the clips
still resolve.

It is **not in git**. Those are binary blobs that never delta-compress, so
committing them would put the whole lot in every clone's history permanently and
grow it with each re-ingest. They are also derived data, not source. So a corpus
travels as a bundle: one tarball holding the database, the videos and the
transcripts.

## Getting the existing Michael Rosen corpus

It is already here. `corpora/michael-rosen/` is committed, so a clone is a
working install and `docker compose up` needs nothing else — compose mounts the
directory read-only at `/corpus` and the entrypoint copies it into the volume
the first time it finds no corpus.

It is committed as loose files rather than one packed bundle for a dull reason:
the bundle is 103 MB and GitHub refuses any single file over 100 MB. As 49 files
none of which exceeds 9 MB, the same bytes go in without complaint, and without
Git LFS and its metered bandwidth.

A bundle is still the right way to move a corpus somewhere else — onto a server,
or into a backup:

```bash
python scripts/corpus.py pack                       # -> corpus-YYYY-MM-DD.tar.zst
python scripts/corpus.py unpack corpus-*.tar.zst
CORPUS_URL=https://example.invalid/corpus.tar.gz docker compose up
```

The entrypoint accepts any of the three: loose files at `/corpus`, a bundle at
`/corpus`, or `CORPUS_URL`. With none of them and no database it prints what is
missing and stops, rather than serving an app that answers every request with
"no clips found".

Generated videos in `output/` stay out of git: about a gigabyte, all
reproducible in seconds.

## Building a new corpus

Any speaker with a decent amount of clear, single-voice footage will work. The
more distinct words, the fewer phoneme splices are needed and the better it
sounds.

**Use a GPU if you have one.** Building a corpus is the only expensive thing
here, and all of the cost is transcription and alignment. Every script takes
`--device auto|cuda|cpu` (or `$MRS_DEVICE`), defaulting to `auto` — CUDA when
there is a usable one, CPU otherwise. On an RTX 3070 Ti, transcription runs at
about 10× realtime against roughly 1× on the CPU, which is the difference
between an afternoon and a week for a large channel.

The catch is that `requirements.txt` installs the **CPU** build of torch on
purpose: the container is headless and a CUDA build drags in gigabytes of
driver payload it would never use. So a GPU ingest wants its own environment:

```bash
python -m venv .venv-cuda
.venv-cuda/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
.venv-cuda/bin/pip install -r requirements.txt
```

On Windows, put that environment somewhere OneDrive does not sync — the sync
engine locks DLLs mid-install and torch fails to load with a `WinError 32`
naming a file that looks entirely innocent.

**1. Ingest.** One video, a local file, or a whole channel:

```bash
python scripts/ingest.py https://www.youtube.com/watch?v=XXXXXXXXXXX
python scripts/ingest.py /path/to/video.mp4 --model medium
python scripts/ingest_channel.py https://www.youtube.com/@SomeChannel/videos --year 2008 --skip-errors
```

This downloads, transcribes with Whisper, and writes one row per word. `--model`
trades speed for accuracy — `small` or `medium` is worth it, since every later
step inherits these labels.

**2. Sharpen the timings.** Whisper's word boundaries run early and wander by
100–350 ms, which makes short clips grab the tail of the previous word:

```bash
python scripts/realign.py --model small                 # stable-ts word timestamps
python scripts/refine_boundaries.py --source-id 3       # CTC alignment, dry run on one
python scripts/refine_boundaries.py --all --apply       # frame-accurate, everything
```

`refine_boundaries` is the one that matters most for splice quality. It needs an
explicit `--all` (or a `--source-id`) to pick targets, and `--apply` to write —
a bare invocation does nothing.

**3. Fix misheard labels** (optional, needs a real transcript). Where Whisper
heard the audio correctly but spelled it wrong, `correct.py` relabels only
high-confidence phonetically-close swaps and never touches timestamps. Put the
true text in `transcripts/<video_id>.txt`, then:

```bash
python scripts/correct.py --all            # dry run over everything with a transcript
python scripts/correct.py --all --apply    # write the changes
```

Both `correct.py` and `refine_boundaries.py` default to a dry run and need
`--apply` to touch the database.

**4. Catch the non-verbal noises** — clicks, pops, spews — which Whisper skips
entirely because they are not words:

```bash
python scripts/find_noises.py              # dry run
python scripts/find_noises.py --apply
```

Note this one is **not generic**: `find_noises.py` carries a hardcoded list of
two Michael Rosen videos, because scanning every source for energetic bursts
turned up mostly breaths and junk, so the build was curated by hand. For another
speaker, edit `CURATED` at the top of `main()` to name the videos worth scanning
and what to label them. Skip the step entirely if you only want words.

**5. Pack it.**

```bash
python scripts/corpus.py pack                  # -> corpus-YYYY-MM-DD.tar.zst
python scripts/corpus.py info corpus-*.tar.zst
```

`pack` checkpoints the SQLite WAL first — tarring a live database with an
unmerged WAL can capture a torn state missing recent writes.

Committing the new videos and the database is how the shipped corpus gets
updated. Bear in mind git keeps every version of a binary forever and cannot be
made to forget one without rewriting history, so add material when it is worth
keeping rather than after every experiment.

If `zstandard` is installed the bundle is `.tar.zst`, otherwise it falls back to
`.tar.gz`. `unpack` reads either.

## Adding to a corpus that is already running

The container carries the ingest dependencies too, so it can work against the
same corpus it serves:

```bash
docker compose run --rm app python scripts/ingest.py <youtube-url>
docker compose run --rm app python scripts/refine_boundaries.py --all --apply
curl -X POST localhost:8765/api/reload
```

`/api/reload` drops the in-memory clip cache so new material is picked up
without a restart.

## Paths are stored portably

Clip paths are stored relative to the corpus directory with forward slashes.
`os.path.join` on Windows writes `downloads\clip.mp4`, which on Linux is not a
directory and a file but a single filename containing a backslash — every lookup
misses and nothing generates. `app/database.py` normalises on read and on write,
and rewrites any legacy separators once at startup. A bundle packed on Windows
therefore unpacks and runs on Linux unchanged.

---

## API

Generation is queued rather than done inside the request. A long sentence takes
minutes, and a request held open that long is killed by any proxy in front of
it — which returned 504 while the work carried on where nobody could collect
it. Submitting returns a job id; poll it for progress.

```bash
id=$(curl -s -X POST localhost:8765/api/generate        -H 'Content-Type: application/json'        -d '{"text":"nice chocolate cake"}' | jq -r .id)
curl -s localhost:8765/api/jobs/$id | jq
```

`?wait=1` keeps the old blocking behaviour, which is easier from a script where
there is no proxy in the way.

One job runs at a time. Generation is ffmpeg-bound and the box is shared, and
two at once is what exhausted the container's thread ceiling before the queue
existed — serialising is the point, not a limit to tune away.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/generate` | `{"text": "..."}` → `202` with a job id. `?wait=1` to block and get the result directly |
| `GET /api/jobs/{id}` | status, stage, progress, and the result once done |
| `GET /api/queue` | how many jobs are queued and running |
| `GET /api/words` | Whole vocabulary with clip counts; the frontend fetches it once for instant checking as you type |
| `GET /api/suggest` | Autocomplete from real spoken runs |
| `POST /api/rate` | Down-vote a splice so it is avoided next time |
| `POST /api/reload` | Drop the clip cache after an ingest or a correction |
| `GET /api/stats` | Corpus totals |

## Layout

```
main.py                       FastAPI app; serves the API, the frontend and output/
app/generate.py               sentence -> clips -> ffmpeg
app/phonemes.py               CMU pronunciations, phoneme-level splice planning
app/forced_align.py           torchaudio CTC alignment for sub-word timings
app/database.py               SQLite schema, corpus selection, portable paths
app/device.py                 CPU or CUDA, chosen once for every model
scripts/ingest.py             download, transcribe, store word clips
scripts/ingest_channel.py     the same, for a whole channel
scripts/realign.py            better word timestamps via stable-ts
scripts/refine_boundaries.py  frame-accurate boundaries via CTC alignment
scripts/correct.py            relabel misheard words from a real transcript
scripts/find_noises.py        pull non-verbal noises out of the gaps
scripts/corpus.py             pack / unpack / migrate / inspect a corpus
corpora/<name>/               one voice: corpus.db + downloads/ + transcripts/
packs/                        drop bundles here to install them on next start
```

## Upgrading an install made before the corpora/ layout

Older installs kept the database loose in the data directory with `downloads/`
beside it. That still works and is still read, but it is the one corpus that
cannot sit next to another, since there is only one `downloads/` to go round.

The container migrates itself on start, so a redeploy is enough. To do it by
hand, or to see what it would do first:

```bash
python scripts/corpus.py migrate            # report only
python scripts/corpus.py migrate --apply    # move it into corpora/michael-rosen/
```

It moves rather than copies, so a 100 MB `downloads/` is never briefly
duplicated, and it is a no-op once done. If you pinned `MRS_CORPUS=default`,
change it to the migrated name — `default` was never really a setting, it was
the name the code gave to the old layout.

## Source material

The shipped corpus comes from publicly posted recordings of Michael Rosen
performing his own children's poetry. The vocabulary is what you would expect
from that: everyday words like *chocolate*, *nice*, *terrible*, *bear*. Nothing
in it is adult or offensive, and no attempt is made to synthesise words it has
never heard — a word with no clips and no viable phoneme splice is reported
missing rather than approximated.

It is a toy for making silly videos out of poetry readings. Keep whatever you
generate in that spirit, and don't use it to put words in anyone's mouth in a
way that misrepresents them.
