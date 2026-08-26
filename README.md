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

The database and the source videos are one unit: the database stores clip
timings that point at files in `downloads/`, so neither is much use alone. The
Michael Rosen corpus is about 110 MB — 48 videos, ~9,500 word clips, ~1,400
distinct words.

It is **not in git**. Those are binary blobs that never delta-compress, so
committing them would put the whole lot in every clone's history permanently and
grow it with each re-ingest. They are also derived data, not source. So a corpus
travels as a bundle: one tarball holding the database, the videos and the
transcripts.

## Getting the existing Michael Rosen corpus

It is already here. `michael_rosen.db` and `downloads/` are committed, so a
clone is a working install and `docker compose up` needs nothing else — compose
mounts them read-only at `/corpus` and the entrypoint copies them into the
volume the first time it finds no database.

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

Clip paths are stored relative to the project root with forward slashes.
`os.path.join` on Windows writes `downloads\clip.mp4`, which on Linux is not a
directory and a file but a single filename containing a backslash — every lookup
misses and nothing generates. `app/database.py` normalises on read and on write,
and rewrites any legacy separators once at startup. A bundle packed on Windows
therefore unpacks and runs on Linux unchanged.

---

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/generate` | `{"text": "..."}` → video URL, plus which words were found, spliced or missing |
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
app/database.py               SQLite schema and portable path handling
scripts/ingest.py             download, transcribe, store word clips
scripts/ingest_channel.py     the same, for a whole channel
scripts/realign.py            better word timestamps via stable-ts
scripts/refine_boundaries.py  frame-accurate boundaries via CTC alignment
scripts/correct.py            relabel misheard words from a real transcript
scripts/find_noises.py        pull non-verbal noises out of the gaps
scripts/corpus.py             pack / unpack / inspect a corpus bundle
```

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
