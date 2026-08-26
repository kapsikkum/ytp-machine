# YTP Machine

Type a sentence, get a video of it being said — cut word by word from a corpus
of Michael Rosen's recorded poetry performances.

Words the corpus has are spliced in directly. Words it doesn't are built from
phonemes taken out of other words, which is where most of the work is: finding
where inside a clip a sound actually begins and ends, so `rice` can be made
from the `r` of one word and the `ice` of another without the seam being
audible.

## How it works

**Ingest** (`scripts/ingest.py`) downloads a video, transcribes it with Whisper,
then refines the word timings with stable-ts. **Alignment**
(`app/forced_align.py`) runs torchaudio's CTC forced aligner, Wav2Vec2 BASE
960h, over each clip to get character-level timings — that is what makes
sub-word cutting possible. **Generation** (`app/generate.py`) resolves a
sentence into clips, preferring the longest real spoken runs it can find so
that a phrase which was actually said comes out sounding like one, and falls
back to phoneme splicing per word. `ffmpeg` does the cutting and concatenation.

Feedback matters: a down-vote on a splice is recorded in `splice_ratings` and
the splicer avoids those clips for that word next time unless it has nothing
else.

## Running it

```bash
docker compose up --build
```

Then open <http://localhost:8765>.

The image needs no GPU. Torch comes from the CPU wheel index — the default one
pulls the CUDA build, which is gigabytes of driver payload a headless server
will never use.

Without Docker:

```bash
pip install -r requirements.txt
python -m uvicorn main:app --port 8765
```

`ffmpeg` must be on `PATH` either way.

## The corpus

The database and the source videos are one unit: the database stores clip
timings that point at files in `downloads/`, so neither is much use alone. It
is roughly 110 MB — 48 videos and about 9,500 word clips covering 1,400 distinct
words.

It is **not in git**. Those are binary blobs that never delta-compress, so
committing them would put the whole lot in every clone's history permanently
and grow it with each re-ingest. They are also not source: they are derived
from YouTube by the ingest scripts.

So the corpus travels as a bundle instead:

```bash
python scripts/corpus.py pack            # -> corpus-YYYY-MM-DD.tar.zst
python scripts/corpus.py info  corpus-*.tar.zst
python scripts/corpus.py unpack corpus-*.tar.zst
```

`unpack` also takes a URL, so a bundle attached to a release is a working
install. `pack` checkpoints the SQLite WAL first — tarring a live database with
an unmerged WAL can capture a torn state missing recent writes.

On a fresh container the entrypoint seeds an empty volume automatically, from a
bundle mounted at `/corpus` or from `CORPUS_URL`. If neither is available it
says so and stops, rather than starting an app that would answer every request
with "no clips found".

Generated videos in `output/` are deliberately left out of bundles — about a
gigabyte, all of it reproducible in seconds.

### Paths are stored portably

Clip paths are stored relative to the project root with forward slashes.
`os.path.join` on Windows writes `downloads\clip.mp4`, which on Linux is not a
directory and a file but a single filename containing a backslash — every
lookup misses and nothing generates. `app/database.py` normalises on read and
on write, and rewrites any legacy separators once at startup.

## Adding videos

The container has the ingest dependencies too, so it can be used against the
same corpus it serves:

```bash
docker compose run --rm app python scripts/ingest.py <youtube-url>
docker compose exec app curl -X POST localhost:8765/api/reload
```

`/api/reload` invalidates the in-memory clip cache so new material is picked up
without a restart.

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
main.py              FastAPI app; serves the API, the frontend and output/
app/generate.py      sentence -> clips -> ffmpeg
app/phonemes.py      CMU pronunciations, phoneme-level splice planning
app/forced_align.py  torchaudio CTC alignment for sub-word timings
app/database.py      SQLite schema and portable path handling
scripts/ingest.py    download, transcribe, refine timings
scripts/corpus.py    pack / unpack / inspect a corpus bundle
```

## Source material

Clips come from publicly posted recordings of Michael Rosen performing his own
children's poetry. The vocabulary is what you would expect from that: everyday
words like *chocolate*, *nice*, *terrible*, *bear*. Nothing in the corpus is
adult or offensive, and no attempt is made to synthesise words it has never
heard — a word with no clips and no viable phoneme splice is reported missing
rather than approximated.

It is a toy for making silly videos out of poetry readings. Please keep whatever
you generate with it in that spirit, and don't use it to put words in anyone's
mouth in a way that misrepresents them.
