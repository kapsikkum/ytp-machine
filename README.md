# YTP Machine

Type a sentence, get a video of it being said, cut word by word from recorded
speech. The shipped corpus is Michael Rosen reading his own poetry. Nothing in
the code is specific to him — point the ingest scripts at any speaker.

Words the corpus has are spliced in directly. Words it doesn't are built from
phonemes taken out of other words, so `rice` can come from the `r` of one word
and the `ice` of another.

## TLDR: get it running

From a clone:

```bash
docker compose up
```

Open <http://localhost:8765> and type a sentence. That is the whole thing — the
Michael Rosen corpus is committed, so a fresh clone already works.

That pulls the published image. Add `--build` only when you have changed the
code, which saves compiling 3 GB that already exists on GHCR.

The image is on GHCR if you would rather not use compose:

```bash
docker pull ghcr.io/kapsikkum/ytp-machine:latest
```

It carries **no corpus** — one has to be mounted, because a voice is 100 MB to
2 GB of video and does not belong inside an image. So point it at one:

```bash
docker run -p 8765:8765 -v ytp-data:/app/data -v ./corpora/michael-rosen:/corpus:ro ghcr.io/kapsikkum/ytp-machine:latest
```

Any of three sources works: a corpus directory or a `.tar.zst` bundle mounted at
`/corpus`, bundles dropped in `/packs`, or `CORPUS_URL` pointing at one over
HTTP. With none of them it says exactly that and stops, rather than serving an
app that answers everything with "no clips found".

With no clone at all, `CORPUS_URL` is the way in — the **corpus** workflow packs
a corpus and attaches it to a release, which is the only place a 100 MB+ bundle
can sit at a stable public URL:

```bash
docker run -p 8765:8765 -v ytp-data:/app/data -e CORPUS_URL=https://github.com/kapsikkum/ytp-machine/releases/download/corpus/michael-rosen.tar.zst ghcr.io/kapsikkum/ytp-machine:latest
```

Run it from the Actions tab, or push a `corpus-*` tag. It packs, unpacks the
result to prove it is whole, and uploads it.

Pin the tag for anything you care about — `:latest` moves under you on the next
push, and every commit is also tagged by its full sha:

```bash
docker pull ghcr.io/kapsikkum/ytp-machine:e4f6b8250ad18f0157acae513c779e83e7d7ca6e
```

No GPU needed to *run* it. You do want one to build a new corpus.

---

## TLDR: train it on a channel

**1. One environment with a GPU**, somewhere your file sync will not touch
(OneDrive locks DLLs mid-install and torch dies with a `WinError 32` naming an
innocent file):

```bash
python -m venv C:/Users/you/.venvs/ytp
```

```bash
C:/Users/you/.venvs/ytp/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
```

```bash
C:/Users/you/.venvs/ytp/Scripts/python.exe -m pip install -r requirements.txt
```

**2. Build it.** One command does download, transcribe, align and pack:

```bash
python scripts/build_corpus.py @SomeChannel --limit 10 --data-dir C:/Users/you/ytp-corpora/some-channel
```

Ten long videos is a good first target — roughly an hour of GPU time and about
5,000 distinct words. `--limit 0` takes the whole channel. Re-running skips what
is already ingested, so a run you interrupt just carries on.

**3. Install it** wherever the server runs:

```bash
python scripts/corpus.py install some-channel.tar.zst --name some-channel
```

Then pick the voice from the dropdown, or `POST /api/corpus`.

**4. Optional, but it is what makes it sound good.** The build leaves a
`pronunciations.csv` in the data directory listing every word it cannot
pronounce, commented out. Uncomment a line and say what the word rhymes with:

```csv
wii,wee
nug,nugget
usb,=letters
```

Then `POST /api/reload`. To rebuild that list later, point it at the corpus:

```bash
MRS_DATA_DIR=C:/Users/you/ytp-corpora/some-channel MRS_CORPUS=some-channel python scripts/verify_corpus.py --unspliceable --write-template
```

---

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

## Markup in the input

Two things are not plain words:

| typed | does |
| --- | --- |
| `~word~` | plays that word backwards |
| `*noise*` | a non-verbal clip (`*spew*`, `*click*`) rather than the spoken word |

Reversal happens at cut time, not in the corpus. Reversed speech is not speech:
Whisper transcribes it into confident nonsense, so a corpus built from a
backwards recording is a set of noises labelled with words nobody said, and it
poisons every later splice. Reversing on the way out costs nothing when unused
and works for any word in any corpus.

A run is one clip, so it is reversed or not as a whole — marking one word ends
the run there rather than quietly reversing its neighbours.

## YTPMV: sing a MIDI file

<http://localhost:8765/ytpmv.html>

Drop in a MIDI file and the corpus plays it: one tile per instrument, the
clip re-triggered on every note, and every note **pitch perfect**. The clip's
wobbling speech pitch is measured frame by frame and laid back down exactly on
the note's frequency (TD-PSOLA, `app/ytpmv/pitch.py`), which keeps the voice's
character while putting it in tune — solo notes measure within a few cents.

You don't need to know any music theory. The file is read for you:

- **key**, tempo, time signature and length, described in plain words
- **what each part is**: tune, backing, chords, bass, or a piece of the drum
  kit (channel 10 is split into kick, snare, hats, toms and cymbals)
- **a word for each part**, picked for the kind of sound it makes: a plosive
  for a kick, a hiss for a hi-hat, a long steady vowel for a tune. Every take
  of the candidate words is measured, so the take picked is the one that holds
  a pitch best.
- **an octave for each part**, so a bass line at E1 is sung where the voice
  can sing it. The notes keep their names, so it is still the same song.

Change any of it: type a word or phrase, cycle through takes, click an
alternative, press ▶ part to hear a part on its own, move an octave, mute a
part, choose `tape` for the classic chipmunk sound instead of `pitch perfect`.
Render goes through the same queue as sentences.

From a terminal:

```bash
python scripts/ytpmv.py song.mid --info
python scripts/ytpmv.py song.mid --part lead=yeah --part kick=boom --max 30
python scripts/ytpmv.py song.mid --only bass --check     # measure the tuning
```

`YTPMV_MAX_SECONDS` caps the length of a render (default 300).

### On a console

`chip=` to the bot, `--chip` from the terminal, or the dropdown plays the song
on an emulated console -- sound and picture both:

```bash
python scripts/ytpmv.py song.mid --chip md          # four-operator FM
python scripts/ytpmv.py song.mid --chip snes        # him, through a SNES
```

| setting | sound | picture |
| --- | --- | --- |
| `md` | the YM2612's FM operators; the voice is gone | 320x224, 9-bit colour |
| `md-voice` | him, out through the Mega Drive's 8-bit DAC | 320x224, 9-bit colour |
| `nes` | the 2A03's pulses, triangle and noise; the voice is gone | 256x240, the 64-colour palette |
| `nes-voice` | him, out through the NES's delta-modulation channel | 256x240, the 64-colour palette |
| `sms` | the SN76489's three squares and noise; the voice is gone | 256x192, 6-bit colour |
| `sms-voice` | him, hammered out of the Master System's volume register | 256x192, 6-bit colour |
| `snes` | him, through the S-DSP's BRR, interpolation and echo | 256x224, 15-bit colour |

The picture is not a separate choice. A Mega Drive soundtrack over a picture
the Mega Drive could never have drawn is two machines, and nobody asking for
one meant that.

The synthesised settings take every part, drums included, and keep only how
long each note lasts and how loud it was. The `-voice` settings keep the whole
trick -- the clip still pitch-tracked onto the exact note -- and play it out
through the channel that console used for speech. The SNES has no other mode:
it synthesises nothing, so it is always him.

A single part can be set with the bot's `tone:<part>=`, to `clean`, `dac`,
`dpcm`, `psg-pcm`, `brr`, or any patch a machine has.

**`app/ytpmv/ym2612.py`** -- the Mega Drive. A model rather than a cycle-exact
emulator like Nuked-OPN2: the log-domain sine and 14-bit operators, the real
envelope generator, all eight algorithms, feedback, detune, 53267 samples a
second, and the output stage, which matters because FM lands a sideband on DC
and without the console's coupling capacitor a song rumbles. Its bass is Sonic
2's own voice `$00`, read out of the disassembly. SMPS numbers operators
backwards from the hardware, so its `op1` is operator 4; get that wrong and
operators 2 and 3 swap, which in a chain is a bright FM bass or a near-pure
sine.

**`app/ytpmv/nes.py`** -- the NES, checked against Blargg's Nes_Snd_Emu. It
generates at four times the output rate and filters on the way down, because
the noise register clocks at up to 447 kHz. The triangle runs out of timer
resolution an octave before the pulses and goes flat at the top of its range;
that is the hardware.

**`app/ytpmv/sms.py`** -- the Master System, checked against SMSPower's notes:
register `$0FE` is 440.4 Hz. It cannot go below 109 Hz, so anything written
lower is raised by octaves. Its documented noise taps are not a maximal-length
polynomial and repeat after 57337 steps, not 65535.

**`app/ytpmv/snes.py`** -- the SNES, checked against fullsnes: the four BRR
predictors and the 512-entry interpolation table, whose taps sum to 2048 at
every phase. BRR works in fifteen-bit integers; fed floats between -1 and 1 it
quantises everything to nothing. Most of the famous muffle is not the chip but
the 64 KB the whole soundtrack had to fit in, which kept samples far below the
DSP's rate -- `stored_hz` models that practice, and says so.

The octave a part is moved by to suit a speaking voice is not applied when a
chip synthesises it, since a chip has no trouble with 49 Hz.

### Balance

Every part is measured and moved to where its job says it should sit, rather
than trusting a table that cannot tell a whisper from a brass patch. On by
default; `--no-balance`, `balance=off`, or the checkbox.

The measurement is the broadcast one (ITU-R BS.1770), and the part that earns
its keep is the gating: a cymbal might be four hits in three minutes, and
averaged across the song it looks nearly silent. Anything balancing on plain
RMS would crank it until those four hits took the roof off. Gated, the same
cymbal played forty times and four times measures within about a decibel.

It nudges rather than levels. Parts inside one role are not meant to be
equally loud -- several decibels between two things both called "rhythm" is
usually somebody's arrangement -- so anything within `DEADBAND` of its target
is left alone and anything past that is brought back by `STRENGTH` of the
excess. Flattening them outright was tried, measured tidier, and sounded
worse. The targets in `app/ytpmv/master.py` are themselves measured, not
guessed: render songs the old way, see what each role came to, take the
median.

### The palettes

`app/ytpmv/screen.py`. Each machine's resolution, and its colours and no
others.

The Mega Drive's eight levels per channel are **not** evenly spaced: measured
off hardware they run 0, 52, 87, 116, 144, 172, 206, 255. The NES has sixty-four
colours burnt in that it could neither mix nor change; there is no single
correct palette for it, since the chip emits composite video and the television
decodes it, so this uses FCEUX's. The Master System is two bits a channel,
taken as even quarters rather than measured. The SNES is five bits a channel on
a straight ramp. Console renders are encoded harder than normal, because at the
usual setting the codec smears the palette the filter just restricted.

## Matrix bot

A bot account that makes videos when asked in a Matrix room:

```
!ytp say nice chocolate cake
!ytp mv                          the last MIDI file posted in the room
!ytp mv info                     what the song is and which words it would use
!ytp mv lead=yeah kick=boom max=60 octave:bass=+1 mode:lead=tape
!ytp voices
!ytp help
```

Post a `.mid` file, then `!ytp mv` (or reply to the file with it). In a direct
message the prefix is optional and plain text is said.

It runs as a second compose service against the same image, talking to the
web app over HTTP — so it uses the same queue, and a bot that falls over cannot
take the site with it. Put its settings in a `.env` beside `docker-compose.yml`:

```bash
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_USER=@ytp:example.org
MATRIX_PASSWORD=...                 # used once; the token is saved to the volume
MATRIX_ALLOWED_USERS=*:example.org  # or MATRIX_ALLOWED_ROOMS=!abc:example.org
MATRIX_ADMINS=@you:example.org      # who may switch the voice for everyone
YTP_PUBLIC_URL=https://ytp.example.org   # optional: link to videos too big to upload
```

```bash
docker compose --profile matrix up -d
docker compose run --rm bot --check-config
```

Invite the bot to a room and it joins. Without either allow-list it answers
anyone who invites it (and logs a warning saying so). Each person gets one
request at a time and a cooldown (`MATRIX_COOLDOWN`, 20 s); song videos are
capped at `MATRIX_MV_MAX_SECONDS` (120). Encrypted rooms are not supported —
the bot leaves them rather than sit somewhere it cannot read.

## Fixing a corpus by hand

<http://localhost:8765/editor.html>

The checker below finds words that are *labelled* wrong. This finds the other
half: words whose **timings** are wrong, which is most of what makes a
generation sound bad. An aligner puts an edge where it stops being confident,
not where the sound starts and stops — that is why "mrs rosen" lost its N and
why "time" once took its T from the word after it.

Pick a source, click a word. You get the video seeked to it, a waveform of the
three seconds around it with its edges drawn on, and the neighbouring words'
edges in grey. Click the waveform to move the start, shift-click to move the
end, or nudge by 10ms and 50ms. Space plays just that clip, so you can hear the
edit before saving it.

Words are flagged when their shape has caused trouble before — very short, very
long, or overlapping the next word. On one ASMR source, 335 of 1,011 clips
overlap their neighbour.

You can also relabel a word Whisper misheard, delete a clip that is unusable,
or add one it skipped entirely. Every edit drops the generator's cache, so the
next sentence uses the new timings. Nothing touches the videos — only the
corpus database, which is what travels in the bundle.

## Checking a corpus

Whisper's mistakes are invisible from inside. A word it mishears is stored
confidently under the wrong spelling, so the corpus looks complete and simply
never produces that word — "rupees" was transcribed as "rubies", the most
quotable word in that corpus, unreachable by typing it.

```bash
python scripts/verify_corpus.py --all
python scripts/verify_corpus.py --source-id 3 --show 40
```

This fetches YouTube's own captions and matches them to the stored labels by
timestamp, reporting **missed** words (caption text with no clip near it, so
speech the transcription skipped) and **disagreements** (both systems heard
something at the same moment and spelled it differently).

Captions are not ground truth. Uploader-written subtitles would be, but most
channels do not publish any; what is available is Google's speech recognition,
with its own failures. The value is that it is a *different* system from
Whisper — the two agreeing means something, and the two disagreeing is worth a
person deciding. Nothing is edited: to act on a finding, put a transcript you
trust in `transcripts/<video_id>.txt` and run `correct.py`.

Repeated disagreements matter far more than one-offs. A word both systems
disagree about every time it is said is a systematic mishearing; a single one is
usually the two of them splitting a phrase differently.

## Teaching it words

Any channel invents words a dictionary has never held — names, in-jokes, brand
names, coinages — and a word with no pronunciation cannot be spliced at all.
There are two places to teach it one:

```
$MRS_DATA_DIR/pronunciations.csv     every corpus on this machine
corpora/<name>/pronunciations.csv    just that one, and packs with its bundle
```

Use the global file. Most of what needs teaching is not specific to a speaker —
"usb", "wii" and "kilometres" are the same words whoever says them — and a
per-corpus copy would have to be maintained again for every channel ingested.
The per-corpus file is for a speaker's own coinages; it overrides the global one
and travels inside the bundle, so a corpus arrives able to say its own words on
a machine that has never heard of them.

Two columns, and the second can be written whichever way suits:

```csv
nug,N AH G          # ARPAbet, if you know it
wii,wee             # or just a word that already sounds right
mcnug,mick nug      # several words are fine
usb,=letters        # read it out letter by letter
hevexum,=skip       # leave it unsayable on purpose
```

The second form is the one to reach for. Nobody should have to learn ARPAbet to
say that "wii" rhymes with "wee".

To find out what is worth adding:

```bash
python scripts/verify_corpus.py --unspliceable --write-template
```

That lists every stored word with no pronunciation, worst first, and writes them
into the global CSV commented out, ready to fill in (`--per-corpus` to keep them
with one corpus instead). It needs no network, and edits take effect on the next
`POST /api/reload`.

Entries here beat everything built in, and a line that cannot be parsed is
reported rather than ignored — silence would leave the word unsayable with
nothing to explain why.

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
| `POST /api/ytpmv/midi` | upload a MIDI file (multipart `midi`) → key, tempo, parts, suggested sounds |
| `POST /api/ytpmv/sample` | `{"text","take"}` → the clip as an instrument, and the note it is on |
| `POST /api/ytpmv/preview` | `{"midi_id","part"}` → a few seconds of one part, audio only |
| `POST /api/ytpmv/render` | `{"midi_id","parts","options"}` → `202` with a job id, polled like the rest |

## Tests

```bash
python tests/test_tokenize.py        # tokeniser: expansions, markers, boundaries
python tests/test_splice_modes.py    # pronunciations, substitutions, cost order
python tests/test_corpus_select.py   # which corpus a run writes into
python tests/test_bundle.py          # a bundle survives a round trip
python tests/test_pitch.py           # YTPMV: a wobbling vowel lands on the note
python tests/test_music.py           # YTPMV: tempo, key and parts from a MIDI file
python tests/test_matrix_commands.py # what the Matrix bot hears in a message
```

These run in CI and need nothing but `num2words`, `nltk`, `zstandard`,
`numpy` and `mido`.

```bash
python tests/test_end_to_end.py      # ~30s: one real video, start to finish
```

That one is not in CI — it downloads a 63-second video, transcribes it, refines
the boundaries and generates sentences from it, so it needs the network, ffmpeg,
torch and a Whisper model. Run it before shipping anything that touches ingest,
clip selection or the ffmpeg builder. It is the only test that exercises those
together, and the bugs that have escaped so far escaped *between* the parts
rather than inside any one of them.

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
app/editor.py                 the corpus editor's API (the only writer
                              outside an ingest)
frontend/editor.html          fix words and timings by hand, with a waveform
app/ytpmv/                    MIDI in, pitch-perfect grid video out
scripts/ytpmv.py              the same from a terminal, with a tuning check
app/matrix_bot.py             the Matrix bot (python -m app.matrix_bot)
frontend/ytpmv.html           the YTPMV page
scripts/verify_corpus.py      check stored labels against YouTube captions
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
