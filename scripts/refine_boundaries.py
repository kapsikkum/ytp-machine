#!/usr/bin/env python3
"""
Refine word-clip boundaries with CTC forced alignment.

Whisper/stable-whisper word timestamps are shifted early and vary by 100-350ms,
so short clips grab the previous word's tail.  This re-aligns each source's word
sequence to its audio with wav2vec2 forced alignment (frame-accurate) and
rewrites start_time/end_time in the DB — fixing short-word offset and making
splices land cleanly. The cuts themselves are placed from the audio around
the aligned letters (see _FRAME), and clips the aligner cannot find their
word in -- mostly noises Whisper heard as words -- are marked so they are
rarely picked (see _MIN_CONFIDENCE).

Usage:
    python scripts/refine_boundaries.py --source-id 36          # one source (dry run)
    python scripts/refine_boundaries.py --source-id 36 --apply
    python scripts/refine_boundaries.py --all --apply
    python scripts/refine_boundaries.py --all --apply --start-from-id 1 --end-at-id 16
"""
import argparse
import math
import os
import re
import sqlite3
import subprocess
import tempfile
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db, get_db

_SR = 16000
_model = None
_labels = None
_DICT = None
_device = "cpu"

# How much audio goes through the model at once. Wav2Vec2 emits a frame per
# 20ms and attends over all of them, so cost grows with the square of this
# number: 40s is ~2,000 frames, which is comfortable on a laptop GPU and still
# gives the aligner plenty of context either side of every word.
_WINDOW_SECONDS = 40.0

# Extra audio at each end of a window, so a word sitting on the boundary is not
# asked to align against a sound that has been cut in half.
_WINDOW_PAD = 0.5

# The aligner says which stretch of sound is which word; it does not say where
# the cut between two words should go. CTC fires a letter as a spike, usually
# a frame or two into its sound, so a word "begins" on its first spike -- after
# a plosive's burst, part way into an S -- and "ends" on its last, before the
# vowel or the final consonant has finished. Cutting there clipped every
# onset, and since each clip then ran up to its neighbour's late onset, it
# carried the start of the next word with it. Measured on older-rosen, three
# quarters of the cuts between connected words sat 20-90ms away from the dip
# between them, and half the words after a pause started within 10dB of their
# loudest.
#
# So the cut is placed from the audio, inside the gap the aligner leaves:
# where there is a pause, a word ends where the quiet begins and the next
# starts where it ends; where there is none, both share the quietest point.
_FRAME = 0.01
# How far a word may reach from its aligned letters to find the quiet: the
# tail of a sonorant rings on well past its last spike, an onset rarely leads
# its first by more than a plosive's closure.
_TAIL_REACH = 0.5
_LEAD_REACH = 0.3
# Quiet is the bottom quarter of the range between the source's noise floor
# and its speech: relative, because a room recording and a studio one have
# floors 30dB apart.
_QUIET_SHARE = 0.25
# Dips within this many dB of the quietest count as equally good (see dip).
_DIP_TIE = 3.0

# A word whose letters the model scores below this on average was not heard
# where Whisper put it. Mostly that is a noise transcribed as a word -- clicks
# as "chk chk chk", a breath as "uh" or "a" -- or a word Whisper stretched
# over seconds of something else.
#
# Mostly, not always, so these are marked rather than removed. They were
# removed at first, and checked against michael-rosen's real transcripts a
# third of what went were words that are there: short ones, "i", "he", "it",
# said so quickly or placed so slightly off that the model cannot find their
# letters. Their scores were no different from the noise's, so no threshold
# separates the two. A mark is the editor's own "badly cut" report, which
# makes a take five times less likely per count: a word with better takes
# all but never uses one of these, a word with no other take still has it,
# and they are listed for review among the low-confidence clips.
_MIN_CONFIDENCE = 0.02
_UNHEARD_REPORTS = 3            # 0.2 ** 3: under one pick in a hundred

# A clip whose length changes by more than this has its phoneme times
# measured again rather than shifted (see refine_source).
_PHONES_KEEP = 0.03


def _group_words(words: list[dict], window: float) -> list[list[int]]:
    """Word indices split into windows of at most *window* seconds.

    Whole words only: a word split across two windows would be aligned twice
    against two half-sounds and land worse than the timestamp it started with.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    start = 0.0
    for i, w in enumerate(words):
        if not current:
            current, start = [i], w["start_time"]
        elif w["end_time"] - start > window:
            groups.append(current)
            current, start = [i], w["start_time"]
        else:
            current.append(i)
    if current:
        groups.append(current)
    return groups


def _load_model(device: str | None = None):
    global _model, _labels, _DICT, _device
    if _model is not None:
        return
    import torchaudio
    from app.device import get as get_device, describe

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    _device = get_device(device)
    _model = bundle.get_model().eval().to(_device)
    _labels = bundle.get_labels()
    _DICT = {c: i for i, c in enumerate(_labels)}
    print(f"  aligner on {describe(_device)}")


def _load_wav(path):
    import numpy as np
    import torch
    with wave.open(path, "rb") as w:
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype("float32") / 32768.0
    return torch.from_numpy(d).unsqueeze(0)


class Envelope:
    """Loudness in 10ms frames, and what counts as quiet in this source."""

    def __init__(self, samples):
        import numpy as np
        hop = int(_SR * _FRAME)
        n = len(samples) // hop
        frames = np.asarray(samples[:n * hop], dtype=np.float32).reshape(n, hop)
        self.db = 10.0 * np.log10((frames ** 2).mean(axis=1) + 1e-10)
        if n:
            floor, speech = np.percentile(self.db, [5, 95])
            self.threshold = floor + _QUIET_SHARE * (speech - floor)
        else:
            self.threshold = 0.0

    def frame(self, t: float) -> int:
        return min(max(int(round(t / _FRAME)), 0), len(self.db))

    def quiet(self, i: int) -> bool:
        return 0 <= i < len(self.db) and self.db[i] < self.threshold

    def pause(self, i: int, step: int) -> bool:
        """Quiet for two frames from *i* on, in the direction of *step*.

        One quiet frame is not a pause: the dip between two syllables, or
        inside a stop, can be that short and is still the word.
        """
        return self.quiet(i) and self.quiet(i + step)

    def dip(self, a: float, b: float, early: bool = False) -> float:
        """The quietest point between *a* and *b*, as a frame edge.

        *early* takes the first of near-equal dips, for when *b* is the first
        letter of a word that starts on a vowel, which CTC fires late:
        "stands up" had a dip where the "u" began and another, as deep, just
        before the aligner's U, and cutting at the second left "stands"
        saying "stands u". Elsewhere the quietest wins; taking the first
        everywhere moved a third of all cuts, some into the word before.
        """
        import numpy as np
        i, j = self.frame(a), self.frame(b)
        if j - i < 2 or j > len(self.db):
            return (a + b) / 2
        seg = self.db[i:j]
        k = i + int(np.argmax(seg <= seg.min() + _DIP_TIE) if early else np.argmin(seg))
        return min(max((k + 0.5) * _FRAME, a), b)

    def sound_ends(self, after: float, limit: float) -> float | None:
        """Where the sound running at *after* first falls quiet, before *limit*."""
        for i in range(self.frame(after), self.frame(min(limit, after + _TAIL_REACH))):
            if self.pause(i, 1):
                return i * _FRAME
        return None

    def sound_starts(self, before: float, limit: float) -> float | None:
        """Where the sound running at *before* began, walking back to *limit*."""
        for i in range(self.frame(before) - 1, self.frame(max(limit, before - _LEAD_REACH)) - 1, -1):
            if self.pause(i, -1):
                return (i + 1) * _FRAME
        return None


def _envelope(wav):
    return Envelope(wav[0].numpy())


def _alignment_text(word: str) -> str | None:
    """English target text without guessing a partial alphanumeric word.

    Numbers can occupy one stored clip even when spoken as several words.
    Leading zeroes and mixed letters/digits have several possible readings,
    so retain their original timestamps instead of aligning a guessed prefix.
    """
    match = re.fullmatch(r"([0-9]+)(st|nd|rd|th)?", word.lower())
    if match:
        digits, suffix = match.groups()
        if len(digits) > 1 and digits.startswith("0"):
            return None
        try:
            from num2words import num2words
            spoken = num2words(int(digits), to="ordinal" if suffix else "cardinal")
        except (ImportError, ValueError, OverflowError, NotImplementedError):
            return None
        return "|".join(spoken.upper().replace(",", "").replace("-", " ").split())
    if any(ch.isdigit() for ch in word):
        return None
    return word.upper()


def _ctc_targets(words: list[dict]) -> tuple[list[int], list[tuple[int, int] | None]]:
    """Encode whole words, retaining the model's trained word separator.

    Without the separator, "a nice" and "an ice" are the same target string;
    the aligner has no evidence of which boundary the transcript intended.
    Ranges exclude separators so word bounds come from letters alone.
    """
    chars: list[int] = []
    ranges: list[tuple[int, int] | None] = []
    for word in words:
        text = _alignment_text(word["word"])
        if not text or any(c not in _DICT for c in text):
            ranges.append(None)
            continue
        cw = [_DICT[c] for c in text]
        if chars:
            chars.append(_DICT["|"])
        start = len(chars)
        chars.extend(cw)
        ranges.append((start, len(chars)))
    return chars, ranges


def _align_window(segment, words: list[dict]) -> list[tuple[float, float, float] | None]:
    """Align one audio window; return window-relative word cores.

    Each core is (first letter, last letter, confidence), the confidence being
    the mean probability of the word's letters along the forced path.
    """
    import torch
    import torchaudio
    from torchaudio.functional import merge_tokens

    chars, ranges = _ctc_targets(words)
    empty = [None] * len(words)
    if not chars:
        return empty
    with torch.inference_mode():
        emission, _ = _model(segment)
        # forced_align expects log probabilities, rather than unnormalised
        # logits. Its returned scores then share the model's probability scale.
        logp = emission.log_softmax(dim=-1)
    # Repeated characters need an intervening blank in the CTC path.
    required = len(chars) + sum(a == b for a, b in zip(chars, chars[1:]))
    if emission.size(1) < required:
        return empty

    targets = torch.tensor([chars], dtype=torch.int32, device=emission.device)
    aligned, scores = torchaudio.functional.forced_align(logp, targets, blank=0)
    spans = merge_tokens(aligned[0].cpu(), scores[0].cpu())
    if len(spans) != len(chars):
        return empty
    ratio = segment.size(1) / emission.size(1) / _SR
    foreign = _foreign_letters(logp, spans, ranges)
    cores = [(spans[rng[0]].start * ratio, (end or spans[rng[1] - 1].end) * ratio,
              sum(math.exp(s.score) for s in spans[rng[0]:rng[1]]) / (rng[1] - rng[0]),
              None if after is None else after * ratio,
              None if before is None else before * ratio)
             if rng is not None else None for rng, (after, before, end) in zip(ranges, foreign)]
    if hasattr(segment, "cpu"):
        cores = _split_at_pauses(cores, segment, logp, spans, ranges, ratio)
    return cores


# The aligner can stretch a word across a pause on to speech the transcript
# does not have: "Brian", a breath, and an unwritten "and", all one word. A
# pause inside a word is not enough to say so -- a stop's closure is one --
# so it takes a pause with a letter the word does not have on one side of it.
# The word is then the side its own letters are heard on.
_INNER_PAUSE = 2 * _FRAME


def _split_at_pauses(cores, segment, logp, spans, ranges, ratio):
    """Cores cut down to one side of any long pause inside them (window-relative)."""
    env = Envelope(segment[0].detach().cpu().numpy())
    if logp.dim() == 3:
        logp = logp[0]
    best = logp.max(dim=-1)
    ids = best.indices.cpu().tolist()
    sure = (best.values > math.log(_FOREIGN_PROB)).cpu().tolist()
    need = int(round(_INNER_PAUSE / _FRAME))
    out = []
    for core, rng in zip(cores, ranges):
        if core is None:
            out.append(core)
            continue
        s, e = core[0], core[1]
        pieces, a, run = [], s, 0
        for i in range(env.frame(s), env.frame(e)):
            if env.quiet(i):
                run += 1
                continue
            if run >= need and (i - run) * _FRAME > a:
                pieces.append((a, (i - run) * _FRAME))
                a = i * _FRAME
            run = 0
        pieces.append((a, e))
        if len(pieces) < 2:
            out.append(core)
            continue
        own = {spans[k].token for k in range(*rng)}

        def heard(p, mine=True):
            lo, hi = int(p[0] / ratio), min(int(p[1] / ratio) + 1, len(ids))
            return sum(1 for t in range(lo, hi)
                       if sure[t] and ids[t] not in (0, _DICT["|"], _DICT.get("'")) and (ids[t] in own) == mine
                       and (t == 0 or ids[t - 1] != ids[t]))

        def complete(p, first, last):
            lo, hi = int(p[0] / ratio), min(int(p[1] / ratio) + 1, len(ids))
            got = {ids[t] for t in range(lo, hi) if sure[t]}
            return first in got and last in got

        k = max(range(len(pieces)), key=lambda j: (heard(pieces[j]) - heard(pieces[j], False), -j))
        # Only a whole word is kept on its own: its first and last letters
        # both heard on that side. A misheard letter beside a "b" or "th"
        # closure would otherwise cut "hamburger" down to "burger".
        if (rng[1] - rng[0] < 2 or not complete(pieces[k], spans[rng[0]].token, spans[rng[1] - 1].token)
                or not any(heard(p, False) for j, p in enumerate(pieces) if j != k)):
            out.append(core)
            continue
        a, b = pieces[k]
        after = min((t for t in (core[3], pieces[k + 1][0] if k + 1 < len(pieces) else None)
                     if t is not None and t >= b), default=None)
        before = max((t for t in (core[4], pieces[k - 1][1] if k > 0 else None)
                      if t is not None and t <= a), default=None)
        out.append((a, b, core[2], after, before))
    return out


# Letters the model hears between two words that the transcript does not
# have: an "erm" nobody wrote down, run straight on from "Brian" with no pause.
# The tail used to walk on through it to the pause after, and the clip said
# "Brian, erm". A letter the word itself ends with (or its neighbour starts
# with) is not counted: CTC often fires a word's last letter twice.
#
# The aligner can also stretch a word over it: "Brian"'s N heard at 151.70,
# the erm's M at 151.98, and the forced path putting its one N at 152.00 so
# the erm is inside the word. When the model heard the word's last letter
# and then a letter the word does not have, before the forced last letter,
# the word ended at the first, and what follows is not it.
_FOREIGN_PROB = 0.5
_FOREIGN_GAP = 0.10


def _foreign_letters(logp, spans, ranges) -> list[tuple[int | None, int | None, int | None]]:
    """Per word, the frames where unwritten speech starts after it and ends before it.

    (first frame of it after the word's letters, frame just past it before
    them, where the word really ends if the aligner stretched it over some),
    each None where there is nothing of the kind.
    """
    if logp.dim() == 3:
        logp = logp[0]
    best = logp.max(dim=-1)
    sep = _DICT["|"]
    # The apostrophe is no sound, and clips are stored without one.
    quote = _DICT.get("'", -1)
    letter = ((best.indices != 0) & (best.indices != sep) & (best.indices != quote)
              & (best.values > math.log(_FOREIGN_PROB))).cpu().tolist()
    ids = best.indices.cpu().tolist()
    # (first frame, end frame, first letter, last letter) of each placed word
    words = [(spans[r[0]].start, spans[r[1] - 1].end, spans[r[0]].token, spans[r[1] - 1].token)
             if r is not None else None for r in ranges]
    placed = [w for w in words if w is not None]

    def spikes(a: int, b: int, own: set[int]) -> list[int]:
        """Frames in [a, b) where a letter other than *own* fires."""
        return [t for t in range(max(a, 0), min(b, len(letter)))
                if letter[t] and ids[t] not in own and (t == 0 or ids[t - 1] != ids[t])]

    def stretched(r) -> tuple[int, int] | None:
        """(true end, first foreign frame) if the last letter was dragged over other speech."""
        if r[1] - r[0] < 2:
            return None
        own = {spans[i].token for i in range(r[0], r[1])}
        last = spans[r[1] - 1]
        heard_end = None
        for t in range(spans[r[1] - 2].end, last.end):
            if not letter[t]:
                continue
            if ids[t] == last.token:
                heard_end = t + 1
            elif ids[t] not in own and heard_end is not None:
                return heard_end, t
        return None

    out: list[tuple[int | None, int | None, int | None]] = []
    k = 0
    for w, r in zip(words, ranges):
        if w is None:
            out.append((None, None, None))
            continue
        prv = placed[k - 1] if k > 0 else None
        nxt = placed[k + 1] if k + 1 < len(placed) else None
        k += 1
        cut = stretched(r)
        if cut is not None:
            out.append((cut[1], None, cut[0]))
            continue
        after = spikes(w[1], nxt[0] if nxt else len(letter), {w[3]} | ({nxt[2]} if nxt else set()))
        before = spikes(prv[1] if prv else 0, w[0], {w[2]} | ({prv[3]} if prv else set()))
        # A word's onset is often heard as a letter it is not spelled with --
        # "once" as WANCE -- so before a word it takes two to be another sound.
        out.append((after[0] if after else None,
                    before[-1] + 1 if len(before) >= 2 else None, None))
    return out


def _core(word: dict, aligned) -> tuple[float, float] | None:
    """The aligned letters of an automatic clip, if they are usable."""
    if aligned is None or word.get("edited"):
        return None
    onset, core_end = aligned[0], aligned[1]
    if not all(math.isfinite(t) for t in (onset, core_end)) or core_end <= onset:
        return None
    return onset, core_end


def _boundary_updates(words: list[dict], fa: list[tuple[float, ...] | None],
                      duration: float, envelope: Envelope | None = None,
                      ) -> tuple[list[tuple[float, float, int]], list[float]]:
    """Place each aligned word's cut within the source and its neighbours.

    Hand-edited clips still supply transcript context, but their boundaries
    are final and also constrain the automatic clips either side of them; so
    does a word the aligner could not place, at its stored times.

    Without an *envelope* the aligned letters are padded by rule. With one,
    the cuts are taken from the audio -- see _FRAME.
    """
    cores = [_core(w, a) for w, a in zip(words, fa)]
    placed: list[tuple[float, float] | None] = [None] * len(words)
    for i, (word, core) in enumerate(zip(words, cores)):
        if core is None:
            continue
        onset, core_end = core
        # Where the neighbours' sound is known to be: their letters, or the
        # times they are stored with when nothing better is known.
        prv = nxt = None
        if i > 0:
            prv = cores[i - 1][1] if cores[i - 1] is not None else words[i - 1]["end_time"]
        if i + 1 < len(words):
            nxt = cores[i + 1][0] if cores[i + 1] is not None else words[i + 1]["start_time"]
        shared_prev = i > 0 and cores[i - 1] is not None
        shared_next = i + 1 < len(words) and cores[i + 1] is not None
        # Unwritten speech either side is a wall like a word that cannot move:
        # the cut stops at the dip before it rather than walking through it.
        # The model only marks it where a letter fires, often late in it --
        # the M of "erm", after the "er" -- so the wall is also kept within
        # _FOREIGN_GAP of the word's own letters.
        after, before = (tuple(fa[i][3:5]) + (None, None))[:2]
        if after is not None and after > core_end:
            after = min(after, core_end + _FOREIGN_GAP)
            if nxt is None or after < nxt:
                nxt, shared_next = after, False
        if before is not None and before < onset:
            before = max(before, onset - _FOREIGN_GAP)
            if prv is None or before > prv:
                prv, shared_prev = before, False
        if envelope is None:
            new_start = max(0.0, onset)
            # Preserve the core where there is less than 20ms between words, but
            # never let a minimum tail override the next word's onset or EOF.
            new_end = min(core_end + 0.30, duration)
            if nxt is not None:
                new_end = min(new_end, max(core_end, nxt - 0.02), nxt)
        else:
            # Never inside the letters themselves, whatever the audio says.
            new_start = min(onset, max(0.0, _place_start(
                envelope, onset, prv, shared_prev, _vowel_first(word))))
            new_end = max(core_end, min(duration, _place_end(
                envelope, core_end, nxt, duration, shared_next,
                i + 1 < len(words) and _vowel_first(words[i + 1]))))
        placed[i] = (new_start, new_end)

    # The two sides of a gap are placed separately and can reach past each
    # other where one finds a pause the other is too far away to see.
    for i in range(len(words) - 1):
        a, b = placed[i], placed[i + 1]
        if a is not None and b is not None and a[1] > b[0]:
            cut = max(cores[i][1], b[0])
            placed[i] = (a[0], cut)
            placed[i + 1] = (max(b[0], min(cut, cores[i + 1][0])), b[1])

    updates: list[tuple[float, float, int]] = []
    shifts: list[float] = []
    for word, span in zip(words, placed):
        if span is None or span[1] - span[0] < 0.04:
            continue
        shifts.append(span[0] - word["start_time"])
        updates.append((round(span[0], 4), round(span[1], 4), word["id"]))
    return updates, shifts


def _vowel_first(word: dict) -> bool:
    return word.get("word", "")[:1].lower() in "aeiou"


def _place_start(env: Envelope, onset: float, prv: float | None, shared: bool,
                 vowel: bool = False) -> float:
    """Where a word whose first letter fires at *onset* begins.

    *prv* is where the previous word's sound is last known to be, and
    *shared* says whether that word is being placed by the same rule -- if
    so, with no quiet between them, both cuts land on the same dip.
    """
    limit = 0.0 if prv is None else min(prv, onset)
    quiet = env.sound_starts(onset, limit)
    if quiet is not None:
        return min(quiet, onset)
    if prv is not None and onset - prv < _LEAD_REACH:
        # Run together with the word before. Stored times of a word this one
        # cannot move are a wall, and the cut stays clear of it.
        cut = env.dip(prv, onset, vowel)
        return cut if shared else max(prv, cut)
    # Sound all the way back, and nothing near: a music bed or a noisy room.
    # Back off from the spike by about a plosive's closure.
    return max(limit, onset - 0.03)


def _place_end(env: Envelope, core_end: float, nxt: float | None, duration: float,
               shared: bool, vowel_next: bool = False) -> float:
    """Where a word whose last letter fires at *core_end* ends."""
    limit = duration if nxt is None else max(nxt, core_end)
    quiet = env.sound_ends(core_end, limit)
    if quiet is not None:
        return max(quiet, core_end)
    if nxt is not None and nxt - core_end < _TAIL_REACH:
        cut = env.dip(core_end, nxt, vowel_next)
        return cut if shared else min(nxt, cut)
    return min(limit, core_end + 0.30)


def refine_source(sid: int, apply: bool) -> tuple[int, int, float]:
    """Re-align one source. Returns (updated, total, median_shift_ms)."""
    with get_db() as conn:
        src = conn.execute("SELECT source_file FROM sources WHERE id=?", (sid,)).fetchone()
        # A word the second opinion (second_opinion.py) did not hear is left
        # where it is and out of the alignment: asked to find a word that is
        # not there, the aligner takes it from its neighbours' sound. It still
        # bounds its neighbours, below -- it may well be there after all.
        try:
            disputed = {r[0] for r in conn.execute(
                "SELECT clip_id FROM boundary_reports WHERE kind='disputed'")}
        except sqlite3.OperationalError:     # a corpus from before boundary reports
            disputed = set()
        everything = [dict(r) for r in conn.execute(
            "SELECT id, word, start_time, end_time, edited FROM word_clips "
            "WHERE source_id=? ORDER BY start_time", (sid,)).fetchall()]
        words = [w for w in everything if w["id"] not in disputed]
    if not src or not words:
        return (0, 0, 0.0)

    from app.database import resolve_path
    # Scratch audio goes to the system temp directory, not the working directory.
    # ffmpeg is handed a path and does not care, but the working directory is
    # usually the repo: a killed run left a 50 MB wav sitting in it, syncing to
    # OneDrive and one `git add -A` away from the history. The temp directory is
    # also the right place for something this size -- it is cleaned up by the
    # system if a process dies before its own cleanup runs.
    with tempfile.NamedTemporaryFile(prefix="mrs-refine-", suffix=".wav", delete=False) as scratch:
        tmp = scratch.name
    try:
        decoded = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", resolve_path(src["source_file"]),
             "-ac", "1", "-ar", str(_SR), tmp], capture_output=True, text=True)
        if decoded.returncode != 0:
            raise RuntimeError(f"Could not decode source {sid}: {(decoded.stderr or '').strip()[-500:]}")
        wav = _load_wav(tmp)
        # Aligned a window at a time, not a source at a time.
        #
        # This used to push the entire source through Wav2Vec2 in one forward
        # pass, which worked only because the original corpus was 22-second
        # poetry clips. Self-attention is quadratic in sequence length, and the
        # model emits a frame per 20ms: a 22-second clip is ~1,100 frames and
        # costs nothing, but a 13-minute video is ~39,000, and the attention
        # matrix alone would be over a billion entries. Long-form sources did
        # not align slowly, they died allocating -- on the GPU and on the CPU
        # both. Windowing makes the cost linear in length instead of quadratic.
        #
        # Words are never split across a window, and each window is padded so a
        # word at the edge still has its surrounding audio to align against.
        fa: list[tuple[float, float] | None] = [None] * len(words)
        groups = _group_words(words, _WINDOW_SECONDS)
        total_samples = wav.size(1)

        for group in groups:
            first, last = words[group[0]], words[group[-1]]
            w_start = max(0.0, first["start_time"] - _WINDOW_PAD)
            w_end = last["end_time"] + _WINDOW_PAD
            s0 = max(0, int(w_start * _SR))
            s1 = min(total_samples, int(w_end * _SR))
            if s1 - s0 < _SR // 10:       # under 100ms of audio, nothing to do
                continue
            segment = wav[:, s0:s1].to(_device)

            # Window-relative back to source-relative.
            for idx, aligned in zip(group, _align_window(segment, [words[i] for i in group])):
                if aligned is None:
                    continue
                fa[idx] = (s0 / _SR + aligned[0], s0 / _SR + aligned[1], *aligned[2:3],
                           *(None if t is None else s0 / _SR + t for t in aligned[3:]))

        if not any(f is not None for f in fa):
            return (0, len(words), 0.0)

        # Words the audio does not bear out are marked and kept where they
        # are. A hand-edited clip is not marked: somebody listened to it.
        doubtful = [w for w, f in zip(words, fa) if f is not None and len(f) > 2
                    and f[2] < _MIN_CONFIDENCE and not w.get("edited")]
        for w, f in zip(words, fa):
            if w in doubtful:
                print(f"    unheard: {w['word']!r} at {w['start_time']:.2f}s "
                      f"(clip {w['id']}, confidence {f[2]:.4f})")

        # Neither a doubtful word nor a disputed one moves, but both still
        # bound their neighbours, like a hand-edited clip: often enough they
        # are real -- an "all" said so fast the aligner barely scores it --
        # and a neighbour that ignored them walked over their sound. "right"
        # came out "alright". A doubtful word stands where the aligner put
        # it, in order; a disputed one, which it never saw, at its stored times.
        aligned = {w["id"]: f for w, f in zip(words, fa)}
        doubted = {w["id"] for w in doubtful}
        placing, pfa = [], []
        for w in everything:
            f = aligned.get(w["id"])
            if w["id"] in disputed or w["id"] in doubted:
                wall = dict(w, edited=1)
                if f is not None and w["id"] in doubted:
                    wall.update(start_time=f[0], end_time=f[1])
                placing.append(wall)
                pfa.append(None)
            else:
                placing.append(w)
                pfa.append(f)

        updates, shifts = _boundary_updates(placing, pfa, total_samples / _SR, _envelope(wav))

        if apply and doubtful:
            with get_db() as conn:
                conn.executemany(
                    "INSERT INTO boundary_reports (clip_id, kind, count) VALUES (?, 'unheard', ?) "
                    "ON CONFLICT(clip_id, kind) DO UPDATE SET count = excluded.count",
                    [(w["id"], _UNHEARD_REPORTS) for w in doubtful])

        if apply and updates:
            with get_db() as conn:
                conn.executemany(
                    "UPDATE word_clips SET start_time=?, end_time=? WHERE id=?", updates)

                # This is the other thing that moves a start_time, and aligned
                # phoneme times are measured from it. Run over an already
                # aligned corpus without this and every sub-word cut is out by
                # however far the boundary moved -- silently, because the
                # times are still there and still look reasonable.
                #
                # Shifted only while the clip keeps its length. Times measured
                # in a window that was too short are squeezed into it --
                # "anthology" cut to 0.3s had all eight phonemes in that 0.3s,
                # and once the word ran its full 0.7s the "-gy" a spliced "G"
                # takes from it was the middle of the word. A clip that grew
                # or shrank is measured again by align_phones.py instead.
                from app.phone_align import outside, shift_stored
                before = {w["id"]: w["end_time"] - w["start_time"] for w in words}
                for (new_start, _e, cid), moved in zip(updates, shifts):
                    row = conn.execute(
                        "SELECT phones FROM word_clips WHERE id=?", (cid,)).fetchone()
                    if row and row["phones"]:
                        shifted = shift_stored(row["phones"], -moved)
                        if (outside(shifted, _e - new_start)
                                or abs((_e - new_start) - before.get(cid, _e - new_start)) > _PHONES_KEEP):
                            shifted = None
                        conn.execute(
                            "UPDATE word_clips SET phones=? WHERE id=?",
                            (shifted, cid))
                    # Character times use the same relative origin, but this
                    # table has no shifting helper. Recompute after a change.
                    conn.execute("DELETE FROM clip_alignments WHERE clip_id=?", (cid,))
                    # Tone is measured inside the window, which just moved.
                    conn.execute("DELETE FROM clip_tone WHERE clip_id=?", (cid,))

        import statistics
        med = statistics.median(shifts) * 1000 if shifts else 0.0
        return (len(updates), len(words), med)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def main() -> None:
    global _WINDOW_SECONDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start-from-id", type=int)
    ap.add_argument("--end-at-id", type=int)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--window", type=float, default=_WINDOW_SECONDS, metavar="SEC",
                    help=f"Seconds of audio aligned at once (default "
                         f"{_WINDOW_SECONDS:g}). Lower it if you run out of memory.")
    from app.device import add_argument as _device_arg
    _device_arg(ap)
    args = ap.parse_args()
    _WINDOW_SECONDS = args.window

    init_db()
    _load_model(args.device)

    with get_db() as conn:
        if args.source_id is not None:
            ids = [args.source_id]
        else:
            q = "SELECT id FROM sources WHERE 1=1"
            p: list = []
            if args.start_from_id is not None:
                q += " AND id>=?"; p.append(args.start_from_id)
            if args.end_at_id is not None:
                q += " AND id<=?"; p.append(args.end_at_id)
            q += " ORDER BY id"
            ids = [r["id"] for r in conn.execute(q, p).fetchall()]

    print(f"Refining {len(ids)} source(s){'  (DRY RUN)' if not args.apply else ''}\n{'-'*54}")
    tot_u = 0
    for sid in ids:
        u, t, med = refine_source(sid, args.apply)
        tot_u += u
        print(f"  source {sid:3d}: {u}/{t} clips refined  (median shift {med:+.0f}ms)")

    if args.apply and tot_u:
        try:
            from app.generate import invalidate_cache
            invalidate_cache()
        except Exception:
            pass
    print(f"\n{tot_u} clips {'updated' if args.apply else '(dry run)'}. "
          f"{'Call POST /api/reload.' if args.apply else 'Add --apply to write.'}")
    if args.apply and tot_u:
        # Phoneme times that no longer fit their moved clip were dropped, and
        # so was every moved clip's tone; until both are measured again the
        # splicer guesses cuts from spelling and sentences pick blind to tone.
        print("Boundaries moved: run scripts/align_phones.py --all and "
              "scripts/measure_tone.py next (build_corpus.py does both).")


if __name__ == "__main__":
    main()
