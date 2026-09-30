"""Read a MIDI file and say, in plain words, what the song is doing.

A YTPMV needs to know what each part *is* before it can pick a word for it:
a bass line wants a long low vowel, a hi-hat wants a hiss, a kick wants a
plosive. General MIDI says most of that outright (program numbers, channel 10
for drums); the rest -- which part is the tune, what key it is in -- is read
from the notes themselves.

Nothing here touches audio.
"""

from __future__ import annotations

import bisect
import io
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import mido

from app.ytpmv.pitch import note_name

DRUM_CHANNEL = 9

GM_PROGRAMS = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    "Nylon Guitar", "Steel Guitar", "Jazz Guitar", "Clean Guitar",
    "Muted Guitar", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
    "Acoustic Bass", "Fingered Bass", "Picked Bass", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass",
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
    "Oboe", "English Horn", "Bassoon", "Clarinet",
    "Piccolo", "Flute", "Recorder", "Pan Flute",
    "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Square Lead", "Saw Lead", "Calliope Lead", "Chiff Lead",
    "Charang Lead", "Voice Lead", "Fifths Lead", "Bass + Lead",
    "New Age Pad", "Warm Pad", "Polysynth Pad", "Choir Pad",
    "Bowed Pad", "Metallic Pad", "Halo Pad", "Sweep Pad",
    "Rain FX", "Soundtrack FX", "Crystal FX", "Atmosphere FX",
    "Brightness FX", "Goblins FX", "Echoes FX", "Sci-fi FX",
    "Sitar", "Banjo", "Shamisen", "Koto",
    "Kalimba", "Bagpipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]

# What a family of programs is called when counting them for the summary.
_FAMILY = ["piano", "chromatic percussion", "organ", "guitar", "bass", "strings",
           "ensemble", "brass", "reed", "pipe", "synth lead", "synth pad",
           "synth effects", "ethnic", "percussive", "sound effects"]

# General MIDI's drum kit, folded into the handful of sounds a grid can show.
# The whole percussion map, not just the middle of it: a kit that reaches for
# timbales, sticks and a metronome bell -- as "Chaos King" does -- put 795 of
# its 800 hits in the leftover "perc" pile, which is one tile playing four
# different instruments.
DRUM_GROUPS = {
    "kick":    {35, 36},
    "snare":   {31, 37, 38, 39, 40},                    # 31 is a stick hit
    "hats":    {42, 44, 46, 54, 69, 70, 82},            # and shakers, which sit the same way
    "toms":    {41, 43, 45, 47, 48, 50,
                60, 61, 62, 63, 64, 65, 66, 67, 68},    # bongos, congas, timbales
    "cymbals": {49, 51, 52, 53, 55, 57, 59},
}
_DRUM_ORDER = ["kick", "snare", "hats", "toms", "cymbals", "perc"]


def drum_group(note: int) -> str:
    for g, notes in DRUM_GROUPS.items():
        if note in notes:
            return g
    return "perc"


# Krumhansl-Kessler key profiles: how strongly each scale degree implies a key.
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


@dataclass
class Note:
    start: float      # seconds
    dur: float
    pitch: int        # MIDI note number
    velocity: int


@dataclass
class Part:
    id: str
    name: str
    track: int
    channel: int
    program: int | None           # None for drums
    notes: list[Note] = field(default_factory=list)
    role: str = "rhythm"          # bass | lead | chords | rhythm | drums:<group>
    drum_group: str | None = None
    pan: float | None = None

    @property
    def is_drums(self) -> bool:
        return self.drum_group is not None

    @property
    def instrument(self) -> str:
        if self.is_drums:
            return {"kick": "Kick drum", "snare": "Snare", "hats": "Hi-hats",
                    "toms": "Toms", "cymbals": "Cymbals", "perc": "Percussion"}[self.drum_group]
        return GM_PROGRAMS[self.program or 0]

    def median_pitch(self) -> float:
        """The pitch the part sits on, weighted by how long each note lasts."""
        if not self.notes:
            return 60.0
        ws = sorted((n.pitch, n.dur) for n in self.notes)
        half = sum(d for _p, d in ws) / 2.0
        acc = 0.0
        for pch, d in ws:
            acc += d
            if acc >= half:
                return float(pch)
        return float(ws[-1][0])

    def common_pitch(self) -> int:
        return Counter(n.pitch for n in self.notes).most_common(1)[0][0] if self.notes else 60

    def polyphony(self) -> tuple[int, float]:
        """(most notes ever sounding at once, typical number while any sound)."""
        ev = sorted([(n.start, 1) for n in self.notes] + [(n.start + n.dur, -1) for n in self.notes],
                    key=lambda e: (e[0], e[1]))
        cur = mx = 0
        weighted = active = 0.0
        last = None
        for t, d in ev:
            if last is not None and cur > 0:
                weighted += cur * (t - last)
                active += t - last
            cur += d
            mx = max(mx, cur)
            last = t
        return mx, (weighted / active if active > 0 else 0.0)

    def summary(self) -> dict:
        mx, typ = self.polyphony()
        lo = min((n.pitch for n in self.notes), default=0)
        hi = max((n.pitch for n in self.notes), default=0)
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "instrument": self.instrument,
            "is_drums": self.is_drums,
            "notes": len(self.notes),
            "range": [note_name(lo), note_name(hi)] if not self.is_drums else None,
            "range_midi": [lo, hi],
            "median_midi": self.median_pitch(),
            "common_midi": self.common_pitch(),
            "pan": round(self.pan, 2) if self.pan is not None else None,
            "polyphony": mx,
            "typical_polyphony": round(typ, 2),
            "first": round(self.notes[0].start, 3) if self.notes else None,
        }


@dataclass
class Song:
    parts: list[Part]
    duration: float
    bpm: float
    time_signature: tuple[int, int]
    key: str
    key_confidence: float
    title: str = ""
    tempo_changes: int = 0

    def part(self, pid: str) -> Part | None:
        return next((p for p in self.parts if p.id == pid), None)

    def describe(self) -> str:
        bpm = self.bpm
        pace = ("slow" if bpm < 76 else "relaxed" if bpm < 100 else "upbeat" if bpm < 128
                else "fast" if bpm < 160 else "very fast")
        mood = "dark" if self.key.endswith("minor") else "bright"
        melodic = [p for p in self.parts if not p.is_drums]
        fams = Counter(_FAMILY[(p.program or 0) // 8] for p in melodic)
        heavy = any(p.program in (29, 30) for p in melodic)
        bits = [fam if n == 1 else f"{n} {fam}{'' if fam.endswith('s') else 's'}"
                for fam, n in fams.most_common()]
        drums = [p for p in self.parts if p.is_drums]
        if drums:
            bits.append(f"{len(drums)} drum sound{'s' if len(drums) > 1 else ''}")
        m, s = divmod(int(round(self.duration)), 60)
        return (f"{pace.capitalize()} ({bpm:.0f} BPM), {self.key}, {mood}"
                f"{' and heavy' if heavy else ''}, {m}:{s:02d} long: " + ", ".join(bits) + ".")

    def summary(self) -> dict:
        return {
            "title": self.title,
            "duration": round(self.duration, 2),
            "bpm": round(self.bpm, 1),
            "tempo_changes": self.tempo_changes,
            "time_signature": f"{self.time_signature[0]}/{self.time_signature[1]}",
            "key": self.key,
            "key_confidence": round(self.key_confidence, 2),
            "description": self.describe(),
            "parts": [p.summary() for p in self.parts],
        }


class _TempoMap:
    def __init__(self, tpb: int, changes: list[tuple[int, int]]):
        self.tpb = abs(tpb) if tpb else 480
        if self.tpb == 0:
            self.tpb = 480
        ch = sorted(changes)
        if not ch or ch[0][0] != 0:
            ch.insert(0, (0, 500000))          # MIDI's default: 120 BPM
        self.ticks = [t for t, _ in ch]
        self.tempi = [v if v > 0 else 500000 for _, v in ch]
        self.secs = [0.0]
        for i in range(1, len(ch)):
            dt = self.ticks[i] - self.ticks[i - 1]
            self.secs.append(self.secs[-1] + dt * self.tempi[i - 1] / 1e6 / self.tpb)

    def seconds(self, tick: int) -> float:
        if self.tpb <= 0:
            self.tpb = 480
        tick = max(0, tick)
        i = max(0, bisect.bisect_right(self.ticks, tick) - 1)
        return self.secs[i] + (tick - self.ticks[i]) * self.tempi[i] / 1e6 / self.tpb

    def main_bpm(self, end_tick: int) -> float:
        """The tempo the song spends longest in."""
        spans: Counter = Counter()
        for i, t in enumerate(self.ticks):
            nxt = self.ticks[i + 1] if i + 1 < len(self.ticks) else max(end_tick, t)
            spans[self.tempi[i]] += (nxt - t) * self.tempi[i]      # time, not beats
        tempo = spans.most_common(1)[0][0] if spans else 500000
        if not tempo or tempo <= 0:
            tempo = 500000
        return 60e6 / tempo


def detect_key(parts: list[Part]) -> tuple[str, float]:
    hist = [0.0] * 12
    for p in parts:
        if p.is_drums:
            continue
        for n in p.notes:
            hist[n.pitch % 12] += n.dur
    if sum(hist) == 0:
        return "no key", 0.0

    def corr(a, b):
        ma, mb = sum(a) / 12, sum(b) / 12
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den if den else 0.0

    scores = []
    for tonic in range(12):
        rot = hist[tonic:] + hist[:tonic]
        scores.append((corr(rot, _MAJOR), f"{_PC[tonic]} major"))
        scores.append((corr(rot, _MINOR), f"{_PC[tonic]} minor"))
    by_name = {name: sc for sc, name in scores}
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    # Confidence: how clearly the winner beats the runner-up, squashed to 0..1.
    conf = max(0.0, min(1.0, best[0])) * min(1.0, 0.5 + (best[0] - second[0]) * 5)

    # A major key and its relative minor use the same seven notes, so counting
    # notes cannot tell them apart -- Megalovania's D minor read as F major.
    # What does is where the music says home is: the note the tune starts on,
    # and the notes the bass starts and ends on. When the relative key scores
    # nearly as well, let those vote.
    t0, mode0 = _PC.index(best[1].split()[0]), best[1].split()[1]
    rel = (t0 + 9) % 12 if mode0 == "major" else (t0 + 3) % 12
    rel_name = f"{_PC[rel]} {'minor' if mode0 == 'major' else 'major'}"
    if best[0] - by_name[rel_name] < 0.15:
        votes = {t0: 0.0, rel: 0.0}
        melodic = [p for p in parts if not p.is_drums and p.notes]
        if melodic:
            low = min(melodic, key=lambda p: p.median_pitch())
            high = max(melodic, key=lambda p: p.median_pitch())
            bass = sorted(low.notes, key=lambda n: (n.start, n.pitch))
            tune = sorted(high.notes, key=lambda n: (n.start, -n.pitch))
            for pc in (bass[0].pitch % 12, bass[-1].pitch % 12, tune[0].pitch % 12):
                if pc in votes:
                    votes[pc] += 1
        votes[t0] += 0.1 * hist[t0] / max(hist)        # the counts only break a tie
        votes[rel] += 0.1 * hist[rel] / max(hist)
        if votes[rel] > votes[t0]:
            best = (by_name[rel_name], rel_name)

    # The profiles find the tonic well and the mode badly whenever one note
    # drones: a riff that sits on E for half its length correlates with E major
    # and E minor almost equally, and the drone decides it by accident. That is
    # how E1M1 -- G natural 74 times, G sharp twice -- came out "E major". The
    # third (and the sixth) above the tonic is what actually tells them apart,
    # so let it, whenever the song plays either.
    tonic = _PC.index(best[1].split()[0])
    minor = hist[(tonic + 3) % 12] + 0.5 * hist[(tonic + 8) % 12]
    major = hist[(tonic + 4) % 12] + 0.5 * hist[(tonic + 9) % 12]
    if minor + major > 0.02 * sum(hist) and abs(minor - major) > 0.25 * (minor + major):
        return f"{_PC[tonic]} {'minor' if minor > major else 'major'}", conf
    return best[1], conf


def _assign_roles(parts: list[Part]) -> None:
    melodic = [p for p in parts if not p.is_drums]
    info = {p.id: (p.median_pitch(), *p.polyphony(), len(p.notes)) for p in melodic}
    for p in parts:
        if p.is_drums:
            p.role = f"drums:{p.drum_group}"
    # A GM bass program is the bass. Only when a song has none is the part
    # found by ear -- the lowest one that plays single notes -- because guitars
    # sit on low E too, and "low and one note at a time" called both of E1M1's
    # guitars basses.
    basses = {p.id for p in melodic if p.program is not None and 32 <= p.program <= 39}
    if not basses:
        low = [p for p in melodic if info[p.id][0] < 48 and info[p.id][2] < 1.5]
        if low and len(melodic) > 1:
            basses = {min(low, key=lambda p: info[p.id][0]).id}
    rest = []
    for p in melodic:
        _med, _mx, typ, _n = info[p.id]
        if p.id in basses:
            p.role = "bass"
        elif typ >= 2.5:
            p.role = "chords"
        else:
            rest.append(p)
    if rest:
        # The tune is the highest part that mostly plays one note at a time;
        # busier parts win a tie, since a melody moves more than an accompaniment.
        lead = max(rest, key=lambda p: (round(info[p.id][0] / 3), info[p.id][3]))
        for p in rest:
            p.role = "lead" if p is lead else "rhythm"


def _chunks(data: bytes):
    """(header, [chunk, ...]) of a MIDI file, each chunk with its 8-byte head."""
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    hlen = int.from_bytes(data[4:8], "big")
    i, out = 8 + hlen, []
    while i + 8 <= len(data):
        n = int.from_bytes(data[i + 4:i + 8], "big")
        out.append(data[i:i + 8 + n])
        i += 8 + n
    return data[:8 + hlen], out


def _one_track(head: bytes, chunk: bytes, clip: bool = True) -> "mido.MidiTrack":
    one = b"MThd" + (6).to_bytes(4, "big") + bytes((0, 1, 0, 1)) + head[12:14] + chunk
    return mido.MidiFile(file=io.BytesIO(one), clip=clip).tracks[0]


def _salvage(data: bytes) -> "mido.MidiFile":
    """The tracks of *data* that read, when one that does not sinks the file.

    A Deltarune arrangement had a stray 0xF4 -- a status byte MIDI never
    defined -- in one short track of fifty-nine, and mido refuses the whole
    file for it. Each track is tried on its own and the broken ones left out.
    """
    head, chunks = _chunks(data)
    good = []
    for chunk in chunks:
        if chunk[:4] != b"MTrk":
            continue
        try:
            good.append(_one_track(head, chunk))
        except (OSError, EOFError, KeyError, ValueError):
            pass                             # a broken track is left out
    if not good:
        raise ValueError("no track in the file could be read")
    mf = mido.MidiFile(type=1, ticks_per_beat=int.from_bytes(head[12:14], "big"))
    mf.tracks.extend(good)
    return mf


# Some writer, given more than sixteen parts, kept counting channels and ORed
# the count straight into the status byte: status = (type << 4) | c with c up
# to 0x7F. Channel 0x34's note-on is then 0xB4, which reads as a controller;
# 0x3F's program change is 0xFF, which reads as a meta event; 0x4D's note-off
# and program change are both 0xCD. mido refuses some of those tracks and
# misreads the rest -- note-offs taken for note-ons, so notes never end and a
# 3:24 song came out 44 minutes long.
#
# Such a track is spotted by failing to read as plain MIDI (mido, without
# clipping: a data byte over 127 is the telltale) and then read again for
# each possible overflow h = c >> 4: a status s can be any type t with
# t | h == s >> 4. That is ambiguous event by event (0xCD: note-off with two
# data bytes, or program change with one), so the whole track is solved at
# once, backwards -- which positions can still reach the end cleanly -- and
# read forwards along a path that does. Running status is not allowed there;
# the writer never used it, and it is what keeps the solve cheap and certain.
# Of the overflows that read, the largest is taken: every status carries the
# bits of h, so a smaller h that also reads has dropped a bit it should keep
# (0x3F's program change, 0xF_, read with h = 2 comes out channel pressure).
#
# A meta event is told from a channel event with status 0xFF by its shape:
# the meta types MIDI defines, at the lengths it defines them, beat the
# channel reading. 0xFF 0x59 0x64 is a note-on, not a key signature of length
# 100 -- mido would refuse the key -- and 0xFF 0x2F 0x00 ends the track only
# where the track ends. An unknown type is kept as a last resort.
_META = {0x00: 2, 0x20: 1, 0x21: 1, 0x2F: 0, 0x51: 3, 0x54: 5, 0x58: 4, 0x59: 2,
         **{t: None for t in (1, 2, 3, 4, 5, 6, 7, 8, 9, 0x7F)}}
# When more than one reading gets to the end, the likeliest is taken. This
# writer puts a program change first and then only notes, so: the program
# change first, then notes, then notes with a velocity over 127 (it writes
# those too, and they are clamped), then anything else. Taking a loud note
# over a program change anywhere but the start is what keeps 0xF8 0x36 0x8A
# a note; taking the program change at the start is what keeps 0xD0 0x50
# 0x86 0x98 0x00 a program change and a delta, not a note and a shorter delta.
# A controller that shares a status with a note-on reads as the note.
_TYPES = (9, 8, 12, 11, 14, 13, 10)


def _vlq(b: bytes, i: int) -> tuple[int | None, int]:
    v = 0
    for _ in range(4):
        if i >= len(b):
            break
        x, i = b[i], i + 1
        v = (v << 7) | (x & 0x7F)
        if x < 0x80:
            return v, i
    return None, i


def _readings(b: bytes, p: int, h: int, first: bool = False) -> list:
    """Every way the event whose status is b[p] reads, as (type, end), likeliest first."""
    s, out = b[p], []
    for i, t in enumerate(_TYPES):
        end = p + (2 if t in (12, 13) else 3)
        note = t in (8, 9)
        if t | h == s >> 4 and end <= len(b) and all(x < 0x80 for x in b[p + 1:end - note]):
            rank = -1 if first and t == 12 else (b[end - 1] >= 0x80) if note else 2
            out.append((rank, i, t, end))
    ln, q = _vlq(b, p + 2) if s == 0xFF else _vlq(b, p + 1) if s in (0xF0, 0xF7) else (None, 0)
    if ln is not None and q + ln <= len(b):
        known = s == 0xFF and b[p + 1] in _META
        if not known:
            out.append((3, 0, "meta", q + ln))           # a sysex or unknown meta, last
        elif _META[b[p + 1]] in (None, ln) and (b[p + 1] != 0x2F or q + ln == len(b)):
            out.append((-2, 0, "meta", q + ln))
    return [(t, end) for _r, _i, t, end in sorted(out)]


def _unoverflow(body: bytes, h: int) -> bytes | None:
    """*body* rewritten as plain MIDI, read with overflow *h*; None if it will not read."""
    n = len(body)
    # Forwards: where each event reachable from the start could end. Then
    # backwards: which of those can still get to the end of the track.
    reach, ends = bytearray(n + 1), {}
    reach[0] = 1
    for pos in range(n):
        if reach[pos]:
            d, p = _vlq(body, pos)
            ends[pos] = [e for _t, e in _readings(body, p, h)] if d is not None and p < n else []
            for e in ends[pos]:
                reach[e] = 1
    if not reach[n]:
        return None
    ok = bytearray(n + 1)
    ok[n] = 1
    for pos in sorted(ends, reverse=True):
        ok[pos] = any(ok[e] for e in ends[pos])
    out, pos, first = bytearray(), 0, True
    while pos < n:
        _d, p = _vlq(body, pos)
        t, end = next(r for r in _readings(body, p, h, first) if ok[r[1]])
        first = first and t == "meta"
        data = body[p + 1:end]
        # The low nibble is the channel. Channel 9 there is not a drum kit --
        # the writer's kits all sit on a true channel 9, and a count of 0x19
        # is its 26th melodic part -- so a rewritten track is never drums.
        ch = body[p] & 0x0F if body[p] & 0x0F != DRUM_CHANNEL else 8
        if t == "meta":
            out += body[pos:end]
        elif t in (8, 9):
            # With h odd a note-on and a note-off share one status, and only
            # the velocity tells them apart: this writer releases with 0. It
            # also stacks the same pitch on itself, so "already sounding"
            # would not mean "being released".
            off = t == 8 or data[1] == 0
            out += body[pos:p] + bytes((0x80 | ch if off else 0x90 | ch, data[0], min(data[1], 127)))
        else:
            out += body[pos:p] + bytes(((t << 4) | ch,)) + data
        pos = end
    return bytes(out)


def _h1(track) -> bool:
    """Whether a track that reads may still be overflowed by 0x10.

    With h = 1 every status still reads: note-ons stay note-ons and the
    program change turns into channel pressure. Pressure with no program
    change, and nothing whose status lacks the 0x10 bit, is that track.
    """
    types = {m.type for m in track if not m.is_meta}
    return "aftertouch" in types and types <= {"note_on", "control_change", "aftertouch", "sysex"}


def _plain(head: bytes, chunk: bytes) -> bool:
    try:
        try:
            track = _one_track(head, chunk, clip=False)
        except (OSError, EOFError, KeyError, ValueError):
            # A velocity over 127 alone, as in "Chaos King", is no overflow.
            if _unoverflow(chunk[8:], 0) is None:
                return False
            track = _one_track(head, chunk)
        return not _h1(track)
    except (OSError, EOFError, KeyError, ValueError):
        return False


def _repair(data: bytes) -> bytes:
    """*data* with every channel-overflowed track rewritten as plain MIDI."""
    try:
        head, chunks = _chunks(data)
    except ValueError:
        return data
    fixed, changed = [], False
    for chunk in chunks:
        new = None
        if chunk[:4] == b"MTrk" and not _plain(head, chunk):
            new = next((r for h in range(7, 0, -1) if (r := _unoverflow(chunk[8:], h)) is not None), None)
        if new is not None:
            chunk, changed = b"MTrk" + len(new).to_bytes(4, "big") + new, True
        fixed.append(chunk)
    return head + b"".join(fixed) if changed else data


def parse(data: bytes, title: str = "") -> Song:
    # clip=True: a data byte over 127 is clamped rather than refused. Files in
    # the wild have them -- one exported "Chaos King" has a velocity out of
    # range -- and mido's default is to raise, which threw away a whole song
    # over one byte in one note.
    #
    # A file that reads strictly reads the same clipped, so it is read once;
    # only one that does not (or looks overflowed by 0x10) is repaired first.
    try:
        mf = mido.MidiFile(file=io.BytesIO(data))
        if any(_h1(t) for t in mf.tracks):
            raise ValueError("overflowed channels")
    except (OSError, EOFError, KeyError, ValueError):
        data = _repair(data)
        try:
            mf = mido.MidiFile(file=io.BytesIO(data), clip=True)
        except (OSError, EOFError, KeyError, ValueError):
            mf = _salvage(data)
    tpb = mf.ticks_per_beat or 480

    tempo_changes: list[tuple[int, int]] = []
    time_sig = (4, 4)
    programs: dict[int, int] = {}
    end_tick = 0
    # (track, channel) -> list of (start_tick, end_tick, pitch, velocity, released)
    raw: dict[tuple[int, int], list[tuple[int, int, int, int, bool]]] = defaultdict(list)
    chan_prog: dict[tuple[int, int], Counter] = defaultdict(Counter)
    names: dict[int, str] = {}
    pans: dict[tuple[int, int], float] = {}

    for ti, track in enumerate(mf.tracks):
        tick = 0
        open_: dict[tuple[int, int], deque[tuple[int, int]]] = defaultdict(deque)
        for msg in track:
            tick += msg.time
            if msg.is_meta:
                if msg.type == "set_tempo":
                    tempo_changes.append((tick, msg.tempo))
                elif msg.type == "time_signature" and tick == 0:
                    time_sig = (msg.numerator, msg.denominator)
                elif msg.type == "track_name" and msg.name.strip():
                    names[ti] = msg.name.strip()
                continue
            if msg.type == "program_change":
                programs[msg.channel] = msg.program
                continue
            if msg.type == "control_change" and msg.control == 10:
                pans[(ti, msg.channel)] = max(-1.0, min(1.0, (msg.value - 64) / 64.0))
                continue
            if msg.type == "note_on" and msg.velocity > 0:
                open_[(msg.channel, msg.note)].append((tick, msg.velocity))
                chan_prog[(ti, msg.channel)][programs.get(msg.channel, 0)] += 1
            elif msg.type in ("note_off", "note_on"):
                q = open_.get((msg.channel, msg.note))
                if q:
                    st, vel = q.popleft()
                    raw[(ti, msg.channel)].append((st, tick, msg.note, vel, True))
            end_tick = max(end_tick, tick)
        for (ch, pitch), q in open_.items():        # never released: end with the track
            for st, vel in q:
                raw[(ti, ch)].append((st, tick, pitch, vel, False))

    tm = _TempoMap(tpb, tempo_changes)
    # A note never released rings to the end of its track, and one stuck note
    # in a track that runs on (or one misread file) made a song last forty
    # minutes. So it rings at most a few seconds past the last note that was
    # properly let go -- or past its own start, if it starts later than that.
    last = max((e for evs in raw.values() for _s, e, _p, _v, rel in evs if rel), default=None)
    last = tm.seconds(last) if last is not None else None

    def end(s: int, e: int, rel: bool) -> float:
        e_s = tm.seconds(e)
        return e_s if rel or last is None else min(e_s, max(tm.seconds(s), last) + 4.0)

    parts: list[Part] = []
    for (ti, ch), evs in sorted(raw.items()):
        evs.sort()
        notes = [Note(tm.seconds(s), max(end(s, e, rel) - tm.seconds(s), 0.0), pch, vel)
                 for s, e, pch, vel, rel in evs]
        p_pan = pans.get((ti, ch))
        if ch == DRUM_CHANNEL:
            groups: dict[str, list[Note]] = defaultdict(list)
            for n in notes:
                groups[drum_group(n.pitch)].append(n)
            for g in _DRUM_ORDER:
                if groups.get(g):
                    parts.append(Part(id=f"t{ti}c{ch}-{g}", name=g, track=ti, channel=ch,
                                      program=None, notes=groups[g], drum_group=g, pan=p_pan))
        else:
            prog = chan_prog[(ti, ch)].most_common(1)[0][0] if chan_prog[(ti, ch)] else 0
            parts.append(Part(id=f"t{ti}c{ch}", name=names.get(ti) or GM_PROGRAMS[prog],
                              track=ti, channel=ch, program=prog, notes=notes, pan=p_pan))

    _assign_roles(parts)

    # Friendlier, unique names: "Distortion Guitar (lead)".
    seen: Counter = Counter()
    for p in parts:
        if not p.is_drums:
            base = f"{p.instrument} ({p.role})"
            seen[base] += 1
            p.name = base if seen[base] == 1 else f"{base} {seen[base]}"

    # Melodic parts first (lead, chords, rhythm, bass), then the kit in kit order.
    order = {"lead": 0, "chords": 1, "rhythm": 2, "bass": 3}
    parts.sort(key=lambda p: (1, _DRUM_ORDER.index(p.drum_group)) if p.is_drums
               else (0, order.get(p.role, 9)))

    key, conf = detect_key(parts)
    duration = max((n.start + n.dur for p in parts for n in p.notes), default=0.0)
    return Song(parts=parts, duration=duration, bpm=tm.main_bpm(end_tick),
                time_signature=time_sig, key=key, key_confidence=conf, title=title,
                tempo_changes=max(0, len(set(tempo_changes)) - 1))


def auto_octave(median_midi: float, sample_midi: float | None) -> int:
    """Octaves to move a part so it sits near the voice singing it.

    Pulling speech two octaves down to a bass E1 turns it into a growl of
    separate clicks; pushing it two up turns it into a whistle. Moving the part
    by whole octaves keeps every note's name -- the harmony is the same song --
    while the voice stays somewhere it can actually sing.
    """
    if sample_midi is None:
        return 0
    return int(max(-3, min(3, round((sample_midi - median_midi) / 12.0))))
