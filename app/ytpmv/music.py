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
from collections import Counter, defaultdict
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
DRUM_GROUPS = {
    "kick":    {35, 36},
    "snare":   {37, 38, 39, 40},
    "hats":    {42, 44, 46},
    "toms":    {41, 43, 45, 47, 48, 50},
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
        self.tpb = tpb
        ch = sorted(changes)
        if not ch or ch[0][0] != 0:
            ch.insert(0, (0, 500000))          # MIDI's default: 120 BPM
        self.ticks = [t for t, _ in ch]
        self.tempi = [v for _, v in ch]
        self.secs = [0.0]
        for i in range(1, len(ch)):
            dt = self.ticks[i] - self.ticks[i - 1]
            self.secs.append(self.secs[-1] + dt * self.tempi[i - 1] / 1e6 / tpb)

    def seconds(self, tick: int) -> float:
        i = bisect.bisect_right(self.ticks, tick) - 1
        return self.secs[i] + (tick - self.ticks[i]) * self.tempi[i] / 1e6 / self.tpb

    def main_bpm(self, end_tick: int) -> float:
        """The tempo the song spends longest in."""
        spans: Counter = Counter()
        for i, t in enumerate(self.ticks):
            nxt = self.ticks[i + 1] if i + 1 < len(self.ticks) else max(end_tick, t)
            spans[self.tempi[i]] += (nxt - t) * self.tempi[i]      # time, not beats
        tempo = spans.most_common(1)[0][0] if spans else 500000
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
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    # Confidence: how clearly the winner beats the runner-up, squashed to 0..1.
    conf = max(0.0, min(1.0, best[0])) * min(1.0, 0.5 + (best[0] - second[0]) * 5)

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


def parse(data: bytes, title: str = "") -> Song:
    mf = mido.MidiFile(file=io.BytesIO(data))
    tpb = mf.ticks_per_beat or 480

    tempo_changes: list[tuple[int, int]] = []
    time_sig = (4, 4)
    programs: dict[int, int] = {}
    end_tick = 0
    # (track, channel) -> list of (start_tick, end_tick, pitch, velocity)
    raw: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    chan_prog: dict[tuple[int, int], Counter] = defaultdict(Counter)
    names: dict[int, str] = {}

    for ti, track in enumerate(mf.tracks):
        tick = 0
        open_: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
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
            if msg.type == "note_on" and msg.velocity > 0:
                open_[(msg.channel, msg.note)].append((tick, msg.velocity))
                chan_prog[(ti, msg.channel)][programs.get(msg.channel, 0)] += 1
            elif msg.type in ("note_off", "note_on"):
                q = open_.get((msg.channel, msg.note))
                if q:
                    st, vel = q.pop(0)
                    raw[(ti, msg.channel)].append((st, tick, msg.note, vel))
            end_tick = max(end_tick, tick)
        for (ch, pitch), q in open_.items():        # never released: end with the track
            for st, vel in q:
                raw[(ti, ch)].append((st, tick, pitch, vel))

    tm = _TempoMap(tpb, tempo_changes)
    parts: list[Part] = []
    for (ti, ch), evs in sorted(raw.items()):
        evs.sort()
        notes = [Note(tm.seconds(s), max(tm.seconds(e) - tm.seconds(s), 0.0), pch, vel)
                 for s, e, pch, vel in evs]
        if ch == DRUM_CHANNEL:
            groups: dict[str, list[Note]] = defaultdict(list)
            for n in notes:
                groups[drum_group(n.pitch)].append(n)
            for g in _DRUM_ORDER:
                if groups.get(g):
                    parts.append(Part(id=f"t{ti}c{ch}-{g}", name=g, track=ti, channel=ch,
                                      program=None, notes=groups[g], drum_group=g))
        else:
            prog = chan_prog[(ti, ch)].most_common(1)[0][0] if chan_prog[(ti, ch)] else 0
            parts.append(Part(id=f"t{ti}c{ch}", name=names.get(ti) or GM_PROGRAMS[prog],
                              track=ti, channel=ch, program=prog, notes=notes))

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
