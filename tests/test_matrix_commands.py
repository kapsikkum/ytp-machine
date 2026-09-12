#!/usr/bin/env python3
"""What the Matrix bot hears in a message, as plain asserts.

    python tests/test_matrix_commands.py

Needs nothing installed: the parser is kept out of the bot so it can be
checked without a homeserver or matrix-nio.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.matrix_commands import is_midi, match_part, parse

failures: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


print("addressing")
check("plain chatter in a room is ignored", parse("nice chocolate cake"), None)
check("the prefix alone asks for help", parse("!ytp").name, "help")
check("the prefix is case-blind", parse("!YTP say hi").text, "hi")
check("a prefix inside a word is not the prefix", parse("!ytpfoo say hi"), None)
check("in a DM plain text is something to say", parse("nice cake", direct=True).name, "say")
check("... and keeps its words", parse("nice cake", direct=True).text, "nice cake")
check("in a DM a verb still works bare", parse("mv lead=yeah", direct=True).name, "mv")
check("a custom prefix", parse("!rosen say hi", prefix="!rosen").text, "hi")
check("quoted reply lines are not the command",
      parse("> <@a:b> something\n!ytp say hello").text, "hello")

print("say")
c = parse("!ytp say nice chocolate cake")
check("say takes the rest as words, captions on by default",
      (c.name, c.text, c.subtitles), ("say", "nice chocolate cake", True))
c = parse("!ytp say --subs nice cake")
check("--subs is still accepted", (c.text, c.subtitles), ("nice cake", True))
c = parse("!ytp say --no-subs nice cake")
check("--no-subs turns captions off", (c.text, c.subtitles), ("nice cake", False))
check("a bare --no-subs says nothing", parse("!ytp say --no-subs").name, "help")
check("say with nothing to say is help", parse("!ytp say").name, "help")
check("~markup~ passes through", parse("!ytp say ~nice~ *spew*").text, "~nice~ *spew*")

print("mv")
check("mv alone", (parse("!ytp mv").name, parse("!ytp mv").parts), ("mv", {}))
check("mv info", parse("!ytp mv info").name, "info")
c = parse("!ytp mv lead=yeah kick=boom max=60")
check("part words", c.parts, {"lead": {"text": "yeah"}, "kick": {"text": "boom"}})
check("song options", c.options, {"max_seconds": 60.0})
c = parse('!ytp mv lead="oh no" bass=hello_there')
check("a quoted phrase", c.parts["lead"]["text"], "oh no")
check("underscores are spaces", c.parts["bass"]["text"], "hello there")
c = parse("!ytp mv octave:bass=+1 mode:lead=tape take:kick=3 volume:hats=0.4")
check("per-part settings", c.parts, {"bass": {"octave": 1}, "lead": {"mode": "tape"},
                                     "kick": {"take": 2 + 1}, "hats": {"volume": 0.4}})
c = parse("!ytp mv hats=mute")
check("a part can be muted", c.parts["hats"], {"mute": True, "visible": False})
c = parse("!ytp mv vary=random jitter=off")
check("how much a part varies between hits", c.options, {"vary": "random", "jitter": False})
check("the ultra mode", parse("!ytp mv vary=ultra").options, {"vary": "ultra"})
check("the whole song through a chip", parse("!ytp mv chip=md").options, {"chip": "md"})
check("keeping his voice, off that machine's sample channel",
      parse("!ytp mv chip=md-voice").options, {"chip": "md-voice"})
check("the other machine", parse("!ytp mv chip=nes").options, {"chip": "nes"})
check("turning it off again", parse("!ytp mv chip=off").options, {"chip": False})
check("a machine we do not have is reported",
      parse("!ytp mv chip=snes").errors, ["chip is one of off, md, md-voice, nes, nes-voice"])
check("an NES voice on one part", parse("!ytp mv tone:bass=triangle").parts,
      {"bass": {"tone": "triangle"}})
check("one part through one of the chip's patches",
      parse("!ytp mv tone:bass=organ").parts, {"bass": {"tone": "organ"}})
check("the old name for the bass patch still works",
      parse("!ytp mv tone:bass=megadrive").parts, {"bass": {"tone": "megadrive"}})
check("a tone that does not exist is reported",
      len(parse("!ytp mv tone:bass=trombone").errors), 1)
check("a variety mode that does not exist is reported",
      parse("!ytp mv vary=sideways").errors, ["vary is one of off, random, rotate, ultra"])
c = parse("!ytp mv speed=1.5 key=-2 labels=on flip=off")
check("speed, key and switches", c.options,
      {"speed": 1.5, "transpose": -2, "labels": True, "flip": False})
c = parse("!ytp mv mode:lead=loud octave:bass=up banana")
check("bad values are reported, not guessed", len(c.errors), 3)

print("other verbs")
check("voices", parse("!ytp voices").name, "voices")
check("voice with a name", (parse("!ytp voice dank-pods").name, parse("!ytp voice dank-pods").text),
      ("voice", "dank-pods"))
check("voice without one lists them", parse("!ytp voice").name, "voices")
check("queue", parse("!ytp queue").name, "queue")
check("an unknown verb is reported", (parse("!ytp dance").name, parse("!ytp dance").text),
      ("unknown", "dance"))

print("midi files and parts")
check("a .mid is MIDI", is_midi("E1M1.MID"), True)
check("so is audio/midi with no name", is_midi(None, "audio/midi"), True)
check("an mp3 is not", is_midi("song.mp3", "audio/mpeg"), False)
parts = [
    {"id": "t2c1", "name": "Overdriven Guitar (lead)", "role": "lead", "instrument": "Overdriven Guitar"},
    {"id": "t1c0", "name": "Distortion Guitar (rhythm)", "role": "rhythm", "instrument": "Distortion Guitar"},
    {"id": "t3c2", "name": "Picked Bass (bass)", "role": "bass", "instrument": "Picked Bass"},
    {"id": "t4c9-hats", "name": "hats", "role": "drums:hats", "instrument": "Hi-hats"},
]
check("by role", match_part("bass", parts)["id"], "t3c2")
check("by drum group", match_part("hats", parts)["id"], "t4c9-hats")
check("by alias", match_part("hihat", parts)["id"], "t4c9-hats")
check("by id", match_part("t1c0", parts)["id"], "t1c0")
check("by instrument word", match_part("distortion", parts)["id"], "t1c0")
check("melody means the lead", match_part("melody", parts)["id"], "t2c1")
check("nonsense matches nothing", match_part("kazoo", parts), None)

print()
print(f"{len(failures)} failures" if failures else "ALL PASS")
for f in failures:
    print(" ", f)
sys.exit(1 if failures else 0)
