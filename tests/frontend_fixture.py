#!/usr/bin/env python3
"""Isolated browser fixture; synthetic media and in-memory API, no corpus writes.

    python tests/frontend_fixture.py --port 8877
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import mimetypes
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlparse
import wave

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.editor import _spectral_bins

EVIL = '<img src=x onerror="window.fixtureInjected=true">'
CLIPS = [{"id": 1, "kind": "word", "word": "cat", "start_time": 0.4, "end_time": 1.4,
          "phones": [["K", .02, .15], ["AE", .15, .7], ["T", .7, .94]],
          "flags": [], "gap_after": .2, "ratings": {}},
         {"id": 2, "kind": "word", "word": "second", "start_time": 1.6, "end_time": 2.4,
          "phones": [], "flags": [], "gap_after": None, "ratings": {}}]

CORPORA = [{"slug": "fixture", "name": "Test voice", "clips": 2, "words": 2, "active": True},
           {"slug": "fixture-other", "name": "Alternate voice", "clips": 3, "words": 3, "active": False}]
CHIPS = ["md", "nes", "sms", "snes", "gb", "gbc", "opl2", "opl3", "sid", "sid8580"]
MIDI_ID = "fixture-song"


def fixture_part(part_id, corpus):
    """Distinct words and pitches expose accidental global-corpus routing."""
    other = corpus == "fixture-other"
    texts = {"lead": "bell" if other else "cat", "bass": "boom" if other else "second",
             "kick": "thump" if other else "pop"}
    text = texts[part_id]
    rec = {"text": text, "take": 0, "octave": -1 if part_id == "lead" else 0,
           "pitch": {"midi": 72 if other else 60, "stability": 12, "voiced": .95, "dur": .6},
           "alternatives": [{"text": "chime" if other else "cat", "take": 1}]}
    settings = {"id": part_id, "text": text, "take": 0, "corpus": corpus,
                "octave": rec["octave"], "mode": "perfect", "volume": 1, "pan": 0,
                "sustain": "note", "mute": False, "visible": True, "program": 0,
                "hit": part_id == "kick"}
    return {"id": part_id, "corpus": corpus, "voice": "Alternate voice" if other else "Test voice",
            "recommended": rec, "settings": settings, "noises": {"pop": 1}, "units": {"aa": 2}}


def fixture_song(corpus, midi_id=MIDI_ID):
    song = {"midi_id": midi_id, "corpus": corpus, "title": "Synthetic trio.mid" if midi_id == MIDI_ID else "Second song.mid",
            "description": "A tiny fixture tune with a lead, bass and kick.", "duration": 12,
            "bpm": 120, "key": "C major", "time_signature": "4/4", "noises": {"pop": 1},
            "units": {"aa": 2}, "parts": []}
    for part_id, role, name in [("lead", "lead", "Lead piano"), ("bass", "bass", "Bass guitar"),
                                ("kick", "drums:kick", "Kick drum")]:
        suggestion = fixture_part(part_id, corpus)
        song["parts"].append({"id": part_id, "role": role, "instrument": name, "notes": 8,
            "range": None if part_id == "kick" else ["C3", "C5"], "polyphony": 1,
            "is_drums": part_id == "kick", "settings": suggestion["settings"],
            "recommended": suggestion["recommended"],
            "chip_tones": {c: "md-fm" for c in CHIPS} | {c + "-voice": "clean" for c in CHIPS},
            "chip_tones_synth": {c: "md-fm" for c in CHIPS} |
                                 {c + "-voice": "md-fm" for c in CHIPS}})
    song["instruments"] = {"voice": [["clean", "Voice as recorded"], ["robot", "Robotic voice"]],
        "synth": ["md-fm", "md-noise"], "programs": ["Acoustic Grand Piano", "Bright Piano"],
        "chips": {c: "Synthetic chip for browser checks" for c in CHIPS},
        "machine_of": {c: "md" for c in CHIPS}, "machines": {"md": {"name": "Mega Drive",
            "melodic": [["md-fm", "FM instrument"]], "drums": [["md-noise", "Noise drum"]]}}}
    return song


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path); route = parsed.path; q = parse_qs(parsed.query)
        if route == "/api/corpora":
            return self.reply({"active": "fixture", "corpora": CORPORA})
        if route == "/api/settings/server":
            return self.reply({"available_encoders":{"libx264":True},"encoder":"libx264",
                               "encode_workers":2,"clip_cache_mb":0,"clip_cache_max_mb":500})
        if route == "/api/edit/sources":
            return self.reply({"corpus": "fixture", "sources": [
                {"id": 1, "title": "Synthetic tone", "clips": 2},
                {"id": 2, "title": "Second source", "clips": 1}]})
        if route in ["/api/edit/source/1", "/api/edit/source/2"]:
            if route.endswith("/1"):
                time.sleep(.15)
            return self.reply({"clips": CLIPS if route.endswith("/1") else [{**CLIPS[0], "word": "different"}]})
        if route.endswith("/align"):
            return self.reply({"available": False, "reason": "Fixture: alignment disabled", "running": False,
                               "pending": 0, "no_pronunciation": 0, "unknown_words": []})
        if route.endswith("/envelope"):
            a=float(q.get("start", [0])[0]); b=min(a+60,float(q.get("end", [3])[0]))
            samples=self.server.samples[max(0,int(a*16000)):max(0,int(b*16000))]
            hop=max(1,len(samples)//900)
            values=[int(np.sqrt(np.mean(samples[i:i+hop]**2))*32768) for i in range(0,len(samples),hop)]
            body={"start":a,"end":b,"peak":max(values,default=0),"values":values}
            if q.get("spectrogram",["false"])[0] == "true": body["spectrogram"]=_spectral_bins(samples)
            return self.reply(body)
        if route.endswith("/video"):
            return self.file(self.server.media, "video/mp4")
        if route == "/api/edit/recipes": return self.reply({"recipes": []})
        if route == "/api/ytpmv/library":
            return self.reply([{"midi_id": mid, "title": title, "added": time.time(),
                "description": "A tiny fixture tune with a lead, bass and kick.", "duration": 12,
                "bpm": 120, "key": "C major", "time_signature": "4/4", "parts": 2,
                "drums": 1, "notes": 24, "bytes": 256, "tempo_changes": 0}
                for mid, title in [(MIDI_ID, 'Synthetic trio.mid'), ('fixture-second', 'Second song.mid')]])
        if route in [f"/api/ytpmv/midi/{MIDI_ID}", "/api/ytpmv/midi/fixture-second"]:
            return self.reply(fixture_song(self.headers.get("X-Corpus", "fixture"), route.rsplit('/', 1)[-1]))
        if route.startswith(f"/api/ytpmv/midi/{MIDI_ID}/part/"):
            part_id = route.rsplit("/", 1)[-1]
            corpus = q.get("corpus", [self.headers.get("X-Corpus", "fixture")])[0]
            if part_id not in {"lead", "bass", "kick"} or corpus not in {c["slug"] for c in CORPORA}:
                return self.reply({"detail": "unknown fixture part or corpus"}, 400)
            return self.reply(fixture_part(part_id, corpus))
        if route == "/api/words":
            return self.reply({"words": {"bell": 3, "boom": 2, "thump": 1}
                if self.headers.get("X-Corpus") == "fixture-other" else {"cat": 2, "second": 1}})
        if route == "/api/fixture/audio": return self.file(self.server.audio, "audio/wav")
        if route == "/api/fixture/video": return self.file(self.server.media, "video/mp4")
        if route == "/api/jobs/fixture-render":
            return self.reply({"id": "fixture-render", "status": "done", "result": {
                "video_url": "/api/fixture/video", "duration": 3, "size": [480, 270], "problems": []}})
        if route == "/api/jobs/fixture-preview":
            return self.reply({"id": "fixture-preview", "status": "done", "result": {
                "audio_url": "/api/fixture/audio", "from": 0, "duration": 3, "problems": []}})
        if route.startswith("/api/edit/splice/"):
            return self.reply({"word": EVIL, "phones": ["K","AE","T"], "exact_total": 0,
                "plan": [{"phones": ["K","AE","T"], "from": EVIL, "start_time": .1, "end_time": .9}],
                "known": True, "recipe": None, "pron": None})
        if route == "/api/edit/splice-sources":
            return self.reply({"sources": [{"word": EVIL, "phones": ["K","AE","T"], "cuts": 0,
                "clips": 2, "longest": 1, "after_vowel": False,
                "clip_list": [{"id":1,"duration":1,"ratings":{}},{"id":2,"duration":.8,"ratings":{}}]}]})
        if route == "/api/corpus/diagnostics":
            return self.reply({"aligned_clips":2,"alignment_coverage":100,"summary": {"clips_with_phones":2,"total_clips":2,"low_confidence_clips":1,
                "ctc_aligned":2,"ctc_percentage":100,
                "negative_margins":1,"unindexed_downloads":1,"words_missing_pronunciations":1,"downvoted_clips":1},
                "unindexed_downloads":[EVIL], "negative_margins":[{"id":1,"word":EVIL,"start_time":0,"end_time":1,"margin":-.1,"reason":EVIL}],
                "low_confidence_clips":[{"id":1,"word":EVIL,"start_time":0,"end_time":1,"reason":EVIL}],
                "words_missing_pronunciations":[EVIL],"downvoted_clips":[{"clip_id":1,"clip_word":EVIL,"target_word":EVIL,"score":-1}]})
        name=route.strip("/") or "editor.html"
        file=ROOT / "frontend" / name
        if file.parent != ROOT / "frontend" or not file.is_file(): return self.reply({"detail":"fixture route not found"},404)
        return self.file(file,mimetypes.guess_type(str(file))[0] or "application/octet-stream")

    def file(self, path, mime):
        body=path.read_bytes(); self.send_response(200);self.send_header("Content-Type",mime)
        self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)

    def do_POST(self):
        route=urlparse(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        data = json.loads(body) if body and self.headers.get("Content-Type", "").startswith("application/json") else {}
        if route == '/api/fixture/shutdown':
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return self.reply({'stopping': True})
        if route == "/api/edit/splice-piece":
            return self.reply({"source_id":1,"clip_id":1,"src_start":.4,"src_end":1.4,"start":.5,"end":1.3})
        if route == "/api/ytpmv/midi":
            return self.reply(fixture_song(self.headers.get("X-Corpus", "fixture")))
        if route == "/api/ytpmv/sample":
            corpus = data.get("corpus") or self.headers.get("X-Corpus", "fixture")
            return self.reply({"corpus": corpus, "take": data.get("take", 0) % 3, "takes": 3,
                "midi": 72 if corpus == "fixture-other" else 60, "stability": 12, "voiced": .95,
                "duration": .6, "sample_url": "/api/fixture/video?corpus=" + corpus,
                "source": 'Synthetic source recording'})
        if route == "/api/ytpmv/preview": return self.reply({"audio_url": "/api/fixture/audio"})
        if route == "/api/ytpmv/render": return self.reply({"id": "fixture-render", "status": "queued"}, 202)
        if route == "/api/ytpmv/preview-mix": return self.reply({"id": "fixture-preview", "status": "queued"}, 202)
        return self.reply({"detail":"fixture write not implemented"},400)

    def do_DELETE(self):
        return self.reply({'cancelled': True, 'status': 'cancelled'})


def main():
    p=argparse.ArgumentParser(); p.add_argument("--port",type=int,default=8877)
    p.add_argument("--seconds", type=float, default=0,
                   help="automatically stop and remove fixture media after this many seconds")
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix="mrs-browser-fixture-") as directory:
        directory=Path(directory); wavpath=directory / "tone.wav"; media=directory / "tone.mp4"
        t=np.arange(48000)/16000
        samples=(.6*np.sin(2*np.pi*1000*t)).astype(np.float32); samples[32000:]=0
        with wave.open(str(wavpath),"wb") as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
            wav.writeframes((samples*32767).astype("<i2").tobytes())
        subprocess.run(["ffmpeg","-v","error","-y","-f","lavfi","-i","color=c=gray:s=480x270:r=25:d=3",
            "-i",str(wavpath),"-c:v","libx264","-preset","ultrafast","-c:a","aac","-shortest",str(media)],check=True)
        server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler);server.samples=samples;server.media=media;server.audio=wavpath
        timer = threading.Timer(args.seconds, server.shutdown) if args.seconds > 0 else None
        if timer:
            timer.daemon = True
            timer.start()
        print(f"Fixture: http://127.0.0.1:{args.port}/editor.html",flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
        finally:
            if timer: timer.cancel()
            server.server_close()


if __name__ == "__main__": main()
