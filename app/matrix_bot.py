"""A Matrix bot that makes videos when asked.

    python -m app.matrix_bot                 run it
    python -m app.matrix_bot --check-config  say what it would do, and stop

It is a client of the web app, not part of it: everything goes through the
same HTTP API and the same one-at-a-time queue as the page, so a bot that
crashes, loops or gets flooded cannot take the site down with it, and a
video it makes is the video the page would have made.

Configuration is all environment:

    MATRIX_HOMESERVER      https://matrix.example.org
    MATRIX_USER            @ytp:example.org
    MATRIX_PASSWORD        used once; the token it gets is kept (see below)
    MATRIX_ACCESS_TOKEN    ... or give a token and no password at all
    MATRIX_ALLOWED_ROOMS   comma-separated room ids the bot will join and answer in
    MATRIX_ALLOWED_USERS   comma-separated user ids, or *:example.org for a server
    MATRIX_ADMINS          users who may switch the voice for everyone
    MATRIX_PREFIX          default !ytp
    MATRIX_COOLDOWN        seconds between one person's requests, default 20
    MATRIX_MV_MAX_SECONDS  longest song video it will make, default 120
    MATRIX_VOICE_POLL      how often to check which voice is live, default 20s.
                           One corpus serves the whole app, so a switch on the
                           web page is announced in every room the bot is in
    YTP_API_URL            where the web app is, default http://app:8765
    YTP_PUBLIC_URL         the web app as the room can reach it, for links to
                           videos too big for the homeserver to take

With neither allow-list set the bot answers anyone who invites it, and says
so in its log. The login is saved to $MRS_DATA_DIR/matrix/session.json so a
restart does not create a new device every time.

Encrypted rooms are not supported: the bot leaves them rather than sit in a
room where it cannot read a word.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import aiohttp
from nio import (AsyncClient, AsyncClientConfig, InviteMemberEvent, LoginResponse,
                 MatrixRoom, MegolmEvent, RoomMessageAudio, RoomMessageFile,
                 RoomMessageText, UploadResponse)

from app.matrix_commands import DEFAULT_PREFIX, HELP, Command, is_midi, match_part, parse

log = logging.getLogger("app.matrix_bot")

_POLL = 2.0
_STATUS_EVERY = 5.0
_JOB_TIMEOUT = 30 * 60
_STAGE = {"waiting": "waiting", "loading": "loading the voice", "resolving": "finding clips",
          "encoding": "encoding", "joining": "joining", "measuring": "measuring words",
          "samples": "cutting samples", "notes": "singing notes", "frames": "drawing frames",
          "finished": "finishing"}


def _csv(name: str) -> set[str]:
    return {x.strip() for x in os.environ.get(name, "").split(",") if x.strip()}


@dataclass
class Config:
    homeserver: str = ""
    user: str = ""
    password: str = ""
    token: str = ""
    device_name: str = "ytp-machine"
    prefix: str = DEFAULT_PREFIX
    api_url: str = "http://app:8765"
    public_url: str = ""
    allowed_rooms: set[str] = field(default_factory=set)
    allowed_users: set[str] = field(default_factory=set)
    admins: set[str] = field(default_factory=set)
    cooldown: float = 20.0
    mv_max_seconds: float = 120.0
    voice_poll: float = 20.0
    session_path: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        data = os.environ.get("MRS_DATA_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return cls(
            homeserver=os.environ.get("MATRIX_HOMESERVER", "").rstrip("/"),
            user=os.environ.get("MATRIX_USER", ""),
            password=os.environ.get("MATRIX_PASSWORD", ""),
            token=os.environ.get("MATRIX_ACCESS_TOKEN", ""),
            device_name=os.environ.get("MATRIX_DEVICE_NAME", "ytp-machine"),
            prefix=os.environ.get("MATRIX_PREFIX", DEFAULT_PREFIX),
            api_url=os.environ.get("YTP_API_URL", "http://app:8765").rstrip("/"),
            public_url=os.environ.get("YTP_PUBLIC_URL", "").rstrip("/"),
            allowed_rooms=_csv("MATRIX_ALLOWED_ROOMS"),
            allowed_users=_csv("MATRIX_ALLOWED_USERS"),
            admins=_csv("MATRIX_ADMINS"),
            cooldown=float(os.environ.get("MATRIX_COOLDOWN", "20")),
            mv_max_seconds=float(os.environ.get("MATRIX_MV_MAX_SECONDS", "120")),
            voice_poll=max(5.0, float(os.environ.get("MATRIX_VOICE_POLL", "20"))),
            session_path=os.path.join(data, "matrix", "session.json"),
        )

    def problems(self) -> list[str]:
        out = []
        if not self.homeserver:
            out.append("MATRIX_HOMESERVER is not set")
        if not self.user:
            out.append("MATRIX_USER is not set")
        if not (self.password or self.token or os.path.exists(self.session_path)):
            out.append("need MATRIX_PASSWORD or MATRIX_ACCESS_TOKEN (no saved session either)")
        return out

    def user_allowed(self, user: str) -> bool:
        if not self.allowed_users:
            return True
        server = user.split(":", 1)[-1]
        return user in self.allowed_users or f"*:{server}" in self.allowed_users

    def room_allowed(self, room_id: str) -> bool:
        return not self.allowed_rooms or room_id in self.allowed_rooms

    def allowed(self, user: str, room_id: str) -> bool:
        # With both lists set, either is enough: an allowed room is open to
        # everyone in it, and an allowed person may use the bot anywhere.
        if self.allowed_rooms and self.allowed_users:
            return room_id in self.allowed_rooms or self.user_allowed(user)
        return self.room_allowed(room_id) and self.user_allowed(user)


class ApiError(Exception):
    pass


class Api:
    """The web app, over HTTP."""

    def __init__(self, base: str):
        self.base = base
        self.http: aiohttp.ClientSession | None = None

    async def _session(self) -> aiohttp.ClientSession:
        if self.http is None or self.http.closed:
            self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))
        return self.http

    async def _read(self, r: aiohttp.ClientResponse):
        try:
            body = await r.json(content_type=None)
        except (ValueError, aiohttp.ContentTypeError):
            body = None
        if r.status >= 400:
            d = (body or {}).get("detail") if isinstance(body, dict) else None
            msg = d.get("message") if isinstance(d, dict) else d
            raise ApiError(msg or f"the video server said {r.status}")
        return body

    async def get(self, path: str):
        s = await self._session()
        async with s.get(self.base + path) as r:
            return await self._read(r)

    async def post(self, path: str, data: dict):
        s = await self._session()
        async with s.post(self.base + path, json=data) as r:
            return await self._read(r)

    async def upload_midi(self, data: bytes, filename: str):
        s = await self._session()
        form = aiohttp.FormData()
        form.add_field("midi", data, filename=filename or "song.mid", content_type="audio/midi")
        async with s.post(self.base + "/api/ytpmv/midi", data=form) as r:
            return await self._read(r)

    async def fetch(self, url_path: str) -> bytes:
        s = await self._session()
        async with s.get(self.base + url_path) as r:
            if r.status >= 400:
                raise ApiError(f"couldn't fetch the video ({r.status})")
            return await r.read()

    async def close(self):
        if self.http and not self.http.closed:
            await self.http.close()


def _probe(path: str) -> dict:
    """Width, height and duration (ms) of a video, for the m.video info block."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height:format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=30).stdout
        j = json.loads(out)
        st = (j.get("streams") or [{}])[0]
        return {"w": int(st.get("width", 0)), "h": int(st.get("height", 0)),
                "duration": int(float(j.get("format", {}).get("duration", 0)) * 1000)}
    except Exception:  # noqa: BLE001 -- the info block is optional
        return {}


def _thumbnail(path: str) -> bytes | None:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "1", "-i", path, "-frames:v", "1",
             "-vf", "scale=480:-2", "-f", "image2", "-c:v", "mjpeg", "-"],
            capture_output=True, timeout=30)
        return proc.stdout or None
    except Exception:  # noqa: BLE001
        return None


class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api = Api(cfg.api_url)
        self.client = AsyncClient(cfg.homeserver, cfg.user,
                                  config=AsyncClientConfig(store_sync_tokens=False,
                                                           encryption_enabled=False))
        self.busy: set[str] = set()                     # users with a request in flight
        self.last_request: dict[str, float] = {}
        self._warned: set[str] = set()                         # rooms told they are encrypted
        self._voice_asked_in: str | None = None                # room that asked for the last switch
        self.midi_by_event: dict[str, tuple[str, str]] = {}    # event id -> (mxc, filename)
        self.last_midi: dict[str, tuple[str, str]] = {}        # room id -> (mxc, filename)
        self.upload_limit: int | None = None

    # ── session ──────────────────────────────────────────────────────────────

    def _load_session(self) -> dict | None:
        try:
            with open(self.cfg.session_path, encoding="utf-8") as fh:
                s = json.load(fh)
            if s.get("user_id") == self.cfg.user and s.get("homeserver") == self.cfg.homeserver:
                return s
        except (OSError, ValueError):
            pass
        return None

    def _save_session(self) -> None:
        os.makedirs(os.path.dirname(self.cfg.session_path), exist_ok=True)
        tmp = self.cfg.session_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"homeserver": self.cfg.homeserver, "user_id": self.client.user_id,
                       "device_id": self.client.device_id,
                       "access_token": self.client.access_token}, fh)
        os.replace(tmp, self.cfg.session_path)
        try:
            os.chmod(self.cfg.session_path, 0o600)
        except OSError:
            pass

    async def login(self) -> None:
        saved = self._load_session()
        if self.cfg.token:
            self.client.restore_login(self.cfg.user, (saved or {}).get("device_id", ""), self.cfg.token)
        elif saved:
            self.client.restore_login(saved["user_id"], saved["device_id"], saved["access_token"])
        else:
            resp = await self.client.login(self.cfg.password, device_name=self.cfg.device_name)
            if not isinstance(resp, LoginResponse):
                raise SystemExit(f"Matrix login failed: {resp}")
            self._save_session()
            log.info("logged in as %s (device %s); session saved", resp.user_id, resp.device_id)
            return
        who = await self.client.whoami()
        if getattr(who, "user_id", None) is None:
            raise SystemExit(f"Matrix token rejected: {who}. Delete {self.cfg.session_path} "
                             "or give a fresh MATRIX_ACCESS_TOKEN.")
        if getattr(who, "device_id", None):
            self.client.device_id = who.device_id
        log.info("resumed session for %s", who.user_id)

    async def _upload_limit(self) -> int | None:
        """The homeserver's upload cap, asked for once."""
        if self.upload_limit is None:
            try:
                url = f"{self.cfg.homeserver}/_matrix/client/v1/media/config"
                headers = {"Authorization": f"Bearer {self.client.access_token}"}
                s = await self.api._session()
                async with s.get(url, headers=headers) as r:
                    if r.status == 404:
                        async with s.get(f"{self.cfg.homeserver}/_matrix/media/v3/config",
                                         headers=headers) as r2:
                            j = await r2.json(content_type=None)
                    else:
                        j = await r.json(content_type=None)
                self.upload_limit = int(j.get("m.upload.size") or 0) or 0
            except Exception:  # noqa: BLE001
                self.upload_limit = 0
        return self.upload_limit or None

    # ── sending ──────────────────────────────────────────────────────────────

    @staticmethod
    def _reply_to(event_id: str | None) -> dict:
        return {"m.relates_to": {"m.in_reply_to": {"event_id": event_id}}} if event_id else {}

    async def say(self, room_id: str, text: str, reply_to: str | None = None,
                  html: str | None = None) -> str | None:
        content = {"msgtype": "m.notice", "body": text, **self._reply_to(reply_to)}
        if html:
            content.update(format="org.matrix.custom.html", formatted_body=html)
        resp = await self.client.room_send(room_id, "m.room.message", content)
        return getattr(resp, "event_id", None)

    async def edit(self, room_id: str, event_id: str | None, text: str) -> None:
        if not event_id:
            return
        await self.client.room_send(room_id, "m.room.message", {
            "msgtype": "m.notice", "body": f"* {text}",
            "m.new_content": {"msgtype": "m.notice", "body": text},
            "m.relates_to": {"rel_type": "m.replace", "event_id": event_id},
        })

    async def react(self, room_id: str, event_id: str, key: str) -> None:
        await self.client.room_send(room_id, "m.reaction", {
            "m.relates_to": {"rel_type": "m.annotation", "event_id": event_id, "key": key}})

    async def send_video(self, room_id: str, video_url: str, caption: str,
                         reply_to: str | None) -> None:
        data = await self.api.fetch(video_url)
        limit = await self._upload_limit()
        if limit and len(data) > limit:
            if self.cfg.public_url:
                await self.say(room_id, f"{caption}\nToo big to upload here "
                               f"({len(data) / 1e6:.0f} MB): {self.cfg.public_url}{video_url}", reply_to)
            else:
                await self.say(room_id, f"That came out at {len(data) / 1e6:.0f} MB, more than this "
                               "server takes. Try a shorter one (max=30).", reply_to)
            return
        with tempfile.TemporaryDirectory(prefix="ytpbot_") as tmp:
            path = os.path.join(tmp, "video.mp4")
            with open(path, "wb") as fh:
                fh.write(data)
            info = {"mimetype": "video/mp4", "size": len(data), **_probe(path)}
            thumb = _thumbnail(path)
        if thumb:
            tresp, _ = await self.client.upload(lambda *_: thumb, content_type="image/jpeg",
                                                filename="thumb.jpg", filesize=len(thumb))
            if isinstance(tresp, UploadResponse):
                info["thumbnail_url"] = tresp.content_uri
                info["thumbnail_info"] = {"mimetype": "image/jpeg", "size": len(thumb)}
        name = os.path.basename(video_url)
        resp, _ = await self.client.upload(lambda *_: data, content_type="video/mp4",
                                           filename=name, filesize=len(data))
        if not isinstance(resp, UploadResponse):
            link = f" {self.cfg.public_url}{video_url}" if self.cfg.public_url else ""
            await self.say(room_id, f"{caption}\nThe upload failed ({resp}).{link}", reply_to)
            return
        await self.client.room_send(room_id, "m.room.message", {
            "msgtype": "m.video", "body": caption or name, "filename": name,
            "url": resp.content_uri, "info": info, **self._reply_to(reply_to)})

    # ── jobs ─────────────────────────────────────────────────────────────────

    async def wait_for(self, job: dict, room_id: str, status_id: str | None, what: str) -> dict:
        jid = job["id"]
        started = time.monotonic()
        last_note = 0.0
        last_text = ""
        while True:
            j = await self.api.get(f"/api/jobs/{jid}")
            if j["status"] == "done":
                return j["result"]
            if j["status"] == "error":
                raise ApiError(j.get("error") or "it went wrong")
            now = time.monotonic()
            if now - started > _JOB_TIMEOUT:
                raise ApiError("that took far too long, so I gave up waiting")
            if now - last_note >= _STATUS_EVERY:
                if j["status"] == "queued":
                    text = f"⏳ {what}: queued" + (f", {j['position']} ahead" if j.get("position") else "")
                else:
                    stage = _STAGE.get(j.get("stage"), j.get("stage", ""))
                    frac = f" {round(100 * j['done'] / j['total'])}%" if j.get("total") else ""
                    text = f"⏳ {what}: {stage}{frac}"
                if text != last_text:
                    await self.edit(room_id, status_id, text)
                    last_text = text
                last_note = now
            await asyncio.sleep(_POLL)

    # ── commands ─────────────────────────────────────────────────────────────

    async def cmd_say(self, room: MatrixRoom, event, cmd: Command) -> None:
        status = await self.say(room.room_id, f"⏳ saying: {cmd.text[:80]}", event.event_id)
        job = await self.api.post("/api/generate", {"text": cmd.text, "subtitles": cmd.subtitles})
        result = await self.wait_for(job, room.room_id, status, "saying it")
        missing = result.get("missing") or []
        note = f" (couldn't say: {', '.join(missing)})" if missing else ""
        await self.send_video(room.room_id, result["video_url"], cmd.text + note, event.event_id)
        await self.edit(room.room_id, status, f"✅ said it{note}")

    async def _midi_for(self, room: MatrixRoom, event) -> tuple[str, str] | None:
        content = event.source.get("content", {})
        reply = (content.get("m.relates_to") or {}).get("m.in_reply_to", {}).get("event_id")
        if reply:
            if reply in self.midi_by_event:
                return self.midi_by_event[reply]
            got = await self.client.room_get_event(room.room_id, reply)
            ev = getattr(got, "event", None)
            if ev is not None:
                c = ev.source.get("content", {})
                name = c.get("filename") or c.get("body")
                if c.get("url") and is_midi(name, (c.get("info") or {}).get("mimetype")):
                    return c["url"], name
        return self.last_midi.get(room.room_id)

    async def _analyse(self, room: MatrixRoom, event) -> dict:
        found = await self._midi_for(room, event)
        if not found:
            raise ApiError("Post a MIDI file here first (or reply to one), then ask again.")
        mxc, name = found
        resp = await self.client.download(mxc=mxc)
        body = getattr(resp, "body", None)
        if not body:
            raise ApiError(f"Couldn't download {name} from the homeserver.")
        return await self.api.upload_midi(body, name)

    async def cmd_info(self, room: MatrixRoom, event, cmd: Command) -> None:
        song = await self._analyse(room, event)
        lines = [f"🎼 {song.get('title') or 'that song'}: {song['description']}"]
        for p in song["parts"]:
            s = p["settings"]
            octave = f" (octave {s['octave']:+d})" if s.get("octave") else ""
            role = p["role"].split(":")[-1]
            lines.append(f"  • {role:8} {p['instrument']:22} → {s.get('text') or '—'}{octave}")
        lines.append(f"Change one with e.g. {self.cfg.prefix} mv lead=yeah kick=boom")
        await self.say(room.room_id, "\n".join(lines), event.event_id,
                       html="<pre>" + "\n".join(lines).replace("&", "&amp;").replace("<", "&lt;") + "</pre>")

    async def cmd_mv(self, room: MatrixRoom, event, cmd: Command) -> None:
        if cmd.errors:
            await self.say(room.room_id, "Didn't understand: " + "; ".join(cmd.errors), event.event_id)
            return
        song = await self._analyse(room, event)
        parts, unknown = [], []
        for name, settings in cmd.parts.items():
            p = match_part(name, song["parts"])
            if p is None:
                unknown.append(name)
            else:
                parts.append({"id": p["id"], **settings})
        if unknown:
            names = ", ".join(sorted({p["role"].split(":")[-1] for p in song["parts"]}))
            await self.say(room.room_id, f"This song has no part called {', '.join(unknown)}. "
                           f"Its parts: {names}.", event.event_id)
            return
        options = {"max_seconds": self.cfg.mv_max_seconds, **cmd.options}
        options["max_seconds"] = min(float(options["max_seconds"]), self.cfg.mv_max_seconds)
        title = song.get("title") or "the song"
        status = await self.say(room.room_id, f"⏳ singing {title}: {song['description']}",
                                event.event_id)
        job = await self.api.post("/api/ytpmv/render",
                                  {"midi_id": song["midi_id"], "parts": parts, "options": options})
        result = await self.wait_for(job, room.room_id, status, f"singing {title}")
        words = ", ".join(f"{p['name'].split(' (')[0]}: {p['text']}" for p in result["parts"]
                          if p.get("text") and not p.get("mute"))
        await self.send_video(room.room_id, result["video_url"], f"{title} — {words}", event.event_id)
        probs = "; ".join(f"{p['name']}: {p['error']}" for p in result.get("problems", []))
        await self.edit(room.room_id, status, "✅ done" + (f" (left out {probs})" if probs else ""))

    async def cmd_voices(self, room: MatrixRoom, event, cmd: Command) -> None:
        data = await self.api.get("/api/corpora")
        lines = [f"{'▶' if c['active'] else ' '} {c['slug']} — {c.get('words', '?')} words"
                 for c in data["corpora"]]
        tail = f"\n{self.cfg.prefix} voice <name> to switch" + (" (admins only)" if self.cfg.admins else "")
        await self.say(room.room_id, "Voices:\n" + "\n".join(lines) + tail, event.event_id)

    async def cmd_voice(self, room: MatrixRoom, event, cmd: Command) -> None:
        # Switching the voice switches it for everyone, the web page included,
        # so it is not something any passer-by in a public room gets to do.
        if self.cfg.admins and event.sender not in self.cfg.admins:
            await self.say(room.room_id, "Only an admin can switch the voice.", event.event_id)
            return
        # Every other room hears about this from the watcher below; this one
        # is getting a direct answer, so it does not need telling twice.
        self._voice_asked_in = room.room_id
        data = await self.api.post("/api/corpus", {"slug": cmd.text.strip()})
        await self.say(room.room_id, f"Now speaking as {data.get('name', cmd.text)} "
                       f"({data.get('words', '?')} words).", event.event_id)

    async def cmd_queue(self, room: MatrixRoom, event, cmd: Command) -> None:
        q = await self.api.get("/api/queue")
        await self.say(room.room_id, f"{q['running']} running, {q['queued']} waiting.", event.event_id)

    async def watch_voice(self) -> None:
        """Tell every room when the voice changes, whoever changed it.

        One corpus is live at a time for the whole app, so somebody switching
        it on the web page changes what the bot says in every room it sits in.
        That used to happen silently, and the next video came back in a voice
        nobody in the room had asked for.

        Polled rather than pushed: the app has nothing to push with, and one
        small request every MATRIX_VOICE_POLL seconds is cheaper than building
        it a way to.
        """
        last: tuple | None = None
        while True:
            try:
                s = await self.api.get("/api/stats")
                now = (s.get("corpus"), s.get("corpus_name"))
                if last is not None and now != last:
                    asked_in, self._voice_asked_in = self._voice_asked_in, None
                    text = (f"The voice is now {s.get('corpus_name')} "
                            f"({s.get('unique_words')} words, {s.get('total_clips')} clips).")
                    for room_id in list(self.client.rooms):
                        if room_id != asked_in:
                            await self.say(room_id, text)
                    log.info("voice changed to %s; told %d room(s)", now[1], len(self.client.rooms))
                last = now
            except (ApiError, aiohttp.ClientError) as exc:
                log.debug("voice check failed: %s", exc)
            except Exception:  # noqa: BLE001 -- this loop outlives everything else
                log.exception("voice watcher stumbled")
            await asyncio.sleep(self.cfg.voice_poll)

    # ── events ───────────────────────────────────────────────────────────────

    def _is_direct(self, room: MatrixRoom) -> bool:
        return room.member_count <= 2

    async def on_invite(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        if event.state_key != self.client.user_id or event.membership != "invite":
            return
        if not self.cfg.allowed(event.sender, room.room_id):
            log.info("ignoring invite to %s from %s (not allowed)", room.room_id, event.sender)
            return
        resp = await self.client.join(room.room_id)
        log.info("joined %s at %s's invite: %s", room.room_id, event.sender, type(resp).__name__)
        joined = self.client.rooms.get(room.room_id)
        if joined is not None and joined.encrypted:
            log.warning("left %s: it is encrypted, and this bot cannot read encrypted rooms", room.room_id)
            await self.client.room_leave(room.room_id)
            return
        await self.say(room.room_id, f"Hello! {self.cfg.prefix} help for what I can do.")

    async def on_file(self, room: MatrixRoom, event) -> None:
        if event.sender == self.client.user_id:
            return
        content = event.source.get("content", {})
        name = content.get("filename") or content.get("body") or ""
        mime = (content.get("info") or {}).get("mimetype")
        if not content.get("url") or not is_midi(name, mime):
            return
        self.midi_by_event[event.event_id] = (content["url"], name)
        self.last_midi[room.room_id] = (content["url"], name)
        if len(self.midi_by_event) > 500:
            for k in list(self.midi_by_event)[:250]:
                del self.midi_by_event[k]
        # A file sent with a caption (Matrix 1.10): body is the caption, filename the name.
        caption = content.get("body") if content.get("filename") and content.get("body") != name else None
        if caption:
            await self.on_text(room, event, caption)

    async def on_encrypted(self, room: MatrixRoom, event: MegolmEvent) -> None:
        """Say, once per room, that it cannot read a word in here.

        An encrypted room looks exactly like an idle one from in here: every
        message arrives as ciphertext this bot has no key for, so it answers
        nothing and there is nothing in the log either. Saying so out loud
        beats leaving somebody typing !ytp at a bot that cannot hear them.
        Encryption cannot be turned off once a room has it on, so the answer
        is always a different room.
        """
        if event.sender == self.client.user_id or room.room_id in self._warned:
            return
        self._warned.add(room.room_id)
        log.warning("%s is encrypted; this bot cannot read it", room.room_id)
        await self.say(room.room_id,
                       "I can't read this room -- it's end-to-end encrypted, and I have no "
                       "keys for it. Make an unencrypted room and invite me there instead.")

    async def on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        await self.on_text(room, event, event.body)

    async def on_text(self, room: MatrixRoom, event, body: str) -> None:
        if event.sender == self.client.user_id:
            return
        cmd = parse(body, self.cfg.prefix, direct=self._is_direct(room))
        if cmd is None:
            return
        if not self.cfg.allowed(event.sender, room.room_id):
            return
        if cmd.name == "help":
            extra = ("\n" + "; ".join(cmd.errors)) if cmd.errors else ""
            await self.say(room.room_id, HELP.format(p=self.cfg.prefix) + extra, event.event_id)
            return
        if cmd.name == "unknown":
            await self.say(room.room_id, f"I don't know {cmd.text!r} — try {self.cfg.prefix} help",
                           event.event_id)
            return
        handler = {"say": self.cmd_say, "mv": self.cmd_mv, "info": self.cmd_info,
                   "voices": self.cmd_voices, "voice": self.cmd_voice,
                   "queue": self.cmd_queue}[cmd.name]
        heavy = cmd.name in ("say", "mv")
        if heavy:
            if event.sender in self.busy:
                await self.say(room.room_id, "Still working on your last one — hang on.", event.event_id)
                return
            wait = self.cfg.cooldown - (time.monotonic() - self.last_request.get(event.sender, -1e9))
            if wait > 0:
                await self.say(room.room_id, f"Give it {wait:.0f}s.", event.event_id)
                return
            self.busy.add(event.sender)
            self.last_request[event.sender] = time.monotonic()
        asyncio.create_task(self._run(handler, room, event, cmd, heavy))

    async def _run(self, handler, room, event, cmd, heavy) -> None:
        try:
            if heavy:
                await self.react(room.room_id, event.event_id, "👀")
            await handler(room, event, cmd)
        except ApiError as exc:
            await self.say(room.room_id, f"❌ {exc}", event.event_id)
        except aiohttp.ClientError as exc:
            log.warning("API unreachable: %s", exc)
            await self.say(room.room_id, "❌ The video server isn't answering right now.", event.event_id)
        except Exception:  # noqa: BLE001 -- one bad request must not stop the bot
            log.exception("command %s failed", cmd.name)
            await self.say(room.room_id, "❌ Something broke. It's in the log.", event.event_id)
        finally:
            if heavy:
                self.busy.discard(event.sender)

    # ── main loop ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.login()
        if not (self.cfg.allowed_rooms or self.cfg.allowed_users):
            log.warning("no MATRIX_ALLOWED_ROOMS or MATRIX_ALLOWED_USERS: answering anyone who invites me")
        # Catch up without answering: anything sent while the bot was down is
        # history, and replying to a morning's worth of commands at once is
        # worse than replying to none.
        first = await self.client.sync(timeout=0, full_state=True)
        log.info("synced (%s); listening for %s", type(first).__name__, self.cfg.prefix)
        # Invites that arrived while it was down are still worth accepting.
        for room_id, room in list(self.client.invited_rooms.items()):
            inviter = getattr(room, "inviter", None) or ""
            if self.cfg.allowed(inviter, room_id):
                await self.client.join(room_id)
                log.info("joined %s (invited by %s while offline)", room_id, inviter)
        self.client.add_event_callback(self.on_message, RoomMessageText)
        self.client.add_event_callback(self.on_file, (RoomMessageFile, RoomMessageAudio))
        self.client.add_event_callback(self.on_invite, InviteMemberEvent)
        self.client.add_event_callback(self.on_encrypted, MegolmEvent)
        watcher = asyncio.create_task(self.watch_voice())

        # Stop when systemd or podman says stop. Without this, sync_forever
        # sits through SIGTERM and every restart waited the full ten seconds
        # for the SIGKILL, leaving the unit "failed" each time it was asked
        # politely to stop.
        syncing = asyncio.current_task()
        loop = asyncio.get_running_loop()
        for sig in ("SIGTERM", "SIGINT"):
            try:
                loop.add_signal_handler(getattr(signal, sig), syncing.cancel)
            except (AttributeError, NotImplementedError):   # Windows has neither
                pass
        try:
            await self.client.sync_forever(timeout=30000, full_state=False)
        except asyncio.CancelledError:
            log.info("asked to stop")
        finally:
            watcher.cancel()
            await self.api.close()
            await self.client.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="The YTP Machine's Matrix bot.")
    ap.add_argument("--check-config", action="store_true",
                    help="print the configuration and whether the web app answers, then stop")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("nio").setLevel(logging.WARNING)
    cfg = Config.from_env()

    if args.check_config:
        print(f"homeserver     {cfg.homeserver or '(unset)'}")
        print(f"user           {cfg.user or '(unset)'}")
        print(f"login          {'token' if cfg.token else 'password' if cfg.password else 'saved session' if os.path.exists(cfg.session_path) else '(none)'}")
        print(f"session file   {cfg.session_path}")
        print(f"prefix         {cfg.prefix}")
        print(f"rooms          {', '.join(sorted(cfg.allowed_rooms)) or 'any'}")
        print(f"users          {', '.join(sorted(cfg.allowed_users)) or 'anyone'}")
        print(f"admins         {', '.join(sorted(cfg.admins)) or 'anyone may switch voice'}")
        print(f"cooldown       {cfg.cooldown:.0f}s   song max {cfg.mv_max_seconds:.0f}s")
        print(f"api            {cfg.api_url}")

        async def ping():
            api = Api(cfg.api_url)
            try:
                s = await api.get("/api/stats")
                return f"answers — voice {s.get('corpus_name')}, {s.get('unique_words')} words"
            except Exception as exc:  # noqa: BLE001
                return f"NOT answering ({exc})"
            finally:
                await api.close()
        print(f"               {asyncio.run(ping())}")
        for p in cfg.problems():
            print(f"problem        {p}")
        return 1 if cfg.problems() else 0

    problems = cfg.problems()
    if problems:
        for p in problems:
            log.error(p)
        return 2
    try:
        asyncio.run(Bot(cfg).run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
