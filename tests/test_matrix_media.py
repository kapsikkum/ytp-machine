"""Bounded Matrix media and shutdown, without a homeserver or messages."""
import asyncio
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import matrix_bot as mb


class Response:
    def __init__(self, chunks=(), length=None, status=200):
        self.chunks, self.content_length, self.status = chunks, length, status
        self.content = self
        self.consumed = 0
        self.closed = False

    async def iter_chunked(self, size):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True


class Session:
    closed = False

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return next(self.responses)


def bot_with_session(session):
    bot = object.__new__(mb.Bot)
    bot.cfg = mb.Config(homeserver="https://homeserver.example", api_url="http://video.example")
    bot.client = SimpleNamespace(user_id="@bot:example", access_token="fixture-token", close=AsyncMock())
    bot.api = mb.Api(bot.cfg.api_url)
    bot.api.http = session
    bot.busy, bot.last_request, bot._tasks = set(), {}, set()
    bot.midi_by_event, bot.last_midi = {}, {}
    bot.say, bot.edit = AsyncMock(), AsyncMock()
    return bot


class MatrixMediaTests(unittest.IsolatedAsyncioTestCase):
    async def test_content_length_refuses_before_reading(self):
        response = Response([b"ignored"], length=100)
        with self.assertRaises(mb.MediaTooLarge):
            await mb._bounded_media(response, 8)
        self.assertEqual(response.consumed, 0)

    async def test_chunked_limit_stops_before_later_chunks(self):
        response = Response([b"12345", b"67890", b"never read"])
        with self.assertRaises(mb.MediaTooLarge):
            await mb._bounded_media(response, 8)
        self.assertEqual(response.consumed, 2)

    async def test_within_limit_is_byte_identical(self):
        self.assertEqual(await mb._bounded_media(Response([b"abc", b"def"]), 6), b"abcdef")

    async def test_midi_routes_via_homeserver_with_finite_timeout(self):
        response = Response([b"MThd"])
        session = Session(response)
        bot = bot_with_session(session)
        self.assertEqual(await bot._download_midi("mxc://untrusted.example/song"), b"MThd")
        url, options = session.requests[0]
        self.assertTrue(url.startswith(bot.cfg.homeserver + "/_matrix/client/v1/media/download/"))
        self.assertEqual(options["timeout"].total, 60)
        self.assertFalse(options["allow_redirects"])
        self.assertEqual(options["headers"], {"Authorization": "Bearer fixture-token"})
        self.assertNotIn("fixture-token", url)
        self.assertTrue(response.closed)

    async def test_legacy_homeserver_fallback(self):
        legacy, valid = Response(status=404), Response([b"midi"])
        session = Session(legacy, valid)
        self.assertEqual(await bot_with_session(session)._download_midi("mxc://server/song"), b"midi")
        self.assertEqual(len(session.requests), 2)
        self.assertIn("/_matrix/media/v3/download/", session.requests[1][0])
        self.assertTrue(legacy.closed and valid.closed)

    async def test_media_redirect_refused(self):
        response = Response(status=302)
        session = Session(response)
        with self.assertRaisesRegex(mb.ApiError, "302"):
            await bot_with_session(session)._download_midi("mxc://server/song")
        self.assertEqual(len(session.requests), 1)
        self.assertTrue(response.closed)

    async def test_midi_limit_applies_without_content_length(self):
        response = Response([b"12345", b"67890", b"extra"])
        bot = bot_with_session(Session(response))
        with patch.object(mb, "_MAX_MIDI_BYTES", 8), self.assertRaises(mb.MediaTooLarge):
            await bot._download_midi("mxc://server/song")
        self.assertEqual(response.consumed, 2)
        self.assertTrue(response.closed)

    async def test_malformed_mxc_never_downloaded(self):
        session = Session()
        bot = bot_with_session(session)
        for mxc in ("https://evil.example/song", "mxc://user:pass@server/song", "mxc://server/a/b", "mxc://[invalid/song"):
            with self.subTest(mxc=mxc), self.assertRaises(mb.ApiError):
                await bot._download_midi(mxc)
        self.assertEqual(session.requests, [])

    async def test_video_fetch_limit_closes_response(self):
        response = Response([b"12345", b"67890"])
        bot = bot_with_session(Session(response))
        with self.assertRaises(mb.MediaTooLarge):
            await bot.api.fetch("/output/video.mp4", max_bytes=8)
        self.assertTrue(response.closed)

    async def test_video_cap_checked_before_fetch_and_link_fallback(self):
        bot = bot_with_session(Session())
        bot.cfg.public_url = "https://public.example"
        bot._upload_limit = AsyncMock(return_value=8)
        bot.api.fetch = AsyncMock(side_effect=mb.MediaTooLarge())
        await bot.send_video("room", "/output/a.mp4", "video", "event")
        bot.api.fetch.assert_awaited_once_with("/output/a.mp4", max_bytes=8)
        self.assertIn("https://public.example/output/a.mp4", bot.say.await_args.args[1])

    async def test_unauthorized_file_cannot_replace_latest_midi(self):
        bot = bot_with_session(Session())
        bot.cfg.allowed_users = {"@allowed:example"}
        bot.last_midi["room"] = ("mxc://server/trusted", "trusted.mid")
        event = SimpleNamespace(sender="@other:example", event_id="event", source={"content": {
            "filename": "untrusted.mid", "url": "mxc://server/untrusted", "body": "!ytp mv"}})
        await bot.on_file(SimpleNamespace(room_id="room"), event)
        self.assertEqual(bot.midi_by_event, {})
        self.assertEqual(bot.last_midi["room"][1], "trusted.mid")
        bot.say.assert_not_awaited()

    async def test_cancelled_job_is_terminal(self):
        bot = bot_with_session(Session())
        bot.api.get = AsyncMock(return_value={"status": "cancelled"})
        with self.assertRaisesRegex(mb.ApiError, "cancelled"):
            await bot.wait_for({"id": "job"}, "room", None, "video")
        bot.api.get.assert_awaited_once()

    async def test_shutdown_cancels_and_awaits_commands_even_on_login_failure(self):
        bot = bot_with_session(Session())
        bot.api.close = AsyncMock()
        bot._listen = AsyncMock(side_effect=RuntimeError("login fixture failure"))
        cancelled = asyncio.Event()
        async def running():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        task = asyncio.create_task(running())
        bot._tasks.add(task)
        await asyncio.sleep(0)
        with self.assertRaises(RuntimeError):
            await bot.run()
        self.assertTrue(task.done() and cancelled.is_set())
        bot.api.close.assert_awaited_once()
        bot.client.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
