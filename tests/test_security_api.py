"""Network-free API origin, callback, maintenance and cleanup regressions."""
import asyncio
import io
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import webhooks


def address(ip):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return family, socket.SOCK_STREAM, 6, "", (ip, 443, 0, 0) if family == socket.AF_INET6 else (ip, 443)


class WebhookTests(unittest.TestCase):
    def test_private_mixed_and_special_addresses_refused(self):
        for ips in (["127.0.0.1"], ["10.0.0.1"], ["169.254.169.254"], ["::1"],
                    ["::ffff:127.0.0.1"], ["224.0.0.1"], ["93.184.216.34", "192.168.1.2"]):
            with self.subTest(ips=ips), patch.object(webhooks.socket, "getaddrinfo", return_value=list(map(address, ips))):
                with self.assertRaises(ValueError):
                    webhooks.validate("https://callback.example/hook")

    def test_ambiguous_urls_refused_before_dns(self):
        for url in ("file:///secret", "https://user:pass@example.org/hook", "https://example.org/#frag",
                    "https://example.org\\@localhost", "https://example.org/\r\nHost: localhost",
                    "https://[fe80::1%25eth0]/", "https://faß.de/hook"):
            with self.subTest(url=url), patch.object(webhooks.socket, "getaddrinfo") as dns:
                with self.assertRaises(ValueError):
                    webhooks.validate(url)
                dns.assert_not_called()

    def test_connection_uses_validated_ip_without_dns_lookup(self):
        sock = MagicMock()
        with patch.object(webhooks.socket, "getaddrinfo", side_effect=[[address("93.184.216.34")],
                                                                     [address("127.0.0.1")]]) as dns:
            target = webhooks.validate("https://callback.example/hook")
            with patch.object(webhooks.socket, "socket", return_value=sock):
                self.assertIs(webhooks._connect(target, 15), sock)
        self.assertEqual(dns.call_count, 1)
        sock.connect.assert_called_once_with(("93.184.216.34", 443))

    def test_https_keeps_hostname_verification_and_sni(self):
        target = webhooks.Target("https", "callback.example", 443, "/", (address("93.184.216.34"),))
        context = MagicMock()
        sock = MagicMock()
        with patch.object(webhooks, "_connect", return_value=sock), patch.object(webhooks.ssl, "create_default_context", return_value=context):
            conn = webhooks._HTTPS(target)
            conn.connect()
        context.wrap_socket.assert_called_once_with(sock, server_hostname="callback.example")

    def test_redirect_not_followed_and_connections_closed(self):
        conn = MagicMock()
        conn.getresponse.return_value.status = 302
        with patch.object(webhooks.socket, "getaddrinfo", return_value=[address("93.184.216.34")]) as dns, \
             patch.object(webhooks, "_HTTP", return_value=conn):
            with self.assertRaisesRegex(OSError, "302"):
                webhooks.deliver("http://callback.example/hook", {})
        self.assertEqual(dns.call_count, 1)
        conn.request.assert_called_once()
        conn.getresponse.return_value.close.assert_called_once()
        conn.close.assert_called_once()


class ApiSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="mrs-api-security-")
        cls.root = Path(cls.tmp.name)
        cls.env = patch.dict(os.environ, {"MRS_DATA_DIR": str(cls.root), "MRS_CORPUS": "a"})
        cls.env.start()
        from app import database, api, jobs
        cls.db, cls.api, cls.jobs = database, api, jobs
        cls.paths = patch.multiple(database, DATA_DIR=str(cls.root), CORPORA_DIR=str(cls.root / "corpora"),
                                   LEGACY_DB=str(cls.root / "legacy.db"), _active=None, _ready=set())
        cls.paths.start()
        for slug in ("a", "b"):
            with database.using_corpus(database._corpus_entry(slug)):
                database.init_db()
        database.set_active("a")
        (cls.root / "frontend").mkdir()
        cwd = os.getcwd()
        try:
            os.chdir(cls.root)
            # Importing main must not launch a sweeper against the real output.
            with patch("threading.Thread.start"):
                import main
            cls.main = main
        finally:
            os.chdir(cwd)
        from fastapi.testclient import TestClient
        cls.web = TestClient(main.app)

        @main.app.post("/api/security-context-probe")
        def probe():
            before = database.active()["slug"]
            database.set_active("b")
            with database.get_db() as conn:
                conn.execute("INSERT OR REPLACE INTO settings VALUES ('security_probe', 'a')")
            return {"before": before, "after": database.active()["slug"]}

        @main.app.get("/api/security-context-async")
        async def async_probe():
            before = database.active()["slug"]
            await asyncio.to_thread(database.set_active, "b")
            return {"before": before, "after": database.active()["slug"]}
        for route in list(main.app.router.routes):
            if getattr(route, "path", "").startswith("/api/security-context"):
                main.app.router.routes.remove(route)
                main.app.router.routes.insert(0, route)

    @classmethod
    def tearDownClass(cls):
        cls.web.close()
        cls.paths.stop()
        cls.env.stop()
        cls.tmp.cleanup()

    def setUp(self):
        self.db.set_active("a")
        self.jobs.end_corpus_maintenance()
        self.worker = patch.object(self.jobs, "_ensure_worker")
        self.worker.start()
        with self.jobs._lock:
            self.jobs._jobs.clear()
            self.jobs._queue.clear()
            self.jobs._order.clear()

    def tearDown(self):
        self.jobs.end_corpus_maintenance()
        self.worker.stop()

    def test_other_browser_origins_refused_and_local_write_accepted(self):
        for headers in ({"sec-fetch-site": "same-site"}, {"sec-fetch-site": "cross-site"},
                        {"origin": "http://testserver:9999"}, {"origin": "null"}):
            self.assertEqual(self.web.post("/api/reload", headers=headers).status_code, 403)
        self.assertEqual(self.web.post("/api/reload", headers={"origin": "http://testserver"}).status_code, 200)

    def test_https_writes_behind_explicitly_trusted_proxy(self):
        from fastapi.testclient import TestClient
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

        proxied = ProxyHeadersMiddleware(self.main.app, trusted_hosts="127.0.0.1,10.88.0.1")
        headers = {"host": "ytp.example", "origin": "https://ytp.example",
                   "sec-fetch-site": "same-origin", "x-forwarded-proto": "https"}
        # Match nginx -> host loopback -> Podman bridge. The route's missing
        # MIDI yields 422 only after the origin guard accepts the request.
        with TestClient(proxied, client=("10.88.0.1", 1234)) as web:
            for endpoint in ("/api/ytpmv/preview-mix", "/api/ytpmv/sample", "/api/generate"):
                self.assertEqual(web.post(endpoint, headers=headers, json={}).status_code, 422)
            for changed in ({"origin": "https://foreign.example"},
                            {"origin": "https://ytp.example:9999"},
                            {"sec-fetch-site": "cross-site"},
                            {"x-forwarded-proto": "http"}):
                self.assertEqual(web.post("/api/ytpmv/preview-mix",
                                          headers={**headers, **changed}, json={}).status_code, 403)
        # Forwarded headers do not let a direct, untrusted client change the
        # scheme used for origin validation.
        with TestClient(proxied, client=("192.0.2.10", 1234)) as web:
            self.assertEqual(web.post("/api/ytpmv/preview-mix", headers=headers, json={}).status_code, 403)

    def test_small_request_budgets_apply_before_parsing(self):
        for endpoint, size in (("/api/ytpmv/midi", 4 << 20), ("/api/generate", 2 << 20)):
            response = self.web.post(endpoint, content=b"x", headers={"content-length": str(size)})
            self.assertEqual(response.status_code, 413)
        response = self.web.post("/api/corpus/import", content=b"x", headers={"content-length": str(4 << 20)})
        self.assertEqual(response.status_code, 415)  # larger bundle budget still requires a multipart file

    def test_urlencoded_upload_form_refused_before_dependency_parser(self):
        for endpoint in ("/api/corpus/import", "/api/ytpmv/midi"):
            response = self.web.post(endpoint, content=b"name=x&" * 5000,
                                     headers={"content-type": "application/x-www-form-urlencoded"})
            self.assertEqual(response.status_code, 415)

    def test_threadpool_write_remains_in_original_corpus(self):
        response = self.web.post("/api/security-context-probe")
        self.assertEqual(response.json(), {"before": "a", "after": "a"})
        self.assertEqual(self.db.active()["slug"], "b")
        for slug, expected in (("a", True), ("b", False)):
            with self.db.using_corpus(slug), self.db.get_db() as conn:
                row = conn.execute("SELECT value FROM settings WHERE key='security_probe'").fetchone()
            self.assertEqual(row is not None, expected)

    def test_async_request_remains_in_original_corpus(self):
        self.assertEqual(self.web.get("/api/security-context-async").json(), {"before": "a", "after": "a"})
        self.assertEqual(self.db.active()["slug"], "b")

    def test_stale_editor_write_refused_and_explicit_voice_read_allowed(self):
        self.assertEqual(self.web.post("/api/reload", headers={"x-corpus": "b"}).status_code, 409)
        response = self.web.get("/api/stats", headers={"x-corpus": "b"})
        self.assertEqual(response.json()["corpus"], "b")
        self.assertEqual(self.db.active()["slug"], "a")

    def test_busy_jobs_and_alignment_prevent_switch(self):
        from app import editor
        for jobs_busy, align_busy in ((True, False), (False, True)):
            with patch.object(self.jobs, "has_live_jobs", return_value=jobs_busy), \
                 patch.object(editor, "alignment_running", return_value=align_busy):
                self.assertEqual(self.web.post("/api/corpus", json={"slug": "b"}).status_code, 409)
                self.assertEqual(self.db.active()["slug"], "a")

    def test_maintenance_blocks_mutations_but_not_reads(self):
        self.jobs.begin_corpus_maintenance()
        self.assertEqual(self.web.post("/api/generate", json={"text": "hello"}).status_code, 409)
        self.assertEqual(self.web.get("/api/queue").status_code, 200)
        self.assertEqual(self.web.get("/api/stats").status_code, 200)

    def test_switch_initializes_and_reports_chosen_voice(self):
        initialized = []
        with patch.object(self.api, "init_db", side_effect=lambda: initialized.append(self.db.active()["slug"])):
            response = self.web.post("/api/corpus", json={"slug": "b"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active"], "b")
        self.assertEqual(initialized, ["b"])
        self.assertEqual(self.db.active()["slug"], "b")

    def test_maintenance_waits_for_already_admitted_write(self):
        from fastapi import HTTPException
        admitted, release = threading.Event(), threading.Event()
        def write():
            with self.api.request_mutation():
                admitted.set()
                release.wait(3)
        worker = threading.Thread(target=write)
        worker.start()
        self.assertTrue(admitted.wait(3))
        try:
            with self.assertRaises(HTTPException) as caught:
                self.api._begin_maintenance()
            self.assertEqual(caught.exception.status_code, 409)
            self.assertFalse(self.jobs.maintenance_in_progress())
        finally:
            release.set()
            worker.join(3)
        self.assertEqual(self.api._active_mutations, 0)

    def test_private_webhook_rejected_before_jobs_queued(self):
        with patch.object(webhooks.socket, "getaddrinfo", return_value=[address("127.0.0.1")]), \
             patch.object(self.jobs, "submit_batch") as submit:
            response = self.web.post("/api/generate/batch", json={"sentences": ["hello"], "webhook_url": "http://local/hook"})
        self.assertEqual(response.status_code, 400)
        submit.assert_not_called()

    def test_cancelled_blocking_request_reports_conflict(self):
        job = self.jobs.Job("cancelled", "hello", status="cancelled")
        job.ended.set()
        with patch.object(self.jobs, "submit", return_value=job):
            response = self.web.post("/api/generate?wait=1", json={"text": "hello"})
        self.assertEqual(response.status_code, 409)

    def test_failed_export_removes_partial_archive_and_releases_reservation(self):
        paths = []
        def pack(slug, path):
            paths.extend([path, path.removesuffix(".tar.zst") + ".tar.gz"])
            for candidate in paths:
                Path(candidate).write_bytes(b"partial")
            raise RuntimeError("bad archive")
        with patch.object(self.api, "_corpus_pack", return_value=type("Pack", (), {"pack_bundle": staticmethod(pack)})):
            self.assertEqual(self.web.get("/api/corpus/export").status_code, 500)
        self.assertTrue(all(not Path(path).exists() for path in paths))
        self.assertFalse(self.jobs.maintenance_in_progress())

    def test_import_runs_off_event_loop_closes_upload_and_hides_server_paths(self):
        from starlette.datastructures import UploadFile
        upload = UploadFile(io.BytesIO(b"archive"), filename="voice.tar")
        installed_on = []
        def install(*args):
            installed_on.append(threading.get_ident())
            return {"slug": "new", "target": "private/server/path", "backup": "private/backup/path"}
        with patch.object(self.api, "_corpus_pack", return_value=type("Pack", (), {"install_bundle": staticmethod(install)})):
            result = asyncio.run(self.api.import_corpus("new", upload, False))
        self.assertNotEqual(installed_on[0], threading.get_ident())
        self.assertEqual(result, {"status": "ok", "slug": "new", "backup_saved": True})
        self.assertTrue(upload.file.closed)
        self.assertFalse(self.jobs.maintenance_in_progress())

    def test_systemexit_import_is_an_error_and_upload_is_closed(self):
        from fastapi import HTTPException
        from starlette.datastructures import UploadFile
        upload = UploadFile(io.BytesIO(b"archive"), filename="voice.tar")
        pack = MagicMock()
        pack.install_bundle.side_effect = SystemExit("private/server/path")
        with patch.object(self.api, "_corpus_pack", return_value=pack):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(self.api.import_corpus("new", upload, False))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertNotIn("private", caught.exception.detail)
        self.assertTrue(upload.file.closed)
        self.assertFalse(self.jobs.maintenance_in_progress())

    def test_http_import_excludes_own_mutation_and_clears_reservation(self):
        pack = MagicMock()
        def install(*args):
            self.assertTrue(self.jobs.maintenance_in_progress())
            return {"slug": "new", "target": "private/path"}
        pack.install_bundle.side_effect = install
        with patch.object(self.api, "_corpus_pack", return_value=pack):
            response = self.web.post("/api/corpus/import", data={"name": "new"},
                                     files={"bundle": ("new.tar", b"archive")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.api._active_mutations, 0)
        self.assertFalse(self.jobs.maintenance_in_progress())


if __name__ == "__main__":
    unittest.main()
