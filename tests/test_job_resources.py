"""Queue reservations and cancellation use bounded, isolated resources."""

import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import database, generate, jobs


class JobResourcesTests(unittest.TestCase):
    def setUp(self):
        self.worker = patch.object(jobs, "_ensure_worker")
        self.worker_mock = self.worker.start()
        with jobs._lock:
            jobs._jobs.clear()
            jobs._order.clear()
            jobs._queue.clear()
            jobs._maintenance = False

    def tearDown(self):
        self.worker.stop()
        with jobs._lock:
            jobs._jobs.clear()
            jobs._order.clear()
            jobs._queue.clear()

    def test_cancelled_reservations_do_not_accumulate(self):
        for _ in range(1000):
            job = jobs.submit("test", corpus="")
            self.assertTrue(jobs.cancel(job.id))
        self.assertEqual(len(jobs._queue), 0)
        self.assertLessEqual(len(jobs._jobs), jobs._KEEP)
        self.assertFalse(jobs.has_live_jobs())

    def test_batch_is_all_or_nothing_and_isolated(self):
        for _ in range(jobs.MAX_QUEUED - 1):
            jobs.submit("waiting", corpus="")
        before = list(jobs._queue)
        with patch.object(database, "active", return_value={"slug": "voice"}):
            with self.assertRaises(jobs.QueueFull):
                jobs.submit_batch(["one", "two"])
        self.assertEqual(list(jobs._queue), before)
        params = {"parts": [{"text": "original"}]}
        job = jobs.submit_ytpmv(params, corpus="")
        params["parts"][0]["text"] = "changed"
        self.assertEqual(job.params["parts"][0]["text"], "original")

    def test_worker_start_failure_rolls_back_reservation(self):
        self.worker_mock.side_effect = RuntimeError("cannot start thread")
        with self.assertRaisesRegex(RuntimeError, "cannot start thread"):
            jobs.submit("test", corpus="")
        self.assertEqual((len(jobs._queue), len(jobs._jobs), len(jobs._order)), (0, 0, 0))

    def test_corpus_guard_blocks_registration_until_maintenance_ends(self):
        entered = threading.Event()
        complete = threading.Event()

        def submit():
            entered.set()
            jobs.submit("test", corpus="")
            complete.set()

        with jobs.corpus_guard():
            thread = threading.Thread(target=submit)
            thread.start()
            self.assertTrue(entered.wait(1))
            self.assertFalse(complete.wait(0.05))
            self.assertFalse(jobs.has_live_jobs())
        thread.join(1)
        self.assertTrue(complete.is_set())
        self.assertTrue(jobs.has_live_jobs())

    def test_maintenance_reservation_blocks_submit_without_holding_lock(self):
        jobs.begin_corpus_maintenance()
        try:
            with self.assertRaises(jobs.CorpusBusy):
                jobs.submit("test", corpus="")
            self.assertEqual(len(jobs._queue), 0)
            observed = []
            thread = threading.Thread(target=lambda: observed.append(jobs.maintenance_in_progress()))
            thread.start()
            thread.join(1)
            self.assertEqual(observed, [True])
        finally:
            jobs.end_corpus_maintenance()
        self.assertFalse(jobs.maintenance_in_progress())
        self.assertEqual(jobs.submit("test", corpus="").status, "queued")

    def test_concurrent_worker_initialization_starts_one_worker(self):
        self.worker.stop()
        real_thread = threading.Thread
        starts = []

        class FakeWorker:
            def __init__(self, **kwargs):
                self.alive = False

            def is_alive(self):
                return self.alive

            def start(self):
                time.sleep(0.01)
                starts.append(self)
                self.alive = True

        with patch.object(jobs, "_worker", None), patch.object(jobs.threading, "Thread", FakeWorker):
            threads = [real_thread(target=jobs._ensure_worker) for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(1)
        self.assertEqual(len(starts), 1)
        self.worker.start()

    def test_missing_corpus_does_not_render_another_voice(self):
        job = jobs.Job("missing", "test", corpus="gone")
        with patch.object(database, "list_corpora", return_value=[]), \
                patch.object(generate, "generate_video") as render:
            jobs._run(job)
        render.assert_not_called()
        self.assertEqual((job.status, job.error_kind), ("error", "request"))
        self.assertTrue(job.ended.is_set())

    def test_job_context_preserves_selected_voice(self):
        a, b = {"slug": "a"}, {"slug": "b"}
        heard = []

        def render(*args, **kwargs):
            heard.append(database.active()["slug"])
            return {"video_url": None}

        with patch.object(database, "_active", b), \
                patch.object(database, "list_corpora", return_value=[a, b]), \
                patch.object(generate, "generate_video", side_effect=render):
            job = jobs.Job("context", "test", corpus="a")
            jobs._run(job)
            self.assertEqual(database.active()["slug"], "b")
        self.assertEqual((heard, job.status), (["a"], "done"))


if __name__ == "__main__":
    unittest.main()
