#!/usr/bin/env python3
"""Parallel operations keep their database, pronunciations and recipes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import database as db, phonemes as ph


class CorpusContextTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="mrs-context-")))
        self.entries = {}
        for slug in ("alpha", "beta"):
            directory = self.root / "corpora" / slug
            directory.mkdir(parents=True)
            self.entries[slug] = {"slug": slug, "dir": str(directory), "db": str(directory / "corpus.db")}
        self.stack.enter_context(patch.object(db, "DATA_DIR", str(self.root)))
        self.stack.enter_context(patch.object(db, "CORPORA_DIR", str(self.root / "corpora")))
        self.stack.enter_context(patch.object(db, "LEGACY_DB", str(self.root / "absent.db")))
        self.stack.enter_context(patch.object(db, "_active", self.entries["alpha"]))
        self.stack.enter_context(patch.object(db, "_ready", set()))
        self.stack.enter_context(patch.object(ph, "_DICT", {}))
        ph.invalidate_user_dict()
        ph.invalidate_recipes()
        self.addCleanup(ph.invalidate_user_dict)
        self.addCleanup(ph.invalidate_recipes)
        for slug, sounds in (("alpha", "B AE"), ("beta", "D IY")):
            entry = self.entries[slug]
            (Path(entry["dir"]) / "pronunciations.csv").write_text(
                f"flumbix,{sounds}\nalias,flumbix\n", encoding="utf-8")
            with db.using_corpus(entry):
                db.init_db()
                with db.get_db() as conn:
                    conn.execute("INSERT INTO splice_recipes VALUES ('custom', ?)",
                                 (json.dumps([{"phones": sounds.split(), "from": [slug]}]),))

    def test_nested_contexts_restore_selection_even_after_failure(self):
        with db.using_corpus("beta"):
            self.assertEqual(db.active()["slug"], "beta")
            with self.assertRaises(ValueError), db.using_corpus("alpha"):
                self.assertEqual(db.active()["slug"], "alpha")
                raise ValueError("fixture")
            self.assertEqual(db.active()["slug"], "beta")
        self.assertEqual(db.active()["slug"], "alpha")
        with self.assertRaises(KeyError), db.using_corpus("missing"):
            pass

    def test_global_selection_cannot_change_an_operation_database(self):
        with db.using_corpus("alpha"):
            db.set_active("beta")
            self.assertEqual(db.active()["slug"], "alpha")
            with db.get_db() as conn:
                conn.execute("INSERT INTO settings VALUES ('pinned', 'alpha')")
        self.assertEqual(db.active()["slug"], "beta")
        with db.get_db() as conn:
            self.assertIsNone(conn.execute("SELECT value FROM settings WHERE key='pinned'").fetchone())

    def test_two_threads_keep_distinct_pronunciations_recipes_and_writes(self):
        ready = threading.Barrier(2)

        def run(slug, expected):
            with db.using_corpus(slug):
                ready.wait(timeout=5)
                for _ in range(80):
                    self.assertEqual(ph.word_to_phonemes("alias"), expected)
                    self.assertEqual(ph.user_entry("flumbix"), ("voice", " ".join(expected)))
                    self.assertEqual(ph.user_recipe("custom")[0][1], [slug])
                with db.get_db() as conn:
                    conn.execute("INSERT INTO settings VALUES ('thread', ?)", (slug,))

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [pool.submit(run, "alpha", ["B", "AE"]), pool.submit(run, "beta", ["D", "IY"])]
            for result in results:
                result.result(timeout=15)
        for slug in self.entries:
            with db.using_corpus(slug), db.get_db() as conn:
                self.assertEqual(conn.execute("SELECT value FROM settings WHERE key='thread'").fetchone()[0], slug)

    def test_reload_invalidates_both_corpora_memoized_answers(self):
        for slug in self.entries:
            with db.using_corpus(slug):
                self.assertIsNotNone(ph.word_to_phonemes("flumbix"))
        for entry in self.entries.values():
            (Path(entry["dir"]) / "pronunciations.csv").write_text("flumbix,K AH\n", encoding="utf-8")
        ph.invalidate_user_dict()
        for slug in self.entries:
            with db.using_corpus(slug):
                self.assertEqual(ph.word_to_phonemes("flumbix"), ["K", "AH"])


if __name__ == "__main__":
    unittest.main()
