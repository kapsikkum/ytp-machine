#!/usr/bin/env python3
"""Corrections, word/noise edits and references preserve unrelated corpus data.

    python tests/test_corpus_metadata.py
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import database as db, editor, forced_align as fa
from scripts import correct, import_transcripts as transcripts


class CorpusMetadataTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="mrs-metadata-")))
        self.stack.enter_context(patch.object(db, "_active", {
            "slug": "test", "dir": str(self.directory), "db": str(self.directory / "corpus.db"),
        }))
        self.stack.enter_context(patch.object(db, "_ready", set()))
        self.stack.enter_context(patch.object(fa, "_char_cache", {}))
        self.selection = Mock()
        self.stack.enter_context(patch.dict(sys.modules, {
            "app.generate": SimpleNamespace(invalidate_cache=self.selection),
        }))
        db.init_db()
        self.cache_dir = self.directory / "cache" / "clips"
        self.cache_dir.mkdir(parents=True)
        with db.get_db() as conn:
            conn.execute("INSERT INTO sources (id,video_id,source_file,title) VALUES (1,'v1',?,?)",
                         ("downloads/source.mp4", "Hot Food"))
            for cid, word, phone in ((1, "cat", "K"), (2, "dog", "D")):
                conn.execute(
                    "INSERT INTO word_clips (id,source_id,word,start_time,end_time,source_file,phones) "
                    "VALUES (?,1,?,1,2,?,?)", (cid, word, "downloads/source.mp4",
                                              json.dumps([[phone, 0.1, 0.2]])))
                conn.execute("INSERT INTO clip_alignments (clip_id,alignment) VALUES (?,?)",
                             (cid, json.dumps([[word[0].upper(), 0.1, 0.2]])))
                name = f"{cid}_0.180.ts"
                (self.cache_dir / name).write_bytes(b"cached test media")
                conn.execute("INSERT INTO clip_cache (clip_id,pad_end,cache_file) VALUES (?,0.18,?)",
                             (cid, name))
                conn.execute("INSERT INTO splice_ratings (word,clip_id,score) VALUES ('target',?,-3)", (cid,))
                conn.execute("INSERT INTO boundary_reports (clip_id,kind) VALUES (?,'starts late')", (cid,))
            conn.execute("INSERT INTO noise_clips (id,source_id,word,start_time,end_time,source_file) "
                         "VALUES (1,1,'click',3,4,'downloads/source.mp4')")

    def ids(self, table):
        with db.get_db() as conn:
            return [row[0] for row in conn.execute(f"SELECT clip_id FROM {table} ORDER BY clip_id")]

    def test_recipe_changes_preserve_clip_alignments_and_render_cache(self):
        with patch("app.phonemes.canonical_phones", return_value=["K"]), \
                patch("app.phonemes.invalidate_recipes"):
            editor.save_recipe("cat", editor.SplicePlan(groups=[editor.SpliceGroup(phones=["K"], source="cat")]))
            editor.delete_recipe("cat")
        self.assertEqual(self.ids("clip_alignments"), [1, 2])
        self.assertEqual(self.ids("clip_cache"), [1, 2])
        self.assertTrue((self.cache_dir / "1_0.180.ts").exists())
        self.assertTrue((self.cache_dir / "2_0.180.ts").exists())
        self.assertEqual(self.selection.call_count, 2)
        self.selection.assert_called_with(alignments=False)

    def test_corrected_word_drops_only_affected_alignment_and_render(self):
        correct._apply([{"id": 1, "old": "cat", "new": "cap"}])
        with db.get_db() as conn:
            row = conn.execute("SELECT * FROM word_clips WHERE id=1").fetchone()
        self.assertEqual(row["word"], "cap")
        self.assertIsNone(row["phones"])
        self.assertEqual(self.ids("clip_alignments"), [2])
        self.assertEqual(self.ids("clip_cache"), [2])
        self.assertFalse((self.cache_dir / "1_0.180.ts").exists())
        self.assertTrue((self.cache_dir / "2_0.180.ts").exists())
        with patch.object(fa, "_align", return_value=[("P", 0.1, 0.2)]) as inference:
            self.assertEqual(fa.char_times(dict(row)), [("P", 0.1, 0.2)])
            inference.assert_called_once()
        self.selection.assert_called_once_with(alignments=False)

    def test_failed_metadata_cleanup_rolls_back_label_and_phone_changes(self):
        with db.get_db() as conn:
            conn.executescript("CREATE TRIGGER prevent_alignment_delete BEFORE DELETE ON clip_alignments "
                               "BEGIN SELECT RAISE(ABORT,'cleanup failed'); END;")
        with self.assertRaises(sqlite3.IntegrityError):
            correct._apply([{"id": 1, "old": "cat", "new": "cap"}])
        with db.get_db() as conn:
            row = conn.execute("SELECT word,phones FROM word_clips WHERE id=1").fetchone()
        self.assertEqual(row["word"], "cat")
        self.assertIsNotNone(row["phones"])
        self.assertEqual(self.ids("clip_alignments"), [1, 2])
        self.selection.assert_not_called()

    def test_noise_deletion_preserves_same_id_word_feedback_and_caches(self):
        editor.delete_clip(1, kind="noise")
        for table in ("splice_ratings", "boundary_reports", "clip_alignments", "clip_cache"):
            self.assertEqual(self.ids(table), [1, 2], table)
        self.assertTrue((self.cache_dir / "1_0.180.ts").exists())
        self.selection.assert_called_once_with(alignments=False)

    def test_noise_boundary_edit_preserves_same_id_word_caches(self):
        editor.edit_clip(1, editor.ClipEdit(start_time=3.1), kind="noise")
        self.assertEqual(self.ids("clip_alignments"), [1, 2])
        self.assertEqual(self.ids("clip_cache"), [1, 2])
        self.selection.assert_called_once_with(alignments=False)

    def test_rating_only_preserves_alignment_render_and_boundary_reports(self):
        editor.set_rating(1, editor.Rating(word="target", score=2))
        for table in ("clip_alignments", "clip_cache", "boundary_reports"):
            self.assertEqual(self.ids(table), [1, 2], table)
        self.assertTrue((self.cache_dir / "1_0.180.ts").exists())
        self.selection.assert_called_once_with(alignments=False)

    def test_word_deletion_removes_only_its_dependent_data(self):
        editor.delete_clip(1, kind="word")
        for table in ("splice_ratings", "boundary_reports", "clip_alignments", "clip_cache"):
            self.assertEqual(self.ids(table), [2], table)
        self.assertFalse((self.cache_dir / "1_0.180.ts").exists())
        self.assertTrue((self.cache_dir / "2_0.180.ts").exists())

    def test_word_rename_clears_its_metadata_and_preserves_other_clips(self):
        editor.edit_clip(1, editor.ClipEdit(word="cap"))
        self.assertEqual(self.ids("clip_alignments"), [2])
        self.assertEqual(self.ids("clip_cache"), [2])
        with db.get_db() as conn:
            row = conn.execute("SELECT word,phones FROM word_clips WHERE id=1").fetchone()
        self.assertEqual(row["word"], "cap")
        self.assertIsNone(row["phones"])

    def test_adding_clip_preserves_existing_alignments(self):
        editor.add_clip(1, editor.NewClip(word="new", start_time=5, end_time=6))
        self.assertEqual(self.ids("clip_alignments"), [1, 2])
        self.assertEqual(self.ids("clip_cache"), [1, 2])
        self.selection.assert_called_once_with(alignments=False)

    def test_correct_default_and_explicit_transcript_directories(self):
        native = self.directory / "transcripts"
        explicit = self.directory / "references"
        for directory, text in ((native, "native reference"), (explicit, "explicit reference")):
            directory.mkdir()
            (directory / "v1.txt").write_text(text, encoding="utf-8")
        for arguments, text in (([], "native reference"), (["--transcripts-dir", str(explicit)], "explicit reference")):
            with patch.object(sys, "argv", ["correct", "--all"] + arguments), \
                    patch.object(correct, "_run_one", return_value=0) as runner, redirect_stdout(io.StringIO()):
                correct.main()
            runner.assert_called_once_with(1, text, False)

    def test_import_default_and_explicit_output_directories(self):
        dump = self.directory / "wiki.json"
        dump.write_text(json.dumps({"parse": {"wikitext": "==Hot Food==\ntrusted text"}}), encoding="utf-8")
        for arguments, output in (([], self.directory / "transcripts"),
                                  (["--out", str(self.directory / "custom")], self.directory / "custom")):
            with patch.object(sys, "argv", ["import", "--json", str(dump), "--write"] + arguments), redirect_stdout(io.StringIO()):
                transcripts.main()
            self.assertEqual((output / "v1.txt").read_text(encoding="utf-8"), "trusted text")

    def test_wiki_nested_headings_stay_in_same_poem(self):
        poems = transcripts.parse_wiki("==Hot Food==\none\n===Verse 2===\ntwo\n==Cake==\nthree")
        self.assertEqual(set(poems), {"Hot Food", "Cake"})
        self.assertEqual(poems["Hot Food"].split(), ["one", "two"])

    def test_failed_transcript_replace_preserves_existing_reference(self):
        target = self.directory / "reference.txt"
        target.write_text("hand checked reference", encoding="utf-8")
        with patch.object(transcripts.os, "replace", side_effect=OSError("replacement failed")):
            with self.assertRaisesRegex(OSError, "replacement failed"):
                transcripts._write_transcript(str(target), "new imported text")
        self.assertEqual(target.read_text(encoding="utf-8"), "hand checked reference")
        self.assertEqual(list(self.directory.glob(".transcript-*.tmp")), [])

    def test_empty_titles_cannot_match_arbitrary_poems(self):
        self.assertIsNone(transcripts.match(transcripts.norm_title("Michael Rosen"), {"Hot Food": "hot food"}))
        self.assertEqual(transcripts.norm_title(None), "")
        self.assertIsNone(transcripts.match("hot food", {"Empty": ""}))
        got = transcripts.match("chocolate cake", {"Chocolate Pudding": "chocolate pudding"})
        self.assertLess(got[1], 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
