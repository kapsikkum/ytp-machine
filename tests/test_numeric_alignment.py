#!/usr/bin/env python3
"""Numeric meaning survives lookup and English alignment target encoding.

    python tests/test_numeric_alignment.py

Uses populated corpus-cache doubles and a small pronunciation dictionary.
Requires num2words, as the tokeniser does; no models, downloads or database
writes are needed.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import generate as g, phonemes as ph
from scripts import refine_boundaries as rb


class NumericTargetsTests(unittest.TestCase):
    def setUp(self):
        self.vocab = {ch: i + 1 for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ|")}
        self.reverse = {token: char for char, token in self.vocab.items()}
        patcher = patch.object(rb, "_DICT", self.vocab)
        patcher.start()
        self.addCleanup(patcher.stop)

    def targets(self, labels):
        words = [{"word": label} for label in labels]
        tokens, ranges = rb._ctc_targets(words)
        return "".join(self.reverse[token] for token in tokens), ranges

    def test_cardinal_targets_keep_the_original_clip_range(self):
        for label, expected in [("0", "ZERO"), ("20", "TWENTY"),
                                ("22", "TWENTY|TWO"), ("100", "ONE|HUNDRED"),
                                ("101", "ONE|HUNDRED|AND|ONE")]:
            with self.subTest(label=label):
                target, ranges = self.targets([label])
                self.assertEqual(target, expected)
                self.assertEqual(ranges, [(0, len(expected))])

    def test_ordinal_target_ends_before_the_next_original_word(self):
        target, ranges = self.targets(["22nd", "cat"])
        self.assertEqual(target, "TWENTY|SECOND|CAT")
        self.assertEqual(ranges, [(0, len("TWENTY|SECOND")),
                                  (len("TWENTY|SECOND|") , len(target))])

    def test_single_word_ordinals(self):
        for label, expected in [("1st", "FIRST"), ("2nd", "SECOND"), ("3rd", "THIRD"),
                                ("7th", "SEVENTH"), ("11th", "ELEVENTH")]:
            with self.subTest(label=label):
                self.assertEqual(self.targets([label]), (expected, [(0, len(expected))]))

    def test_ambiguous_alphanumeric_and_leading_zeroes_do_not_align_partial_text(self):
        for label in ["v8", "ps5", "i30", "50km", "35l", "F-150", "007", "000"]:
            with self.subTest(label=label):
                self.assertEqual(self.targets([label]), ("", [None]))
        self.assertEqual(self.targets(["a", "v8", "cat"]), ("A|CAT", [(0, 1), None, (2, 5)]))

    def test_unknown_alphabet_is_not_silently_truncated(self):
        self.assertEqual(self.targets(["caf\u00e9"]), ("", [None]))


class NumericLookupTests(unittest.TestCase):
    def setUp(self):
        # These aliases reproduce the real fault: empty-corpus tokeniser tests
        # expanded correctly while a populated corpus preferred the alias.
        for patcher in (patch.object(g, "_ensure_cache", lambda: None),
                        patch.object(g, "_clips_by_word_cache", {
                            key: [{}] for key in ["20", "50", "35l", "v8", "80s", "rosens"]}),
                        patch.object(g, "_known_spelling", lambda word: word)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_decimal_input_does_not_reuse_integer_alias(self):
        self.assertEqual(g._expand_token("2.0"), ["two", "point", "zero"])
        self.assertEqual(g._expand_token("3.5l"), ["three", "point", "five", "litres"])

    def test_currency_and_percent_keep_the_unit_with_existing_digit_clip(self):
        for token, expected in [("$50", ["fifty", "dollars"]),
                                ("\u00a350", ["fifty", "pounds"]),
                                ("\u20ac50", ["fifty", "euros"]),
                                ("50%", ["fifty", "percent"])]:
            with self.subTest(token=token):
                self.assertEqual(g._expand_token(token), expected)

    def test_exact_plain_and_alphanumeric_clips_still_match(self):
        for token, key in [("20", "20"), ("50", "50"), ("v8", "v8"),
                           ("80s", "80s"), ("rosen's", "rosens")]:
            with self.subTest(token=token):
                self.assertEqual(g._expand_token(token), [key])


class OrdinalPhonesTests(unittest.TestCase):
    def setUp(self):
        self.dictionary = {
            "first": ["F", "ER", "S", "T"],
            "second": ["S", "EH", "K", "AH", "N", "D"],
            "third": ["TH", "ER", "D"],
            "twenty": ["T", "W", "EH", "N", "T", "IY"],
            "two": ["T", "UW"],
        }
        ph.word_to_phonemes.cache_clear()
        for patcher in (patch.object(ph, "_DICT", {w: [phones] for w, phones in self.dictionary.items()}),
                        patch.object(ph, "_user_dict", {})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(ph.word_to_phonemes.cache_clear)

    def test_ordinal_labels_use_ordinal_pronunciation(self):
        for label, expected in [("1st", self.dictionary["first"]),
                                ("2nd", self.dictionary["second"]),
                                ("3rd", self.dictionary["third"]),
                                ("22nd", self.dictionary["twenty"] + self.dictionary["second"])]:
            with self.subTest(label=label):
                self.assertEqual(ph.word_to_phonemes(label), expected)

    def test_cardinal_lookup_stays_cardinal(self):
        self.assertEqual(ph.word_to_phonemes("22"), self.dictionary["twenty"] + self.dictionary["two"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
