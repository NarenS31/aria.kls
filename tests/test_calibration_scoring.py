"""Tests for the calibration pre-check scorer.

No rater has returned a sheet, so these fill the blind calibration CSVs
synthetically. The cases that matter are the diagnostic ones: a rater who misses
too many should be told to re-read the rubric, and an item both raters miss the
same way should be surfaced as evidence against the CORPUS label rather than
against the raters.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "eval"))

from score_calibration import REVIEW_THRESHOLD, read_sheet, score  # noqa: E402

ANNOTATION_DIR = os.path.join(REPO_ROOT, "eval", "data", "annotation")
KEY_PATH = os.path.join(ANNOTATION_DIR, "annotation_private_key.json")
BLANK_SHEET = os.path.join(ANNOTATION_DIR, "rater1_calibration.csv")


def load_calibration_key():
    with open(KEY_PATH, encoding="utf-8") as fh:
        return json.load(fh)["calibration_round"]


def fill(key, tmpdir, name, answers):
    """Write a filled copy of the blind calibration sheet."""
    with open(BLANK_SHEET, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0])
    path = os.path.join(tmpdir, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["predicted_category"] = answers.get(row["item_id"], "")
            row["confidence"] = "4" if row["predicted_category"] else ""
            writer.writerow(row)
    return path


class CalibrationScoringTests(unittest.TestCase):
    def setUp(self):
        self.key = load_calibration_key()
        self.truth = {i: m["true_category"] for i, m in self.key["items"].items()}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_scorer(self, answers1, answers2):
        p1 = fill(self.key, self.tmp.name, "r1.csv", answers1)
        p2 = fill(self.key, self.tmp.name, "r2.csv", answers2)
        return score(self.key, {"rater1": read_sheet(p1, "rater1"),
                                "rater2": read_sheet(p2, "rater2")})

    def test_the_calibration_set_covers_every_category(self):
        categories = {m["true_category"] for m in self.key["items"].values()}
        self.assertEqual(len(categories), 7)
        self.assertEqual(len(self.key["items"]), 14)

    def test_perfect_recovery_clears_the_gate(self):
        report = self.run_scorer(self.truth, self.truth)
        for label in ("rater1", "rater2"):
            self.assertEqual(report["per_rater"][label]["recovery_rate"], 1.0)
            self.assertFalse(
                report["per_rater"][label]["needs_review_before_main_sheet"])
        self.assertEqual(report["items_both_raters_missed_identically"], [])
        self.assertIn("proceed to the main sheet", report["verdict"])

    def test_a_weak_rater_is_told_to_re_read_before_the_main_sheet(self):
        wrong = dict(self.truth)
        for item_id in list(wrong)[:6]:          # 6/14 wrong -> 57%
            wrong[item_id] = "RUSHING" if wrong[item_id] != "RUSHING" else "FLOW"
        report = self.run_scorer(wrong, self.truth)
        self.assertLess(report["per_rater"]["rater1"]["recovery_rate"],
                        REVIEW_THRESHOLD)
        self.assertTrue(
            report["per_rater"]["rater1"]["needs_review_before_main_sheet"])
        self.assertFalse(
            report["per_rater"]["rater2"]["needs_review_before_main_sheet"])
        self.assertIn("Re-read the rubric", report["verdict"])

    def test_both_raters_missing_identically_indicts_the_corpus_label(self):
        target = next(i for i, c in self.truth.items() if c != "FLOW")
        both = dict(self.truth)
        both[target] = "FLOW"
        report = self.run_scorer(both, both)
        flagged = report["items_both_raters_missed_identically"]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["item_id"], target)
        self.assertEqual(flagged[0]["both_said"], "FLOW")
        self.assertIn("the corpus label is the thing to doubt", report["verdict"])

    def test_disagreeing_raters_are_not_flagged_as_a_corpus_problem(self):
        target = next(iter(self.truth))
        a = dict(self.truth); a[target] = "FLOW"
        b = dict(self.truth); b[target] = "STUCK"
        report = self.run_scorer(a, b)
        self.assertEqual(report["items_both_raters_missed_identically"], [])

    def test_blank_answers_are_counted_not_scored_as_wrong(self):
        partial = dict(self.truth)
        for item_id in list(partial)[:3]:
            partial[item_id] = ""
        report = self.run_scorer(partial, self.truth)
        stats = report["per_rater"]["rater1"]
        self.assertEqual(stats["n_blank"], 3)
        self.assertEqual(stats["n_coded"], 11)
        self.assertEqual(stats["recovery_rate"], 1.0)

    def test_no_kappa_is_emitted(self):
        report = self.run_scorer(self.truth, self.truth)
        serialized = json.dumps(report).lower()
        self.assertNotIn('"kappa"', serialized)
        self.assertIn("no_kappa_reported", report)



class RubricVariantTests(unittest.TestCase):
    """The calibration rubric must be the full rubric minus the examples only.

    If any definition, marker or tie-breaker differed between the two, the
    calibration round and the main round would have been coded against
    different instruments and could not be compared.
    """

    FULL = os.path.join(ANNOTATION_DIR, "rubric.md")
    LEAN = os.path.join(ANNOTATION_DIR, "rubric_calibration.md")

    def setUp(self):
        for path in (self.FULL, self.LEAN):
            if not os.path.exists(path):
                self.skipTest("run eval/build_annotation_study.py first")
        self.full = open(self.FULL, encoding="utf-8").read().splitlines()
        self.lean = open(self.LEAN, encoding="utf-8").read().splitlines()

    def test_calibration_rubric_has_no_worked_examples(self):
        self.assertEqual([l for l in self.lean if l.startswith("> ")], [])
        self.assertEqual(len([l for l in self.full if l.startswith("> ")]), 14)

    def test_headings_are_identical(self):
        self.assertEqual([l for l in self.full if l.startswith("#")],
                         [l for l in self.lean if l.startswith("#")])

    def test_every_definition_and_tie_breaker_survives(self):
        for prefix in ("*Typical markers:", "*Most often confused with:", "| "):
            with self.subTest(prefix=prefix):
                self.assertEqual([l for l in self.full if l.startswith(prefix)],
                                 [l for l in self.lean if l.startswith(prefix)])

    def test_each_rubric_carries_its_own_round_banner_on_line_one(self):
        self.assertEqual(self.full[0],
                         "**MAIN STUDY — definitions plus worked examples**")
        self.assertEqual(
            self.lean[0],
            "**CALIBRATION ROUND — definitions only, no examples**")

    def test_neither_rubric_carries_the_other_round_banner(self):
        self.assertNotIn(self.lean[0], self.full)
        self.assertNotIn(self.full[0], self.lean)

    def test_the_banner_is_not_a_blockquote(self):
        # A "> " banner would be indistinguishable from a worked example to the
        # strip logic, and would silently survive into the calibration rubric.
        for line in (self.full[0], self.lean[0]):
            with self.subTest(line=line):
                self.assertFalse(line.startswith("> "))

    def test_the_only_other_differences_are_the_footer_and_the_banner(self):
        import difflib
        non_quote_full = [l for l in self.full if not l.startswith("> ")]
        non_quote_lean = [l for l in self.lean if not l.startswith("> ")]
        delta = [d for d in difflib.unified_diff(
            non_quote_full, non_quote_lean, lineterm="", n=0)
            if d.startswith(("+", "-")) and not d.startswith(("---", "+++"))]
        banners = {"**MAIN STUDY — definitions plus worked examples**",
                   "**CALIBRATION ROUND — definitions only, no examples**"}
        for line in delta:
            body = line[1:]
            self.assertTrue(
                body.strip() == ""
                or body in banners
                or body.startswith("*Category definitions and markers"),
                f"unexpected difference between the rubrics: {body[:80]!r}")

if __name__ == "__main__":
    unittest.main()
