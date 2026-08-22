"""Tests for the RQ2 response analysis.

No real responses exist yet, so these build a Google Forms-shaped export by
hand. The point is not to check arithmetic — it is to check the unblinding,
which is the one thing that fails silently. Every synthetic rater here scores
the scaffolded response 5 and the answer-giving response 2, written into
whichever of the A/B columns actually holds that condition for that item. If
the analysis ignored the answer key and assumed A is always scaffolded, both
means would land near 3.5 instead of at 5 and 2, and the planted effect would
vanish. That is exactly the failure a real run could not detect.
"""

import csv
import json
import os
import tempfile
import unittest

from eval.rq2_analysis import SAME, SCAFFOLDED, ANSWER_GIVING, analyse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH = os.path.join(REPO_ROOT, "data", "rq2_answer_key.json")

SCAFFOLDED_SCORE = 5
ANSWER_GIVING_SCORE = 2


def load_key():
    with open(KEY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def write_export(path, key, n_raters=6, blanks=False, break_headers=False,
                 jitter=False):
    """Write a CSV shaped like a Google Forms response export.

    Column order follows the form: timestamp, consent, then five columns per
    item in item order, then the three closing questions. The per-item headers
    repeat verbatim, which is the whole reason the analysis matches them by
    position.
    """
    q = key["questions"]
    by_number = {v["form_item_number"]: (pid, v) for pid, v in key["key"].items()}
    numbers = sorted(by_number)

    headers = ["Timestamp", "By continuing, you confirm you're voluntarily "
                            "participating and understand your responses are "
                            "anonymous."]
    for _ in numbers:
        headers += [q["which_helps"], q["rate_a"], q["rate_b"],
                    q["when_stuck"], q["free_text"]]
    headers += [q["overall"], q["grade"], q["math_relationship"]]
    if break_headers:
        headers[3] = "Rate Response A - a totally different question"

    rows = []
    for rater in range(n_raters):
        row = [f"2026/09/0{rater + 1} 10:00:00", "I understand and agree to continue"]
        for position, number in enumerate(numbers):
            _, item = by_number[number]
            scaffolded_letter = item["scaffolded_letter"]
            answer_letter = item["answer_giving_letter"]

            # Flat scores prove the unblinding; jittered ones give the paired
            # tests something to work with, since a set of identical
            # differences has no variance for a t-test to divide by.
            if jitter:
                scaffolded_score = 5 - ((rater + position) % 2)
                answer_giving_score = 1 + ((rater + 2 * position) % 3)
            else:
                scaffolded_score = SCAFFOLDED_SCORE
                answer_giving_score = ANSWER_GIVING_SCORE

            rating_a = (scaffolded_score if scaffolded_letter == "A"
                        else answer_giving_score)
            rating_b = (scaffolded_score if scaffolded_letter == "B"
                        else answer_giving_score)

            # One rater leaves one item's ratings blank, to exercise the
            # incomplete-pair path without losing the rest of their answers.
            if blanks and rater == 0 and position == 0:
                rating_a = rating_b = ""

            row += [
                scaffolded_letter,          # Q5: most helpful -> scaffolded
                str(rating_a),
                str(rating_b),
                answer_letter,              # Q8: when stuck -> answer-giving
                "" if position else "The second one just told me the answer.",
            ]
        row += ["Depended on the item", "11th", "I struggle with specific areas"]
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


class Rq2AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.key = load_key()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def analyse(self, **kwargs):
        path = os.path.join(self.tmp.name, "responses.csv")
        write_export(path, self.key, **kwargs)
        return analyse(path, KEY_PATH)

    def test_unblinding_recovers_the_planted_effect(self):
        results = self.analyse(n_raters=6)
        ratings = results["ratings"]
        self.assertEqual(ratings[SCAFFOLDED]["mean"], float(SCAFFOLDED_SCORE))
        self.assertEqual(ratings[ANSWER_GIVING]["mean"], float(ANSWER_GIVING_SCORE))
        self.assertEqual(ratings["complete_pairs"], 60)
        self.assertEqual(
            ratings["mean_difference_scaffolded_minus_answer_giving"],
            float(SCAFFOLDED_SCORE - ANSWER_GIVING_SCORE),
        )

    def test_letters_are_not_conditions(self):
        """A run that ignored the key would average the two letters together."""
        results = self.analyse(n_raters=6)
        letters = {v["scaffolded_letter"] for v in self.key["key"].values()}
        self.assertEqual(letters, {"A", "B"}, "the fixture needs both arrangements")
        self.assertNotAlmostEqual(
            results["ratings"][SCAFFOLDED]["mean"],
            results["ratings"][ANSWER_GIVING]["mean"],
        )

    def test_preferences_are_reported_by_condition_not_by_letter(self):
        results = self.analyse(n_raters=6)
        self.assertEqual(results["most_helpful"]["rates"][SCAFFOLDED], 1.0)
        self.assertEqual(results["most_helpful"]["rates"][ANSWER_GIVING], 0.0)
        self.assertEqual(results["when_stuck"]["rates"][ANSWER_GIVING], 1.0)
        self.assertEqual(results["when_stuck"]["rates"][SAME], 0.0)

    def test_divergence_is_counted_and_given_a_direction(self):
        results = self.analyse(n_raters=6)
        divergence = results["divergence"]
        self.assertEqual(divergence["item_level"]["rate"], 1.0)
        self.assertEqual(divergence["raters_with_any_divergence"], 6)
        self.assertEqual(
            divergence["learning_says_scaffolded_but_stressed_wants_answer"], 60
        )
        self.assertEqual(
            divergence["learning_says_answer_but_stressed_wants_scaffolded"], 0
        )

    def test_significance_test_runs_on_the_rater_level_pairing(self):
        results = self.analyse(n_raters=6, jitter=True)
        test = results["tests"]["rater_level"]
        self.assertEqual(test["n_pairs"], 6)
        self.assertGreater(test["mean_difference"], 0)
        self.assertIsNotNone(test["wilcoxon"]["p_value"])
        self.assertLess(test["wilcoxon"]["p_value"], 0.05)
        # Every rater favoured scaffolded, so all signed ranks are positive.
        self.assertEqual(test["wilcoxon"]["rank_biserial_r"], 1.0)
        self.assertIsNotNone(test["paired_t"]["p_value"])
        self.assertLess(test["paired_t"]["p_value"], 0.05)
        self.assertIsNotNone(test["paired_t"]["cohens_dz"])
        self.assertIsNone(test["paired_t"]["note"])

    def test_zero_variance_differences_refuse_the_t_test_but_keep_wilcoxon(self):
        results = self.analyse(n_raters=6)
        test = results["tests"]["rater_level"]
        self.assertEqual(test["mean_difference"], 3.0)
        self.assertIsNone(test["paired_t"]["p_value"])
        self.assertIn("no variance in the differences", test["paired_t"]["note"])
        self.assertIsNotNone(test["wilcoxon"]["p_value"])

    def test_identical_scores_do_not_crash_the_test(self):
        path = os.path.join(self.tmp.name, "flat.csv")
        write_export(path, self.key, n_raters=4)
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        # Flatten every rating to 3 so all paired differences are zero.
        for row in rows[1:]:
            for i, value in enumerate(row):
                if value in {str(SCAFFOLDED_SCORE), str(ANSWER_GIVING_SCORE)}:
                    row[i] = "3"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)

        results = analyse(path, KEY_PATH)
        test = results["tests"]["rater_level"]
        self.assertEqual(test["mean_difference"], 0.0)
        self.assertIsNone(test["wilcoxon"]["p_value"])
        self.assertIn("identically", test["wilcoxon"]["note"])

    def test_missing_ratings_drop_only_that_pair(self):
        results = self.analyse(n_raters=6, blanks=True)
        self.assertEqual(results["ratings"]["complete_pairs"], 59)
        self.assertEqual(results["meta"]["n_raters"], 6)
        self.assertEqual(results["tests"]["rater_level"]["n_pairs"], 6)

    def test_per_item_rows_cover_every_item_in_form_order(self):
        results = self.analyse(n_raters=6)
        rows = results["per_item"]
        self.assertEqual(len(rows), self.key["n_items"])
        self.assertEqual([r["form_item_number"] for r in rows],
                         list(range(1, self.key["n_items"] + 1)))
        for row in rows:
            self.assertEqual(row["difference"],
                             float(SCAFFOLDED_SCORE - ANSWER_GIVING_SCORE))

    def test_a_mismatched_export_fails_loudly(self):
        path = os.path.join(self.tmp.name, "wrong.csv")
        write_export(path, self.key, break_headers=True)
        with self.assertRaises(SystemExit) as caught:
            analyse(path, KEY_PATH)
        self.assertIn("Column mapping failed", str(caught.exception))

    def test_answer_key_matches_the_frozen_set(self):
        frozen_path = os.path.join(REPO_ROOT, "data", "rq2_pairs_FROZEN.json")
        with open(frozen_path, encoding="utf-8") as fh:
            frozen = json.load(fh)
        self.assertEqual(self.key["n_items"], len(frozen["pairs"]))
        for number, pair in enumerate(frozen["pairs"], 1):
            entry = self.key["key"][pair["problem_id"]]
            self.assertEqual(entry["form_item_number"], number)
            self.assertEqual(entry["A"], pair["display_order"]["shown_as_A"])
            self.assertNotEqual(entry["A"], entry["B"])


if __name__ == "__main__":
    unittest.main()
