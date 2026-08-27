"""Leakage assertions for the problem-disjoint evaluation split.

These run against the live corpus, not a fixture: the point is to fail if anyone
regenerates the data, changes the pool, or edits the partition logic in a way
that lets a problem appear in more than one split. A green run here is what
licenses calling the test estimate leakage-free.
"""

import hashlib
import json
import os
import sys
import unittest
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ROOT = os.path.join(REPO_ROOT, "eval")
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from build_problem_disjoint_split import (  # noqa: E402
    DEFAULT_SEED,
    GENERATORS,
    SPLIT_SIZES,
    assert_no_leakage,
    build,
    normalize_problem,
    problem_hash,
)
from matched_generator_eval import MATCH_SEED, match_cells  # noqa: E402
from metacognition.analyzer import COGNITIVE_STATES  # noqa: E402

RESULTS_DIR = os.path.join(EVAL_ROOT, "data", "eval", "problem_disjoint_20260826")


class ProblemDisjointSplitTest(unittest.TestCase):
    """The split itself, rebuilt from the corpus on every run."""

    @classmethod
    def setUpClass(cls):
        cls.records, cls.manifest = build(DEFAULT_SEED)
        cls.by_split = {s: [r for r in cls.records if r["split"] == s]
                        for s in ("train", "val", "test")}

    # -- the core leakage assertion ---------------------------------

    def test_no_problem_id_crosses_splits(self):
        seen = {}
        for r in self.records:
            prev = seen.setdefault(r["problem_id"], r["split"])
            self.assertEqual(prev, r["split"],
                             f"{r['problem_id']} appears in both {prev} and {r['split']}")

    def test_no_normalized_problem_hash_crosses_splits(self):
        by_hash = {}
        for r in self.records:
            prev = by_hash.setdefault(r["problem_hash"], r["split"])
            self.assertEqual(prev, r["split"],
                             f"hash {r['problem_hash'][:12]} in {prev} and {r['split']}")

    def test_train_test_problem_overlap_is_zero(self):
        train = {r["problem_id"] for r in self.by_split["train"]}
        test = {r["problem_id"] for r in self.by_split["test"]}
        self.assertEqual(train & test, set())

    def test_val_test_problem_overlap_is_zero(self):
        val = {r["problem_id"] for r in self.by_split["val"]}
        test = {r["problem_id"] for r in self.by_split["test"]}
        self.assertEqual(val & test, set())

    def test_train_val_problem_overlap_is_zero(self):
        train = {r["problem_id"] for r in self.by_split["train"]}
        val = {r["problem_id"] for r in self.by_split["val"]}
        self.assertEqual(train & val, set())

    def test_no_utterance_text_crosses_splits(self):
        """Catches paraphrase/persona/generator variants that share a rendering."""
        by_text = {}
        for r in self.records:
            prev = by_text.setdefault(r["text"], r["split"])
            self.assertEqual(prev, r["split"],
                             "an utterance appears in two splits")

    def test_assert_no_leakage_helper_agrees(self):
        assignment = {r["problem_id"]: r["split"] for r in self.records}
        report = assert_no_leakage(self.records, assignment)
        for key, overlap in report.items():
            if "|" in key:
                self.assertEqual(overlap, 0, f"{key} overlap is {overlap}")

    def test_leakage_assertion_actually_fires(self):
        """A planted leak must raise, or a clean run proves nothing.

        Relabelling one test utterance with a train problem's ID leaves its
        normalized-problem hash counted under both splits — exactly the kind of
        mislabelling the assertion exists to catch.
        """
        assignment = {r["problem_id"]: r["split"] for r in self.records}
        test_pid = sorted(p for p, s in assignment.items() if s == "test")[0]
        train_pid = sorted(p for p, s in assignment.items() if s == "train")[0]

        planted = [dict(r) for r in self.records]
        victim = next(r for r in planted if r["problem_id"] == test_pid)
        victim["problem_id"] = train_pid

        with self.assertRaises(AssertionError):
            assert_no_leakage(planted, assignment)

    # -- shape, coverage, determinism -------------------------------

    def test_partition_sizes(self):
        counts = Counter(p["split"] for p in self.manifest["problems"])
        self.assertEqual(dict(counts), dict(SPLIT_SIZES))

    def test_every_problem_assigned_exactly_once(self):
        ids = [p["problem_id"] for p in self.manifest["problems"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 50)

    def test_all_seven_states_present_in_val_and_test(self):
        for split in ("val", "test"):
            present = {r["cognitive_state"] for r in self.by_split[split]}
            self.assertEqual(present, set(COGNITIVE_STATES), f"{split} is missing a class")

    def test_all_seven_states_present_per_generator_in_test(self):
        for gen in GENERATORS:
            present = {r["cognitive_state"] for r in self.by_split["test"]
                       if r["generator"] == gen}
            self.assertEqual(present, set(COGNITIVE_STATES),
                             f"{gen} is missing a class in test")

    def test_all_four_generators_present_in_test(self):
        self.assertEqual({r["generator"] for r in self.by_split["test"]}, set(GENERATORS))

    def test_split_is_deterministic(self):
        again, _ = build(DEFAULT_SEED)
        self.assertEqual({r["record_id"]: r["split"] for r in self.records},
                         {r["record_id"]: r["split"] for r in again})

    def test_problem_ids_are_seed_independent(self):
        """P## is keyed to the problem hash, so a different seed must not rename it."""
        other, _ = build(DEFAULT_SEED + 1)
        a = {r["problem_hash"]: r["problem_id"] for r in self.records}
        b = {r["problem_hash"]: r["problem_id"] for r in other}
        self.assertEqual(a, b)

    def test_normalization_folds_typographic_variants(self):
        self.assertEqual(normalize_problem("How many  moles of H₂O?"),
                         normalize_problem("how many moles of H₂O?"))
        self.assertEqual(problem_hash(" A  B "), problem_hash("a b"))

    def test_pool_covers_the_whole_corpus_without_double_counting(self):
        self.assertEqual(len(self.records), 8750)
        self.assertEqual(len({r["text"] for r in self.records}), 8750)


class MatchedGeneratorTest(unittest.TestCase):
    """The matched cross-generator sample must be exactly balanced and deterministic."""

    @classmethod
    def setUpClass(cls):
        records, cls.manifest = build(DEFAULT_SEED)
        cls.test_rows = [r for r in records if r["split"] == "test"]
        cls.matched, cls.info = match_cells(cls.test_rows, MATCH_SEED)
        cls.by_gen = {g: [r for r in cls.matched if r["generator"] == g]
                      for g in GENERATORS}

    def test_equal_total_support_per_generator(self):
        sizes = {g: len(rows) for g, rows in self.by_gen.items()}
        self.assertEqual(len(set(sizes.values())), 1, f"unequal support: {sizes}")

    def test_identical_per_state_support(self):
        ref = Counter(r["cognitive_state"] for r in self.by_gen[GENERATORS[0]])
        for g in GENERATORS[1:]:
            self.assertEqual(Counter(r["cognitive_state"] for r in self.by_gen[g]), ref,
                             f"{g} per-state support differs")

    def test_identical_per_cell_support(self):
        """The strong claim: matched problem-by-problem AND label-by-label."""
        ref = Counter(r["matched_cell"] for r in self.by_gen[GENERATORS[0]])
        for g in GENERATORS[1:]:
            self.assertEqual(Counter(r["matched_cell"] for r in self.by_gen[g]), ref,
                             f"{g} per-cell support differs")

    def test_every_state_survives_matching(self):
        for g in GENERATORS:
            present = {r["cognitive_state"] for r in self.by_gen[g]}
            self.assertEqual(present, set(COGNITIVE_STATES), f"{g} lost a class")

    def test_matched_examples_are_all_test_problems(self):
        allowed = set(self.manifest["distributions"]["test"]["problem_ids"])
        self.assertEqual({r["problem_id"] for r in self.matched} - allowed, set())

    def test_no_example_selected_twice(self):
        keys = [(r["source_file"], r["source_line"]) for r in self.matched]
        self.assertEqual(len(keys), len(set(keys)))

    def test_matching_is_deterministic(self):
        again, info = match_cells(self.test_rows, MATCH_SEED)
        self.assertEqual([(r["source_file"], r["source_line"]) for r in self.matched],
                         [(r["source_file"], r["source_line"]) for r in again])
        self.assertEqual(self.info["n_per_generator"], info["n_per_generator"])

    def test_a_different_match_seed_changes_the_draw(self):
        """Otherwise the seed is not actually driving the sampling."""
        other, _ = match_cells(self.test_rows, MATCH_SEED + 1)
        self.assertNotEqual([(r["source_file"], r["source_line"]) for r in self.matched],
                            [(r["source_file"], r["source_line"]) for r in other])

    def test_k_never_exceeds_what_a_generator_has(self):
        available = {}
        for r in self.test_rows:
            available.setdefault((r["problem_id"], r["cognitive_state"]), Counter())
            available[(r["problem_id"], r["cognitive_state"])][r["generator"]] += 1
        for entry in self.info["cell_audit"]:
            key = (entry["problem_id"], entry["cognitive_state"])
            for g in GENERATORS:
                self.assertLessEqual(entry["matched_k"], available[key][g])

    def test_dropped_cells_are_reported_not_hidden(self):
        self.assertEqual(self.info["n_cells_used"] + self.info["n_cells_dropped"],
                         self.info["n_cells_total"])
        self.assertEqual(len(self.info["dropped_cells"]), self.info["n_cells_dropped"])


class MatchedArtefactTest(unittest.TestCase):
    """The written matched artefacts must agree with a fresh rebuild."""

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(RESULTS_DIR, "matched_metrics.json")
        if not os.path.exists(cls.path):
            raise unittest.SkipTest("matched results not generated yet; "
                                    "run eval/matched_generator_eval.py")
        with open(cls.path, encoding="utf-8") as fh:
            cls.metrics = json.load(fh)

    def test_reported_support_is_identical_across_generators(self):
        support = self.metrics["support"]
        ref = support[GENERATORS[0]]
        for g in GENERATORS[1:]:
            self.assertEqual(support[g], ref)

    def test_reported_n_matches_a_fresh_match(self):
        records, _ = build(self.metrics["split"]["seed"])
        _, info = match_cells([r for r in records if r["split"] == "test"],
                              self.metrics["matching"]["seed"])
        self.assertEqual(info["n_per_generator"],
                         self.metrics["matching"]["n_per_generator"])
        self.assertEqual(info["n_cells_dropped"],
                         self.metrics["matching"]["n_cells_dropped"])

    def test_every_arm_scored_on_the_same_n(self):
        ns = {g: self.metrics["conditions"][g]["n_examples"] for g in GENERATORS}
        self.assertEqual(len(set(ns.values())), 1, f"unequal n: {ns}")

    def test_manifest_rows_match_reported_n(self):
        import csv as _csv
        path = os.path.join(RESULTS_DIR, "matched_manifest.csv")
        with open(path, encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        self.assertEqual(len(rows), self.metrics["matching"]["n_total"])
        per_gen = Counter(r["generator"] for r in rows)
        self.assertEqual(set(per_gen.values()),
                         {self.metrics["matching"]["n_per_generator"]})

    def test_manifest_carries_no_utterance_text(self):
        """The manifest is a reproducibility record, not a copy of the corpus."""
        path = os.path.join(RESULTS_DIR, "matched_manifest.json")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for ex in data["examples"]:
            self.assertNotIn("text", ex)
            self.assertNotIn("think_aloud", ex)


class ResultsArtefactTest(unittest.TestCase):
    """The written manifest must agree with a freshly rebuilt split."""

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(RESULTS_DIR, "split_manifest.json")
        if not os.path.exists(cls.path):
            raise unittest.SkipTest("results not generated yet; "
                                    "run eval/problem_disjoint_eval.py")
        with open(cls.path, encoding="utf-8") as fh:
            cls.manifest = json.load(fh)

    def test_manifest_matches_a_fresh_rebuild(self):
        _, fresh = build(self.manifest["seed"])
        self.assertEqual(
            {p["problem_id"]: p["split"] for p in self.manifest["problems"]},
            {p["problem_id"]: p["split"] for p in fresh["problems"]})

    def test_manifest_leakage_report_is_all_zero(self):
        for key, overlap in self.manifest["leakage_check"].items():
            if "|" in key:
                self.assertEqual(overlap, 0, f"{key} = {overlap}")

    def test_metrics_use_the_manifest_split(self):
        metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
        if not os.path.exists(metrics_path):
            self.skipTest("metrics.json not generated yet")
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)
        manifest_test = set(self.manifest["distributions"]["test"]["problem_ids"])
        self.assertEqual(set(metrics["split"]["test_problem_ids"]), manifest_test)

        # Every generator arm must be scored on exactly the same test problems.
        for gen in GENERATORS:
            cond = metrics["conditions"][f"C_generator::{gen}"]
            self.assertEqual(set(cond["problem_ids"]), manifest_test,
                             f"{gen} was not scored on the manifest test problems")

        # And the problem-disjoint arm must not touch a train or val problem.
        forbidden = (set(self.manifest["distributions"]["train"]["problem_ids"])
                     | set(self.manifest["distributions"]["val"]["problem_ids"]))
        b = metrics["conditions"]["B_problem_disjoint"]
        self.assertEqual(set(b["problem_ids"]) & forbidden, set())

    def test_checksums_match_the_written_artefacts(self):
        checks = os.path.join(RESULTS_DIR, "CHECKSUMS.sha256")
        if not os.path.exists(checks):
            self.skipTest("checksums not generated yet")
        with open(checks, encoding="utf-8") as fh:
            for line in fh:
                digest, name = line.strip().split("  ", 1)
                path = (os.path.join(RESULTS_DIR, name) if os.sep not in name
                        else os.path.join(REPO_ROOT, name))
                if not os.path.exists(path):
                    continue
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                self.assertEqual(h.hexdigest(), digest, f"{name} changed since the run")


if __name__ == "__main__":
    unittest.main()
