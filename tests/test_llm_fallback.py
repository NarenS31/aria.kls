"""Validation for the LLM-fallback transfer experiment.

Two jobs. First, prove the experiment ran on the untouched canonical example set:
the manifest, the 339-per-generator balance, the 70 matched cells and the
problem-disjointness must all be exactly what the canonical run froze. Second,
pin the few-shot leakage finding in place, so a later edit that quietly drops or
launders it fails here rather than passing unnoticed into a paper.
"""

import csv
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

from build_problem_disjoint_split import DEFAULT_SEED, GENERATORS, build  # noqa: E402
from llm_fallback_eval import (  # noqa: E402
    CANONICAL_DIR,
    build_block,
    fewshot_audit,
    load_manifest_examples,
)
from matched_generator_eval import MATCH_SEED, match_cells  # noqa: E402
from metacognition.analyzer import COGNITIVE_STATES, CognitiveStateAnalyzer  # noqa: E402
import metacognition.analyzer as analyzer_mod  # noqa: E402

RESULTS_DIR = os.path.join(EVAL_ROOT, "data", "eval", "llm_fallback_20260827")
METRICS = os.path.join(RESULTS_DIR, "fallback_metrics.json")

# Canonical matched numbers, from problem_disjoint_20260826/matched_metrics.json.
# Hard-coded on purpose: the control must reproduce these, and reading them from
# the same file the control writes would make the check circular.
CANONICAL_MATCHED = {
    "llama3.1:8b": (0.8820, 0.8796),
    "mistral:7b": (0.8142, 0.8104),
    "gemma2:9b": (0.8378, 0.8317),
    "phi3:medium": (0.6313, 0.6226),
}


class FrozenManifestTest(unittest.TestCase):
    """The experiment must run on exactly the canonical example set."""

    @classmethod
    def setUpClass(cls):
        cls.examples, cls.split_manifest = load_manifest_examples()
        cls.by_gen = {g: [e for e in cls.examples if e["generator"] == g]
                      for g in GENERATORS}

    def test_manifest_checksum_unchanged(self):
        """The canonical manifest must be byte-identical to what it recorded."""
        path = os.path.join(CANONICAL_DIR, "matched_manifest.csv")
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        checks = os.path.join(CANONICAL_DIR, "CHECKSUMS.sha256")
        recorded = {}
        with open(checks, encoding="utf-8") as fh:
            for line in fh:
                digest, name = line.rstrip("\n").split("  ", 1)
                recorded[name] = digest
        self.assertEqual(h.hexdigest(), recorded["matched_manifest.csv"],
                         "the canonical matched manifest has changed on disk")

    def test_total_is_1356(self):
        self.assertEqual(len(self.examples), 1356)

    def test_339_per_generator(self):
        for g in GENERATORS:
            self.assertEqual(len(self.by_gen[g]), 339, f"{g} is not 339")

    def test_70_cells_all_matched(self):
        cells = {e["matched_cell"] for e in self.examples}
        self.assertEqual(len(cells), 70)
        ref = Counter(e["matched_cell"] for e in self.by_gen[GENERATORS[0]])
        for g in GENERATORS[1:]:
            self.assertEqual(Counter(e["matched_cell"] for e in self.by_gen[g]), ref,
                             f"{g} cell support differs")

    def test_identical_per_condition_support(self):
        ref = Counter(e["cognitive_state"] for e in self.by_gen[GENERATORS[0]])
        for g in GENERATORS[1:]:
            self.assertEqual(Counter(e["cognitive_state"] for e in self.by_gen[g]), ref)

    def test_no_problem_level_leakage_in_the_evaluated_set(self):
        """Every scored example must belong to a held-out TEST problem."""
        test_ids = set(self.split_manifest["distributions"]["test"]["problem_ids"])
        train_val = (set(self.split_manifest["distributions"]["train"]["problem_ids"])
                     | set(self.split_manifest["distributions"]["val"]["problem_ids"]))
        got = {e["problem_id"] for e in self.examples}
        self.assertTrue(got <= test_ids, "a non-test problem entered the example set")
        self.assertEqual(got & train_val, set())

    def test_labels_match_the_manifest(self):
        with open(os.path.join(CANONICAL_DIR, "matched_manifest.csv"),
                  encoding="utf-8") as fh:
            rows = {(r["source_file"], r["source_line"]): r for r in csv.DictReader(fh)}
        for e in self.examples:
            row = rows[(e["source_file"], str(e["source_line"]))]
            self.assertEqual(e["cognitive_state"], row["cognitive_state"])
            self.assertEqual(e["problem_id"], row["problem_id"])

    def test_rehydration_agrees_with_a_fresh_match(self):
        records, _ = build(DEFAULT_SEED)
        fresh, _ = match_cells([r for r in records if r["split"] == "test"], MATCH_SEED)
        self.assertEqual(
            sorted((r["source_file"], r["source_line"]) for r in fresh),
            sorted((e["source_file"], e["source_line"]) for e in self.examples))


class FewShotLeakageTest(unittest.TestCase):
    """The leakage finding must stay visible and the clean block must stay clean."""

    @classmethod
    def setUpClass(cls):
        records, cls.split_manifest = build(DEFAULT_SEED)
        cls.audit = fewshot_audit(records, cls.split_manifest)
        cls.selected = cls.audit["_selected"]

    def test_every_fewshot_example_resolves(self):
        """An unresolved example is an unquantified leakage risk, not a rounding error."""
        self.assertEqual(self.audit["by_split"].get("UNRESOLVED", 0), 0)
        self.assertEqual(self.audit["n_examples_in_prompt"], 56)

    def test_shipped_block_is_contaminated(self):
        self.assertTrue(self.audit["leakage"])
        self.assertGreater(self.audit["by_split"].get("test", 0), 0)
        self.assertEqual(self.audit["test_problems_in_prompt"], ["P14", "P39"])

    def test_shipped_block_matches_the_imported_one(self):
        """Re-rendering must reproduce what the analyzer actually built at import."""
        rendered = build_block(self.selected, {"train", "val", "test", "UNRESOLVED"})
        self.assertEqual(rendered, analyzer_mod._FEWSHOT_BLOCK)

    def test_decontaminated_block_has_no_heldout_problems(self):
        held_out = (set(self.split_manifest["distributions"]["test"]["problem_ids"])
                    | set(self.split_manifest["distributions"]["val"]["problem_ids"]))
        for item, entry in self.selected:
            if entry["split"] != "train":
                continue
            self.assertNotIn(entry["problem_id"], held_out)

    def test_decontaminated_block_is_a_strict_subset(self):
        """B' may only remove examples — never add, rewrite or re-select one."""
        decon = build_block(self.selected, {"train"})
        shipped = build_block(self.selected, {"train", "val", "test", "UNRESOLVED"})
        decon_lines = [l for l in decon.splitlines() if l.startswith("[")]
        shipped_lines = [l for l in shipped.splitlines() if l.startswith("[")]
        self.assertTrue(set(decon_lines) <= set(shipped_lines))
        self.assertEqual(len(decon_lines), 25)
        self.assertEqual(len(shipped_lines), 56)

    def test_classifier_source_is_unmodified(self):
        """The experiment must not have edited the frozen analyzer."""
        import subprocess
        out = subprocess.run(
            ["git", "diff", "--stat", "e9f0c56", "--", "eval/metacognition/analyzer.py"],
            cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, "", "analyzer.py differs from the frozen commit")


class ResultsArtefactTest(unittest.TestCase):
    """The written metrics must agree with the report and with the canonical run."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(METRICS):
            raise unittest.SkipTest("results not generated yet; "
                                    "run eval/llm_fallback_eval.py")
        with open(METRICS, encoding="utf-8") as fh:
            cls.m = json.load(fh)

    def test_control_reproduces_canonical_per_generator_metrics(self):
        a = self.m["configs"].get("A_lexical_only")
        self.assertIsNotNone(a, "the lexical-only control did not run")
        for gen, (acc, f1) in CANONICAL_MATCHED.items():
            got = a["by_generator"][gen]
            self.assertAlmostEqual(got["accuracy"], acc, places=4,
                                   msg=f"{gen} accuracy drifted from canonical")
            self.assertAlmostEqual(got["macro_f1"], f1, places=4,
                                   msg=f"{gen} macro-F1 drifted from canonical")

    def test_control_makes_no_llm_calls(self):
        a = self.m["configs"]["A_lexical_only"]
        self.assertEqual(a["fallback_activation"]["n_triggered"], 0)

    def test_every_config_scored_the_same_1356_examples(self):
        for name, c in self.m["configs"].items():
            self.assertEqual(c["n_examples"], 1356, f"{name} scored {c['n_examples']}")
            self.assertEqual(c["n_unique_problems"], 10, name)
            for g in GENERATORS:
                self.assertEqual(c["by_generator"][g]["n_examples"], 339, f"{name}/{g}")

    def test_manifest_sha_recorded_and_current(self):
        path = os.path.join(REPO_ROOT, self.m["manifest"]["source"])
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        self.assertEqual(h.hexdigest(), self.m["manifest"]["sha256"])

    def test_leakage_audit_is_recorded_in_the_artefact(self):
        fa = self.m["fewshot_audit"]
        self.assertTrue(fa["leakage"])
        self.assertEqual(fa["by_split"].get("test"), 17)
        self.assertEqual(fa["by_split"].get("val"), 14)
        self.assertEqual(fa["by_split"].get("train"), 25)

    def test_contaminated_arm_is_labelled_as_such(self):
        b = self.m["configs"].get("B_hybrid_shipped")
        if b is None:
            self.skipTest("config B not run")
        self.assertIn("CONTAMINATED", b["fewshot_block"])

    def test_forced_arm_is_labelled_an_ablation(self):
        c = self.m["configs"].get("C_forced_llm_ablation")
        if c is None:
            self.skipTest("config C not run")
        self.assertIn("ablation", c["route"].lower())
        self.assertEqual(c["fallback_activation"]["n_triggered"], 1356)

    def test_activation_counts_are_internally_consistent(self):
        for name, c in self.m["configs"].items():
            act = c["fallback_activation"]
            by_cond = sum(v["triggered"] for v in c["by_condition_activation"].values())
            self.assertEqual(act["n_triggered"], by_cond, f"{name} condition split")
            by_gen = sum(c["by_generator"][g]["fallback_triggered"] for g in GENERATORS)
            self.assertEqual(act["n_triggered"], by_gen, f"{name} generator split")

    def test_per_example_csv_matches_the_json(self):
        path = os.path.join(RESULTS_DIR, "per_example_predictions.csv")
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for name, c in self.m["configs"].items():
            sub = [r for r in rows if r["config"] == name]
            self.assertEqual(len(sub), 1356, name)
            correct = sum(1 for r in sub if r["correct"] == "True")
            self.assertAlmostEqual(correct / len(sub), c["accuracy"], places=4, msg=name)
            trig = sum(1 for r in sub if r["fallback_triggered"] == "True")
            self.assertEqual(trig, c["fallback_activation"]["n_triggered"], name)

    def test_change_counts_add_up(self):
        for name, ch in self.m.get("changes", {}).items():
            self.assertEqual(ch["n_changed"],
                             ch["corrected"] + ch["broke"] + ch["wrong_to_wrong"], name)
            self.assertEqual(ch["net"], ch["corrected"] - ch["broke"], name)

    def test_report_numbers_match_the_json(self):
        report = os.path.join(RESULTS_DIR, "FALLBACK_RESULTS.md")
        if not os.path.exists(report):
            self.skipTest("FALLBACK_RESULTS.md not written yet")
        text = open(report, encoding="utf-8").read()
        for name, c in self.m["configs"].items():
            self.assertIn(f"{c['accuracy']:.4f}", text,
                          f"{name} accuracy {c['accuracy']:.4f} absent from the report")
        for bad in ("/Users/", "narensara", "@gmail"):
            self.assertNotIn(bad, text, f"report leaks {bad}")


if __name__ == "__main__":
    unittest.main()
