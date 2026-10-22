"""Code test for scripts/annotation_kappa.py. The annotator files are generated in a temp directory from a SYNTHETIC key and are
labelled CODE TEST ONLY: nothing here is, or may be mistaken for, a real annotation."""
import csv
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import annotation_kappa as K  # noqa: E402

CATS = ["PLANNING", "FLOW", "CONFUSED", "RUSHING", "FRUSTRATED", "STUCK", "INSIGHT"]


def synthetic_key(n=150):
    items = {}
    for i in range(n):
        items[f"ITEM_{i + 1:03d}"] = {"true_category": CATS[i % 7], "contains_own_label_word": i % 5 == 0}
    return {"categories": CATS, "n_items": n, "items": items, "_README": "CODE TEST ONLY: synthetic key"}


def write_sheet(path, labels, n=150):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "problem_text", "think_aloud", "predicted_category", "confidence"])
        for i in range(n):
            w.writerow([f"ITEM_{i + 1:03d}", "CODE TEST ONLY", "CODE TEST ONLY", labels[i], 3])


class Kappa(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(K.cohen_kappa(list("aabb"), list("aabb")), 1.0)
        # textbook 2x2: po = 0.7, pe = 0.5 -> kappa 0.4
        a = ["y"] * 25 + ["y"] * 10 + ["n"] * 5 + ["n"] * 10 + ["n"] * 0
        b = ["y"] * 25 + ["n"] * 10 + ["y"] * 5 + ["n"] * 10
        k = K.cohen_kappa(a, b)
        po = sum(x == y for x, y in zip(a, b)) / len(a)
        pa, pb = a.count("y") / len(a), b.count("y") / len(b)
        pe = pa * pb + (1 - pa) * (1 - pb)
        self.assertAlmostEqual(k, (po - pe) / (1 - pe))
        try:
            from sklearn.metrics import cohen_kappa_score
            self.assertAlmostEqual(k, cohen_kappa_score(a, b), places=10)
        except ImportError:
            pass

    def test_chance_level_is_near_zero(self):
        rng = random.Random(1)
        a = [rng.choice(CATS) for _ in range(5000)]
        b = [rng.choice(CATS) for _ in range(5000)]
        self.assertLess(abs(K.cohen_kappa(a, b, CATS)), 0.03)

    def test_end_to_end_on_synthetic_annotators(self):
        rng = random.Random(7)
        key = synthetic_key()
        truth = [key["items"][f"ITEM_{i + 1:03d}"]["true_category"] for i in range(150)]
        r1 = [t if rng.random() < 0.8 else rng.choice(CATS) for t in truth]
        r2 = [t if rng.random() < 0.8 else rng.choice(CATS) for t in truth]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "key.json").write_text(json.dumps(key))
            write_sheet(d / "CODE_TEST_ONLY_r1.csv", r1)
            write_sheet(d / "CODE_TEST_ONLY_r2.csv", r2)
            p = subprocess.run([sys.executable, str(ROOT / "scripts" / "annotation_kappa.py"), "--rater1", str(d / "CODE_TEST_ONLY_r1.csv"),
                                "--rater2", str(d / "CODE_TEST_ONLY_r2.csv"), "--key", str(d / "key.json"), "--out", str(d / "out.json"),
                                "--label", "CODE TEST ONLY (synthetic annotators)"], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            out = json.loads((d / "out.json").read_text())
            self.assertEqual(out["n_items"], 150)
            self.assertAlmostEqual(out["inter_rater"]["kappa"], K.cohen_kappa(r1, r2, CATS))
            self.assertGreater(out["inter_rater"]["kappa"], 0.5)
            lo, hi = out["inter_rater"]["kappa_ci95"]
            self.assertLessEqual(lo, out["inter_rater"]["kappa"])
            self.assertGreaterEqual(hi, out["inter_rater"]["kappa"])
            self.assertEqual(out["items_with_own_label_word"]["n"] + out["items_without_own_label_word"]["n"], 150)
            self.assertIn("CODE TEST ONLY", out["label"])

    def test_refusals(self):
        key = synthetic_key()
        good = [CATS[i % 7] for i in range(150)]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_sheet(d / "ok.csv", good)
            write_sheet(d / "blank.csv", [""] + good[1:])
            write_sheet(d / "bad.csv", ["NOPE"] + good[1:])
            with self.assertRaises(ValueError):
                K.read_sheet(d / "blank.csv", CATS, 150)
            with self.assertRaises(ValueError):
                K.read_sheet(d / "bad.csv", CATS, 150)
            with self.assertRaises(ValueError):
                K.read_sheet(d / "ok.csv", CATS, 149)
            r = K.read_sheet(d / "ok.csv", CATS, 150)
            with self.assertRaises(ValueError):                     # a copied sheet is not an independent rater
                K.analyse(r, dict(r), key, "x")


if __name__ == "__main__":
    unittest.main()
