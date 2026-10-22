"""A9 checks on SYNTHETIC sequences: the train-only rank transform, eligibility, and the structure of the comparison (no ASSISTments data)."""
import unittest

import numpy as np

from aria_pace import hazard as H
from aria_pace.labels import first_mastery


def synth(n_students=300, seed=11):
    rng = np.random.RandomState(seed)
    out = []
    for u in range(1, n_students + 1):
        for _ in range(2):
            n = int(rng.randint(3, 16))
            c = (rng.uniform(size=n) < rng.uniform(0.3, 0.9)).astype(np.int8)
            rt = np.exp(rng.normal(9.5, 1.0, size=n))
            rt[rng.uniform(size=n) < 0.01] = -5.0                       # clock artefacts
            out.append(H.XSeq(u, f"k{rng.randint(0, 3)}", c, rng.poisson(0.3, size=n).astype(float), rt, n, first_mastery(c)))
    return out


class Rank(unittest.TestCase):
    def test_rank_is_fitted_on_train_only_and_monotone(self):
        tr = synth(100, 1)
        r = H.RankLogRT().fit(tr)
        x = np.array([100.0, 1000.0, 10000.0, 100000.0])
        t = r.transform(x)
        self.assertTrue(np.all(np.diff(t) > 0))
        self.assertTrue(np.all((t >= 0) & (t <= 1)))
        self.assertAlmostEqual(r.transform(np.array([-3.0]))[0], r.transform(np.array([np.exp(r.median)]))[0], places=2)   # missing -> median
        far = r.transform(np.array([1e12]))[0]
        self.assertGreater(far, 0.99)


class Eligibility(unittest.TestCase):
    def test_k_window(self):
        seqs = synth(60, 2)
        for k in H.KS:
            for q, y in H.eligible(seqs, k):
                self.assertGreaterEqual(q.n, k)
                self.assertGreater(q.m, k)
                self.assertIn(y, (0, 1))


class Comparison(unittest.TestCase):
    def test_structure_and_that_noise_is_not_promoted(self):
        tr, va = synth(300, 3), synth(300, 4)
        out = H.run(tr, va, 10, log=lambda s: None)
        self.assertEqual(out["status"], H.STATUS)
        self.assertEqual(set(out["per_k"]), {str(k) for k in H.KS})
        scored = [r for r in out["per_k"].values() if r["models"]]
        self.assertGreater(len(scored), 5)
        for r in scored:
            self.assertEqual(set(r["models"]), set(H.FEATURE_SETS))
            self.assertEqual(r["in_frozen_policy_window"], r["k"] >= 3)
        self.assertFalse(out["promotion_rule"]["full_model_promoted"])       # random response times and random outcomes: nothing to promote


if __name__ == "__main__":
    unittest.main()
