"""A7 checks on SYNTHETIC sequences (no ASSISTments data): folds are disjoint and cover the pool, a held-out student's data cannot change that
fold's calibrator or tau, the result has the expected shape, and the pooled curve is monotone in tau."""
import copy
import unittest

import numpy as np

from aria_pace import cv_tradeoff as CV
from aria_pace import mastery_sens as MS
from aria_pace import policy as P
from aria_pace.labels import first_mastery
from aria_pace.loader import Sequence


def synthetic_pool(n_students=160, seed=3):
    rng = np.random.RandomState(seed)
    seqs = []
    for u in range(1, n_students + 1):
        for a in range(2):
            n = int(rng.randint(3, 16))
            p = rng.uniform(0.3, 0.9)
            c = (rng.uniform(size=n) < p).astype(np.int8)
            seqs.append(Sequence(u, a, f"k{rng.randint(0, 3)}", c, n, first_mastery(c), np.zeros(n, dtype=np.int8)))
    return seqs


FROZEN = {"quiet": 0.57, "balanced": 0.19, "supportive": 0.09}
DEFS = MS.DEFS[:2]            # D3 (control) and D4 keep the test fast


class Folds(unittest.TestCase):
    def test_folds_are_disjoint_and_cover(self):
        ids = list(range(1, 303))
        folds = CV.make_folds(ids)
        flat = [u for f in folds for u in f]
        self.assertEqual(sorted(flat), ids)
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(len(folds), 5)
        self.assertEqual(folds, CV.make_folds(ids))                         # deterministic

    def test_fit_cal_split_is_a_partition_of_the_remaining_students(self):
        rest = list(range(1, 201))
        fit, cal = CV.fit_cal_split(rest, 2)
        self.assertEqual(sorted(fit + cal), rest)
        self.assertEqual(len(fit), 150)


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seqs = synthetic_pool()
        cls.res = CV.run(cls.seqs, 10, FROZEN, defs=DEFS, log=lambda s: None)

    def test_shape_and_status(self):
        self.assertEqual(self.res["status"], CV.STATUS)
        for d in DEFS:
            r = self.res["definitions"][d.name]
            self.assertEqual(len(r["saving"]), len(P.TAU_GRID))
            self.assertEqual(set(r["chosen_tau_pooled_result"]), set(P.PROFILES))
            for prof in P.PROFILES:
                self.assertEqual(len(r["tau_movement"][prof]["fold_taus"]), CV.N_FOLDS)

    def test_saving_falls_as_tau_rises(self):
        r = self.res["definitions"]["D3"]
        s = np.array(r["saving"], float)
        self.assertTrue(np.all(np.diff(s[~np.isnan(s)]) >= -1e-12))          # a higher threshold never fires more often
        self.assertAlmostEqual(s[0], 0.0, places=9) if r["tau_grid"][0] == 0.0 else None

    def test_heldout_students_cannot_change_their_own_fold_choices(self):
        ids = sorted({q.user_id for q in self.seqs})
        folds = CV.make_folds(ids)
        f = 1
        victim = folds[f][0]
        changed = []
        for q in self.seqs:
            if q.user_id == victim:
                c = (1 - q.correct).astype(np.int8)                             # flip every response of a held-out student
                q = Sequence(q.user_id, q.assignment_id, q.kc, c, q.n, first_mastery(c), q.first_action)
            changed.append(q)
        res2 = CV.run(changed, 10, FROZEN, defs=DEFS, log=lambda s: None)
        for d in DEFS:
            for prof in P.PROFILES:
                a = self.res["definitions"][d.name]["tau_movement"][prof]["fold_taus"][f]
                b = res2["definitions"][d.name]["tau_movement"][prof]["fold_taus"][f]
                self.assertEqual(a, b, f"{d.name} {prof}: held-out data changed the tau chosen for its own fold")
            self.assertEqual(self.res["definitions"][d.name]["fold_platt"][f], res2["definitions"][d.name]["fold_platt"][f])


if __name__ == "__main__":
    unittest.main()
