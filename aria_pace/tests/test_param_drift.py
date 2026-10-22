"""A8 checks on SYNTHETIC sequences: the grouping rule, the spread summary, and that own-skill and pooled scoring agree exactly when no skill has own parameters."""
import unittest

import numpy as np

from aria_pace import baselines as B
from aria_pace import param_drift as PD
from aria_pace import policy as P
from aria_pace.labels import first_mastery
from aria_pace.loader import Sequence


def pool(n_students=120, seed=5):
    rng = np.random.RandomState(seed)
    seqs = []
    for u in range(1, n_students + 1):
        for a in range(2):
            n = int(rng.randint(3, 16))
            kc = f"k{rng.randint(0, 3)}"
            p = {"k0": 0.8, "k1": 0.5, "k2": 0.3}[kc] + rng.uniform(-0.1, 0.1)
            c = (rng.uniform(size=n) < p).astype(np.int8)
            seqs.append(Sequence(u, a, kc, c, n, first_mastery(c), np.zeros(n, dtype=np.int8)))
    return seqs


class Grouping(unittest.TestCase):
    def test_rule_is_first_match_in_order(self):
        self.assertEqual(PD.group_of("Equation Solving Two or Fewer Steps"), "algebra-type")
        self.assertEqual(PD.group_of("Order of Operations All"), "algebra-type")
        self.assertEqual(PD.group_of("Pythagorean Theorem"), "geometry/measurement")
        self.assertEqual(PD.group_of("Circle Graph"), "data/probability")
        self.assertEqual(PD.group_of("Addition and Subtraction Fractions"), "number/arithmetic")
        self.assertEqual(PD.group_of("Write Linear Equation from Graph"), "algebra-type")       # algebra keywords are tested first
        self.assertEqual(PD.group_of("Something Else Entirely"), "other")


class Spread(unittest.TestCase):
    def test_spread_counts_only_skills_with_enough_sequences(self):
        seqs = pool()
        bkt = B.BKT(10).fit(seqs)
        names = {"k0": "Equation Solving", "k1": "Area", "k2": "Probability"}
        s = PD.spread(bkt.kc_fits, 10, names, len(seqs))
        self.assertEqual(s["n_skills_fitted"], 3)
        for p in PD.PARAMS:
            row = s["parameters"][p]
            self.assertLessEqual(row["min"], row["median"])
            self.assertLessEqual(row["median"], row["max"])
            self.assertLessEqual(row["q25"], row["q75"])
        self.assertEqual(set(s["by_group"]), {"algebra-type", "geometry/measurement", "data/probability"})
        self.assertIsInstance(s["by_group"]["algebra-type"]["parameters"], str)      # fewer than 5 skills: not summarised


class Decisions(unittest.TestCase):
    def test_no_flips_when_no_skill_has_own_parameters(self):
        seqs = pool()
        bkt = B.BKT(10_000).fit(seqs)                       # nothing reaches min_seq, so every skill is scored with the pooled parameters
        res = PD.decisions(seqs, bkt, {}, {"intercept": 0.0, "slope": 1.0}, {"quiet": 0.5, "balanced": 0.3, "supportive": 0.1})
        for prof in P.PROFILES:
            self.assertEqual(res["profiles"][prof]["flips"], 0)
            self.assertAlmostEqual(res["profiles"][prof]["own_skill_params"]["reduction_vs_always_help"],
                                   res["profiles"][prof]["pooled_params"]["reduction_vs_always_help"], places=12)

    def test_flips_are_possible_when_skills_differ(self):
        seqs = pool()
        bkt = B.BKT(10).fit(seqs)
        res = PD.decisions(seqs, bkt, {}, {"intercept": 0.0, "slope": 1.0}, {"quiet": 0.5, "balanced": 0.3, "supportive": 0.1})
        self.assertGreater(res["decision_cells_with_verified_error"], 0)
        self.assertTrue(any(res["profiles"][p]["flips"] > 0 for p in P.PROFILES))


if __name__ == "__main__":
    unittest.main()
