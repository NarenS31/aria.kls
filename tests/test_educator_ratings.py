import unittest

from eval.analyze_educator_ratings import binary_value, weighted_kappa


class EducatorRatingAnalysisTests(unittest.TestCase):
    def test_perfect_constant_agreement_is_one(self):
        self.assertEqual(weighted_kappa([4, 4, 4], [4, 4, 4], [1, 2, 3, 4, 5]), 1.0)

    def test_disagreement_is_below_perfect(self):
        score = weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5])
        self.assertLess(score, 1.0)

    def test_binary_parser_rejects_non_binary_values(self):
        with self.assertRaises(ValueError):
            binary_value("maybe")


if __name__ == "__main__":
    unittest.main()
