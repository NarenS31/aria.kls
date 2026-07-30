import unittest

from eval.power_analysis import required_sample


class PowerAnalysisTests(unittest.TestCase):
    def test_more_clustering_requires_more_participants(self):
        individual = required_sample(standardized_effect=0.3)
        clustered = required_sample(
            standardized_effect=0.3, cluster_size=20, icc=0.08
        )
        self.assertGreater(
            clustered["required_total"], individual["required_total"]
        )

    def test_pretest_precision_reduces_required_sample(self):
        raw = required_sample(standardized_effect=0.3)
        adjusted = required_sample(
            standardized_effect=0.3, pretest_r_squared=0.4
        )
        self.assertLess(adjusted["required_total"], raw["required_total"])

    def test_attrition_increases_recruitment_target(self):
        no_attrition = required_sample(standardized_effect=0.4)
        attrition = required_sample(
            standardized_effect=0.4, attrition=0.2
        )
        self.assertGreater(
            attrition["required_total"], no_attrition["required_total"]
        )


if __name__ == "__main__":
    unittest.main()
