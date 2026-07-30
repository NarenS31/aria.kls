import unittest

from eval.research_stats import (
    compare_conditions,
    holm_adjust,
    paired_episode_differences,
)


class ResearchStatisticsTests(unittest.TestCase):
    def test_turns_are_averaged_before_episode_difference(self):
        rows = [
            {"episode_id": "a", "condition": "full", "score": 1},
            {"episode_id": "a", "condition": "full", "score": 0},
            {"episode_id": "a", "condition": "base", "score": 0},
            {"episode_id": "b", "condition": "full", "score": 1},
            {"episode_id": "b", "condition": "base", "score": 0},
        ]

        differences = paired_episode_differences(
            rows, "full", "base", "score"
        )

        self.assertEqual(differences, {"a": 0.5, "b": 1.0})

    def test_holm_adjustment_is_monotonic_and_bounded(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})

        self.assertEqual(adjusted["a"], 0.03)
        self.assertEqual(adjusted["b"], 0.06)
        self.assertEqual(adjusted["c"], 0.2)

    def test_comparison_reports_paired_episode_count(self):
        rows = []
        for episode in range(8):
            rows.extend([
                {
                    "episode_id": str(episode),
                    "condition": "full",
                    "score": 1.0,
                },
                {
                    "episode_id": str(episode),
                    "condition": "base",
                    "score": 0.0,
                },
            ])

        result = compare_conditions(
            rows,
            treatment="full",
            control="base",
            metrics=["score"],
        )

        self.assertEqual(result["score"]["n_paired_episodes"], 8)
        self.assertEqual(result["score"]["mean_difference"], 1.0)


if __name__ == "__main__":
    unittest.main()
