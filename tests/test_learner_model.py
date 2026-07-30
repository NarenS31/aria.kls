import tempfile
import unittest
from pathlib import Path

from agent.learner_model import LearnerModelStore


class LearnerModelTests(unittest.TestCase):
    def test_correct_grounded_evidence_increases_skill_mastery(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LearnerModelStore(Path(directory) / "learner.json")
            before = store.state(
                student="Naren",
                topic="Algebra",
                skills=["distribution"],
            )

            after = store.observe(
                student="Naren",
                topic="Algebra",
                skills=["distribution"],
                misconception="Distribute only to the first term.",
                correct=True,
                grounded_progress=True,
                self_correction=True,
                misconception_persisted=False,
                evidence_strength=1.0,
            )

            self.assertGreater(after["mastery_mean"], before["mastery_mean"])
            self.assertEqual(after["skills"]["distribution"]["observations"], 1)

    def test_uninformative_turn_barely_changes_mastery(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LearnerModelStore(Path(directory) / "learner.json")

            after = store.observe(
                student="Naren",
                topic="Algebra",
                skills=["distribution"],
                misconception="",
                correct=None,
                grounded_progress=False,
                self_correction=False,
                misconception_persisted=None,
                evidence_strength=0.0,
            )

            self.assertGreater(after["mastery_mean"], 0.45)
            self.assertLess(after["mastery_mean"], 0.5)
            self.assertGreater(
                after["skills"]["distribution"]["ci_95"][1]
                - after["skills"]["distribution"]["ci_95"][0],
                0.5,
            )

    def test_students_have_separate_posteriors(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LearnerModelStore(Path(directory) / "learner.json")
            store.observe(
                student="Naren",
                topic="Algebra",
                skills=["distribution"],
                misconception="",
                correct=True,
                grounded_progress=True,
                self_correction=False,
                misconception_persisted=None,
                evidence_strength=1.0,
            )

            naren = store.state(
                student="Naren", topic="Algebra", skills=["distribution"]
            )
            alex = store.state(
                student="Alex", topic="Algebra", skills=["distribution"]
            )

            self.assertGreater(naren["mastery_mean"], alex["mastery_mean"])


if __name__ == "__main__":
    unittest.main()
