import json
import unittest
from pathlib import Path

from agent.student_intent import (
    classify_student_intent,
    warm_student_intent_model,
)
from agent.student_understanding import understand_student_turn


class StudentIntentModelTests(unittest.TestCase):
    def test_reported_help_phrase_uses_trained_model(self):
        result = classify_student_intent("i need help")

        self.assertEqual(result.label, "HELP_REQUEST")
        self.assertGreater(result.confidence, 0.9)
        self.assertIn("dialogue model", result.evidence)

    def test_model_can_be_preloaded_before_first_student_turn(self):
        self.assertTrue(warm_student_intent_model())

    def test_typo_in_first_attempt_is_not_mathematical_planning(self):
        result = classify_student_intent("this is my first attemot")

        self.assertEqual(result.label, "ATTEMPT_META")
        self.assertGreater(result.confidence, 0.9)
        self.assertFalse(result.contains_reasoning)

    def test_correction_and_reasoning_can_coexist(self):
        result = classify_student_intent(
            "Wait I meant subtract 3 from both sides"
        )

        self.assertEqual(result.label, "SELF_CORRECTION")
        self.assertTrue(result.contains_reasoning)

    def test_model_metadata_discloses_weak_labels_and_license(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "student_intent_metadata.json"
        )
        metadata = json.loads(path.read_text())

        self.assertEqual(metadata["license"], "CC BY-NC-SA 4.0")
        self.assertFalse(metadata["commercial_use"])
        self.assertIn("not human ground truth", metadata["label_provenance"])
        self.assertGreater(metadata["train_examples_used"], 10_000)

    def test_semantic_router_handles_varied_natural_student_language(self):
        cases = {
            "im completely lost on where to go next": "HELP_REQUEST",
            "i have not even begun this one": "ATTEMPT_META",
            "this problem is making me mad": "FRUSTRATION",
            "can you put that in normal words": "CLARIFICATION_REQUEST",
            "did i mess anything up": "CONFIRMATION_REQUEST",
            "switch me to a harder question": "CONTROL_REQUEST",
            "scratch that, i forgot a negative": "SELF_CORRECTION",
            "i might divide both sides next": "UNCERTAINTY",
            "i would combine those two sentences": "REASONING",
            "x = -4": "SHORT_ANSWER",
            "alright": "SOCIAL",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = understand_student_turn(text)
                self.assertEqual(result.intent, expected)

    def test_contextual_language_fallback_is_off_by_default_for_latency(self):
        result = understand_student_turn("that thing")

        self.assertEqual(result.source, "fast_intent_model")
        self.assertTrue(result.ambiguous)


if __name__ == "__main__":
    unittest.main()
