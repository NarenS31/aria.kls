import unittest

from agent.reasoning_moves import observe_reasoning_moves
from agent.student_understanding import understand_student_turn


class ObservableReasoningMoveTests(unittest.TestCase):
    def codes(self, text):
        return {move.code for move in observe_reasoning_moves(text)}

    def test_multilabel_plan_step_and_justification(self):
        codes = self.codes(
            "First I would subtract 2x because I want the x terms together."
        )
        self.assertTrue({"PLAN", "STRATEGY_STEP", "JUSTIFICATION"} <= codes)

    def test_uncertainty_does_not_erase_academic_step(self):
        codes = self.codes("Maybe I should distribute the 3 first.")
        self.assertIn("UNCERTAINTY", codes)
        self.assertIn("STRATEGY_STEP", codes)

    def test_help_and_affect_can_coexist(self):
        codes = self.codes("I'm frustrated and lost. Can you give me a hint?")
        self.assertIn("AFFECT", codes)
        self.assertIn("HELP_SEEKING", codes)

    def test_first_attempt_is_task_meta_not_a_plan(self):
        codes = self.codes("this is my first attempt")
        self.assertIn("TASK_META", codes)
        self.assertNotIn("PLAN", codes)
        self.assertNotIn("OFF_TASK", codes)

    def test_evidence_spans_are_exact_substrings(self):
        text = "Wait, I meant divide both sides."
        for move in observe_reasoning_moves(text):
            self.assertEqual(text[move.start:move.end], move.evidence)

    def test_understanding_exposes_observable_moves(self):
        parsed = understand_student_turn(
            "I think x is 6 because both sides balance."
        )
        self.assertIn("JUSTIFICATION", parsed.observable_moves)
        self.assertTrue(parsed.move_evidence)


if __name__ == "__main__":
    unittest.main()
