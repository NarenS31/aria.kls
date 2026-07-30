import unittest
from collections import deque

from agent.reasoning import ARIAAgent
from agent.student_understanding import understand_student_turn
from eval.metacognition.transfer import TransferDetector


class SituationModelTests(unittest.TestCase):
    def agent(self):
        agent = object.__new__(ARIAAgent)
        agent.profile = {}
        agent._coaching_trace = deque(maxlen=8)
        agent._current_problem_ctx = {
            "task_id": "algebra-1",
            "problem": "Solve for x: 2x + 3 = 11",
            "answer": "x = 4",
            "solution_steps": [
                "Subtract 3 from both sides to get 2x = 8.",
                "Divide both sides by 2 to get x = 4.",
            ],
            "key_ideas": ["inverse operations", "equivalent equations"],
            "common_misconceptions": [
                "Adding 3 instead of subtracting 3 from both sides."
            ],
        }
        return agent

    def situation(self, agent, text):
        understanding = understand_student_turn(
            text,
            problem=agent._current_problem_ctx["problem"],
            recent_turns=list(agent._coaching_trace),
        )
        return agent._build_situation_model(text, understanding)

    def test_correct_named_step_is_grounded_in_answer_key(self):
        situation = self.situation(
            self.agent(),
            "First I will subtract 3 from both sides.",
        )
        self.assertTrue(situation.student_has_named_next_step)
        self.assertIs(situation.proposed_step_is_correct, True)
        self.assertEqual(situation.task_step_index, 0)

    def test_unknown_is_not_silently_converted_to_incorrect(self):
        situation = self.situation(
            self.agent(),
            "I will rewrite this in a different form.",
        )
        self.assertTrue(situation.student_has_named_next_step)
        self.assertIsNone(situation.proposed_step_is_correct)

    def test_repeated_wrong_strategy_is_localized(self):
        agent = self.agent()
        text = "I will add 3 to both sides."
        agent._coaching_trace.append({
            "student": text,
            "strategy_signature": agent._strategy_signature(text),
        })
        situation = self.situation(agent, text)
        action, response, intervened = agent._situation_policy_response(
            text,
            situation,
            justification_present=False,
        )
        self.assertTrue(situation.same_strategy_repeated)
        self.assertIs(situation.proposed_step_is_correct, False)
        self.assertEqual(action, "localize_error")
        self.assertIn("specific point", response)
        self.assertTrue(intervened)

    def test_low_confidence_abstains(self):
        agent = self.agent()
        situation = self.situation(agent, "okay")
        action, response, intervened = agent._situation_policy_response(
            "okay",
            situation,
            justification_present=False,
        )
        self.assertLess(situation.aria_confidence, 0.5)
        self.assertEqual(action, "observe")
        self.assertFalse(intervened)
        self.assertNotIn("?", response)


class StrictTransferTests(unittest.TestCase):
    def test_transfer_requires_later_execution_on_different_task(self):
        detector = TransferDetector("unit-test")
        detector._last_task_id = "task-a"

        candidate = detector.detect(
            "First I will subtract 3 from both sides.",
            aria_previous_prompt="",
            turn=1,
            task_id="task-b",
            moves_detected=["PLAN", "STRATEGY_STEP"],
            task_content_referenced=True,
            strategy_executed=False,
            persist=False,
        )
        self.assertTrue(candidate["transfer_candidate"])
        self.assertFalse(candidate["transfer_confirmed"])

        execution = detector.detect(
            "Subtracting 3 gives 2x = 8.",
            aria_previous_prompt="",
            turn=2,
            task_id="task-b",
            moves_detected=["STRATEGY_STEP"],
            task_content_referenced=True,
            strategy_executed=True,
            persist=False,
        )
        self.assertTrue(execution["transfer_confirmed"])
        self.assertEqual(execution["confirms_candidate_turn"], 1)

    def test_prompted_planning_is_separate_and_not_transfer(self):
        detector = TransferDetector("unit-test")
        detector._last_task_id = "task-a"
        record = detector.detect(
            "First I will compare the two claims.",
            aria_previous_prompt="What is your plan?",
            turn=3,
            task_id="task-b",
            moves_detected=["PLAN"],
            task_content_referenced=True,
            persist=False,
        )
        self.assertTrue(record["prompted_planning"])
        self.assertFalse(record["transfer_candidate"])

    def test_generic_plan_without_task_reference_is_not_transfer(self):
        detector = TransferDetector("unit-test")
        detector._last_task_id = "task-a"
        record = detector.detect(
            "Let me think first.",
            aria_previous_prompt="",
            turn=4,
            task_id="task-b",
            moves_detected=["PLAN"],
            task_content_referenced=False,
            persist=False,
        )
        self.assertFalse(record["transfer_candidate"])


if __name__ == "__main__":
    unittest.main()
