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


class StudentQuoteAnchorTests(unittest.TestCase):
    """The anchor is quoted back to the student verbatim, so it has to be a
    readable fragment of their reasoning — never a lone connective."""

    def test_leading_connective_after_opener_is_not_the_whole_anchor(self):
        # "Ok" is stripped as an opener, which used to leave "but then ..." with
        # "then" tripping the clause-boundary break on the second word and
        # shipping 'You wrote "but."' to a real student.
        anchor = ARIAAgent._student_quote_anchor(
            "Ok but then if there are students with exactly 90, "
            "don't we need to include those too?"
        )
        self.assertNotEqual(anchor.lower(), "but")
        self.assertGreaterEqual(len(anchor.split()), 4)
        self.assertEqual(anchor, "but then if there are students with exactly")

    def test_connective_still_cuts_a_clause_once_the_anchor_has_content(self):
        self.assertEqual(
            ARIAAgent._student_quote_anchor(
                "I think I should subtract 5 first because it is inside the "
                "parentheses."
            ),
            "I should subtract 5 first",
        )

    def test_existing_anchors_are_unchanged(self):
        for text, expected in (
            ("Wait, I did it like this... 2(x +",
             "Wait, I did it like this... 2(x +"),
            ("Um, I think... we just add them up.",
             "Um, I think... we just add them up"),
            ("I need to choose effect.", "I need to choose effect"),
            ("Wait, I see subtracting 2x gives 3x + 7 = 25.",
             "subtracting 2x gives 3x + 7 = 25"),
        ):
            with self.subTest(text=text):
                self.assertEqual(ARIAAgent._student_quote_anchor(text), expected)

    def test_trailing_question_mark_is_not_carried_into_the_quote(self):
        # The coach's own question follows the quote, so a "?" the student
        # happened to end on would give one response two question marks.
        anchor = ARIAAgent._student_quote_anchor(
            "I think... we gotta include all the numbers or something?"
        )
        self.assertEqual(anchor, "we gotta include all the numbers or something")
        self.assertNotIn("?", anchor)

    def test_only_the_trailing_terminal_mark_is_stripped(self):
        # An internal "?" is part of what the student wrote and stays; only the
        # anchor's own last character comes off.
        anchor = ARIAAgent._student_quote_anchor(
            "I got it... Is the answer just 90? Like, isn't 90 always right?"
        )
        self.assertEqual(anchor, "I got it... Is the answer just 90")
        self.assertEqual(
            ARIAAgent._student_quote_anchor("Is that right? I really am not sure."),
            "Is that right? I really am not sure",
        )

    def test_trailing_exclamation_and_period_are_stripped_too(self):
        self.assertEqual(
            ARIAAgent._student_quote_anchor("I finally see the pattern here!"),
            "I finally see the pattern here",
        )
        self.assertEqual(
            ARIAAgent._student_quote_anchor("I subtracted 3 from both sides."),
            "I subtracted 3 from both sides",
        )

    def test_empty_input_falls_back_instead_of_quoting_nothing(self):
        self.assertEqual(ARIAAgent._student_quote_anchor("   "), "what you wrote")


class SituationPolicyLengthTests(unittest.TestCase):
    """Scaffolded replies acknowledge the attempt before questioning it, which
    also narrows the length gap against the RQ2 answer-giving condition."""

    def agent(self):
        return SituationModelTests.agent(SituationModelTests())

    def _response(self, text, **kwargs):
        agent = kwargs.pop("agent", None) or self.agent()
        understanding = understand_student_turn(
            text,
            problem=agent._current_problem_ctx["problem"],
            recent_turns=list(agent._coaching_trace),
        )
        situation = agent._build_situation_model(text, understanding)
        return agent._situation_policy_response(
            text, situation, justification_present=kwargs.get(
                "justification_present", False)
        )

    def test_intervening_responses_acknowledge_before_questioning(self):
        for text in (
            "I don't know, maybe something with the 3?",
            "First I will subtract 3 from both sides.",
        ):
            with self.subTest(text=text):
                action, response, intervened = self._response(text)
                anchor = ARIAAgent._student_quote_anchor(text)
                self.assertTrue(intervened)
                self.assertGreaterEqual(len(response.split()), 25)
                # The acknowledgment asks nothing: the coach still contributes
                # exactly one question mark. Any other one is the student's own,
                # carried in inside the quote.
                self.assertEqual(response.count("?") - anchor.count("?"), 1)
                self.assertLessEqual(len(response), 520)
                if "?" not in anchor:
                    self.assertTrue(
                        ARIAAgent._response_passes_grounding(response, text)
                    )


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
