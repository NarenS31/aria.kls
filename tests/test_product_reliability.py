import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from agent.reasoning import ARIAAgent, offline_coaching_response
from ui import app


class FakeAgent:
    THINK_ALOUD_PROMPT = "Think through this out loud."

    def chat_stream(self, message):
        yield "What "
        yield "do you know first?"

    def chat(self, message):
        return "What do you know first?"

    def generate_think_aloud_problem(self, topic_override=None):
        return {
            "problem": "Solve for x: 2x + 3 = 11",
            "topic": "Algebra",
            "answer": "x = 4",
            "solution_steps": ["Subtract 3 from both sides.", "Divide both sides by 2."],
            "key_ideas": ["inverse operations"],
            "common_misconceptions": ["Dividing before isolating 2x."],
        }

    def think_aloud_turn(self, text):
        return {
            "state": "PLANNING",
            "evidence": "a stated first step",
            "question": "What will you do after isolating the variable?",
            "intervened": True,
            "intervention_state": "PLANNING",
            "escalated": False,
        }


class FakeEmotionTracker:
    def analyze(self, message):
        return None


class ProductReliabilityTests(unittest.TestCase):
    def setUp(self):
        app._aria_agent = FakeAgent()
        app._emotion_tracker = FakeEmotionTracker()

    def test_chat_accepts_empty_or_missing_history(self):
        updates = list(app.respond_stream("  I need a plan  ", None))
        self.assertEqual(updates[-1][0], "")
        self.assertEqual(
            updates[-1][1][-1]["content"], "What do you know first?"
        )

    def test_think_aloud_keeps_text_when_analysis_fails(self):
        with patch.object(
            app._aria_agent, "think_aloud_turn", side_effect=RuntimeError
        ):
            _, message, text = app.submit_think_aloud(
                "My first step is subtracting 3."
            )
        self.assertIn("try again", message)
        self.assertEqual(text, "My first step is subtracting 3.")

    def test_new_problem_has_a_no_model_fallback(self):
        with patch.object(
            app._aria_agent,
            "generate_think_aloud_problem",
            side_effect=RuntimeError,
        ):
            problem, text, _, response = app.new_think_problem()
        self.assertIn("Solve for x", problem)
        self.assertEqual(text, "")
        self.assertIn("Start when", response)

    def test_generated_problem_has_answer_key_context(self):
        agent = object.__new__(ARIAAgent)
        agent.profile = {"subjects": ["algebra"], "default_difficulty": "medium"}
        agent.model = "missing-model"
        agent._meta_tracker = None
        agent._calib_pending = None
        agent._current_problem_ctx = {}

        with patch("agent.reasoning.ollama.chat", side_effect=RuntimeError):
            problem = agent.generate_think_aloud_problem("algebra")

        self.assertEqual(problem["subject"], "Math")
        self.assertTrue(problem["answer"])
        self.assertIn("solution_steps", problem)
        self.assertIn("common_misconceptions", problem)
        self.assertTrue(problem["task_id"])

    def test_committed_question_bank_has_100_reviewable_tasks(self):
        bank = ARIAAgent._load_question_bank()

        self.assertEqual(len(bank), 100)
        self.assertEqual(sum(item["subject"] == "Math" for item in bank), 50)
        self.assertEqual(sum(item["subject"] == "English" for item in bank), 30)
        self.assertEqual(sum(item["subject"] == "Science and Coding" for item in bank), 20)
        self.assertEqual(len({item["id"] for item in bank}), 100)
        self.assertTrue(all(len(item["solution_steps"]) >= 3 for item in bank))

    def test_try_another_problem_does_not_repeat_immediately(self):
        agent = object.__new__(ARIAAgent)
        agent.profile = {"subjects": ["math"], "default_difficulty": "medium"}
        agent._meta_tracker = None
        agent._calib_pending = None
        agent._current_problem_ctx = {}
        agent._last_problem_task_id = ""

        first = agent.generate_think_aloud_problem("Math")
        second = agent.generate_think_aloud_problem("Math")

        self.assertNotEqual(first["task_id"], second["task_id"])

    def test_problem_aware_fallback_uses_expected_first_step(self):
        agent = object.__new__(ARIAAgent)
        agent.model = "missing-model"
        agent.profile = {}
        agent._current_problem_ctx = {
            "problem": "Solve for x: 3(x - 4) = 2x + 5",
            "answer": "x = 17",
            "solution_steps": ["Distribute 3 across x - 4 to get 3x - 12."],
            "key_ideas": ["distribution"],
            "common_misconceptions": ["Forgetting to multiply both terms inside the parentheses by 3."],
        }

        with patch("agent.reasoning.ollama.chat", side_effect=RuntimeError):
            response = agent._problem_aware_coaching_response(
                "I think I subtract 4 first and get 3x - 4", "CONFUSED", "What is one fact?"
            )

        self.assertIn("distribution", response)
        self.assertIn("3 times -4", response)
        self.assertNotIn("answer is", response.lower())

    def test_english_problem_uses_rubric_style_coaching(self):
        agent = object.__new__(ARIAAgent)
        agent.profile = {}
        agent.model = "missing-model"
        agent._current_problem_ctx = agent._fallback_problem_for_topic("english")

        response = agent._problem_aware_coaching_response(
            "I have a quote but I do not know what to say after it.",
            "STUCK",
            "What is the next step?",
        )

        self.assertIn("analysis", response.lower())
        self.assertIn("quote", response.lower())
        self.assertNotIn("correct answer", response.lower())

    def test_english_bank_task_uses_its_specific_coaching_move(self):
        agent = object.__new__(ARIAAgent)
        agent.profile = {}
        agent.model = "missing-model"
        task = next(
            item for item in agent._load_question_bank()
            if item["id"] == "english-revision-01"
        )
        agent._current_problem_ctx = task

        response = agent._problem_aware_coaching_response(
            "I would join them with just a comma.",
            "CONFUSED",
            "What should you check?",
        )

        self.assertIn("complete sentences", response)
        self.assertIn("conjunction", response)

    def test_live_coach_prompt_combines_problem_student_profile_and_trace(self):
        agent = object.__new__(ARIAAgent)
        agent.model = "test-model"
        agent.profile = {
            "name": "Naren",
            "grade": 12,
            "dynamic_problem_coaching": True,
            "learning_style": "step_by_step",
            "goals": ["become independent at algebra"],
            "misconceptions": {
                "algebra": ["sometimes distributes to only the first term"]
            },
        }
        agent._current_problem_ctx = {
            "problem": "Solve for x: 3(x - 4) = 2x + 5",
            "topic": "Algebra",
            "subject": "Math",
            "answer": "x = 17",
            "solution_steps": ["Distribute 3 across both terms."],
            "key_ideas": ["distribution"],
            "common_misconceptions": ["Forgetting to distribute to -4."],
        }
        agent._coaching_trace = []
        agent._recent_coaching_responses = []
        agent._coaching_turn_index = 0
        captured = []

        def fake_chat(**kwargs):
            captured.append(kwargs["messages"][1]["content"])
            return SimpleNamespace(
                message=SimpleNamespace(
                    content='You wrote "subtract 4 first." Which two terms must the 3 multiply?'
                )
            )

        with patch("agent.reasoning.ollama.chat", side_effect=fake_chat):
            response = agent._problem_aware_coaching_response(
                "I think I subtract 4 first.", "PLANNING", "What should you check?"
            )

        prompt = captured[0]
        self.assertIn("3(x - 4) = 2x + 5", prompt)
        self.assertIn("I think I subtract 4 first.", prompt)
        self.assertIn("Student: Naren", prompt)
        self.assertIn("sometimes distributes to only the first term", prompt)
        self.assertIn("distribution", response)
        self.assertIn("3 times -4", response)

    def test_repeat_guard_makes_identical_model_drafts_distinct(self):
        agent = object.__new__(ARIAAgent)
        agent.model = "test-model"
        agent.profile = {
            "name": "Naren",
            "dynamic_problem_coaching": True,
        }
        agent._current_problem_ctx = {
            "problem": "Solve for x: 3(x - 4) = 2x + 5",
            "topic": "Algebra",
            "subject": "Math",
            "answer": "x = 17",
            "solution_steps": ["Distribute 3 across both terms."],
            "key_ideas": ["distribution"],
            "common_misconceptions": ["Forgetting to distribute to -4."],
        }
        agent._coaching_trace = []
        agent._recent_coaching_responses = []
        agent._coaching_turn_index = 0
        same_draft = SimpleNamespace(
            message=SimpleNamespace(
                content='You wrote "subtract 4 first." Which two terms must the 3 multiply?'
            )
        )

        with patch("agent.reasoning.ollama.chat", return_value=same_draft):
            first = agent._problem_aware_coaching_response(
                "I think I subtract 4 first.", "PLANNING", "What should you check?"
            )
            second = agent._problem_aware_coaching_response(
                "I think I subtract 4 first.", "PLANNING", "What should you check?"
            )

        self.assertNotEqual(first, second)
        self.assertGreaterEqual(
            agent._last_coaching_meta["semantic_repeats_blocked"], 1
        )

    def test_grounding_guard_rejects_invented_student_action(self):
        bad_draft = (
            "You've successfully subtracted 3 from both sides. "
            "Next, what happens if you multiply both sides by 2?"
        )

        self.assertFalse(
            ARIAAgent._response_passes_grounding(
                bad_draft,
                "I want to divide by 0.5 before dealing with the plus 3.",
            )
        )

    def test_student_anchor_keeps_the_reasoning_move_not_the_filler(self):
        anchor = ARIAAgent._student_quote_anchor(
            "I think I should subtract 5 first because it is inside the parentheses."
        )

        self.assertEqual(anchor, "I should subtract 5 first")
        self.assertEqual(
            ARIAAgent._student_quote_anchor(
                "Wait, I see subtracting 2x gives 3x + 7 = 25."
            ),
            "subtracting 2x gives 3x + 7 = 25",
        )

    def test_state_detector_distinguishes_seeing_a_number_from_insight(self):
        agent = object.__new__(ARIAAgent)
        agent._current_problem_ctx = {
            "problem": "Solve for x: 6x - 4 = 2(x + 8)"
        }

        mistaken = agent._quick_thinking_state(
            "I divide by 2 because I see a 2 on the right."
        )
        corrected = agent._quick_thinking_state(
            "Wait, I see I need to distribute the 2 to both terms."
        )

        self.assertEqual(mistaken["state"], "CONFUSED")
        self.assertEqual(corrected["state"], "INSIGHT")

    def test_keyed_inverse_operation_conflict_is_not_labeled_flow(self):
        agent = object.__new__(ARIAAgent)
        agent._current_problem_ctx = {
            "problem": "Solve for x: 5x + 7 = 2x + 25",
            "solution_steps": [
                "Subtract 2x from both sides to get 3x + 7 = 25."
            ],
        }

        result = agent._quick_thinking_state(
            "I should add 2x to both sides to get the x terms together."
        )

        self.assertEqual(result["state"], "CONFUSED")
        self.assertIn("conflicts", result["evidence"])

    def test_inverse_operation_coaching_does_not_state_the_operation(self):
        agent = object.__new__(ARIAAgent)
        agent.profile = {}
        agent.model = "missing-model"
        agent._current_problem_ctx = next(
            item for item in agent._load_question_bank()
            if item["id"] == "math-algebra-02"
        )

        response = agent._problem_aware_coaching_response(
            "I should add 2x to both sides.",
            "CONFUSED",
            "What should you check?",
        )

        self.assertIn("inverse operation", response.lower())
        self.assertNotIn("subtract 2x", response.lower())

    def test_completed_intermediate_step_advances_the_coaching(self):
        agent = object.__new__(ARIAAgent)
        agent.profile = {}
        agent.model = "missing-model"
        agent._current_problem_ctx = next(
            item for item in agent._load_question_bank()
            if item["id"] == "math-algebra-08"
        )

        response = agent._problem_aware_coaching_response(
            "I subtracted 3 from both sides, so 0.5x = 5.",
            "INSIGHT",
            "What should you check?",
        )

        self.assertIn("0.5x = 5", response)
        self.assertIn("multiplying x by 0.5", response)
        self.assertNotIn("subtract 3 from both sides. which part", response.lower())

    def test_help_request_gets_immediate_problem_orientation(self):
        agent = object.__new__(ARIAAgent)
        agent.profile = {}
        agent.model = "missing-model"
        agent._current_problem_ctx = {
            "id": "math-test",
            "problem": "Solve for x: 6x - 4 = 2(x + 8)",
            "topic": "Algebra",
            "answer": "x = 5",
            "solution_steps": ["Distribute 2 across x + 8."],
            "key_ideas": ["distribution", "equivalent equations"],
            "common_misconceptions": ["Moving terms before distributing."],
        }

        response = agent._problem_aware_coaching_response(
            "i need help", "STUCK", "What should you check?"
        )

        self.assertIn("2(x + 8)", response)
        self.assertIn("which terms", response.lower())
        self.assertNotIn("revisiting", response.lower())
        self.assertNotIn("you wrote", response.lower())

    def test_first_attempt_meta_talk_is_stuck_not_planning(self):
        agent = object.__new__(ARIAAgent)
        agent._current_problem_ctx = {
            "problem": "Solve for x: 6x - 4 = 2(x + 8)",
            "solution_steps": ["Distribute 2 across x + 8."],
        }

        result = agent._quick_thinking_state("this is my first attemot")

        self.assertEqual(result["state"], "STUCK")
        self.assertEqual(result["intent"], "ATTEMPT_META")

    def test_interface_does_not_call_an_unstarted_student_stuck(self):
        original = list(app._think_states)
        try:
            app._think_states = [{
                "state": "STUCK",
                "intent": "ATTEMPT_META",
                "evidence": "the student has not attempted the problem",
            }]

            panel = app._state_panel_html()

            self.assertIn("Getting started", panel)
            self.assertNotIn("<strong>Stuck</strong>", panel)
        finally:
            app._think_states = original

    def test_first_attempt_clarification_keeps_problem_specific_start(self):
        agent = object.__new__(ARIAAgent)
        agent.profile = {}
        agent.model = "missing-model"
        agent._coaching_trace = deque([{
            "student": "i need help",
            "state": "STUCK",
            "aria": "Start with the grouped expression.",
        }], maxlen=8)
        agent._recent_coaching_responses = deque([
            "Start with the grouped part 2(x + 8). Which terms must 2 multiply?"
        ], maxlen=20)
        agent._current_problem_ctx = {
            "id": "math-test",
            "problem": "Solve for x: 6x - 4 = 2(x + 8)",
            "topic": "Algebra",
            "answer": "x = 5",
            "solution_steps": ["Distribute 2 across x + 8."],
            "key_ideas": ["distribution", "equivalent equations"],
            "common_misconceptions": ["Moving terms before distributing."],
        }

        response = agent._problem_aware_coaching_response(
            "this is my first attemot", "STUCK", "What should you check?"
        )

        self.assertIn("not made a reasoning attempt", response.lower())
        self.assertIn("2(x + 8)", response)
        self.assertNotIn("revisiting", response.lower())

    def test_offline_coach_never_returns_a_solution_dump(self):
        response = offline_coaching_response("I am stuck on this equation")
        self.assertIn("What is one fact", response)
        self.assertNotIn("answer is", response.lower())


if __name__ == "__main__":
    unittest.main()
