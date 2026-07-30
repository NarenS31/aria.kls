import json
import tempfile
import unittest
from pathlib import Path

from agent.intervention_pipeline import (
    ClosedLoopInterventionPipeline,
    InterventionOutcomeStore,
    semantic_similarity,
)


class InterventionPipelineTests(unittest.TestCase):
    def test_signature_changes_with_student_reasoning(self):
        common = {
            "student": "Naren",
            "task_id": "math-01",
            "topic": "Algebra",
            "problem_step": "Distribute first.",
            "misconception": "Skipped distribution.",
            "state": "CONFUSED",
            "style": "step_by_step",
            "prior_outcome": "none",
        }
        first = ClosedLoopInterventionPipeline.build_signature(
            student_anchor="subtract four first",
            **common,
        )
        second = ClosedLoopInterventionPipeline.build_signature(
            student_anchor="distribute to both terms",
            **common,
        )

        self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def test_model_candidate_parser_requires_known_strategies(self):
        raw = json.dumps({
            "candidates": [
                {"text": 'You wrote "divide first." What must happen to the parentheses?', "strategy": "error_localization"},
                {"text": "Invalid strategy.", "strategy": "random"},
            ]
        })

        parsed = ClosedLoopInterventionPipeline.parse_model_candidates(raw)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["strategy"], "error_localization")

    def test_semantic_repeat_is_rejected_even_when_wording_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = ClosedLoopInterventionPipeline(
                Path(directory) / "outcomes.jsonl"
            )
            signature = pipeline.build_signature(
                student="Naren",
                task_id="math-01",
                topic="Algebra",
                problem_step="Distribute first.",
                student_anchor="subtract 4 first",
                misconception="Skipped distribution.",
                state="CONFUSED",
                style="step_by_step",
                prior_outcome="none",
            )
            candidates = [
                {
                    "text": 'You wrote "subtract 4 first." Which terms inside the parentheses does 3 multiply?',
                    "strategy": "error_localization",
                    "source": "model",
                },
                {
                    "text": 'Your decision point is "subtract 4 first." What operation must happen before subtraction?',
                    "strategy": "verification",
                    "source": "model",
                },
            ]
            recent = [
                'You wrote "subtract 4 first." Which terms in the parentheses should the 3 multiply?'
            ]

            selected, meta = pipeline.select(
                candidates=candidates,
                student_input="I think I subtract 4 first.",
                recent_responses=recent,
                key_ideas=["distribution"],
                correct_answer="x = 17",
                validator=lambda response, _: response.count("?") == 1,
                signature=signature,
                state="CONFUSED",
            )

            self.assertIsNotNone(selected)
            self.assertEqual(meta["semantic_repeats_blocked"], 1)
            self.assertIn("operation", selected)

    def test_next_turn_updates_strategy_effectiveness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcomes.jsonl"
            pipeline = ClosedLoopInterventionPipeline(path)
            pipeline.pending = {
                "signature": {"fingerprint": "abc"},
                "strategy": "error_localization",
                "state_before": "STUCK",
                "student_input": "idk",
                "response": "What does the coefficient multiply?",
            }

            outcome = pipeline.observe_next_turn(
                state="INSIGHT",
                student_input="Wait, I see that it multiplies both terms.",
            )

            self.assertTrue(outcome["effective"])
            self.assertTrue(outcome["recovered"])
            self.assertGreater(
                pipeline.outcomes.effectiveness("error_localization"), 0.5
            )
            self.assertTrue(path.exists())

    def test_strategy_learning_is_student_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InterventionOutcomeStore(
                Path(directory) / "outcomes.jsonl"
            )
            for effective in (True, True, True):
                store.record({
                    "strategy": "verification",
                    "effective": effective,
                    "signature": {"student": "Naren", "topic": "Algebra"},
                })
            for effective in (False, False, False):
                store.record({
                    "strategy": "verification",
                    "effective": effective,
                    "signature": {"student": "Alex", "topic": "Algebra"},
                })

            naren_rate = store.effectiveness(
                "verification", student="Naren", topic="Algebra"
            )
            alex_rate = store.effectiveness(
                "verification", student="Alex", topic="Algebra"
            )

            self.assertGreater(naren_rate, alex_rate)

    def test_similarity_detects_near_duplicate_interventions(self):
        similarity = semantic_similarity(
            "Which terms inside the parentheses does 3 multiply?",
            "Which terms in the parentheses should the 3 multiply?",
        )

        self.assertGreater(similarity, 0.78)

    def test_wait_by_itself_is_not_counted_as_learning_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = ClosedLoopInterventionPipeline(
                Path(directory) / "outcomes.jsonl"
            )
            pipeline.pending = {
                "signature": {
                    "student": "Naren",
                    "topic": "Algebra",
                    "key_ideas": ["distribution"],
                    "problem_step": "Multiply the coefficient by both terms.",
                    "misconception": "Distributed to one term.",
                },
                "strategy": "error_localization",
                "state_before": "STUCK",
                "student_input": "I do not know.",
                "response": "Which terms must the coefficient multiply?",
            }

            outcome = pipeline.observe_next_turn(
                state="CONFUSED",
                state_confidence=0.9,
                student_input="Wait.",
            )

            self.assertTrue(outcome["correction_language"])
            self.assertFalse(outcome["grounded_progress"])
            self.assertFalse(outcome["self_correction"])
            self.assertFalse(outcome["effective"])

    def test_policy_audit_reports_context_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = ClosedLoopInterventionPipeline(
                Path(directory) / "outcomes.jsonl"
            )
            signature = pipeline.build_signature(
                student="Naren",
                task_id="math-01",
                topic="Algebra",
                problem_step="Distribute first.",
                student_anchor="subtract four",
                misconception="Skipped distribution.",
                state="CONFUSED",
                style="stepwise",
                prior_outcome="none",
                key_ideas=["distribution"],
                mastery_band="developing",
            )

            _, meta = pipeline.select(
                candidates=[{
                    "text": 'You wrote “subtract four.” Which terms should the coefficient multiply?',
                    "strategy": "error_localization",
                    "source": "verified",
                }],
                student_input="I subtract four.",
                recent_responses=[],
                key_ideas=["distribution"],
                correct_answer="x = 3",
                validator=lambda response, _: True,
                signature=signature,
                state="CONFUSED",
            )

            evidence = meta["ranked_candidates"][0]["policy_evidence"]
            self.assertEqual(evidence["observations"], 0)
            self.assertEqual(evidence["posterior_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
