import tempfile
import unittest
from pathlib import Path

from metacognition.interaction_logger import (
    InteractionLogger,
    tone_proxy,
)
from metacognition.jitai import (
    JITAIContext,
    RuleBasedJITAIPolicy,
    build_jitai_intervention,
    topic_stats_from_learning_graph,
)


class TestInteractionLogger(unittest.TestCase):
    def test_log_query_interaction_intervention_and_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = InteractionLogger(Path(tmp) / "aria.sqlite3")
            tables = set(logger.tables())
            self.assertIn("interaction_events", tables)
            self.assertIn("intervention_events", tables)
            self.assertIn("intervention_outcomes", tables)

            interaction_id = logger.log_interaction(
                session_id="s1",
                profile_id="alex",
                topic="algebra",
                student_message="ugh I don't get it",
                cognitive_state="CONFUSED",
                analyzer_confidence=0.84,
                response_latency_ms=12000,
                time_since_previous_ms=240000,
            )
            self.assertGreater(interaction_id, 0)

            intervention_id = logger.log_intervention(
                session_id="s1",
                profile_id="alex",
                topic="algebra",
                trigger_condition="repeated_confused_or_stuck",
                intervention_type="task_chunking",
                intervention_text="Break it down.",
                rule_id="jitai.repeated_confusion",
                rationale="test",
            )
            self.assertGreater(intervention_id, 0)

            outcome_id = logger.log_outcome(
                intervention_id=intervention_id,
                outcome="ignored",
                next_cognitive_state="STUCK",
            )
            self.assertGreater(outcome_id, 0)
            self.assertEqual(logger.get_ignored_intervention_count("s1"), 1)

            recent = logger.get_recent_interventions("s1")
            self.assertEqual(recent[0]["intervention_type"], "task_chunking")

            stats = logger.get_topic_intervention_stats("alex", "algebra")
            self.assertEqual(stats["by_type"]["task_chunking"]["ignored"], 1)

    def test_tone_proxy_is_heuristic(self):
        tone = tone_proxy("ugh this is stupid, I give up")
        self.assertEqual(tone.label, "strong_negative")
        self.assertLess(tone.score, -0.5)


class TestJITAIPolicy(unittest.TestCase):
    def test_repeated_confusion_returns_hint_without_high_struggle(self):
        decision = RuleBasedJITAIPolicy().decide(
            JITAIContext(
                profile_id="alex",
                topic="algebra",
                cognitive_state="CONFUSED",
                consecutive_state_count=2,
            )
        )
        self.assertTrue(decision.should_intervene)
        self.assertEqual(decision.intervention_type, "hint")

    def test_high_struggle_changes_repeated_confusion_to_task_chunking(self):
        decision = RuleBasedJITAIPolicy().decide(
            JITAIContext(
                profile_id="alex",
                topic="algebra",
                cognitive_state="CONFUSED",
                consecutive_state_count=2,
                topic_stats={
                    "confidence": 0.25,
                    "study_count": 4,
                    "struggle_count": 3,
                    "struggle_rate": 0.75,
                },
            )
        )
        self.assertTrue(decision.should_intervene)
        self.assertEqual(decision.intervention_type, "task_chunking")
        self.assertIn("high struggle", decision.rationale)

    def test_frustration_returns_reset_or_encouragement(self):
        decision = RuleBasedJITAIPolicy().decide(
            JITAIContext(
                profile_id="alex",
                topic="algebra",
                cognitive_state="FRUSTRATED",
                consecutive_state_count=2,
                tone_proxy_score=-0.7,
            )
        )
        self.assertTrue(decision.should_intervene)
        self.assertEqual(decision.intervention_type, "break_suggestion")

    def test_flow_does_not_interrupt(self):
        decision = RuleBasedJITAIPolicy().decide(
            JITAIContext(
                profile_id="alex",
                topic="algebra",
                cognitive_state="FLOW",
            )
        )
        self.assertFalse(decision.should_intervene)
        self.assertEqual(decision.intervention_type, "none")

    def test_build_intervention_maps_decision_to_text(self):
        decision = RuleBasedJITAIPolicy().decide(
            JITAIContext(
                profile_id="alex",
                topic="algebra",
                cognitive_state="STUCK",
                consecutive_state_count=2,
                topic_stats={"confidence": 0.2},
            )
        )
        intervention = build_jitai_intervention(decision, "STUCK")
        self.assertEqual(intervention["type"], "task_chunking")
        self.assertIn("first sentence", intervention["text"])

    def test_learning_graph_topic_stats_helper(self):
        class FakeGraph:
            def __init__(self):
                self.nodes = {
                    "algebra": {
                        "confidence": 0.3,
                        "study_count": 5,
                        "struggle_count": 4,
                        "study_hours": [21, 22],
                        "explanation_styles": {"step_by_step": 2},
                    }
                }

            def has_node(self, key):
                return key in self.nodes

        stats = topic_stats_from_learning_graph(FakeGraph(), "Algebra")
        self.assertTrue(stats["high_struggle"])
        self.assertEqual(stats["topic_key"], "algebra")


if __name__ == "__main__":
    unittest.main()
