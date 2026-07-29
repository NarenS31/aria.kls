import json
import tempfile
import unittest
from pathlib import Path

from metacognition.analyzer import CognitiveStateAnalyzer
from metacognition.interaction_logger import (
    InteractionEvent,
    InteractionLogger,
    InterventionEvent,
    InterventionOutcome,
)
from metacognition.interventions import (
    MetacognitiveInterventionGenerator,
    extract_concept,
)
from metacognition.jitai import (
    JITAIContext,
    RULE_CITATIONS,
    RuleBasedJITAIPolicy,
)
from metacognition.timing import TIMING_RULES


class FullFlowTests(unittest.TestCase):
    def test_confused_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = InteractionLogger(Path(tmp) / "flow.sqlite3")
            analyzer = CognitiveStateAnalyzer(use_llm=False)
            generator = MetacognitiveInterventionGenerator(
                student_name="full_flow_test", seed=7
            )
            generator._path = str(Path(tmp) / "interventions.json")

            student_input = "ugh I don't get it"
            analysis = analyzer.analyze(student_input)
            self.assertEqual(analysis["state"], "CONFUSED")

            decision = RuleBasedJITAIPolicy().decide(
                JITAIContext(
                    profile_id="student",
                    topic="quadratics",
                    cognitive_state=analysis["state"],
                )
            )
            self.assertTrue(decision.should_intervene)

            intervention = generator.generate(
                analysis["state"], student_input=student_input
            )
            self.assertTrue(intervention["text"])

            intervention_id = logger.log_intervention(
                InterventionEvent(
                    session_id="session",
                    profile_id="student",
                    topic="quadratics",
                    trigger_condition=decision.trigger_condition,
                    intervention_type=decision.intervention_type,
                    intervention_text=intervention["text"],
                    rule_id=decision.rule_id,
                    rationale=decision.rationale,
                    citation_keys=decision.citation_keys,
                )
            )
            logger.log_interaction(
                InteractionEvent(
                    session_id="session",
                    profile_id="student",
                    topic="quadratics",
                    student_message=student_input,
                    aria_response=intervention["text"],
                    cognitive_state=analysis["state"],
                    analyzer_confidence=analysis["confidence"],
                )
            )
            logger.log_outcome(
                InterventionOutcome(
                    intervention_id=intervention_id,
                    outcome="responded",
                )
            )

            log_entry = logger.get_recent_interventions("session", limit=1)[0]
            self.assertEqual(log_entry["intervention_text"], intervention["text"])
            self.assertEqual(log_entry["rule_id"], decision.rule_id)
            self.assertTrue(log_entry["citation_keys"])

    def test_confused_prompt_uses_specific_concept(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = MetacognitiveInterventionGenerator("concept_test", seed=4)
            generator._path = str(Path(tmp) / "concept.json")
            student_input = "I'm confused about the quadratic formula"
            self.assertEqual(extract_concept(student_input), "the quadratic formula")
            prompts = [
                generator.generate(
                    "CONFUSED",
                    consecutive_count=1,
                    student_input=student_input,
                )["text"]
                for _ in range(6)
            ]
            concept_prompts = [p for p in prompts if "quadratic formula" in p.lower()]
            self.assertGreaterEqual(len(concept_prompts), 4)

    def test_rotation_avoids_immediate_repetition_and_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = MetacognitiveInterventionGenerator("rotation_test", seed=3)
            generator._path = str(Path(tmp) / "rotation.json")
            first = generator.generate("STUCK")["text"]
            second = generator.generate("STUCK")["text"]
            self.assertNotEqual(first, second)
            self.assertGreaterEqual(len(generator.interventions_used_this_session), 2)

            generator.reset_session()
            self.assertEqual(generator.interventions_used_this_session, set())
            self.assertEqual(generator._consecutive, 0)

    def test_all_timing_rules_have_valid_citation_keys(self):
        citations_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "research"
            / "citations.json"
        )
        citations = json.loads(citations_path.read_text(encoding="utf-8"))["citations"]
        for rule in TIMING_RULES.values():
            self.assertIn(rule["citation_key"], citations)
        for keys in RULE_CITATIONS.values():
            for key in keys:
                self.assertIn(key, citations)


if __name__ == "__main__":
    unittest.main()
