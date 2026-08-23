"""Tests for grain-aware timing thresholds.

The point of the grain split is that one set of absolute millisecond cutoffs
cannot serve both a quick exchange and a long reasoning chain. The external
validation measured the gap: a whole ASSISTments problem has a median elapsed
time of 31,000 ms against 4,000 ms for a single Fractions Lab step, and under
the single global cutoff 84.4% of multi-step rows landed above the STUCK
threshold. These tests pin the behaviour that fixes it, and — just as
importantly — pin that a caller who does not ask for a grain gets exactly what
it got before.
"""

import unittest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "eval"))

from metacognition.timing import (  # noqa: E402
    DEFAULT_GRAIN,
    DEFAULT_OPTIMAL,
    GRAIN_RESPONSE_TIME_MS,
    INTERACTION_GRAINS,
    NEGATIVE_STATES,
    TIMING_RULES_BY_GRAIN,
    InterventionTimer,
    infer_interaction_grain,
    normalize_grain,
    response_time_state,
)


class BackwardCompatibilityTests(unittest.TestCase):
    """Nothing may move for a caller that does not pass a grain."""

    def setUp(self):
        self.timer = InterventionTimer("test_grain_compat")

    def test_default_grain_is_atomic(self):
        self.assertEqual(DEFAULT_GRAIN, "atomic")

    def test_atomic_thresholds_are_the_previous_defaults(self):
        self.assertEqual(TIMING_RULES_BY_GRAIN["atomic"], DEFAULT_OPTIMAL)

    def test_atomic_response_time_cutoffs_are_unchanged(self):
        # The original single global set, which was calibrated for chat turns.
        self.assertEqual(GRAIN_RESPONSE_TIME_MS["atomic"],
                         {"rushing_below": 500, "planning_above": 5_000,
                          "stuck_above": 10_000})

    def test_omitting_the_grain_matches_asking_for_atomic(self):
        for state in NEGATIVE_STATES:
            with self.subTest(state=state):
                self.assertEqual(self.timer.recommended_wait(state),
                                 self.timer.recommended_wait(state, grain="atomic"))

    def test_positive_states_are_still_never_gated(self):
        for grain in INTERACTION_GRAINS:
            with self.subTest(grain=grain):
                self.assertEqual(self.timer.recommended_wait("FLOW", grain=grain), 1)
                self.assertTrue(self.timer.should_intervene("FLOW", 1, grain=grain))


class GrainNormalizationTests(unittest.TestCase):
    """Timing gates a live turn, so a bad grain must degrade, not raise."""

    def test_unknown_and_empty_grains_fall_back_to_the_default(self):
        for value in ("nonsense", "", None, "ATOMIC-ish", 0):
            with self.subTest(value=value):
                self.assertEqual(normalize_grain(value), DEFAULT_GRAIN)

    def test_case_and_underscore_variants_are_accepted(self):
        for value in ("MULTI-STEP", "multi_step", " Multi-Step "):
            with self.subTest(value=value):
                self.assertEqual(normalize_grain(value), "multi-step")

    def test_a_bad_grain_does_not_raise_in_the_gate(self):
        timer = InterventionTimer("test_grain_badinput")
        self.assertIsInstance(
            timer.recommended_wait("STUCK", grain="not-a-grain"), int)


class GrainInferenceTests(unittest.TestCase):
    """A turn is only called a chain on positive evidence that it is one."""

    def test_move_count_drives_the_classification(self):
        self.assertEqual(infer_interaction_grain(move_count=1), "atomic")
        self.assertEqual(infer_interaction_grain(move_count=2), "multi-step")
        self.assertEqual(infer_interaction_grain(moves=["PLAN", "STRATEGY_STEP"]),
                         "multi-step")
        self.assertEqual(infer_interaction_grain(moves=["PLAN"]), "atomic")

    def test_turn_length_is_the_fallback_when_moves_are_absent(self):
        self.assertEqual(infer_interaction_grain(text="I think it is four"),
                         "atomic")
        self.assertEqual(infer_interaction_grain(text=" ".join(["word"] * 45)),
                         "multi-step")

    def test_no_evidence_means_atomic(self):
        self.assertEqual(infer_interaction_grain(), "atomic")
        self.assertEqual(infer_interaction_grain(text="", move_count=None), "atomic")

    def test_response_time_is_not_used_to_infer_the_grain(self):
        # The grain selects which response-time cutoffs apply, so inferring it
        # from response time would be circular. A long pause on a short turn
        # must stay atomic.
        self.assertEqual(
            infer_interaction_grain(text="idk", move_count=1), "atomic")


class ResponseTimeReadingTests(unittest.TestCase):
    """The same elapsed time means different things at different grains."""

    def test_thirty_seconds_is_stuck_atomically_but_ordinary_multi_step(self):
        self.assertEqual(response_time_state(30_000, "atomic"), "STUCK")
        self.assertIsNone(response_time_state(30_000, "multi-step"))

    def test_multi_step_still_detects_a_genuinely_long_pause(self):
        self.assertEqual(response_time_state(90_000, "multi-step"), "STUCK")
        self.assertEqual(response_time_state(40_000, "multi-step"), "PLANNING")

    def test_rushing_is_detected_at_each_grain(self):
        self.assertEqual(response_time_state(200, "atomic"), "RUSHING")
        self.assertEqual(response_time_state(400, "multi-step"), "RUSHING")

    def test_times_between_the_cutoffs_imply_nothing(self):
        self.assertIsNone(response_time_state(3_000, "atomic"))
        self.assertIsNone(response_time_state(20_000, "multi-step"))

    def test_missing_and_invalid_times_are_handled(self):
        for value in (None, 0, -5, "", "abc", float("nan")):
            with self.subTest(value=value):
                result = response_time_state(value, "atomic")
                self.assertIn(result, (None, "RUSHING", "STUCK", "PLANNING"))
        self.assertIsNone(response_time_state(None))
        self.assertIsNone(response_time_state(0))
        self.assertIsNone(response_time_state("abc"))

    def test_multi_step_cutoffs_are_strictly_larger(self):
        atomic = GRAIN_RESPONSE_TIME_MS["atomic"]
        multi = GRAIN_RESPONSE_TIME_MS["multi-step"]
        for key in atomic:
            with self.subTest(key=key):
                self.assertGreater(multi[key], atomic[key])


class GrainWaitTests(unittest.TestCase):
    """Multi-step turns run ~8x longer, so the same wait is far more elapsed time."""

    def setUp(self):
        self.timer = InterventionTimer("test_grain_waits")

    def test_multi_step_never_waits_longer_than_atomic(self):
        for state in NEGATIVE_STATES:
            with self.subTest(state=state):
                self.assertLessEqual(
                    self.timer.recommended_wait(state, grain="multi-step"),
                    self.timer.recommended_wait(state, grain="atomic"))

    def test_states_that_waited_two_turns_now_act_a_turn_earlier(self):
        for state in ("CONFUSED", "STUCK"):
            with self.subTest(state=state):
                self.assertEqual(self.timer.recommended_wait(state, grain="atomic"), 2)
                self.assertEqual(
                    self.timer.recommended_wait(state, grain="multi-step"), 1)

    def test_the_gate_fires_a_turn_earlier_at_multi_step_grain(self):
        self.assertFalse(self.timer.should_intervene("STUCK", 1, grain="atomic"))
        self.assertTrue(self.timer.should_intervene("STUCK", 1, grain="multi-step"))

    def test_waits_stay_inside_the_bucket_range(self):
        for grain in INTERACTION_GRAINS:
            for state in NEGATIVE_STATES:
                for profile in ("hyperactive", "inattentive", "combined", ""):
                    with self.subTest(grain=grain, state=state, profile=profile):
                        wait = self.timer.recommended_wait(state, profile, grain)
                        self.assertGreaterEqual(wait, 1)
                        self.assertLessEqual(wait, 4)


if __name__ == "__main__":
    unittest.main()
