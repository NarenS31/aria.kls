"""Wrong-answer and threshold matrix for the supported bank; not a learning evaluation."""
import itertools
from unittest import mock

import pytest

from agent.demo_bank import load_problems
from agent.demo_pace import PaceTracker, load_config
from agent.demo_tutor import DemoTutor
from agent.demo_workspace import Workspace
from agent.math_check import false_claims, reveals_answer

PROBLEMS = load_problems()
TUTOR = DemoTutor(problems=PROBLEMS)
PROFILES = ("quiet", "balanced", "supportive")
MISTAKES = {
    "alg-01": ["x = 7", "x = -17", "3x - 4 = 2x + 5", "x - 12 = 17"],
    "alg-05": ["x = 1", "x = -2", "x = 5", "x = 2 or x = 4"],
    "trig-01": ["0", "2", "-1", "sqrt(3)"],
    "trig-04": ["cos(x)", "sin(x)", "1", "sin(x)/cos(x)^2"],
    "calc-01": ["12x^3 - 10x - 9", "12x^3 - 10x + 7x", "3x^3 - 5x + 7"],
    "calc-02": ["2x*cos(2x) + 2x^2*sin(2x)", "2x*cos(2x) - x^2*sin(2x)"],
    "calc-06": ["6x*(x^2+1)^3", "3*(x^2+1)", "6*(x^2+1)^2"],
    "calc-07": ["2x^3 - 4x^2 + x", "12x^3 - 2x^2 + x", "2x^3 - 2x^2 - x"],
}


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("pid", list(MISTAKES))
def test_common_mistakes_across_profiles(pid, profile):
    p = TUTOR.problems[pid]
    for wrong in MISTAKES[pid]:
        w = Workspace(TUTOR, pid)
        w.session.set_profile(profile)
        assert w.submit(wrong, 0)
        r = w.events[-1].result
        assert r.status == "wrong", (pid, wrong, r.status)
        assert not r.accepted
        assert r.meta["pace_reason"] == "wrong_before_window"
        assert r.intervened  # All three preserve the early-attempt safety rule.
        assert not reveals_answer(r.reply, p.truth, [p.prompt, wrong]).leaked
        assert not false_claims(r.reply, p.truth, [p.prompt, wrong])
        assert w.submit(p.msg_correct(), w.revision)
        assert w.solved


@pytest.mark.parametrize("profile,first", [("quiet", 6), ("balanced", 4), ("supportive", 3)])
def test_two_valid_steps_then_four_wrong_steps(profile, first):
    w = Workspace(TUTOR, "calc-01")
    w.session.set_profile(profile)
    inputs = ["12x^3", "12x^3 - 10x"] + ["12x^3 - 10x - 9"] * 4
    for text in inputs:
        w.submit(text, w.revision)
    hints = [i + 1 for i, e in enumerate(w.events) if e.result.intervened]
    assert hints[0] == first, (profile, hints)
    assert [e.result.meta["pace_profile"] for e in w.events] == [profile] * 6
    if not w.hint(w.revision):
        assert "no more authored hints" in w.notice
    assert len(w.session.pace.obs) == 6


def test_all_policy_histories_to_nine_observations_preserve_gate():
    cfg = load_config()
    checked = 0
    for n in range(1, 10):
        for seq in itertools.product(("step", "wrong"), repeat=n):
            tracker = PaceTracker(cfg)
            for status in seq:
                tracker.update(status)
            for profile in PROFILES:
                d = tracker.snapshot(profile, seq[-1])
                expected = seq[-1] == "wrong" and (d.risk >= d.tau if cfg.k_min <= n <= cfg.k_max else True)
                assert d.speak == expected
                checked += 1
    assert checked == 3066


@pytest.mark.parametrize("profile", PROFILES)
def test_wrong_after_decision_window_always_gets_support(profile):
    tracker = PaceTracker(load_config())
    for _ in range(10):
        tracker.update("step")
    d = tracker.decide("wrong", profile)
    assert d.speak and d.reason == "wrong_after_window"


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("pid", [p.id for p in PROBLEMS])
def test_full_bank_authored_wrong_and_correct_answers(pid, profile):
    p = TUTOR.problems[pid]
    w = Workspace(TUTOR, pid)
    w.session.set_profile(profile)
    assert w.submit(p.msg_wrong(), 0)
    assert w.events[-1].result.status == "wrong"
    r = w.events[-1].result
    assert not reveals_answer(r.reply, p.truth, [p.prompt, p.msg_wrong()]).leaked
    assert not false_claims(r.reply, p.truth, [p.prompt, p.msg_wrong()])
    assert w.submit(p.msg_correct(), 1)
    assert w.solved


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("delta,expected", [(-0.000001, False), (0, True), (0.000001, True)])
def test_threshold_boundary(profile, delta, expected):
    cfg = load_config()
    tracker = PaceTracker(cfg)
    tracker.obs = [1, 1, 0]
    with mock.patch.object(tracker, "risk", return_value=cfg.taus[profile] + delta):
        assert tracker.snapshot(profile, "wrong").speak == expected
