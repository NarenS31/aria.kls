"""Teacher view graphics: the chart mirrors the session events, and the other-settings strip agrees with the live decisions."""
import re

import pytest

import ui  # noqa: F401  (import the package first: eval/ui would otherwise shadow it once eval/ is on sys.path)
from agent.demo_bank import load_problems
from agent.demo_tutor import DemoTutor
from agent.demo_workspace import Workspace
from ui.teacher_chart import ORDER, bars, caption, chart_svg, first_hint, other_settings, risk_series, teacher_graphics
from ui.workspace_app import teacher_html

SEQ = ["3x - 12 = 2x + 5", "x - 12 = 5"] + ["x = 7"] * 4
FIRST_HINT_TRY = {"supportive": 3, "balanced": 4, "quiet": 6}      # the first, second and fourth x = 7


def run(profile, seq=SEQ, hint_after=None):
    w = Workspace(DemoTutor(problems=load_problems(), log_path=None, pace_profile=profile), "alg-01")
    for i, text in enumerate(seq, 1):
        assert w.submit(text, w.revision)
        if hint_after == i:
            assert w.hint(w.revision)
    return w


def count(svg, cls):
    return len(re.findall(rf'class="{cls}"', svg))


@pytest.mark.parametrize("profile", ORDER)
def test_chart_has_one_point_per_checked_step_with_the_stored_risk(profile):
    w = run(profile)
    tries, requested = risk_series(w)
    steps = [e for e in w.events if e.kind == "step"]
    assert len(tries) == len(steps) == 6 and requested == []
    assert [t["risk"] for t in tries] == [e.result.meta["pace_risk"] for e in steps]          # the values Decision details uses
    svg = chart_svg(w, tries, requested)
    assert count(svg, "rc-dot") == 6
    assert re.findall(r'data-risk="([\d.]+)"', svg) == [f"{e.result.meta['pace_risk']:.4f}" for e in steps]


@pytest.mark.parametrize("profile", ORDER)
def test_decision_markers_match_the_session_events(profile):
    w = run(profile)
    tries, requested = risk_series(w)
    svg = chart_svg(w, tries, requested)
    steps = [e for e in w.events if e.kind == "step"]
    assert count(svg, "rc-mk-right") == sum(e.result.status in ("step", "partial", "correct") for e in steps) == 2
    assert count(svg, "rc-mk-wrong") == sum(e.result.status == "wrong" for e in steps) == 4
    gave = [e for e in steps if e.result.intervened and e.result.meta["intervention_type"] in ("hint", "task_chunking")]
    held = [e for e in steps if e.result.status == "wrong" and not e.result.intervened]
    assert count(svg, "rc-mk-hint") == len(gave) and count(svg, "rc-mk-held") == len(held)
    assert len(gave) + len(held) == 4                                                          # every wrong step is one or the other
    assert next(t["n"] for t in tries if t["hint"]) == FIRST_HINT_TRY[profile]


def test_requested_hint_has_its_own_marker_after_its_step():
    w = run("quiet", ["3x - 12 = 2x + 5", "x - 12 = 5"], hint_after=1)
    tries, requested = risk_series(w)
    assert requested == [1] and len(tries) == 2
    svg = chart_svg(w, tries, requested)
    assert count(svg, "rc-mk-req") == 1 and count(svg, "rc-mk-hint") == 0
    assert "asked for a hint after try 1" in caption(w, tries, requested)


def test_bars_come_from_the_pace_config_and_the_active_one_is_bold():
    w = run("balanced")
    cfg = {p["key"]: p["tau"] for p in w.tutor.pace_cfg.profiles()}
    assert {k: t for k, _, t in bars(w)} == cfg
    svg = chart_svg(w, *risk_series(w))
    assert count(svg, "rc-bar rc-bar-on rc-stroke-balanced") == 1 and svg.count("rc-bar-off") == 2
    for tau in cfg.values():
        assert f"{tau:.0%}" in svg


@pytest.mark.parametrize("profile", ORDER)
def test_other_settings_strip_agrees_with_the_live_decisions(profile):
    w = run(profile)
    rows = other_settings(w)
    live = [t["hint"] for t in risk_series(w)[0]]
    assert [t["hint"] for t in rows[profile]] == live                                          # same as what actually happened
    assert {k: first_hint(v) for k, v in rows.items()} == FIRST_HINT_TRY                       # one answer sequence, three settings
    # and each other setting equals what a live session under that setting really does
    for other in ORDER:
        assert [t["hint"] for t in rows[other]] == [t["hint"] for t in risk_series(run(other))[0]]


def test_building_the_teacher_view_does_not_change_the_live_session():
    w = run("balanced")
    before = (w.revision, len(w.events), list(w.session.pace.obs), w.session.profile, dict(w.session.move_hints), len(w.session.log))
    html_out = teacher_html(w)
    teacher_graphics(w)
    assert (w.revision, len(w.events), list(w.session.pace.obs), w.session.profile, dict(w.session.move_hints), len(w.session.log)) == before
    assert "Stuck risk over this attempt" in html_out and "Same answers, other settings" in html_out


def test_text_alternative_and_plain_labels():
    w = run("balanced")
    out = teacher_html(w)
    assert out.count('<li>Try ') == 6 and "stuck risk 15%" in out and "ARIA held the hint" in out
    for label in ("steps checked", "steps that were correct", "hints the student asked for", "hints ARIA gave", "Full event log", "Where to follow up"):
        assert label in out
    assert "words, not math" not in out and "incorrect</td>" in out and "correct step</td>" in out
    assert 'href="#try-3"' in out and 'id="try-3"' in out
    assert "does not establish independent mastery" in out


def test_empty_attempt_still_renders():
    w = Workspace(DemoTutor(problems=load_problems(), log_path=None), "alg-01")
    out = teacher_html(w)
    assert "No checked steps yet" in out and "Where to follow up" in out
