"""Product behavior is separate from the frozen policy evaluation."""
import json
from unittest import mock

from agent.demo_bank import load_problems
from agent.demo_progress import completed_moves
from agent.demo_tutor import DemoTutor
from agent.demo_workspace import Workspace
from agent.math_check import false_claims, reveals_answer


def workspace(pid="alg-01"):
    return Workspace(DemoTutor(problems=load_problems()), pid)


def test_equation_progress_skips_completed_operations_not_hint_count():
    w = workspace()
    assert w.submit("x - 12 = 5", 0)
    assert completed_moves(w.session)[:2] == [True, True]
    assert w.hint(1)
    assert "Subtract 2x" not in w.events[-1].result.reply
    assert "12" in w.events[-1].result.reply


def test_hint_does_not_advance_math_or_skip_uncompleted_move():
    w = workspace()
    assert w.hint(0)
    assert len(w.session.pace.obs) == 0
    first = w.events[-1].result.meta["move_index"]
    assert w.hint(1)
    assert w.events[-1].result.meta["move_index"] == first
    assert len(w.session.pace.obs) == 0
    assert not w.hint(2)
    assert "no more authored hints" in w.notice


def test_duplicate_guard_allows_deliberate_identical_attempt():
    w = workspace()
    assert w.submit("x = 7", 0)
    assert not w.submit("x = 7", 0)
    assert len(w.events) == 1
    assert w.submit("x = 7", 1)
    assert len(w.events) == 2


def test_revision_replays_active_branch_and_keeps_original():
    w = workspace()
    w.submit("x = 7", 0)
    w.hint(1)
    w.submit("x = 17", 2)
    assert w.solved
    original_meta = dict(w.events[0].result.meta)
    assert w.submit("x - 12 = 5", 3, edit=0)
    assert not w.solved
    assert len(w.events) == 1
    assert len(w.branches[0]["events"]) == 3
    assert w.branches[0]["events"][0].result.meta == original_meta
    assert len(w.session.pace.obs) == 1
    assert w.summary()["outcome"] == "in_progress"


def test_revision_does_not_duplicate_file_log():
    w = workspace()
    with mock.patch.object(w.tutor.turn_log, "write") as write:
        w.submit("x = 7", 0)
        w.submit("x - 12 = 5", 1)
        assert write.call_count == 2
        w.submit("3x - 12 = 2x + 5", 2, edit=1)
        assert write.call_count == 3


def test_retry_preserves_history_and_graph_clear_is_distinct():
    w = workspace()
    w.submit("x = 7", 0)
    before = w.session.graph.snapshot()
    w.change_problem("alg-01", retry=True)
    assert w.session.graph.snapshot() == before
    assert len(w.session.pace.obs) == 0
    assert len(w.archive) == 1
    w.clear()
    assert not w.session.graph.has_data()
    assert not w.archive and not w.events


def test_completion_not_repeated_and_next_problem_preserves_work():
    w = workspace()
    assert w.submit("x = 17", 0)
    assert w.solved
    assert not w.submit("x = 17", 1)
    assert not w.hint(1)
    w.change_problem("calc-01")
    assert not w.solved
    assert w.archive[0]["summary"]["outcome"] == "solved"


def test_failure_keeps_active_work_and_does_not_update_estimate():
    w = workspace()
    w.submit("x - 12 = 5", 0)
    with mock.patch.object(type(w.session), "_analyze", side_effect=RuntimeError("failure")):
        assert not w.submit("x = 17", 1)
    assert len(w.events) == 1
    assert len(w.session.pace.obs) == 1
    assert "input is still here" in w.notice


def test_sessions_isolated_and_unknown_input_not_an_incorrect_attempt():
    a, b = workspace(), workspace()
    a.submit("asdf", 0)
    assert a.events[0].result.status == "none"
    assert len(a.session.pace.obs) == 0
    assert not b.events and not b.session.log


def test_requested_hint_below_threshold_and_profile_snapshot():
    w = workspace()
    w.session.set_profile("quiet")
    for text in ("3x - 12 = 2x + 5", "x - 12 = 5", "x = 7"):
        w.submit(text, w.revision)
    assert not w.events[-1].result.intervened
    n = len(w.session.pace.obs)
    assert w.hint(w.revision)
    assert len(w.session.pace.obs) == n
    assert w.events[-1].result.meta["hint_source"] == "requested"
    w.session.set_profile("supportive")
    assert all(e.profile == "quiet" for e in w.events)
    assert w.summary()["counts"]["requested_hints"] == 1


def test_featured_hints_never_leak_or_lie():
    for p in load_problems(featured_only=True):
        w = workspace(p.id)
        for _ in range(2):
            w.hint(w.revision)
            reply = w.events[-1].result.reply
            assert not reveals_answer(reply, p.truth, [p.prompt]).leaked, (p.id, reply)
            assert not false_claims(reply, p.truth, [p.prompt]), (p.id, reply)


def test_math_rendering_and_quiet_work_are_not_chat():
    from ui.workspace_app import math_text, work_md, teacher_html
    w = workspace()
    w.submit("x - 12 = 5", 0)
    assert "Watching" not in work_md(w.events)
    assert "Valid step" in work_md(w.events)
    assert work_md([]) == ""
    assert "\\frac" in math_text("1/2")
    assert "Where to follow up" in teacher_html(w)
    assert "x - 12 = 5" not in json.dumps(w.summary())


def test_workspace_completion_hides_composer_and_revision_restores_it():
    from ui.workspace_app import build_workspace
    app = build_workspace(DemoTutor(problems=load_problems()))
    callbacks = {f.fn.__name__: f for f in app.fns.values() if f.fn is not None}
    initial = callbacks["start"].fn()
    assert len(initial) == len(callbacks["start"].outputs)
    w = initial[0]
    updates = list(callbacks["send_ui"].fn("x = 17", w, 0, "new"))
    assert updates[0][12]["value"] == "Checking…"
    assert updates[0][12]["interactive"] is False
    complete = updates[-1]
    assert complete[12]["value"] == "Check step"
    assert len(complete) == len(callbacks["send_ui"].outputs)
    assert complete[11]["visible"] is True
    assert complete[18]["visible"] is False
    revision = callbacks["pick_edit"].fn("0", w)
    assert revision[18]["visible"] is True


def test_checking_failure_restores_button_and_preserves_draft():
    from ui.workspace_app import build_workspace
    app = build_workspace(DemoTutor(problems=load_problems()))
    callbacks = {f.fn.__name__: f for f in app.fns.values() if f.fn is not None}
    w = callbacks["start"].fn()[0]
    with mock.patch.object(w, "submit", side_effect=RuntimeError("test failure")):
        updates = list(callbacks["send_ui"].fn("x = 7", w, 0, "new"))
    final = updates[-1]
    assert final[9]["value"] == "x = 7"
    assert final[9]["interactive"] is True
    assert final[12]["value"] == "Check step"
    assert final[12]["interactive"] is True


def test_profile_change_never_replaces_a_draft_being_typed():
    from ui.workspace_app import build_workspace
    app = build_workspace(DemoTutor(problems=load_problems()))
    callbacks = {f.fn.__name__: f for f in app.fns.values() if f.fn is not None}
    w = callbacks["start"].fn()[0]
    updates = callbacks["set_profile"].fn("quiet", w, "old draft", "new")
    assert w.session.profile == "quiet"
    assert "value" not in updates[9]
    assert "value" not in updates[15]


def test_expired_ui_session_recovers_on_selected_problem_without_scoring_draft():
    from ui.workspace_app import build_workspace
    app = build_workspace(DemoTutor(problems=load_problems()))
    callbacks = {f.fn.__name__: f for f in app.fns.values() if f.fn is not None}
    restarted = callbacks["restart"].fn(None, "alg-05")
    assert restarted[0].session.problem.id == "alg-05"
    checked = list(callbacks["send_ui"].fn("x = 7", None, 5, "new", "alg-05"))[-1]
    assert checked[0].session.problem.id == "alg-05"
    assert checked[0].events == []
    assert checked[9]["value"] == "x = 7"
    hint = callbacks["ask"].fn(None, 5, "x = 7", "new", "alg-05")
    assert hint[0].session.problem.id == "alg-05"
    assert hint[0].events == []
    revision = callbacks["pick_edit"].fn("3", None, "alg-05")
    assert revision[0].session.problem.id == "alg-05"
