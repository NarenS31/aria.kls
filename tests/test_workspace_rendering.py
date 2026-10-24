"""The workspace shows what was written, never what it evaluates to (e.g. cos(pi/4)^2 must not become 1/2)."""
import json
import re
from collections import Counter
from pathlib import Path

import pytest

import ui  # noqa: F401  (import the package first: eval/ui would otherwise shadow it once eval/ is on sys.path)
from agent.demo_bank import load_problems
from agent.demo_tutor import DemoTutor
from agent.demo_workspace import Workspace
from ui.workspace_app import math_text, problem_md, work_md

PROBLEMS = json.loads((Path(__file__).resolve().parent.parent / "data/demo_problems.json").read_text())


def written_tokens(text):
    """Names and numbers as typed, order-free: 'cos(pi/4)^2' -> cos, pi, 4, 2."""
    return Counter(re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text))


def displayed_tokens(md):
    tex = "".join(re.findall(r"\$\$(.*?)\$\$", md, re.S))
    tex = re.sub(r"\\(left|right|cdot|frac)\b", " ", tex)
    return Counter(re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", tex))


def shown_math(md):
    return "".join(re.findall(r"\$\$(.*?)\$\$", md, re.S))


def test_the_bank_has_all_thirty_problems():
    assert len(PROBLEMS) == 30


@pytest.mark.parametrize("spec", PROBLEMS, ids=[p["id"] for p in PROBLEMS])
def test_problem_card_shows_the_written_problem_not_its_value(spec):
    problem = next(p for p in load_problems() if p.id == spec["id"])
    md = problem_md(problem)
    written = spec["equation"] if spec["kind"] == "solve" else spec["check"]["of"]
    assert "$$" in md, f"{spec['id']} fell back to plain text: {md}"
    assert displayed_tokens(md) == written_tokens(written), (spec["id"], shown_math(md), written)
    # every displayed name and number really is in the problem as authored
    assert not displayed_tokens(md) - written_tokens(spec["prompt"]), (spec["id"], spec["prompt"])


def test_evaluate_cos_pi_over_4_squared_is_not_shown_as_one_half():
    problem = next(p for p in load_problems() if p.id == "trig-03")
    md = problem_md(problem)
    assert r"\left(\cos{\left(\frac{\pi}{4} \right)}\right)^{2}" in md
    assert r"\frac{1}{2}" not in md


@pytest.mark.parametrize("step", [
    "sqrt(2)/2", "x = sqrt(2)/2", "2/4", "6/3", "1/2 + 1/3", "x/2 + 1/2", "(2x - 1)(x + 4)", "cos(pi/4)^2", "sin(pi/6) + cos(pi/3)",
    "3(x - 4) = 2x + 5", "x = 17", "x = -3/4", "(x + 1)(x - 1)/(x^2 - 1)", "ln(x^2 + 1)", "2*3/6", "tan(pi/4)*cos(pi/3)",
])
def test_student_steps_are_shown_as_written(step):
    md = math_text(step)
    assert md.startswith("$$"), md
    assert displayed_tokens(md) == written_tokens(step), (step, shown_math(md))


@pytest.mark.parametrize("step,evaluated", [
    ("sqrt(2)/2", r"\frac{1}{\sqrt{2}}"), ("2/4", r"\frac{1}{2}"), ("6/3", "2"), ("cos(pi/4)^2", r"\frac{1}{2}"),
    ("1/2 + 1/3", r"\frac{5}{6}"), ("(2x - 1)(x + 4)", "2x^{2}+7x-4"), ("x = 6/3", "x=2"), ("sin(pi/6) + cos(pi/3)", "1"),
])
def test_steps_are_never_simplified_or_expanded(step, evaluated):
    assert shown_math(math_text(step)).replace(" ", "") != evaluated.replace(" ", "")


def test_unparseable_text_falls_back_to_plain_escaped_text():
    for text in ["I think x = 5", "hello <b>world</b>", "x = 1 = 2"]:
        md = math_text(text)
        assert md.startswith('<p class="literal">') and "<b>" not in md


def test_requested_hint_is_open_and_directly_after_its_step():
    w = Workspace(DemoTutor(problems=load_problems(), log_path=None, pace_profile="quiet"), "alg-01")
    assert w.submit("3x - 12 = 2x + 5", w.revision)
    assert w.hint(w.revision)
    assert w.submit("x - 12 = 5", w.revision)
    md = work_md(w.events)
    step1, hint, step2 = md.index("Step 1"), md.index('<details class="hint-history" open>'), md.index("Step 2")
    assert step1 < hint < step2
    assert "Requested hint" in md[hint:step2]


def test_problem_labels_keep_every_bracket_of_the_written_prompt():
    from ui.workspace_app import problem_label
    for spec in PROBLEMS:
        label = problem_label(spec["prompt"])
        assert label.count("(") == label.count(")") == spec["prompt"].count("(") == spec["prompt"].count(")"), (spec["id"], label)
    assert problem_label("Differentiate f(x) = e^(2x)").endswith("e^(2x)")
    assert problem_label("Evaluate cos(pi/4)^2").endswith("cos(π/4)²")
