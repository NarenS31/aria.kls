"""Compare the legacy token-overlap leak filter with the symbolic one (agent/math_check.py).

Legacy rule = agent/intervention_pipeline.py:431-435 (answer tokens must appear as a contiguous
run in the candidate's tokens).  For each demo problem we write the *answer* in several forms a
model could plausibly emit, and several non-leaking control strings, then count what each
filter catches / wrongly blocks.   Usage:  python scripts/compare_leak_filters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.demo_bank import load_problems  # noqa: E402
from agent.intervention_pipeline import _tokens  # noqa: E402
from agent.math_check import number_words, reveals_answer  # noqa: E402


def legacy_leaks(reply: str, correct_answer: str, student: str) -> bool:
    answer_norm = " ".join(_tokens(correct_answer))
    return bool(answer_norm and answer_norm in " ".join(_tokens(reply)) and answer_norm not in " ".join(_tokens(student)))


def sup(s: str) -> str:
    return s.replace("^2", "²").replace("^3", "³")


def variants(p):
    t = p.truth
    out = {}
    if t.kind == "expr":
        a = t.answer
        txt = t.answer_text()
        out["exact"] = f"The derivative is {txt}."
        out["reordered terms"] = f"Try {sp.sstr(sp.Add(*reversed(sp.Add.make_args(sp.expand(a)))), order='none').replace('**', '^').replace('*', '')}"
        out["python style (* and **)"] = f"Answer: {sp.sstr(a)}"
        out["unicode superscripts"] = f"It comes out to {sup(txt)}"
        out["LaTeX"] = f"So the result is ${sp.latex(a)}$"
        if not t.form_sensitive:      # for simplify/expand problems a *longer* rewrite is a hint, not the answer
            out["unsimplified form"] = f"That gives {p.spec['unsimplified'].replace(' + C', '')}"
    else:
        vals = sorted(t.solutions, key=float)
        out["x = value"] = " and ".join(f"x = {sp.sstr(v)}" for v in vals)
        out["x equals value"] = " and ".join(f"x equals {sp.sstr(v)}" for v in vals)
        u = p.spec["unsimplified"]
        u = u if isinstance(u, list) else [u]
        out["unevaluated (e.g. 34/2)"] = " and ".join(f"x = {v}" for v in u)
        ints = [v for v in vals if v.is_Integer and abs(int(v)) > 10]
        if ints:
            out["spelled out"] = f"x is {number_words(int(ints[0]))[0]}"
    return out


def controls(p):
    t = p.truth
    c = {"generic hint": "What is one small clue in the problem you can use first?",
         "restates problem": f"You are working on: {p.prompt}"}
    for i, m in enumerate(p.moves):
        c[f"verified next step {i+1}"] = f"{m['do']}. What do you get?" + (f" (Check: {m['result']})" if "result" in m and "claim" not in m and t.kind == "solve" else "")
    c["step number"] = "Step 1: look at the first term. Step 2: ..."
    return c


def main():
    probs = load_problems()
    tally: dict = {}
    ctrl = {"legacy": 0, "symbolic": 0, "n": 0}
    for p in probs:
        ctx = [p.prompt]
        for name, reply in variants(p).items():
            r = tally.setdefault(name, {"n": 0, "legacy": 0, "symbolic": 0})
            r["n"] += 1
            r["legacy"] += legacy_leaks(reply, p.truth.answer_text(), p.prompt)
            r["symbolic"] += reveals_answer(reply, p.truth, ctx).leaked
        for name, reply in controls(p).items():
            ctrl["n"] += 1
            ctrl["legacy"] += legacy_leaks(reply, p.truth.answer_text(), p.prompt)
            ctrl["symbolic"] += reveals_answer(reply, p.truth, ctx).leaked
    print(f"{'leak form (answer written as...)':34s} {'n':>3s} {'legacy catches':>15s} {'symbolic catches':>17s}")
    tn = tl = ts = 0
    for name, r in sorted(tally.items()):
        print(f"{name:34s} {r['n']:3d} {r['legacy']:7d} ({100*r['legacy']/r['n']:3.0f}%) {r['symbolic']:9d} ({100*r['symbolic']/r['n']:3.0f}%)")
        tn += r["n"]; tl += r["legacy"]; ts += r["symbolic"]
    print(f"{'ALL LEAK VARIANTS':34s} {tn:3d} {tl:7d} ({100*tl/tn:3.0f}%) {ts:9d} ({100*ts/tn:3.0f}%)")
    print(f"\nnon-leaking controls: n={ctrl['n']}  false blocks  legacy={ctrl['legacy']}  symbolic={ctrl['symbolic']}")


if __name__ == "__main__":
    main()
