"""Demo problem bank: authored problems + SymPy ground truth (see agent/math_check.py).

`data/demo_problems.json` holds, per problem: prompt, the authored answer (or equation),
terms/steps for partial credit, and scripted student utterances for the harness.
`load_problems()` builds a SymPy `Truth` for each one and **independently re-derives
the answer with SymPy** (diff / integrate / expand / simplify / numeric value), so a typo
in the JSON fails loudly at load time instead of in front of the audience.

Problems marked `"featured": true` are the short list shown in the live demo (UI default; `load_problems(featured_only=True)`).
For them the re-derivation is mandatory, not optional: an expression problem must carry a `check` op, and a solve problem
must get its roots from `sp.solveset` on the equation (authored roots, which are only substituted back, are not accepted).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sympy as sp

from agent.math_check import Truth, check_answer, equation_equivalent, equivalent, normalize, to_expr

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_problems.json"


@dataclass
class DemoProblem:
    id: str
    level: str
    topic: str
    prompt: str
    truth: Truth
    spec: dict
    moves: list = field(default_factory=list)

    @property
    def featured(self) -> bool:
        return bool(self.spec.get("featured"))

    # ---- scripted student utterances (used by the harness; the tutor never sees these as labels)
    def msg_correct(self) -> str:
        return self._fmt(self.truth.answer_text() if self.truth.kind == "expr" else None, solve_all=True)

    def msg_wrong(self) -> str:
        return self._fmt(self.spec["wrong"])

    def msg_unsimplified(self) -> str:
        return self._fmt(self.spec["unsimplified"])

    def msg_step(self) -> Optional[str]:
        return self.spec.get("step_ok")

    def msg_latex(self) -> str:
        """The correct answer as a LaTeX-wrapped message, the way a pasted answer might arrive."""
        if self.truth.kind == "expr":
            return f"${sp.latex(self.truth.answer)}$"
        vals = sorted(self.truth.solutions, key=float)
        return " , ".join(f"${self.truth.var} = {sp.latex(v)}$" for v in vals)

    def msg_restate(self) -> str:
        return self.spec["equation"].replace("=", " = ", 1) if self.truth.kind == "solve" else f"f(x) = {self.spec['given'][0]}"

    def _fmt(self, value, solve_all: bool = False) -> str:
        if self.truth.kind == "solve":
            if solve_all:
                return " or ".join(f"{self.truth.var} = {sp.sstr(v).replace('**', '^')}" for v in sorted(self.truth.solutions, key=float))
            vals = value if isinstance(value, list) else [value]
            return " or ".join(f"{self.truth.var} = {v}" for v in vals)
        return f"My answer is {value}"


def _derive(spec: dict, truth: Truth):
    """Re-derive the answer from the problem itself, with SymPy. -> (description, reference) or None if the spec has no `check`."""
    chk = spec.get("check")
    if truth.kind != "expr" or not chk:
        return None
    x = truth.var
    src = to_expr(normalize(chk["of"]))
    if src is None:
        return f"check.of unparseable: {chk['of']!r}", None
    op = chk["op"]
    ref = {"diff": lambda: sp.diff(src, x), "integrate": lambda: sp.integrate(src, x),
           "expand": lambda: sp.expand(src), "simplify": lambda: sp.simplify(src),
           "value": lambda: sp.nsimplify(sp.N(src, 30))}[op]()
    return op, ref


def rederive(p: "DemoProblem") -> dict:
    """What SymPy gets from the problem alone, next to the authored answer (used by tests and the demo audit)."""
    t = p.truth
    if t.kind == "solve":
        l, r = t.given_eq
        got = sp.solveset(l - r, t.var, sp.S.Reals)
        return {"id": p.id, "how": f"solveset({l - r} = 0)", "sympy": sorted(got, key=float), "authored": sorted(t.solutions, key=float)}
    d = _derive(p.spec, t)
    op, ref = d if d else ("none", None)
    return {"id": p.id, "how": f"{op}({p.spec.get('check', {}).get('of')})", "sympy": ref, "authored": t.answer}


def _check_op(spec: dict, truth: Truth) -> Optional[str]:
    d = _derive(spec, truth)
    if d is None:
        return None
    op, ref = d
    if ref is None:
        return op
    if not equivalent(truth.answer, ref, truth.mod_const):
        return f"authored answer {truth.answer} != SymPy {op} result {ref}"
    return None


_PROMPT_EXPR = re.compile(r"^(?:Differentiate\s+f\(x\)\s*=\s*|Find the integral of\s+|Simplify\s+|Evaluate\s+|Expand\s+)(.+?)(?:\s+dx)?$")
_PROMPT_EQ = re.compile(r"^Solve for x:\s*(.+)$")


def _featured_errors(p: "DemoProblem") -> list:
    """Extra requirements for the problems shown live: the answer must be re-derived by SymPy, not just authored."""
    errs, t = [], p.truth
    if t.kind == "expr":
        if not p.spec.get("check"):
            errs.append("featured expression problem has no `check` op, so its answer is not re-derived by SymPy")
        else:                                   # the expression being differentiated/integrated/... must be the one in the prompt
            m = _PROMPT_EXPR.match(p.prompt)
            shown = to_expr(normalize(m.group(1))) if m else None
            src = to_expr(normalize(p.spec["check"]["of"]))
            if shown is None or src is None or not equivalent(shown, src):
                errs.append(f"`check.of` {p.spec['check']['of']!r} is not the expression shown in the prompt {p.prompt!r}")
    else:
        m = _PROMPT_EQ.match(p.prompt)
        if m and "=" in m.group(1):
            a, b = (to_expr(normalize(side)) for side in m.group(1).split("=", 1))
            if a is None or b is None or not equation_equivalent(a, b, *t.given_eq):
                errs.append(f"the equation in the prompt {p.prompt!r} is not the one the answer was derived from")
        else:
            errs.append(f"cannot read the equation out of the prompt {p.prompt!r}")
        if "solutions" in p.spec:
            errs.append("featured solve problem uses authored `solutions`; roots must come from sp.solveset on the equation")
        l, r = t.given_eq
        got = sp.solveset(l - r, t.var, sp.S.Reals)
        if not isinstance(got, sp.FiniteSet) or set(got) != set(t.solutions):
            errs.append(f"solveset {got} != roots used by the tutor {set(t.solutions)}")
        for v in t.solutions:
            if abs(complex(sp.N((l - r).subs(t.var, v), 30))) > 1e-12:
                errs.append(f"root {v} does not satisfy the equation")
    if not p.moves:
        errs.append("featured problem has no authored hint moves")
    return errs


def verify(p: DemoProblem) -> list:
    """Return a list of problems with the authored data (empty == consistent)."""
    errs = []
    e = _check_op(p.spec, p.truth)
    if e:
        errs.append(e)
    if p.featured:
        errs += _featured_errors(p)
    t = p.truth
    expect = [
        (p.msg_correct(), {"correct"}, "correct answer"),
        (p.msg_unsimplified(), {"correct"}, "unsimplified answer"),
        (p.msg_wrong(), {"wrong"}, "wrong answer"),
        (p.msg_restate(), {"restate", "step"}, "restatement"),
    ]
    if p.msg_step():
        expect.append((p.msg_step(), {"partial", "step"}, "step_ok"))
    for msg, ok, label in expect:
        st = check_answer(msg, t).status
        if st not in ok:
            errs.append(f"{label} {msg!r} classified {st!r}, expected {sorted(ok)}")
    for m in p.moves:
        if "claim" in m:
            l, r = (to_expr(normalize(s)) for s in m["claim"].split("=", 1))
            if l is None or r is None or not equivalent(l, r):
                errs.append(f"move claim {m['claim']!r} is not a true identity")
            continue
        if "result" in m:
            st = check_answer(m["result"], t).status
            if st not in {"partial", "step", "correct", "restate"} and not (t.kind == "solve" and "=" in m["result"]):
                errs.append(f"move result {m['result']!r} classified {st!r}")
            if t.kind == "solve" and "=" in m["result"]:
                l, r = (to_expr(normalize(s)) for s in m["result"].split("=", 1))
                if not equation_equivalent(l, r, *t.given_eq):
                    errs.append(f"move result {m['result']!r} is not equivalent to the given equation")
    return errs


def load_problems(path: Path | str | None = None, check: bool = True, featured_only: bool = False) -> list:
    raw = json.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    out = []
    for spec in raw:
        if featured_only and not spec.get("featured"):
            continue
        truth = Truth.from_spec({**spec, "var": spec.get("var", "x")})
        p = DemoProblem(spec["id"], spec["level"], spec["topic"], spec["prompt"], truth, spec, spec.get("moves", []))
        if check:
            errs = verify(p)
            if errs:
                raise ValueError(f"{p.id}: " + "; ".join(errs))
        out.append(p)
    return out
