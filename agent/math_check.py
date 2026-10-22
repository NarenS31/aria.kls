"""Symbolic ground truth for the summit demo (SymPy).

Everything the demo says about "is this right?" or "did the tutor give it away?"
goes through this module, never through string matching.

Public API
----------
    truth = Truth.from_spec({...})            # authored problem -> SymPy truth
    check_answer(text, truth) -> Verdict      # status: correct|partial|step|restate|wrong|none
    reveals_answer(reply, truth, context=())  # -> LeakResult (symbolic, not token overlap)
    false_claims(reply, truth, context=())    # -> list[str] of math claims in `reply` that SymPy refutes
    to_expr(text) -> sympy expr | None        # guarded parser (never evals prose)
    render(expr) -> str                       # SymPy -> plain text a student can read

Design rules
------------
* Parsing is guarded: length cap, character whitelist, name whitelist, exponent cap.
  Free text is never handed to ``parse_expr`` without passing those guards.
* Prose-embedded bare expressions are never classified *wrong*. Only an "explicit"
  answer (a message that is pure math, an ``f'(x) = ...`` / ``answer is ...`` /
  ``x = ...`` form, or an equation) can be wrong.  Restating the problem is neutral.
* Equivalence = numeric sampling (decisive reject, ≥4 valid points to accept) with
  ``simplify(a - b) == 0`` as the confirmation/fallback; so unsimplified and reordered
  forms of the right answer are accepted.
"""
from __future__ import annotations

import functools
import itertools
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

MAX_INPUT_CHARS = 400          # per candidate expression
MAX_OPS = 400                  # parsed expression size cap
MAX_CANDIDATES = 12            # expressions/equations examined per message (more is not a real student step)
_SYMS = {c: sp.Symbol(c, real=True) for c in "xyzt"}
_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "ln": sp.log, "log": sp.log,
    "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs,
}
_LOCALS = {**_SYMS, **_FUNCS, "pi": sp.pi, "e": sp.E}
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_SAMPLE_POINTS = [0.21, 0.37, 0.55, 0.73, 0.91, 1.43, 1.87, 2.31, 2.77]

# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
_SUP_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_ANSWER_LABEL = re.compile(          # "answer is", f'(x) =, y' =, dy/dx =  -> a deliberate answer
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)"
    r"|\b(?:[fgy]\s*['′]+\s*(?:\(\s*[a-z]\s*\))?|dy\s*/\s*dx)\s*=",
    re.I,
)
_DEF_LABEL = re.compile(r"\b[fgy]\s*(?:\(\s*[a-z]\s*\))?\s*=", re.I)   # f(x) =, y =  -> a definition


def normalize(text: str) -> str:
    """Unicode/LaTeX -> plain ascii math. Keeps ``^`` (converted later)."""
    s = (text or "")
    s = re.sub(r"([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)", lambda m: "^(" + m.group(1).translate(_SUP_MAP) + ")", s)
    for a, b in (("−", "-"), ("–", "-"), ("—", "-"), ("×", "*"), ("·", "*"), ("⋅", "*"),
                 ("÷", "/"), ("π", "pi"), ("√", "sqrt"), ("′", "'"), ("’", "'"),
                 ("≈", "="), ("→", " "), ("⇒", " "), ("✅", " "), ("“", " "), ("”", " "), ('"', " "), ("«", " "), ("»", " ")):
        s = s.replace(a, b)
    # LaTeX
    s = s.replace("$", " ")
    s = re.sub(r"\\(?:left|right|,|!|;|quad|qquad)", " ", s)
    s = re.sub(r"\\text\s*\{[^{}]*\}", " ", s)
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"^(\1)", s)
    for _ in range(4):
        s2 = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\\(cdot|times)", "*", s)
    s = re.sub(r"\\(sin|cos|tan|ln|log|exp|sqrt|pi)\b", r"\1", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.lower()
    s = re.sub(r"\bis equal to\b|\bequals\b", "=", s)
    s = re.sub(r"\s*(?:\+|plus)\s*c\b(?!\w)", " ", s)             # "+ C" on antiderivatives
    s = _ANSWER_LABEL.sub(" ans= ", s)
    s = _DEF_LABEL.sub(" def= ", s)
    # sin^2(x) -> (sin(x))**2
    s = re.sub(r"\b(sin|cos|tan)\s*\^\s*\(?(\d+)\)?\s*(\((?:[^()]|\([^()]*\))*\))", r"(\1\3)^\2", s)
    return s


_TOKEN_OK = re.compile(r"(?:ans=|def=|pi|e\^|sin|cos|tan|ln|log|exp|sqrt|abs|[0-9xyzt+\-*/^().=,±])+")


def _split_tokens(s: str) -> list[tuple[str, bool]]:
    """-> [(token, breaks_run_after)], trailing sentence punctuation removed."""
    out: list[tuple[str, bool]] = []
    for raw in s.split():
        brk = raw[-1:] in {",", ";", ":"} and len(raw) > 1
        tok = raw.rstrip(".,;:!?")
        out.append((tok, brk))
    return out


def _is_mathy(tok: str) -> bool:
    return bool(tok) and bool(_TOKEN_OK.fullmatch(tok))


def _math_runs(s: str) -> tuple[list[str], bool]:
    """Maximal runs of consecutive math-looking tokens + whether the whole message is math."""
    runs: list[str] = []
    cur: list[str] = []
    all_math = True
    seen = False
    for tok, brk in _split_tokens(s):
        if _is_mathy(tok):
            seen = True
            cur.append(tok)
            if brk:
                runs.append(" ".join(cur))
                cur = []
        else:
            if tok:
                all_math = False
            if cur:
                runs.append(" ".join(cur))
                cur = []
    if cur:
        runs.append(" ".join(cur))
    runs = [r for r in runs if re.search(r"[0-9xyzt]|pi", r.replace("ans=", ""))]
    return runs, (all_math and seen)


def prose_of(text: str) -> str:
    """The words in `text` with every math-looking token removed (what the language-marker
    analyzer should see: its repeated-phrase rule misfires on symmetric expressions)."""
    keep = []
    for tok in (text or "").split():
        core = normalize(tok).strip().rstrip(".,;:!?")
        if core and _is_mathy(core) and re.search(r"[0-9xyzt]|pi|=|[-+*/^]", core):
            continue
        keep.append(tok)
    return " ".join(keep)


# --------------------------------------------------------------------------- #
# guarded parsing
# --------------------------------------------------------------------------- #
_ALLOWED_CHARS = re.compile(r"[0-9a-z+\-*/^().,\s]*")
_ALPHA_RUN = re.compile(r"[a-z]+")


def _segmentable(word: str) -> bool:
    if word in _LOCALS:
        return True
    if not word:
        return True
    for name in ("sin", "cos", "tan", "sqrt", "exp", "abs", "log", "ln", "pi"):
        if word.startswith(name) and _segmentable(word[len(name):]):
            return True
    return word[0] in _SYMS and _segmentable(word[1:]) or (word[0] == "e" and len(word) == 1)


def _segments(word: str) -> list:
    if word in _LOCALS or not word:
        return [word] if word else []
    for name in ("sin", "cos", "tan", "sqrt", "exp", "abs", "log", "ln", "pi"):
        if word.startswith(name):
            rest = _segments(word[len(name):])
            if rest is not None:
                return [name] + rest
    if word[0] in _SYMS or word == "e":
        rest = _segments(word[1:])
        if rest is not None:
            return [word[0]] + rest
    return None


def to_expr(text: str) -> Optional[sp.Expr]:
    """Parse a *single* math expression (already normalised). None if unsafe/unparseable."""
    s = (text or "").strip()
    if not s or len(s) > MAX_INPUT_CHARS or not _ALLOWED_CHARS.fullmatch(s):
        return None
    if re.search(r"[a-z]\s*\.|\.\s*[a-z]|\d{13,}|,\s*,|\*\*\s*\*\*", s):
        return None
    if re.search(r"(?:\^|\*\*)\s*\(?\s*-?\d{3,}", s):        # x^1000, 9**999
        return None
    if re.search(r"\d\s*(?:\^|\*\*)\s*\(?\s*\d+\s*(?:\^|\*\*)", s):         # 9^9^9
        return None
    for w in _ALPHA_RUN.findall(s):
        if not _segmentable(w):
            return None
    s = _ALPHA_RUN.sub(lambda m: " ".join(_segments(m.group(0)) or [m.group(0)]), s)
    try:
        e = parse_expr(s, local_dict=dict(_LOCALS), transformations=_TRANSFORMS, evaluate=True)
    except Exception:  # noqa: BLE001 - any parse failure means "not math"
        return None
    if not isinstance(e, sp.Expr):
        return None
    if sp.count_ops(e) > MAX_OPS:
        return None
    e = e.xreplace({f: sp.nsimplify(f, rational=True) for f in e.atoms(sp.Float)})
    if e.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None
    return e


# --------------------------------------------------------------------------- #
# candidates
# --------------------------------------------------------------------------- #
@dataclass
class Cand:
    kind: str                       # "expr" | "eq" | "assign"
    explicit: bool                  # a deliberate answer attempt (can be judged wrong)
    expr: Optional[sp.Expr] = None  # expr
    lhs: Optional[sp.Expr] = None   # eq
    rhs: Optional[sp.Expr] = None   # eq
    var: str = ""                   # assign
    values: tuple = ()              # assign
    raw: str = ""
    labeled: bool = False           # written as "answer is ..." / f'(x) = ...
    lraw: str = ""                  # eq: raw text of each side
    rraw: str = ""

    def sides(self) -> list:
        """[(expr, raw text)] for every math expression this candidate states."""
        if self.kind == "expr":
            return [(self.expr, self.raw)]
        if self.kind == "eq":
            return [(self.lhs, self.lraw), (self.rhs, self.rraw)]
        return []


def _parse_values(rhs: str) -> Optional[tuple]:
    if "±" in rhs:
        v = to_expr(rhs.replace("±", " ").strip())
        return None if v is None else (v, -v)
    v = to_expr(rhs)
    return None if v is None else (v,)


def extract_candidates(text: str) -> list[Cand]:
    return list(_extract_cached(text))


@functools.lru_cache(maxsize=256)
def _extract_cached(text: str) -> tuple:
    """Pure function of the message, and the same message is examined several times per turn (verdict, terms,
    leak-check context), so it is cached. Candidates are read-only."""
    return tuple(_extract(text))


def _extract(text: str) -> list[Cand]:
    s = normalize(text)
    runs, whole = _math_runs(s)
    cands: list[Cand] = []
    for run in runs:
        if len(cands) >= MAX_CANDIDATES:
            break
        labeled = run.startswith("ans=")
        defn = run.startswith("def=")
        body = run[4:].strip() if (labeled or defn) else run
        body = body.strip(" ,")
        if not body:
            continue
        parts = [p.strip() for p in body.split("=")]
        if labeled or defn or (len(parts) == 2 and parts[0] == ""):
            rhs = parts[-1] if parts[-1] else (parts[-2] if len(parts) > 1 else "")
            e = to_expr(rhs)
            if e is not None:
                cands.append(Cand("expr", True, expr=e, raw=run, labeled=labeled or not defn))
            continue
        if len(parts) == 1:
            e = to_expr(parts[0])
            if e is not None:
                cands.append(Cand("expr", whole, expr=e, raw=run))
            continue
        # one or more '='
        sides = [to_expr(p) if "±" not in p else None for p in parts]
        if len(parts) == 2 and re.fullmatch(r"[xyzt]", parts[0]) and parts[1]:
            vals = _parse_values(parts[1])
            if vals is not None:
                cands.append(Cand("assign", True, var=parts[0], values=vals, raw=run))
                continue
        for (a, b), (pa, pb) in zip(zip(sides, sides[1:]), zip(parts, parts[1:])):
            if len(cands) >= MAX_CANDIDATES:
                break
            if a is not None and b is not None:
                cands.append(Cand("eq", True, lhs=a, rhs=b, raw=f"{pa} = {pb}", lraw=pa, rraw=pb))
    return cands


# --------------------------------------------------------------------------- #
# equivalence
# --------------------------------------------------------------------------- #
def _numeric_values(expr: sp.Expr, syms: list[sp.Symbol]) -> list[Optional[complex]]:
    try:
        f = sp.lambdify(syms, expr, modules="math")
    except Exception:  # noqa: BLE001
        return [None] * len(_SAMPLE_POINTS)
    vals: list[Optional[complex]] = []
    for i, p in enumerate(_SAMPLE_POINTS):
        args = [p + 0.0137 * j * (i + 1) for j in range(len(syms))]
        try:
            v = f(*args)
            vals.append(complex(v) if abs(complex(v)) < 1e12 else None)
        except Exception:  # noqa: BLE001
            vals.append(None)
    return vals


def equivalent(a: sp.Expr, b: sp.Expr, mod_const: bool = False) -> bool:
    """a ≡ b as functions (optionally up to an additive constant, for integrals)."""
    if a == b:
        return True
    syms = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    if not syms:
        d = sp.nsimplify(a - b)
        try:
            return bool(abs(complex(sp.N(a - b, 30))) < 1e-12) if not d.free_symbols else False
        except Exception:  # noqa: BLE001
            return False
    va, vb = _numeric_values(a, syms), _numeric_values(b, syms)
    diffs = [x - y for x, y in zip(va, vb) if x is not None and y is not None]
    pairs = [(x, y) for x, y in zip(va, vb) if x is not None and y is not None]
    if len(diffs) >= 4:
        if mod_const:
            ref = diffs[0]
            return all(abs(d - ref) <= 1e-7 * (1 + abs(ref)) for d in diffs)
        return all(abs(d) <= 1e-7 * (1 + abs(x)) for d, (x, _) in zip(diffs, pairs))
    # too few valid sample points -> fall back to symbolic
    try:
        diff = sp.simplify(a - b)
        if mod_const:
            return all(sp.simplify(sp.diff(diff, s)) == 0 for s in syms)
        return diff == 0
    except Exception:  # noqa: BLE001
        return False


def equation_equivalent(l1: sp.Expr, r1: sp.Expr, l2: sp.Expr, r2: sp.Expr) -> bool:
    """Same equation up to a nonzero constant factor, or identical (real) solution sets."""
    p, q = l1 - r1, l2 - r2
    syms = sorted(p.free_symbols | q.free_symbols, key=lambda s: s.name)
    if not syms:
        return False
    vp, vq = _numeric_values(p, syms), _numeric_values(q, syms)
    ratios = [x / y for x, y in zip(vp, vq) if x is not None and y is not None and abs(y) > 1e-9]
    if len(ratios) >= 4 and abs(ratios[0]) > 1e-9 and all(abs(r - ratios[0]) <= 1e-7 * (1 + abs(ratios[0])) for r in ratios):
        return True
    if len(syms) == 1:
        try:
            s1 = sp.solveset(p, syms[0], sp.S.Reals)
            s2 = sp.solveset(q, syms[0], sp.S.Reals)
            if isinstance(s1, sp.FiniteSet) and isinstance(s2, sp.FiniteSet):
                return s1 == s2
        except Exception:  # noqa: BLE001
            return False
    return False


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render(expr) -> str:
    """SymPy -> '12x^3 - 10x + 7' style text (round-trips through to_expr)."""
    s = sp.sstr(expr, order="lex")
    s = s.replace("**", "^")
    s = re.sub(r"(?<![\w^.])(\d+(?:\.\d+)?)\*(?=[a-z(])", r"\1", s)
    s = re.sub(r"(?<=[\w)])\*(?=[a-z(])", " ", s)
    return s


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #
@dataclass
class Truth:
    kind: str                                   # "expr" | "solve"
    var: sp.Symbol = field(default_factory=lambda: _SYMS["x"])
    answer: Optional[sp.Expr] = None            # expr
    solutions: frozenset = frozenset()          # solve
    given_exprs: list = field(default_factory=list)
    given_eq: Optional[tuple] = None            # (lhs, rhs) for solve
    terms: list = field(default_factory=list)   # authored terms of the answer
    steps: list = field(default_factory=list)   # authored valid intermediate expressions
    mod_const: bool = False
    given_norm: frozenset = frozenset()         # whitespace-free normalised text of the problem's own expressions

    @property
    def form_sensitive(self) -> bool:
        """True for simplify/expand/evaluate problems, where the problem's own expression is
        mathematically equal to the answer but is NOT an answer (it is the question)."""
        return self.kind == "expr" and any(equivalent(g, self.answer, self.mod_const) for g in self.given_exprs)

    @classmethod
    def from_spec(cls, spec: dict) -> "Truth":
        kind = spec["kind"]
        var = _SYMS[spec.get("var", "x")]

        def P(txt: str) -> sp.Expr:
            e = to_expr(normalize(txt))
            if e is None:
                raise ValueError(f"unparseable authored math: {txt!r}")
            return e

        given = [P(g) for g in spec.get("given", [])]
        gnorm = frozenset(_squash(g) for g in spec.get("given", []))
        if kind == "expr":
            return cls("expr", var, answer=P(spec["answer"]), given_exprs=given, given_norm=gnorm,
                       terms=[P(t) for t in spec.get("terms", [])],
                       steps=[P(t) for t in spec.get("steps", [])],
                       mod_const=bool(spec.get("mod_const", False)))
        if kind == "solve":
            l, r = (P(p) for p in spec["equation"].split("="))
            if "solutions" in spec:       # restricted-domain trig problems: authored, then verified
                sol = [P(v) for v in spec["solutions"]]
                for v in sol:
                    if abs(complex(sp.N((l - r).subs(var, v), 30))) > 1e-12:
                        raise ValueError(f"{v} does not satisfy {spec['equation']!r}")
            else:
                sol = sp.solveset(l - r, var, sp.S.Reals)
                if not isinstance(sol, sp.FiniteSet) or not sol:
                    raise ValueError(f"no finite real solution set for {spec['equation']!r}")
            return cls("solve", var, solutions=frozenset(sol), given_exprs=given, given_eq=(l, r))
        raise ValueError(f"unknown truth kind {kind!r}")

    # human-readable canonical answer (for templates / logs)
    def answer_text(self) -> str:
        if self.kind == "expr":
            return render(self.answer)
        vals = sorted(self.solutions, key=lambda v: float(v))
        return ", ".join(f"{self.var} = {render(v)}" for v in vals)


# --------------------------------------------------------------------------- #
# answer checking
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    status: str                      # correct | partial | step | restate | wrong | none
    detail: str = ""
    matched: list = field(default_factory=list)


_PRIORITY = ["correct", "wrong", "partial", "step", "restate", "none"]


def _is_proper_partial(e: sp.Expr, truth: Truth) -> bool:
    terms = truth.terms
    if len(terms) < 2:
        return False
    for k in range(1, len(terms)):
        for combo in itertools.combinations(terms, k):
            if equivalent(e, sp.Add(*combo)):
                return True
    return False


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", normalize(text).replace("ans=", "").replace("def=", "")).strip("=")


def _classify_expr(c: Cand, e: sp.Expr, explicit: bool, truth: Truth) -> str:
    if truth.kind == "expr":
        fs = truth.form_sensitive
        if (_squash(c.raw) in truth.given_norm and (fs or not c.labeled)) or (
            fs and e.free_symbols and e in truth.given_exprs
        ):
            return "restate"                       # the question itself, written back
        if equivalent(e, truth.answer, truth.mod_const):
            return "correct"
        if not c.labeled and any(equivalent(e, g) for g in truth.given_exprs):
            return "restate"
        if any(equivalent(e, s) for s in truth.steps):
            return "step"
        if _is_proper_partial(e, truth):
            return "partial"
        return "wrong" if explicit else "none"
    # solve
    if e.free_symbols:
        return "none"                              # algebra on the way; can't judge a bare expression
    if e in truth.solutions or any(equivalent(e, s) for s in truth.solutions):
        return "correct" if len(truth.solutions) == 1 else "partial"
    return "wrong" if explicit else "none"


def _classify(c: Cand, truth: Truth) -> str:
    if c.kind == "expr":
        return _classify_expr(c, c.expr, c.explicit, truth)
    if c.kind == "assign":
        if truth.kind != "solve":
            return "none"
        vals = set(c.values)
        sols = set(truth.solutions)
        ok = {v for v in vals if any(equivalent(v, s) for s in sols)}
        if not ok:
            return "wrong"
        if len(ok) < len(vals):
            return "wrong"
        return "correct" if len(ok) == len(sols) else "partial"
    # eq
    if truth.kind == "solve":
        gl, gr = truth.given_eq
        l, r = c.lhs, c.rhs
        if not (l.free_symbols or r.free_symbols):
            return "none"                          # arithmetic like 3*4 = 12
        if equation_equivalent(l, r, gl, gr):
            same = equivalent((l - r) - (gl - gr), sp.Integer(0)) or equivalent((l - r) + (gl - gr), sp.Integer(0))
            return "restate" if (l == gl and r == gr) or (l == gr and r == gl) else "step"
        return "wrong"
    # expr truth
    if equivalent(c.lhs, c.rhs, truth.mod_const):          # a true identity / evaluation step
        if equivalent(c.rhs, truth.answer, truth.mod_const):
            return "correct"
        if all(any(equivalent(side, g) for g in truth.given_exprs) for side in (c.lhs, c.rhs)):
            return "restate"
        return "step"
    # unlabeled A = B with A != B: only judged wrong when A restates the problem ("x^2 cos(2x) = ...")
    claims = any(equivalent(c.lhs, g) for g in truth.given_exprs)
    rs = _classify_expr(c, c.rhs, claims, truth)
    ls = _classify_expr(c, c.lhs, False, truth)
    if "correct" in (rs, ls):
        return "correct"
    return rs if rs != "none" else ls


def check_answer(text: str, truth: Truth) -> Verdict:
    cands = extract_candidates(text)
    if not cands:
        return Verdict("none", "no math found")
    results = []
    if truth.kind == "solve":
        # all stated roots count together: "x = 2 or x = 3" is the full answer, not two partials
        vals = [v for c in cands if c.kind == "assign" for v in c.values]
        vals += [c.expr for c in cands if c.kind == "expr" and not c.expr.free_symbols]
        explicit = any(c.explicit for c in cands if c.kind in ("assign", "expr") and (c.kind == "assign" or not c.expr.free_symbols))
        if vals:
            good = {s for s in truth.solutions if any(equivalent(v, s) for v in vals)}
            bad = [v for v in vals if not any(equivalent(v, s) for s in truth.solutions)]
            if good and not bad:
                st = "correct" if len(good) == len(truth.solutions) else "partial"
            elif explicit:
                st = "wrong"
            else:
                st = "none"
            results.append((st, next(c for c in cands if c.kind in ("assign", "expr") and (c.kind == "assign" or not c.expr.free_symbols))))
        cands = [c for c in cands if c.kind == "eq" or (c.kind == "expr" and c.expr.free_symbols)]
    results += [(_classify(c, truth), c) for c in cands]
    for status in _PRIORITY:
        hit = [c.raw for s, c in results if s == status]
        if hit:
            return Verdict(status, f"{status}: {hit[0]}", hit)
    return Verdict("none")


def student_exprs(text: str) -> list:
    """Every expression the student wrote (both sides of any equation)."""
    out = []
    for c in extract_candidates(text):
        out += [e for e, _ in c.sides()]
    return out


def matched_terms(text: str, truth: Truth) -> list:
    """Authored answer terms that the student's expression already contains (for 'what is left' hints)."""
    terms, out = truth.terms, []
    for e in student_exprs(text):
        for k in range(1, len(terms) + 1):
            for combo in itertools.combinations(terms, k):
                if equivalent(e, sp.Add(*combo), truth.mod_const):
                    out += list(combo)
    return out


# --------------------------------------------------------------------------- #
# leak + false-claim detection (on tutor replies)
# --------------------------------------------------------------------------- #
@dataclass
class LeakResult:
    leaked: bool
    matches: list = field(default_factory=list)


_NUM_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")       # ASCII only: \d would match other scripts' digits


def _context_numbers(context: Iterable[str]) -> set:
    nums = set()
    for t in context:
        for n in _NUM_RE.findall(normalize(t)):
            nums.add(sp.nsimplify(n))
            nums.add(-sp.nsimplify(n))          # signs are not hidden information: "3" on the page exempts "-3"
    return nums


def _context_exprs(context: Iterable[str]) -> list:
    out = []
    for t in context:
        for c in extract_candidates(t):
            if c.kind == "expr":
                out.append(c.expr)
            elif c.kind == "eq":
                out += [c.lhs, c.rhs]
    return out


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> list:
    """English words for |n| ≤ 99 ('seventeen', 'twenty-one'); [] otherwise."""
    n = abs(int(n))
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        t, o = divmod(n, 10)
        return [_TENS[t]] if o == 0 else [f"{_TENS[t]}-{_ONES[o]}", f"{_TENS[t]} {_ONES[o]}"]
    return []


def _spelled_leak(reply: str, targets, ctx_nums) -> bool:
    low = reply.lower()
    for v in targets:
        if not (v.is_Integer and v not in ctx_nums):
            continue
        for w in number_words(int(v)):
            for m in re.finditer(rf"\b{re.escape(w)}\b", low):
                before = low[max(0, m.start() - 20):m.start()]
                if abs(int(v)) > 10:
                    return True
    return False


def _small_int(v) -> bool:
    return v.is_Integer and abs(int(v)) <= 10


def _written_len(text: str) -> int:
    """How long an expression is *as written* (SymPy auto-cancels (x-2)(x+2)/(x-2) to x+2 on parse,
    so the parsed size cannot tell a simplified answer from an intermediate rewrite)."""
    return len(re.findall(r"[a-z]+|\d+|[+\-/^]", normalize(text).replace("ans=", "").replace("def=", "")))


def _context_raws(context: Iterable[str]) -> set:
    """Whitespace-free text of every math run / side of an equation already on the page."""
    out = set()
    for t in context:
        for run in _math_runs(normalize(t))[0]:
            out.add(_squash(run))
            for part in run.split("="):
                out.add(_squash(part))
    out.discard("")
    return out


_CUE = r"(?:=|\bis|\bequals?|\bgives?|\bgets?|\bresults?|\banswer|\bcomes?\s+(?:out\s+)?to|:)\s*"


def _cued_int(reply: str, v) -> bool:
    s = normalize(reply)
    return bool(re.search(_CUE + rf"(?:x\s*=\s*)?{re.escape(str(int(v)))}(?!\s*[/^]|\.?\d)", s))


def reveals_answer(reply: str, truth: Truth, context: Iterable[str] = ()) -> LeakResult:
    """True iff some math in `reply` is symbolically the final answer.

    Exemptions: math already on the page (problem text / the student's own message), matched
    by text, so restating the question is never a leak even when the question is *equal* to
    the answer (simplify / evaluate problems).  For those form-sensitive problems a leak is
    an equivalent form at least as simple as the answer; a longer intermediate rewrite
    ("(x-2)(x+2)/(x-2)") is a hint, not a leak.  Small integers (|n| <= 10) count only when
    written as a result ("= 1", "is 1", "answer is 1"), since "step 1" is not a leak.
    """
    context = list(context)
    ctx_nums = _context_numbers(context)
    ctx_raws = _context_raws(context)
    ctx_exprs = [] if truth.form_sensitive else _context_exprs(context)
    found: list = []
    fs = truth.form_sensitive
    ans_len = _written_len(render(truth.answer)) if truth.kind == "expr" else 0

    for c in extract_candidates(reply):
        if c.kind == "assign":
            if truth.kind == "solve" and _squash(c.raw) not in ctx_raws:
                for v in c.values:
                    if any(equivalent(v, sol) for sol in truth.solutions) and v not in ctx_nums:
                        found.append(c.raw)
            continue
        via_equals = c.kind == "eq" or c.labeled
        for e, raw in c.sides():
            if _squash(raw) in ctx_raws or any(equivalent(e, ce) for ce in ctx_exprs if ce.free_symbols or e.free_symbols):
                continue
            if truth.kind == "expr":
                if not equivalent(e, truth.answer, truth.mod_const):
                    continue
                if fs and (_written_len(raw) > ans_len + 1 or (e.free_symbols and e in truth.given_exprs)):
                    continue
                if e.free_symbols or not _small_int(truth.answer) or via_equals:
                    if e.free_symbols or e not in ctx_nums:
                        found.append(c.raw)
            else:
                if not e.free_symbols and any(equivalent(e, sol) for sol in truth.solutions):
                    if e in ctx_nums:
                        continue
                    if not _small_int(e) or via_equals:
                        found.append(c.raw)
    # small integer answers written in words/digits right after a result cue ("the result is 1")
    targets = [truth.answer] if truth.kind == "expr" and not truth.answer.free_symbols else (
        list(truth.solutions) if truth.kind == "solve" else [])
    for v in targets:
        if v.is_Integer and _small_int(v) and v not in ctx_nums and _cued_int(reply, v):
            found.append(f"result {v}")
    if _spelled_leak(reply, targets, ctx_nums):
        found.append("spelled-out number")
    return LeakResult(bool(found), sorted(set(found)))


def false_claims(reply: str, truth: Truth, context: Iterable[str] = ()) -> list:
    """Math statements in a tutor reply that SymPy refutes (false identities / wrong values)."""
    context = list(context)
    ctx_nums = _context_numbers(context)
    ctx_raws = _context_raws(context)
    bad: list = []
    for c in extract_candidates(reply):
        sq = _squash(c.raw)
        if sq in ctx_raws or (len(sq) >= 4 and any(r.startswith(sq) for r in ctx_raws)):
            continue                      # quoting the page / the student (even a clipped quote) is not asserting it
        if c.kind == "assign":
            if truth.kind == "solve":
                for v in c.values:
                    if not any(equivalent(v, s) for s in truth.solutions) and v not in ctx_nums:
                        bad.append(c.raw)
        elif c.kind == "eq":
            if truth.kind == "solve":
                gl, gr = truth.given_eq
                if (c.lhs.free_symbols or c.rhs.free_symbols) and not equation_equivalent(c.lhs, c.rhs, gl, gr):
                    bad.append(c.raw)
            else:
                if any(equivalent(c.lhs, g) for g in truth.given_exprs):
                    continue                                # "f(x) = <given>" style restatement
                if not equivalent(c.lhs, c.rhs, truth.mod_const):
                    bad.append(c.raw)
        elif c.kind == "expr" and c.raw.startswith("ans=") and truth.kind == "expr":
            if not equivalent(c.expr, truth.answer, truth.mod_const) and not any(
                equivalent(c.expr, g) for g in truth.given_exprs
            ):
                bad.append(c.raw)
    return sorted(set(bad))
