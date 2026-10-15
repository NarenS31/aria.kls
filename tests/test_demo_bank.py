"""Step F: the featured demo bank - 6-8 problems across levels, every answer re-derived by SymPy (independently of the loader)."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations

from agent.demo_bank import DEFAULT_PATH, load_problems, rederive
from agent.demo_tutor import DemoTutor

RAW = json.loads(Path(DEFAULT_PATH).read_text(encoding="utf-8"))
FEATURED = load_problems(featured_only=True)
X = sp.Symbol("x")
TR = standard_transformations + (implicit_multiplication_application, convert_xor)
MUST = {"Differentiate f(x) = 3x^4 - 5x^2 + 7x - 9", "Differentiate f(x) = x^2 cos(2x)"}


def S(txt):
    """Independent parser for this test (not agent.math_check)."""
    return parse_expr(txt.replace("^", "**"), local_dict={"x": X, "e": sp.E, "pi": sp.pi}, transformations=TR)


def same(a, b):
    d = sp.simplify(a - b)
    return d == 0 or all(abs(complex(d.subs(X, v))) < 1e-9 for v in (0.3, 0.7, 1.1, 1.9))


class FeaturedSet(unittest.TestCase):
    def test_size_levels_topics_and_required_problems(self):
        self.assertTrue(6 <= len(FEATURED) <= 8, len(FEATURED))
        self.assertTrue(MUST <= {p.prompt for p in FEATURED})
        self.assertEqual({p.level for p in FEATURED}, {"intro", "core", "advanced"})
        self.assertEqual({p.topic for p in FEATURED}, {"algebra", "trig", "calculus"})
        self.assertGreaterEqual(sum(p.level == "advanced" for p in FEATURED), 2)
        self.assertEqual(len({p.id for p in FEATURED}), len(FEATURED))

    def test_default_load_is_unchanged_and_featured_is_a_subset(self):
        allp = load_problems()
        self.assertEqual(len(allp), 30)
        self.assertEqual([p.id for p in FEATURED], [p.id for p in allp if p.featured])

    def test_every_answer_is_rederived_by_sympy_independently(self):
        for p in FEATURED:
            spec = p.spec
            if spec["kind"] == "expr":
                src = S(spec["check"]["of"])
                ans = S(spec["answer"])
                op = spec["check"]["op"]
                if op == "diff":
                    self.assertTrue(same(sp.diff(src, X), ans), p.id)
                elif op == "integrate":
                    self.assertTrue(same(sp.diff(ans, X), src), p.id)                  # antiderivative: differentiate it back
                    self.assertTrue(spec.get("mod_const"), p.id)
                elif op == "expand":
                    self.assertTrue(same(sp.expand(src), ans), p.id)
                elif op in ("simplify", "value"):
                    self.assertTrue(same(src, ans), p.id)
                else:
                    self.fail(f"{p.id}: unknown op {op}")
            else:
                lhs, rhs = (S(side) for side in spec["equation"].split("="))
                roots = sp.solveset(lhs - rhs, X, sp.S.Reals)
                self.assertEqual({float(r) for r in roots}, {float(r) for r in p.truth.solutions}, p.id)
                for r in roots:
                    self.assertEqual(sp.simplify((lhs - rhs).subs(X, r)), 0, p.id)

    def test_prompt_shows_the_same_problem_that_was_derived(self):
        for p in FEATURED:
            if p.spec["kind"] == "expr":
                shown = p.prompt.split("=", 1)[1] if p.prompt.startswith("Differentiate") else p.prompt.split(" ", 1)[1]
                shown = shown.replace("Find the integral of", "").replace("the integral of", "").replace(" dx", "").strip()
                self.assertTrue(same(S(shown), S(p.spec["check"]["of"])), (p.id, shown))
            else:
                lhs, rhs = (S(side) for side in p.prompt.split(":", 1)[1].split("="))
                l2, r2 = (S(side) for side in p.spec["equation"].split("="))
                self.assertTrue(same(lhs - rhs, l2 - r2) or same(lhs - rhs, -(l2 - r2)), p.id)

    def test_rederive_report_agrees_with_authored(self):
        for p in FEATURED:
            r = rederive(p)
            if p.truth.kind == "solve":
                self.assertEqual([float(v) for v in r["sympy"]], [float(v) for v in r["authored"]], p.id)
            else:
                gap = sp.simplify(r["sympy"] - r["authored"])
                # integrals agree up to the constant of integration; everything else must agree exactly
                self.assertTrue(gap == 0 or (p.spec.get("mod_const") and sp.simplify(sp.diff(gap, X)) == 0), (p.id, gap))


class TyposFailLoudly(unittest.TestCase):
    def load(self, mutate):
        raw = copy.deepcopy(RAW)
        mutate({p["id"]: p for p in raw})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bank.json"
            path.write_text(json.dumps(raw))
            return load_problems(path)

    def test_wrong_authored_answer(self):
        with self.assertRaisesRegex(ValueError, "calc-01"):
            self.load(lambda b: b["calc-01"].update(answer="12x^3-10x+8"))

    def test_featured_expression_without_a_check_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "calc-02.*not re-derived"):
            self.load(lambda b: b["calc-02"].pop("check"))

    def test_check_source_must_be_the_prompt_expression(self):
        def typo(b):
            b["calc-02"]["check"]["of"] = "x^2 cos(3x)"
            b["calc-02"]["answer"] = "2x cos(3x) - 3x^2 sin(3x)"          # consistent with the typo'd check, but not with the prompt
        with self.assertRaisesRegex(ValueError, "calc-02.*not the expression shown in the prompt"):
            self.load(typo)

    def test_featured_solve_needs_a_derivation_not_authored_roots(self):
        with self.assertRaisesRegex(ValueError, "alg-05.*authored"):
            self.load(lambda b: b["alg-05"].update(solutions=["2", "3"]))

    def test_prompt_equation_must_match_the_derived_one(self):
        with self.assertRaisesRegex(ValueError, "alg-01.*prompt"):
            self.load(lambda b: b["alg-01"].update(prompt="Solve for x: 3(x - 4) = 2x + 6"))

    def test_featured_without_hint_moves_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "trig-01.*moves"):
            self.load(lambda b: b["trig-01"].update(moves=[]))

    def test_non_featured_problems_keep_their_old_leniency(self):
        self.load(lambda b: b["calc-03"].pop("check"))                          # not featured: loads as before


class LiveDemoUsesTheShortList(unittest.TestCase):
    def test_every_featured_problem_is_playable_end_to_end(self):
        t = DemoTutor(problems=FEATURED)
        for p in FEATURED:
            s = t.new_session(p.id)
            self.assertTrue(s.turn("I don't know").intervened, p.id)
            r = s.turn(p.msg_correct())
            self.assertEqual(r.status, "correct", p.id)
            self.assertTrue(s.turn(p.msg_wrong()).intervened, p.id)

    def test_ui_lists_the_featured_problems(self):
        import gradio as gr
        spec = importlib.util.spec_from_file_location("aria_demo_app_f", Path(__file__).resolve().parent.parent / "ui" / "demo_app.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        demo = mod.build_demo_ui(DemoTutor(problems=FEATURED))
        dd = [b for b in demo.blocks.values() if isinstance(b, gr.Dropdown)]
        self.assertEqual(len(dd), 1)
        self.assertEqual([c[0] if isinstance(c, (list, tuple)) else c for c in dd[0].choices], [p.prompt for p in FEATURED])


if __name__ == "__main__":
    unittest.main()
