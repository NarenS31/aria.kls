"""Unit tests for the SymPy ground-truth layer (agent/math_check.py, agent/demo_bank.py)."""
import time
import unittest

from agent.demo_bank import load_problems
from agent.math_check import Truth, check_answer, false_claims, reveals_answer, to_expr, render, normalize

DERIV = Truth.from_spec({"kind": "expr", "answer": "12x^3-10x+7", "given": ["3x^4-5x^2+7x-9"],
                         "terms": ["12x^3", "-10x", "7"]})
PRODUCT = Truth.from_spec({"kind": "expr", "answer": "2x cos(2x) - 2x^2 sin(2x)", "given": ["x^2 cos(2x)"],
                           "terms": ["2x cos(2x)", "-2x^2 sin(2x)"]})
LINEAR = Truth.from_spec({"kind": "solve", "equation": "3(x-4) = 2x+5"})
QUAD = Truth.from_spec({"kind": "solve", "equation": "x^2-5x+6 = 0"})
INTEG = Truth.from_spec({"kind": "expr", "answer": "2x^3-2x^2+x", "given": ["6x^2-4x+1"], "mod_const": True,
                         "terms": ["2x^3", "-2x^2", "x"]})
SIMPLIFY = Truth.from_spec({"kind": "expr", "answer": "x+2", "given": ["(x^2-4)/(x-2)"]})


def status(msg, truth):
    return check_answer(msg, truth).status


class CorrectnessTests(unittest.TestCase):
    def test_equivalent_forms_are_correct(self):
        for msg in ["12x^3 - 10x + 7", "7 - 10x + 12x³", "f'(x) = -10x + 12x^3 + 7", "12*x**3 - 10*x + 7",
                    "12x^3-10x+7+0", "the answer is 12x^3 - 10x + 7", r"12x^{3}-10x+7",
                    "I think it's 12x^3 - 10x + 7 maybe", "3*4x^3 - 10x + 7"]:
            self.assertEqual(status(msg, DERIV), "correct", msg)

    def test_wrong_answers(self):
        for msg in ["12x^3 - 10x - 9", "6x + 5", "f'(x) = 12x^4 - 10x^2 + 7", "my answer is 12x^3 - 10x + 8"]:
            self.assertEqual(status(msg, DERIV), "wrong", msg)

    def test_partial_progress_is_not_wrong(self):
        for msg in ["12x^3", "12x^3 - 10x", "the derivative of 3x^4 is 12x^3"]:
            self.assertEqual(status(msg, DERIV), "partial", msg)

    def test_restatement_is_neutral(self):
        for msg in ["f(x) = 3x^4 - 5x^2 + 7x - 9", "3x^4-5x^2+7x-9", "I need to differentiate 3x^4 - 5x^2 + 7x - 9"]:
            self.assertEqual(status(msg, DERIV), "restate", msg)

    def test_prose_is_never_wrong(self):
        for msg in ["I don't know", "asdf qwer zxcv jkl;", "just tell me the answer", "I will use the power rule on each term",
                    "what's the weather like in Charlotte today?", "", "   \n\t ", "ugh this makes no sense"]:
            self.assertIn(status(msg, DERIV), ("none", "restate"), msg)

    def test_product_rule(self):
        self.assertEqual(status("2x cos(2x) - 2x^2 sin(2x)", PRODUCT), "correct")
        self.assertEqual(status("-2x^2 sin(2x) + 2x cos(2x)", PRODUCT), "correct")
        self.assertEqual(status("2x cos(2x) + x^2 (-2 sin(2x))", PRODUCT), "correct")
        self.assertEqual(status("2x cos(2x) + 2x^2 sin(2x)", PRODUCT), "wrong")
        self.assertEqual(status("2x cos(2x)", PRODUCT), "partial")
        self.assertEqual(status("-2x^2sin(2x)", PRODUCT), "partial")

    def test_linear_equation(self):
        for msg in ["x = 17", "x=17", "so x equals 17", "x = 34/2", "17", "x = 17.0", "The answer is 17"]:
            self.assertEqual(status(msg, LINEAR), "correct", msg)
        for msg in ["x = 170", "x = -17", "x = 17.5", "x = 7"]:
            self.assertEqual(status(msg, LINEAR), "wrong", msg)
        self.assertEqual(status("3x - 12 = 2x + 5", LINEAR), "step")
        self.assertEqual(status("3(x - 4) = 2x + 5", LINEAR), "restate")
        self.assertEqual(status("x - 12 = 5", LINEAR), "step")
        self.assertEqual(status("3x - 12 = 2x - 5", LINEAR), "wrong")
        self.assertEqual(status("3(x-4) = 2x+5", LINEAR), "restate")

    def test_multiple_roots(self):
        self.assertEqual(status("x = 2 or x = 3", QUAD), "correct")
        self.assertEqual(status("x = 3, x = 2", QUAD), "correct")
        self.assertEqual(status("x = 2", QUAD), "partial")
        self.assertEqual(status("x = 2 and x = 4", QUAD), "wrong")
        self.assertEqual(status("(x-2)(x-3) = 0", QUAD), "step")

    def test_integral_up_to_constant(self):
        self.assertEqual(status("x - 2x^2 + 2x^3 + C", INTEG), "correct")
        self.assertEqual(status("2x^3 - 2x^2 + x + 5", INTEG), "correct")
        self.assertEqual(status("6x^3 - 4x^2 + x", INTEG), "wrong")

    def test_form_sensitive_problem_restatement_is_not_the_answer(self):
        self.assertEqual(status("(x^2-4)/(x-2)", SIMPLIFY), "restate")
        self.assertEqual(status("x + 2", SIMPLIFY), "correct")
        self.assertEqual(status("(x-2)(x+2)/(x-2)", SIMPLIFY), "correct")
        self.assertEqual(status("x - 2", SIMPLIFY), "wrong")

    def test_latex_and_unicode(self):
        self.assertEqual(status(r"f'(x)=12x^{3}-10x+7", DERIV), "correct")
        self.assertEqual(status(r"\frac{d}{dx}\left(3x^{4}-5x^{2}\right)=12x^{3}-10x", DERIV), "partial")
        self.assertEqual(status("f′(x) = 12x³ − 10x + 7", DERIV), "correct")


class SafetyTests(unittest.TestCase):
    def test_no_code_execution_or_blowup(self):
        t0 = time.perf_counter()
        for evil in ["__import__('os').system('echo hi')", "x^1000000000", "9^9^9^9", "(" * 5000, "x" * 100000,
                     "open('/etc/passwd')", "x.__class__", "lambda: 1", "1/0", "0^-1", "sin(" * 300 + "x" + ")" * 300,
                     "😀😀😀", "\x00\x01", "x = " + "9" * 5000]:
            self.assertIn(status(evil, DERIV), ("none", "wrong", "restate"), evil[:30])
            self.assertIn(status(evil, LINEAR), ("none", "wrong", "restate"), evil[:30])
            reveals_answer(evil, DERIV)
        self.assertLess(time.perf_counter() - t0, 8.0)

    def test_to_expr_guards(self):
        self.assertIsNone(to_expr("x**1000"))
        self.assertIsNone(to_expr("import os"))
        self.assertIsNone(to_expr("the answer"))
        self.assertEqual(to_expr("2x"), to_expr("2*x"))


class LeakTests(unittest.TestCase):
    CTX = ["Differentiate f(x) = 3x^4 - 5x^2 + 7x - 9"]

    def test_leaks_in_every_notation(self):
        for reply in ["The derivative is 12x^3 - 10x + 7.", "Try 7 - 10x + 12x³", "Answer: 12*x**3 - 10*x + 7",
                      r"So $12x^{3} - 10x + 7$", "f'(x) = -10x + 12x^3 + 7", "it is 12x^3-10x+7+0"]:
            self.assertTrue(reveals_answer(reply, DERIV, self.CTX).leaked, reply)

    def test_hints_do_not_leak(self):
        for reply in ["Apply the power rule to 3x^4. What do you get?", "The derivative of 3x^4 is 12x^3.",
                      "What is one small clue in the problem you can use first?", "Step 1: look at the first term.",
                      "You are working on: Differentiate f(x) = 3x^4 - 5x^2 + 7x - 9"]:
            self.assertFalse(reveals_answer(reply, DERIV, self.CTX).leaked, reply)

    def test_solve_leaks(self):
        ctx = ["Solve for x: 3(x - 4) = 2x + 5"]
        for reply in ["So x = 17", "x equals seventeen", "x = 34/2", "the answer is 17", "x=17."]:
            self.assertTrue(reveals_answer(reply, LINEAR, ctx).leaked, reply)
        for reply in ["Try x = 5", "Subtract 2x from both sides: 3x - 12 - 2x = 5", "Distribute the 3 first.", "x - 12 = 5"]:
            self.assertFalse(reveals_answer(reply, LINEAR, ctx).leaked, reply)

    def test_student_own_answer_echo_is_exempt(self):
        self.assertFalse(reveals_answer("Yes, 12x^3 - 10x + 7 works.", DERIV, ["12x^3 - 10x + 7"]).leaked)

    def test_form_sensitive_leak_rule(self):
        ctx = ["Simplify (x^2 - 4)/(x - 2)"]
        self.assertTrue(reveals_answer("It simplifies to x + 2", SIMPLIFY, ctx).leaked)
        self.assertFalse(reveals_answer("Factor the top: (x - 2)(x + 2)/(x - 2)", SIMPLIFY, ctx).leaked)
        self.assertFalse(reveals_answer("You are working on (x^2 - 4)/(x - 2).", SIMPLIFY, ctx).leaked)


class FalseClaimTests(unittest.TestCase):
    def test_detects_false_identities(self):
        self.assertTrue(false_claims("2 + 2 = 5", DERIV))
        self.assertTrue(false_claims("3x - 12 = 2x - 5 so x = 18", LINEAR))
        self.assertTrue(false_claims("f'(x) = 6x + 5", DERIV))

    def test_true_statements_pass(self):
        self.assertEqual(false_claims("2 + 2 = 4", DERIV), [])
        self.assertEqual(false_claims("Subtract 2x: x - 12 = 5", LINEAR), [])
        self.assertEqual(false_claims("3x - 12 = 2x + 5", LINEAR), [])
        self.assertEqual(false_claims("f(x) = 3x^4 - 5x^2 + 7x - 9", DERIV), [])


class BankTests(unittest.TestCase):
    def test_bank_loads_and_self_verifies(self):
        probs = load_problems()          # raises if any authored answer disagrees with SymPy
        self.assertEqual(len(probs), 30)
        self.assertEqual({p.topic for p in probs}, {"algebra", "trig", "calculus"})
        ids = {p.id for p in probs}
        self.assertEqual(len(ids), 30)

    def test_render_roundtrip(self):
        for p in load_problems():
            if p.truth.kind == "expr":
                self.assertIsNotNone(to_expr(normalize(render(p.truth.answer))), p.id)
                self.assertEqual(check_answer(render(p.truth.answer), p.truth).status, "correct", p.id)


if __name__ == "__main__":
    unittest.main()
