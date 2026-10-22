"""Step D: input hardening, bounded latency, warm LLM client with graceful fallback, per-turn log with no student text."""
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from agent import demo_log
from agent import demo_tutor as DT
from agent.demo_bank import load_problems
from agent.demo_input import MAX_INPUT_CHARS, sanitize_input
from agent.math_check import reveals_answer

PROBS = load_problems()
BY_ID = {p.id: p for p in PROBS}
Q = "why is there a minus sign?"


class _Unprintable:
    def __str__(self):
        raise RuntimeError("no str")


def hostile():
    nest = lambda f, n, c="x": __import__("functools").reduce(lambda s, _: f.format(s), range(n), c)  # noqa: E731
    return {
        "none": None, "int": 42, "nan": float("nan"), "bytes": b"x^2 \xff\xfe", "dict": {"a": 1}, "list": ["x", 2],
        "unprintable": _Unprintable(), "empty": "", "spaces": " \t\n ", "nul": "x\x00^2\x00", "ctrl": "".join(map(chr, range(32))),
        "surrogate": "x^2 \ud800 + 1", "rtl": "‮2^x‬", "zerowidth": "x​^​2", "combining": "x" + "́" * 900,
        "emoji": "😀" * 400 + "x^2", "fullwidth": "ｘ＾２＋３", "arabic_digits": "٣x²", "devanagari": "x^३", "huge": "x" * 1_000_000,
        "huge_digits": "9" * 5000, "parens": "(" * 5000, "tower": "9^9^9^9^9", "big_pow": "x^99999999", "sin_nest": "sin(" * 300 + "x" + ")" * 300,
        "div0": "1/0", "zero_zero": "0^0", "log0": "log(0)", "sqrt_neg": "sqrt(-1)", "eq_flood": "x=" * 500, "nl_flood": "x\n" * 600,
        "frac_bomb": "\\frac{" * 400, "label_bomb": "answer is " * 100, "mixed": "我不知道 x^2 مرحبا", "html": "<script>alert(1)</script>",
        "format": "{do} {bank} %s %(x)s {0}", "latex_dollars": "$" * 900, "frac_nest": nest("1/(1+{})", 60),
    }


class Sanitizer(unittest.TestCase):
    def test_never_raises_and_always_returns_short_text(self):
        for name, v in hostile().items():
            text, flags = sanitize_input(v)
            self.assertIsInstance(text, str, name)
            self.assertLessEqual(len(text), MAX_INPUT_CHARS, name)
            self.assertTrue(all(isinstance(f, str) and f.replace("_", "").isalpha() for f in flags), (name, flags))

    def test_flags_and_normalisation(self):
        self.assertEqual(sanitize_input(None), ("", ["empty", "non_text"]))
        self.assertIn("non_text", sanitize_input(5)[1])
        self.assertIn("bytes", sanitize_input(b"x")[1])
        self.assertIn("unprintable", sanitize_input(_Unprintable())[1])
        self.assertEqual(sanitize_input("٣x²")[0], "3x²")                      # digits -> ASCII, superscript kept
        self.assertEqual(sanitize_input("ｘ＾２")[0], "x^2")
        self.assertEqual(sanitize_input("a\x00b​c‮d")[0], "a bcd")
        self.assertEqual(sanitize_input("  \t\n ")[0], "")
        t, f = sanitize_input("q" * 5000)
        self.assertEqual(len(t), MAX_INPUT_CHARS)
        self.assertIn("truncated", f)

    def test_ordinary_math_is_untouched(self):
        for s in ("12x^3 - 10x + 7", "x = 3 or x = -2", "f'(x) = 2x cos(2x) - 2x^2 sin(2x)", "x² − 5", "answer is π/2"):
            self.assertEqual(sanitize_input(s)[0], s)


class TurnsNeverCrash(unittest.TestCase):
    tutor = DT.DemoTutor(problems=PROBS)

    def test_hostile_battery_on_every_level(self):
        for pid in ("alg-01", "calc-02", "trig-04"):
            for name, v in hostile().items():
                t0 = time.perf_counter()
                r = self.tutor.new_session(pid).turn(v)
                self.assertIsInstance(r.reply, str, (pid, name))
                self.assertIsInstance(r.intervened, bool, (pid, name))
                self.assertLess(time.perf_counter() - t0, DT.MATH_BUDGET_S + 1.5, (pid, name))
                self.assertFalse(reveals_answer(r.reply, BY_ID[pid].truth, [BY_ID[pid].prompt]).leaked, (pid, name))

    def test_unreadable_turn_is_quiet_and_flagged_not_a_verdict(self):
        r = self.tutor.new_session("alg-01").turn("٣x² ‮")
        self.assertIn("unicode_digits", r.meta["input_flags"])
        with mock.patch.object(DT.DemoSession, "_analyze", side_effect=RuntimeError("boom")):
            r = self.tutor.new_session("alg-01").turn("x = 3")
        self.assertFalse(r.intervened)
        self.assertEqual(r.state, "UNKNOWN")
        self.assertTrue(r.watch_note.startswith("Watching"))
        self.assertEqual(r.meta["fallback"], "error:RuntimeError")
        self.assertEqual(r.meta["rule_id"], "system.fallback")

    def test_over_budget_symbolic_check_falls_back_and_session_recovers(self):
        heavy = " ; ".join("(x+%d)^8=x^8+%d" % (i, i) for i in range(30))
        t = DT.DemoTutor(problems=PROBS)
        with mock.patch.object(DT, "MATH_BUDGET_S", 0.15), mock.patch.object(DT, "MAX_STUCK_MATH", 1):
            s = t.new_session("alg-01")
            t0 = time.perf_counter()
            r = s.turn(heavy)
            self.assertLess(time.perf_counter() - t0, 1.0)
            self.assertEqual(r.meta["fallback"], "math_timeout")
            self.assertFalse(r.intervened)
            t1 = time.perf_counter()                      # one abandoned check is still running: refuse, do not queue behind it
            r2 = s.turn(heavy + " ")
            self.assertLess(time.perf_counter() - t1, 0.5)
            self.assertEqual(r2.meta["fallback"], "math_busy")
        deadline = time.time() + 20
        while t._math_stuck and time.time() < deadline:
            time.sleep(0.1)
        self.assertEqual(t._math_stuck, 0)
        r3 = s.turn("x = 17")                              # normal service resumes
        self.assertEqual(r3.status, "correct")


class Reset(unittest.TestCase):
    def test_one_click_reset_clears_the_attempt(self):
        t = DT.DemoTutor(problems=PROBS)
        s = t.new_session("calc-01")
        s.turn("I don't know"); s.turn("12x^3"); s.turn("still stuck")
        self.assertGreater(s.hint_level, 0)
        a0 = s.attempt
        s.reset()
        self.assertEqual((s.log, s.hint_level, s.achieved, s.prev_state, s.consecutive), ([], 0, [], None, 0))
        self.assertEqual(s.attempt, a0 + 1)
        self.assertEqual(s.turn("I don't know").meta["turn"], 1)

    def test_ui_reset_handler(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("aria_demo_app", Path(__file__).resolve().parent.parent / "ui" / "demo_app.py")
        mod = importlib.util.module_from_spec(spec)           # by path: importing `ui` can resolve to eval/ui once eval.* is imported
        spec.loader.exec_module(mod)
        make_handlers = mod.make_handlers
        t = DT.DemoTutor(problems=PROBS)
        start, send, reset = make_handlers(t, list(t.problems.values()))
        session, hist, *_ = start(BY_ID["alg-01"].prompt)
        hist, session, chat, badge, _ = send("I don't know", hist, session)
        self.assertEqual(len(hist), 2)
        hist, session, chat, badge, _ = reset(session)
        self.assertEqual(hist, [])
        self.assertIn("READY", badge)
        for weird in (None, "", "\x00" * 10, "٣x²", "😀" * 2000):
            hist, session, chat, badge, _ = send(weird, hist, session)
            self.assertIsInstance(chat, str)


class WarmLLMClient(unittest.TestCase):
    def fake_ollama(self, chat=None, generate=None):
        mod = types.ModuleType("ollama")
        mod.created = []

        class Client:
            def __init__(self, timeout=None):
                self.timeout = timeout
                mod.created.append(timeout)
                self.chat_calls, self.gen_calls = [], []

            def chat(self, **kw):
                self.chat_calls.append(kw)
                mod.last = self
                return chat(**kw) if chat else {"message": {"content": "Because the derivative of cos is minus sin."}}

            def generate(self, **kw):
                self.gen_calls.append(kw)
                mod.last_gen = self
                if generate:
                    return generate(**kw)
                return {}
        mod.Client = Client
        return mod

    def test_model_loaded_once_client_reused_generation_capped(self):
        mod = self.fake_ollama()
        with mock.patch.dict(sys.modules, {"ollama": mod}):
            t = DT.DemoTutor(problems=PROBS, llm_path=True)
            self.assertTrue(t.warm_up())
            self.assertEqual(mod.last_gen.gen_calls[0]["keep_alive"], DT.LLM_KEEP_ALIVE)
            self.assertEqual(mod.created, [DT.LLM_WARMUP_TIMEOUT_S, DT.LLM_TIMEOUT_S])
            for _ in range(3):
                r = t.new_session("calc-02").turn(Q)
                self.assertTrue(r.meta["llm_used"])
                self.assertEqual(r.meta["llm_outcome"], "ok")
            self.assertEqual(len(mod.created), 2)                       # no new client per turn
            c = mod.last.chat_calls[-1]
            self.assertEqual(c["options"]["num_predict"], 80)
            self.assertEqual(c["keep_alive"], DT.LLM_KEEP_ALIVE)
            self.assertEqual(DT.LLM_TIMEOUT_S, 10.0)

    def test_unreachable_ollama_means_templates_without_stalling(self):
        mod = self.fake_ollama(generate=lambda **kw: (_ for _ in ()).throw(ConnectionError("down")))
        with mock.patch.dict(sys.modules, {"ollama": mod}):
            t = DT.DemoTutor(problems=PROBS, llm_path=True)
            self.assertFalse(t.warm_up())
            t0 = time.perf_counter()
            r = t.new_session("calc-02").turn(Q)
            self.assertLess(time.perf_counter() - t0, 0.5)
            self.assertFalse(r.meta["llm_used"])
            self.assertEqual(r.meta["llm_outcome"], "cooldown")
            self.assertTrue(r.intervened and r.reply)

    def test_timeout_opens_the_breaker_then_busy_then_recovers(self):
        calls = []

        def slow(*_a):
            calls.append(1)
            time.sleep(0.4)
            return "Because the derivative of cos is minus sin."
        with mock.patch.object(DT, "LLM_TIMEOUT_S", 0.05), mock.patch.object(DT, "LLM_COOLDOWN_S", 0.0):
            t = DT.DemoTutor(problems=PROBS, llm_path=True, llm_fn=slow)
            r1 = t.new_session("calc-02").turn(Q)
            self.assertEqual(r1.meta["llm_outcome"], "timeout")
            t0 = time.perf_counter()
            r2 = t.new_session("calc-02").turn(Q)                       # first call still running: do not queue behind it
            self.assertLess(time.perf_counter() - t0, 0.2)
            self.assertEqual(r2.meta["llm_outcome"], "busy")
            self.assertEqual(len(calls), 1)
            time.sleep(0.5)
            with mock.patch.object(DT, "LLM_TIMEOUT_S", 2.0):
                r3 = t.new_session("calc-02").turn(Q)
            self.assertTrue(r3.meta["llm_used"])
        with mock.patch.object(DT, "LLM_TIMEOUT_S", 0.05), mock.patch.object(DT, "LLM_COOLDOWN_S", 30.0):
            t = DT.DemoTutor(problems=PROBS, llm_path=True, llm_fn=slow)
            t.new_session("calc-02").turn(Q)
            n = len(calls)
            r = t.new_session("calc-02").turn(Q)
            self.assertEqual(r.meta["llm_outcome"], "cooldown")
            self.assertEqual(len(calls), n)                              # cooldown: the model is not called at all

    def test_output_cut_by_the_token_cap_ends_on_a_full_sentence(self):
        self.assertEqual(DT._trim_to_sentence("It maps angles to points. It also gives sine and cos"), "It maps angles to points.")
        self.assertEqual(DT._trim_to_sentence("Already complete."), "Already complete.")
        self.assertEqual(DT._trim_to_sentence("no punctuation at all here"), "no punctuation at all here")
        t = DT.DemoTutor(problems=PROBS, llm_path=True, llm_fn=lambda *_: "Because the derivative of cos is minus sin. And then the chain ru")
        self.assertEqual(t.llm_answer(BY_ID["calc-02"], Q), "Because the derivative of cos is minus sin.")

    def test_rejected_output_does_not_open_the_breaker(self):
        t = DT.DemoTutor(problems=PROBS, llm_path=True, llm_fn=lambda *_: "The answer is 2x cos(2x) - 2x^2 sin(2x).")
        r = t.new_session("calc-02").turn(Q)
        self.assertEqual(r.meta["llm_outcome"], "rejected")
        self.assertFalse(r.meta["llm_used"])
        self.assertEqual(t.llm_answer(BY_ID["calc-02"], Q), None)
        self.assertEqual(t.last_llm_outcome, "rejected")


class PerTurnLog(unittest.TestCase):
    def test_log_has_one_line_per_turn_and_no_student_or_tutor_text(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "logs" / "turns.jsonl"
            t = DT.DemoTutor(problems=PROBS, log_path=path)
            s = t.new_session("calc-01")
            secrets = ["zebra-quokka-7 is my favourite", "my name is Priya Raman", "12x^3 - 10x + 7", "ugh this makes no sense", "٣x² ‮"]
            replies = []
            for m in secrets:
                replies.append(s.turn(m).reply)
            s.reset()
            replies.append(s.turn("I don't know").reply)
            text = path.read_text()
            rows = [json.loads(line) for line in text.splitlines()]
            self.assertEqual(len(rows), len(secrets) + 1)
            for secret in secrets + [r for r in replies if r]:
                self.assertNotIn(secret, text)
            for tok in ("zebra", "quokka", "Priya", "Raman", "12x^3"):
                self.assertNotIn(tok, text)
            self.assertEqual(len({r["session_id"] for r in rows}), 1)
            self.assertEqual([r["attempt"] for r in rows], [1] * len(secrets) + [2])
            self.assertEqual(rows[-1]["turn"], 1)
            for r in rows:
                self.assertLessEqual(set(r) - {"ts"}, set(demo_log.FIELDS))
                for k in ("student_chars", "reply_chars", "latency_ms", "correctness", "derived_state", "rule_id", "problem_id"):
                    self.assertIn(k, r)
            self.assertEqual(rows[0]["student_chars"], len(secrets[0]))
            self.assertIn("unicode_digits", rows[4]["input_flags"])

    def test_free_text_cannot_reach_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            lg = demo_log.TurnLog(Path(d) / "t.jsonl")
            lg.write({"template": "my name is Priya", "problem_id": "alg-01", "rule_id": "jitai.no_action", "reply": "hello", "text": "secret",
                      "input_flags": ["truncated", "has space"]})
            row = json.loads((Path(d) / "t.jsonl").read_text())
            self.assertEqual(row["template"], "invalid")
            self.assertEqual(row["problem_id"], "alg-01")
            self.assertEqual(row["input_flags"], ["truncated"])
            self.assertNotIn("reply", row)
            self.assertNotIn("text", row)

    def test_logging_failure_never_costs_the_student_a_reply(self):
        with tempfile.TemporaryDirectory() as d:
            t = DT.DemoTutor(problems=PROBS, log_path=Path(d))          # a directory, so every write fails
            r = t.new_session("alg-01").turn("I don't know")
            self.assertTrue(r.intervened and r.reply)
            self.assertGreater(t.turn_log.errors, 0)

    def test_default_is_no_file_log(self):
        self.assertIsNone(DT.DemoTutor(problems=PROBS).turn_log.path)


if __name__ == "__main__":
    unittest.main()
