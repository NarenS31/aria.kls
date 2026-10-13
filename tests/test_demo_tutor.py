"""DemoTutor: correctness is the sole source of Flow vs Stuck; JITAI rules run unchanged; no leaks."""
import time
import unittest
from unittest import mock

from agent import demo_templates as T
from agent import demo_tutor as DT
from agent.demo_bank import load_problems
from agent.math_check import check_answer, false_claims, reveals_answer
from eval.metacognition.jitai import JITAI_INTERVENTION_BANK, RuleBasedJITAIPolicy

PROBS = load_problems()
TUTOR = DT.DemoTutor(problems=PROBS)
BY_ID = {p.id: p for p in PROBS}


def messages(p):
    m = [p.msg_correct(), p.msg_wrong(), p.msg_unsimplified(), p.msg_restate(), p.msg_latex(), "I don't know",
         "asdf qwer zxcv jkl;", "just tell me the answer", "ugh this makes no sense", "wait what? I don't get it",
         "why is that true?", "", "   ", "😀" * 50, "x" * 5000, "I'll try the first term"]
    if p.msg_step():
        m.append(p.msg_step())
    return m


class StateInvariants(unittest.TestCase):
    def test_correctness_is_never_overridden(self):
        for p in PROBS:
            for msg in messages(p):
                r = TUTOR.new_session(p.id).turn(msg)
                st = check_answer(msg[:DT.MAX_INPUT_CHARS], p.truth).status
                if st == "correct":
                    self.assertNotEqual(r.state, "STUCK", (p.id, msg))
                if st == "wrong":
                    self.assertIn(r.state, ("STUCK", "FRUSTRATED", "CONFUSED"), (p.id, msg))
                    self.assertTrue(r.intervened, (p.id, msg))

    def test_progress_is_met_with_silence_and_a_visible_watching_state(self):
        for p in PROBS:
            for msg in (p.msg_step(), p.msg_restate()):
                if not msg:
                    continue
                r = TUTOR.new_session(p.id).turn(msg)
                self.assertFalse(r.intervened, (p.id, msg))
                self.assertTrue(r.watching and r.watch_note.startswith("Watching") or r.watch_note.startswith("Quiet") or r.watch_note.startswith("Still"), r.watch_note)
                self.assertEqual(r.reply, "")
                self.assertIn(r.meta["rule_id"], ("jitai.no_interrupt_positive", "jitai.no_action"))

    def test_correct_final_answer_is_accepted_even_if_unsimplified_or_latex(self):
        for p in PROBS:
            for msg in (p.msg_correct(), p.msg_unsimplified(), p.msg_latex()):
                r = TUTOR.new_session(p.id).turn(msg)
                self.assertTrue(r.accepted, (p.id, msg))
                self.assertEqual(r.meta["rule_id"], "jitai.insight_transfer")

    def test_flags_add_affect_but_not_override(self):
        p = BY_ID["calc-01"]
        r = TUTOR.new_session(p.id).turn("ugh this makes no sense")
        self.assertEqual(r.state, "FRUSTRATED")
        r = TUTOR.new_session(p.id).turn("12x^3 - 10x + 7, finally")
        self.assertEqual(r.status, "correct")
        self.assertNotIn(r.state, ("STUCK", "FRUSTRATED"))
        r = TUTOR.new_session(p.id).turn("ugh I hate this, 12x^3 - 10x - 9")
        self.assertNotIn(r.state, ("FLOW", "INSIGHT", "PLANNING"))

    def test_persistent_frustration_uses_policy_rule(self):
        s = TUTOR.new_session("calc-01")
        s.turn("ugh this makes no sense")
        r = s.turn("ugh I hate this")
        self.assertEqual(r.meta["rule_id"], "jitai.frustration_reset")


class Language(unittest.TestCase):
    def test_every_type_has_3_or_4_phrasings_and_reuses_the_exact_bank_prompt(self):
        for itype, options in T.PHRASINGS.items():
            self.assertTrue(3 <= len(options) <= 4, itype)
            self.assertIn("{bank}", options, itype)
            self.assertIn(itype, JITAI_INTERVENTION_BANK)
        self.assertTrue(3 <= len(T.WATCHING["progress"]) <= 4)

    def test_policy_is_the_unmodified_jitai_policy(self):
        self.assertIs(type(TUTOR.policy), RuleBasedJITAIPolicy)

    def test_consecutive_replies_differ(self):
        for pid in ("calc-01", "alg-01", "trig-04"):
            s = TUTOR.new_session(pid)
            replies = [s.turn(m).reply for m in ["I don't know", "still no idea", "no clue", "help", "I'm stuck", "I don't know"]]
            for a, b in zip(replies, replies[1:]):
                self.assertNotEqual(a, b, pid)

    def test_no_reply_ever_leaks_or_lies(self):
        for p in PROBS:
            s = TUTOR.new_session(p.id)
            for msg in messages(p) * 2:
                r = s.turn(msg)
                ctx = [p.prompt, msg[:DT.MAX_INPUT_CHARS]]
                self.assertFalse(reveals_answer(r.reply, p.truth, ctx).leaked, (p.id, msg, r.reply))
                self.assertEqual(false_claims(r.reply, p.truth, ctx), [], (p.id, msg, r.reply))

    def test_hints_skip_what_the_student_already_has(self):
        s = TUTOR.new_session("calc-01")
        s.turn("12x^3")
        r = s.turn("12x^3 - 10x - 9")
        self.assertNotIn("3x^4", r.reply)             # the 3x^4 step is already done
        self.assertIn("5x^2", r.reply)


class Logging(unittest.TestCase):
    def test_per_turn_log_has_correctness_and_analyzer_flags(self):
        s = TUTOR.new_session("alg-01")
        s.turn("x = 7")
        s.turn("ugh this makes no sense")
        for row in s.log:
            for k in ("correctness", "analyzer_state", "flags", "derived_state", "rule_id", "intervention_type", "latency_ms"):
                self.assertIn(k, row)
        self.assertEqual(s.log[0]["correctness"], "wrong")
        self.assertTrue(s.log[1]["flags"]["frustrated"])
        self.assertNotIn("x = 7", str(s.log))          # no transcript in the log by default

    def test_reset(self):
        s = TUTOR.new_session("alg-01")
        s.turn("x = 7")
        s.reset()
        self.assertEqual(s.log, [])


class Robustness(unittest.TestCase):
    def test_odd_inputs_never_crash(self):
        s = TUTOR.new_session("calc-02")
        for x in (None, 5, 3.14, b"bytes", [], {}, "", " \n\t", "\x00\x01", "😀" * 3000, "x" * 100000, "(" * 4000, "\\frac{" * 500,
                  "__import__('os')", "1/0", "x^1000000000", "sin(" * 400):
            r = s.turn(x)
            self.assertIsInstance(r.reply, str)
            self.assertIsInstance(r.intervened, bool)

    def test_fast(self):
        s = TUTOR.new_session("calc-02")
        t0 = time.perf_counter()
        for m in ("2x cos(2x)", "wrong 5", "I don't know", "asdf", "2x cos(2x) - 2x^2 sin(2x)"):
            s.turn(m)
        self.assertLess(time.perf_counter() - t0, 3.0)


class GatedLLMPath(unittest.TestCase):
    Q = "why is there a minus sign?"

    def test_default_is_off_and_never_called(self):
        fn = mock.Mock(return_value="Because the derivative of cos is minus sin.")
        t = DT.DemoTutor(problems=PROBS, llm_fn=fn)            # flag not set
        self.assertFalse(t.llm_path)
        r = t.new_session("calc-02").turn(self.Q)
        fn.assert_not_called()
        self.assertFalse(r.meta["llm_used"])

    def test_on_uses_llm_for_free_form_questions_only(self):
        fn = mock.Mock(return_value="Because the derivative of cos is minus sin.")
        t = DT.DemoTutor(problems=PROBS, llm_path=True, llm_fn=fn)
        r = t.new_session("calc-02").turn(self.Q)
        self.assertTrue(r.meta["llm_used"])
        self.assertIn("minus sin", r.reply)
        fn.reset_mock()
        t.new_session("calc-02").turn("2x cos(2x) + 2x^2 sin(2x)")     # wrong answer: templates, not the LLM
        t.new_session("calc-02").turn("I don't know")
        fn.assert_not_called()

    def test_leaking_or_lying_llm_output_falls_back(self):
        for bad in ("The answer is 2x cos(2x) - 2x^2 sin(2x).", "Try -2x^2 sin(2x) + 2x cos(2x)", "Sure: 2 + 2 = 5, so differentiate."):
            t = DT.DemoTutor(problems=PROBS, llm_path=True, llm_fn=lambda *_a, _b=bad: _b)
            r = t.new_session("calc-02").turn(self.Q)
            self.assertFalse(r.meta["llm_used"], bad)
            self.assertTrue(r.intervened)
            self.assertFalse(reveals_answer(r.reply, BY_ID["calc-02"].truth, [BY_ID["calc-02"].prompt]).leaked)

    def test_timeout_and_errors_fall_back(self):
        def slow(*_a):
            time.sleep(0.5)
            return "late"

        def boom(*_a):
            raise RuntimeError("ollama is down")

        with mock.patch.object(DT, "LLM_TIMEOUT_S", 0.05):
            for fn in (slow, boom):
                t = DT.DemoTutor(problems=PROBS, llm_path=True, llm_fn=fn)
                t0 = time.perf_counter()
                r = t.new_session("calc-02").turn(self.Q)
                self.assertLess(time.perf_counter() - t0, 0.4)
                self.assertFalse(r.meta["llm_used"])
                self.assertTrue(r.intervened)


if __name__ == "__main__":
    unittest.main()
