"""Pre-freeze fix 1a: affect flags need an explicit feeling word in the student's own words.

Before the fix a words-only restatement ("so I need sin of pi over 6 plus cos of pi over 3") was flagged CONFUSED
(prose_of() strips the math and the analyzer's repeated-phrase rule fires on "sin of over plus cos of over"), and neutral
"why do / why does" questions were flagged FRUSTRATED by the analyzer's marker at eval/metacognition/analyzer.py:263.
"""
import unittest

from agent import demo_tutor as DT
from agent.demo_bank import load_problems

PROBS = load_problems()
TUTOR = DT.DemoTutor(problems=PROBS)
AFFECT = ("FRUSTRATED", "CONFUSED")


def turn(pid, text):
    return TUTOR.new_session(pid).turn(text)


class RestatementsAreNeverAffect(unittest.TestCase):
    CASES = [
        ("trig-01", "so I need sin of pi over 6 plus cos of pi over 3"),
        ("trig-01", "I have to find sin of pi over 6 plus cos of pi over 3"),
        ("alg-01", "the problem says three times x minus four equals two x plus five"),
        ("calc-01", "I need to differentiate three x to the fourth minus five x squared plus seven x minus nine"),
        ("calc-02", "so it is x squared times cosine of two x and I take the derivative"),
    ]

    def test_words_only_restatement_stays_quiet_and_unflagged(self):
        for pid, msg in self.CASES:
            r = turn(pid, msg)
            self.assertFalse(r.meta["flags"]["confused"], msg)
            self.assertFalse(r.meta["flags"]["frustrated"], msg)
            self.assertNotIn(r.state, AFFECT, msg)
            self.assertFalse(r.intervened, msg)                       # ARIA does not step in on a restatement

    def test_symbolic_restatement_never_raises_an_affect_flag(self):
        analysis = {"state": "FRUSTRATED", "confidence": 0.99, "state_scores": {"FRUSTRATED": 4.5, "CONFUSED": 3.0}}
        f = DT._flags(analysis, "ugh hate this f(x) = 3x^4 - 5x^2", status="restate")
        self.assertFalse(f["frustrated"] or f["confused"])
        f = DT._flags(analysis, "ugh hate this", status="none")                     # same words, not a restatement: still flagged
        self.assertTrue(f["frustrated"])

    def test_repeated_phrase_alone_is_not_enough(self):
        analysis = {"state": "CONFUSED", "confidence": 0.87, "state_scores": {"CONFUSED": 2.0}}
        self.assertFalse(DT._flags(analysis, "sin of over plus cos of over")["confused"])


class RealFeelingsAreStillCaught(unittest.TestCase):
    def test_frustrated(self):
        for pid, msg in (("calc-07", "ugh I hate integrals"), ("calc-01", "this is so frustrating"),
                         ("calc-01", "ugh this makes no sense, I keep getting it wrong and I hate this")):
            r = turn(pid, msg)
            self.assertTrue(r.meta["flags"]["frustrated"], msg)
            self.assertEqual(r.state, "FRUSTRATED", msg)
            self.assertTrue(r.intervened, msg)

    def test_confused(self):
        for pid, msg in (("trig-04", "wait I'm confused which identity"), ("alg-01", "I don't get it"), ("alg-05", "huh? that is confusing")):
            r = turn(pid, msg)
            self.assertTrue(r.meta["flags"]["confused"], msg)
            self.assertEqual(r.state, "CONFUSED", msg)
            self.assertTrue(r.intervened, msg)

    def test_feelings_never_override_correctness(self):
        r = turn("calc-01", "ugh I hate this, 12x^3 - 10x + 7")
        self.assertEqual(r.status, "correct")
        self.assertNotIn(r.state, ("STUCK", "FRUSTRATED"))


class FreeFormQuestionsAreQuestions(unittest.TestCase):
    QUESTIONS = [
        ("alg-01", "Why do we have to do the same thing to both sides?"),
        ("calc-01", "Why does the power go down by one when we differentiate?"),
        ("trig-01", "What is the unit circle actually for?"),
        ("calc-02", "How does the chain rule work?"),
        ("alg-05", "What does factoring even mean?"),
        ("trig-04", "Why is sine squared plus cosine squared equal to one?"),
    ]

    def test_not_flagged_as_frustrated_or_confused_by_the_flags(self):
        for pid, msg in self.QUESTIONS:
            r = turn(pid, msg)
            self.assertFalse(r.meta["flags"]["frustrated"], msg)
            self.assertFalse(r.meta["flags"]["confused"], msg)
            self.assertEqual(r.meta["kind"], "question", msg)
            self.assertTrue(r.intervened, msg)                          # a question deserves an answer

    def test_every_topic_can_reach_the_llm_path_when_it_is_on(self):
        calls = []
        t = DT.DemoTutor(problems=PROBS, llm_path=True, llm_fn=lambda p, q: calls.append(q) or "It is the same idea on both sides of the equals sign.")
        for pid, msg in self.QUESTIONS[:3]:
            r = t.new_session(pid).turn(msg)
            self.assertTrue(r.meta["llm_used"], (pid, msg))
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
