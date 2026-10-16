"""Step E: teacher summary = state timeline + hints given + where the student got stuck, and never a transcript."""
import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock

from agent import demo_summary as S
from agent import demo_tutor as DT
from agent.demo_bank import load_problems

PROBS = load_problems()
BY_ID = {p.id: p for p in PROBS}
TUTOR = DT.DemoTutor(problems=PROBS)
SECRETS = ["zebra-quokka-7", "Priya Raman", "my cat Mr Whiskers"]


def load_ui():
    spec = importlib.util.spec_from_file_location("aria_demo_app_e", Path(__file__).resolve().parent.parent / "ui" / "demo_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def play(pid, msgs, tutor=TUTOR):
    s = tutor.new_session(pid)
    replies = [s.turn(m).reply for m in msgs]
    return s, replies


SCRIPT = ["I don't know, zebra-quokka-7", "12x^3", "ugh this makes no sense, says Priya Raman", "12x^3 - 10x - 9", "12x^3 - 10x", "12x^3 - 10x + 7"]


class Content(unittest.TestCase):
    def setUp(self):
        self.s, self.replies = play("calc-01", SCRIPT)
        self.sm = S.summarize(self.s)

    def test_timeline_has_one_row_per_turn_with_state_verdict_and_action(self):
        tl = self.sm["timeline"]
        self.assertEqual([t["turn"] for t in tl], list(range(1, len(SCRIPT) + 1)))
        self.assertEqual(tl[0]["state"], "STUCK")
        self.assertEqual(tl[0]["verdict"], "none")
        self.assertIn(tl[1]["verdict"], ("partial", "step"))
        self.assertEqual(tl[1]["action"], "none")                         # quiet on a correct step
        self.assertEqual(tl[3]["verdict"], "wrong")
        self.assertNotEqual(tl[3]["action"], "none")
        self.assertEqual(tl[-1]["verdict"], "correct")
        self.assertEqual(self.sm["outcome"], "solved")
        self.assertEqual(self.sm["solved_at_turn"], len(SCRIPT))
        for t in tl:
            self.assertTrue(t["why"] and t["state_label"] and t["verdict_label"] and t["action_label"])

    def test_hints_are_tied_to_authored_steps_and_counted(self):
        hs = self.sm["hints"]
        self.assertGreaterEqual(len(hs), 2)
        self.assertEqual(self.sm["counts"]["stepped_in"], len(hs))
        self.assertEqual(self.sm["counts"]["turns"], len(SCRIPT))
        self.assertEqual(self.sm["counts"]["stayed_quiet"] + self.sm["counts"]["stepped_in"], len(SCRIPT))
        steps = [m["do"] for m in BY_ID["calc-01"].moves]
        for h in hs:
            if h["step_index"] >= 0:
                self.assertEqual(h["step"], steps[h["step_index"]])
        self.assertEqual(self.sm["hints"][0]["step"], steps[0])             # first hint pointed at the first authored step
        self.assertEqual(self.sm["counts"]["hints_given"], sum(self.sm["counts"]["by_type"].get(k, 0) for k in S.HINT_TYPES))

    def test_where_the_student_got_stuck(self):
        stretches = self.sm["stuck"]["stretches"]
        self.assertTrue(stretches)
        first = stretches[0]
        self.assertEqual(first["from_turn"], 1)
        self.assertTrue(first["resolved"])                                    # the step on turn 2 got them moving
        self.assertEqual(first["step"], BY_ID["calc-01"].moves[0]["do"])
        lg = self.sm["stuck"]["longest"]                                       # turns 3-4 beat the one-turn stretch at turn 1
        self.assertEqual((lg["from_turn"], lg["to_turn"], lg["resolved"]), (3, 4, True))
        h = S.headline(self.sm)
        self.assertIn("turns 3 to 4 (2 turns)", h)
        self.assertIn("Apply the power rule to 5x^2", h)                       # 12x^3 was already produced, so the next step is 5x^2

    def test_progress_through_authored_steps(self):
        st = self.sm["steps"]
        self.assertEqual([x["done"] for x in st], [True, True])               # 12x^3 and -10x were both produced
        self.assertEqual(st[0]["hinted_at_turns"], [1])

    def test_reflection_after_a_correct_answer_is_not_tied_to_an_authored_step(self):
        refl = [h for h in self.sm["hints"] if h["type"] == "reflection"]
        self.assertTrue(refl)
        for h in refl:
            self.assertEqual((h["step_index"], h["step"]), (-1, None))
        for st in self.sm["steps"]:
            self.assertNotIn(self.sm["solved_at_turn"], st["hinted_at_turns"])      # no step was "hinted" on the turn the student solved it

    def test_unresolved_stretch_and_no_stuck_cases(self):
        s, _ = play("calc-01", ["I don't know", "still no idea", "help"])
        sm = S.summarize(s)
        self.assertEqual(sm["outcome"], "in_progress")
        lg = sm["stuck"]["longest"]
        self.assertEqual((lg["from_turn"], lg["to_turn"], lg["resolved"]), (1, 3, False))
        self.assertIn("still stuck at the end", S.headline(sm))
        s, _ = play("calc-01", ["12x^3", "12x^3 - 10x"])
        self.assertEqual(S.headline(S.summarize(s)), "No stuck turns in this attempt.")

    def test_empty_and_reset(self):
        s = TUTOR.new_session("alg-01")
        self.assertEqual(S.summarize(s)["outcome"], "not_started")
        self.assertEqual(S.headline(S.summarize(s)), "No attempt yet.")
        s.turn("I don't know")
        s.reset()
        sm = S.summarize(s)
        self.assertEqual((sm["outcome"], sm["attempt"], sm["counts"]["turns"]), ("not_started", 2, 0))

    def test_unreadable_turn_is_shown_as_such(self):
        s = TUTOR.new_session("alg-01")
        with mock.patch.object(DT.DemoSession, "_analyze", side_effect=RuntimeError("boom")):
            s.turn("x = 3")
        t = S.summarize(s)["timeline"][0]
        self.assertEqual((t["state"], t["action"], t["fallback"]), ("UNKNOWN", "none", "error:RuntimeError"))

    def test_llm_answer_is_not_attributed_to_an_authored_step(self):
        t = DT.DemoTutor(problems=PROBS, llm_path=True, llm_fn=lambda *_: "Because the derivative of cos is minus sin.")
        s, _ = play("calc-02", ["why is there a minus sign?"], tutor=t)
        h = S.summarize(s)["hints"][0]
        self.assertIsNone(h["step"])
        self.assertEqual(h["step_index"], -1)


class NeverATranscript(unittest.TestCase):
    def test_no_student_text_and_no_tutor_text_anywhere(self):
        ui = load_ui()
        s, replies = play("calc-01", SCRIPT)
        sm = S.summarize(s)
        blob = json.dumps(sm) + ui.summary_html(sm)
        for secret in SECRETS + ["12x^3 - 10x - 9", "ugh this makes no sense", "I don't know"]:
            self.assertNotIn(secret, blob)
        for r in replies:
            if r:
                self.assertNotIn(r, blob)
                self.assertNotIn(r, ui.summary_html(sm).replace("&#x27;", "'"))
        self.assertNotIn("student_chars", blob)

    def test_hint_label_when_no_authored_step_is_left(self):
        ui = load_ui()
        s, _ = play("calc-02", ["I don't know", "2x cos(2x)", "2x cos(2x) + 2x^2 sin(2x)"])     # the only authored step is done by turn 2
        sm = S.summarize(s)
        h = sm["hints"][-1]
        self.assertEqual((h["step"], h["source"]), (None, "template"))
        page = ui.summary_html(sm)
        self.assertIn("a general prompt (every authored step was already done)", page)
        self.assertNotIn("a free-form question", page)
        t = DT.DemoTutor(problems=PROBS, llm_path=True, llm_fn=lambda *_: "Because the derivative of cos is minus sin.")
        s, _ = play("calc-02", ["why is there a minus sign?"], tutor=t)
        self.assertIn("a free-form question", ui.summary_html(S.summarize(s)))

    def test_html_sections_and_escaping(self):
        ui = load_ui()
        s, _ = play("calc-01", SCRIPT)
        page = ui.summary_html(S.summarize(s))
        for h in ("State timeline", "Hints given", "Authored steps", "Where the student got stuck", "no chat transcript"):
            self.assertIn(h, page)
        evil = S.summarize(s)
        evil["steps"][0]["step"] = "<script>alert(1)</script>"
        evil["hints"][0]["step"] = "<img src=x onerror=alert(1)>"
        evil["problem"]["prompt"] = "<b>x</b>"
        page = ui.summary_html(evil)
        self.assertNotIn("<script>", page)
        self.assertNotIn("<img", page)
        self.assertNotIn("<b>x</b>", page)
        self.assertIn("No attempt to summarise yet", ui.summary_html(None))

    def test_ui_handler_reflects_the_live_session_and_survives_hostile_input(self):
        ui = load_ui()
        t = DT.DemoTutor(problems=PROBS)
        start, send, reset = ui.make_handlers(t, list(t.problems.values()))
        show = ui.make_summary_handler()
        session, hist, *_ = start(BY_ID["calc-01"].prompt)
        self.assertIn("No attempt to summarise yet", show(session))
        for m in ("I don't know", "12x^3", None, "٣x² ‮", "x" * 5000, "12x^3 - 10x + 7"):
            hist, session, *_ = send(m, hist, session)
        page = show(session)
        self.assertIn("Solved at turn", page)
        self.assertIn("Apply the power rule to 3x^4", page)
        self.assertIn("No attempt to summarise yet", show(reset(session) and session))
        self.assertIn("No attempt to summarise yet", show(None))


if __name__ == "__main__":
    unittest.main()
