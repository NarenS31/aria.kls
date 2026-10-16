"""Step G: 'Would you step in?' activity. FIXTURE_TURNS below are test fixtures, NOT the real moments (the owner supplies those)."""
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_teacher_taps as AN  # noqa: E402
import make_teacher_moments as MK  # noqa: E402
from agent import teacher_activity as TA  # noqa: E402

FIXTURE_TURNS = [
    {"id": "m1", "problem_id": "calc-01", "history": [], "message": "I don't know"},
    {"id": "m2", "problem_id": "calc-01", "history": ["12x^3"], "message": "12x^3 - 10x"},
    {"id": "m3", "problem_id": "trig-01", "history": [], "message": "so I need sin of pi over 6 plus cos of pi over 3"},
    {"id": "m4", "problem_id": "alg-01", "history": [], "message": "x = 7"},
    {"id": "m5", "problem_id": "calc-02", "history": [], "message": "ugh I hate this"},
    {"id": "m6", "problem_id": "alg-05", "history": [], "message": "(x - 2)(x - 3) = 0"},
    {"id": "m7", "problem_id": "trig-04", "history": [], "message": "why is sine squared plus cosine squared one?"},
    {"id": "m8", "problem_id": "calc-07", "history": ["2x^3 - 2x^2"], "message": "2x^3 - 2x^2 + x"},
]
MOMENTS = MK.build(FIXTURE_TURNS)
BY_ID = {m["id"]: m for m in MOMENTS}


def tmp_log():
    d = tempfile.mkdtemp()
    return TA.TapLog(Path(d) / "taps.csv"), Path(d) / "taps.csv"


def run_visitor(act, decisions, types=None, agree="agree"):
    """decisions: list per moment of 'step_in'/'stay_quiet'."""
    act.consent(True)
    for d in decisions:
        act.decide(d)
        if d == "step_in":
            act.choose_type((types or {}).get(act.moment["id"], "hint"))
        act.agree(agree)


class MomentGeneration(unittest.TestCase):
    def test_ground_truth_is_the_tutors_logged_decision_and_properties_are_computed(self):
        self.assertEqual(len(MOMENTS), 8)
        self.assertEqual(BY_ID["m1"]["aria_decision"], "step_in")                     # "I don't know"
        self.assertEqual(BY_ID["m2"]["aria_decision"], "stay_quiet")                  # correct intermediate step
        self.assertEqual(BY_ID["m3"]["aria_decision"], "stay_quiet")                  # words-only restatement (fix 1a)
        self.assertTrue(BY_ID["m3"]["words_only"] and not BY_ID["m3"]["verified_wrong"])
        self.assertTrue(BY_ID["m4"]["verified_wrong"] and not BY_ID["m4"]["words_only"])
        self.assertEqual(BY_ID["m4"]["aria_decision"], "step_in")                     # never quiet on a verified wrong step
        self.assertEqual(BY_ID["m5"]["aria_state"], "FRUSTRATED")
        self.assertTrue(BY_ID["m5"]["words_only"])
        for m in MOMENTS:
            if m["verified_wrong"]:
                self.assertEqual(m["aria_decision"], "step_in", m["id"])
            if m["aria_decision"] == "step_in":
                self.assertIn(m["aria_type"], TA.TYPES)
                self.assertTrue(m["aria_reply"])
            else:
                self.assertEqual((m["aria_type"], m["aria_reply"]), ("", ""))

    def test_replay_is_deterministic_and_matches_a_fresh_tutor(self):
        again = MK.build(FIXTURE_TURNS)
        self.assertEqual([(m["id"], m["aria_decision"], m["aria_type"], m["aria_reply"]) for m in again],
                         [(m["id"], m["aria_decision"], m["aria_type"], m["aria_reply"]) for m in MOMENTS])

    def test_type_mapping(self):
        self.assertEqual([TA.aria_category(t) for t in ("hint", "task_chunking", "reflection", "encouragement", "break_suggestion", "zzz")],
                         ["hint", "hint", "prompt", "encouragement", "break", "other"])

    def test_loader_validates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(json.dumps(MOMENTS))
            self.assertEqual(len(TA.load_moments(p)), 8)
            p.write_text(json.dumps(MOMENTS[:7]))
            with self.assertRaisesRegex(ValueError, "expected 8"):
                TA.load_moments(p)
            bad = json.loads(json.dumps(MOMENTS))
            bad[0]["aria_decision"] = "maybe"
            p.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "aria_decision"):
                TA.load_moments(p)
            dup = json.loads(json.dumps(MOMENTS))
            dup[1]["id"] = dup[0]["id"]
            p.write_text(json.dumps(dup))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                TA.load_moments(p)


class RealMoments(unittest.TestCase):
    """data/teacher_moments.json is the owner's real set (turns in data/teacher_moments_turns.json), regenerated through the tagged tutor."""
    GOLDEN = {  # ARIA's decision at generation time (build summit-2026-10-24); a change here means the tutor's behaviour changed
        "M1": ("stay_quiet", ""), "M2": ("stay_quiet", ""), "M3": ("step_in", "hint"), "M4": ("stay_quiet", ""),
        "M5": ("stay_quiet", ""), "M6": ("step_in", "encouragement"), "M7": ("step_in", "hint"), "M8": ("step_in", "hint")}

    @classmethod
    def setUpClass(cls):
        cls.moments = TA.load_moments(ROOT / "data" / "teacher_moments.json")
        cls.turns = json.loads((ROOT / "data" / "teacher_moments_turns.json").read_text(encoding="utf-8"))

    def test_eight_moments_in_the_owners_order_with_the_recorded_decisions(self):
        self.assertEqual([m["id"] for m in self.moments], [f"M{i}" for i in range(1, 9)])
        self.assertEqual({m["id"]: (m["aria_decision"], m["aria_type"]) for m in self.moments}, self.GOLDEN)

    def test_committed_file_equals_a_fresh_replay_of_the_turns(self):
        fresh = MK.build(self.turns)
        strip = lambda ms: [{k: v for k, v in m.items() if k != "build"} for m in ms]          # noqa: E731
        self.assertEqual(strip(fresh), strip(json.loads((ROOT / "data" / "teacher_moments.json").read_text(encoding="utf-8"))))

    def test_turns_are_the_owners_verbatim(self):
        self.assertEqual([(t["id"], t["problem_id"], t["history"], t["message"]) for t in self.turns], [
            ("M1", "calc-01", ["first I'll differentiate term by term"], "12x^3 - 10x"), ("M2", "alg-03", ["move the 2x over"], "3x - 3 = 9"),
            ("M3", "calc-02", ["product rule"], "2x cos(2x) + 2x^2 sin(2x)"), ("M4", "alg-05", [], "I think I'll try factoring first"),
            ("M5", "trig-01", [], "so I need sin of pi over 6 plus cos of pi over 3"), ("M6", "calc-07", [], "ugh I hate integrals"),
            ("M7", "alg-05", [], "can you just tell me the answer"), ("M8", "alg-02", [], "x = 11")])

    def test_properties_and_the_verified_wrong_rule(self):
        self.assertEqual({m["id"] for m in self.moments if m["words_only"]}, {"M4", "M5", "M6", "M7"})
        self.assertEqual({m["id"] for m in self.moments if m["verified_wrong"]}, {"M3", "M8"})
        for m in self.moments:
            if m["verified_wrong"]:
                self.assertEqual(m["aria_decision"], "step_in", m["id"])            # never quiet on a verified wrong step
            self.assertEqual(m["build"]["tag"], "summit-2026-10-24")
            self.assertIn(m["aria_type"], ("",) + TA.TYPES)

    def test_six_moments_are_in_the_aria_comparison_and_only_hint_and_encouragement_occur(self):
        in_scope = [m["id"] for m in self.moments if not m["verified_wrong"]]
        self.assertEqual(in_scope, ["M1", "M2", "M4", "M5", "M6", "M7"])
        self.assertEqual(sorted({m["aria_type"] for m in self.moments if m["aria_decision"] == "step_in"}), ["encouragement", "hint"])
        self.assertEqual(sum(m["aria_decision"] == "step_in" for m in self.moments if not m["verified_wrong"]), 2)       # M6, M7

    def test_generated_from_the_tagged_tutor(self):
        info = MK.build_info()
        if info["commit"] == "unknown":
            self.skipTest("tag summit-2026-10-24 not present in this checkout")
        self.assertTrue(info["tutor_files_match_tag"], "tutor files differ from the tag: the moments would not describe the tagged build")
        self.assertTrue(info["commit"].startswith("e09218c"))


class Flow(unittest.TestCase):
    def test_skip_records_nothing(self):
        log, path = tmp_log()
        a = TA.Activity(MOMENTS, log)
        a.consent(False)
        self.assertEqual(a.phase, "skipped")
        a.decide("step_in")                                  # ignored after skip
        self.assertFalse(path.exists())

    def test_full_run_writes_exactly_one_valid_row_per_moment(self):
        log, path = tmp_log()
        a = TA.Activity(MOMENTS, log)
        a.set_profile("teacher", "11-20", "middle_6-8")
        run_visitor(a, ["step_in", "stay_quiet"] * 4, types={"m1": "hint", "m4": "prompt", "m6": "break", "m8": "other"}, agree="disagree")
        self.assertEqual(a.phase, "done")
        rows = list(csv.DictReader(open(path, newline="")))
        self.assertEqual(open(path).readline().strip().split(","), TA.COLUMNS)
        self.assertEqual([r["moment_id"] for r in rows], [m["id"] for m in MOMENTS])
        self.assertEqual({r["session_id"] for r in rows}, {a.session_id})
        self.assertEqual(len(a.session_id), 12)
        r1, r2 = rows[0], rows[1]
        self.assertEqual((r1["teacher_decision"], r1["teacher_type"], r1["aria_decision"], r1["aria_type"], r1["agree"]), ("step_in", "hint", "step_in", BY_ID["m1"]["aria_type"], "disagree"))
        self.assertEqual((r2["teacher_decision"], r2["teacher_type"], r2["aria_decision"], r2["aria_type"]), ("stay_quiet", "", "stay_quiet", ""))
        self.assertTrue(all((r["role"], r["years_band"], r["grade_band"]) == ("teacher", "11-20", "middle_6-8") for r in rows))
        self.assertTrue(all(r["timestamp"] for r in rows))

    def test_no_student_text_or_aria_reply_ever_reaches_the_file(self):
        log, path = tmp_log()
        a = TA.Activity(MOMENTS, log)
        run_visitor(a, ["step_in"] * 8)
        blob = path.read_text()
        for m in MOMENTS:
            for text in [m["message"], m["problem_prompt"], m["aria_reply"], m["aria_reason"]] + m["history"]:
                if text and len(text) > 8:
                    self.assertNotIn(text, blob)

    def test_reveal_only_after_the_visitor_answers_and_order_is_enforced(self):
        log, path = tmp_log()
        a = TA.Activity(MOMENTS, log)
        self.assertIsNone(a.reveal())
        a.consent(True)
        self.assertIsNone(a.reveal())                        # not yet: no answer given
        a.agree("agree")                                     # out of order: ignored
        a.choose_type("hint")                                # out of order: ignored
        self.assertEqual((a.phase, a.i, a.rows_written), ("decision", 0, 0))
        a.decide("step_in")
        self.assertIsNone(a.reveal())                        # still choosing the type
        a.choose_type("not-a-type")
        self.assertEqual(a.phase, "type")
        a.choose_type("hint")
        r = a.reveal()
        self.assertEqual((r["teacher_decision"], r["teacher_type"], r["aria_decision"]), ("step_in", "hint", BY_ID["m1"]["aria_decision"]))
        a.agree("maybe")                                     # invalid: ignored
        self.assertEqual((a.phase, a.rows_written), ("reveal", 0))
        a.agree("agree")
        self.assertEqual((a.i, a.rows_written), (1, 1))

    def test_stay_quiet_skips_the_type_question(self):
        a = TA.Activity(MOMENTS, tmp_log()[0])
        a.consent(True)
        a.decide("stay_quiet")
        self.assertEqual(a.phase, "reveal")

    def test_profile_defaults_and_invalid_values_fall_back(self):
        a = TA.Activity(MOMENTS, tmp_log()[0])
        self.assertEqual(set(a.profile.values()), {TA.PREFER_NOT})
        a.set_profile("<script>", "forty", "kindergarten")
        self.assertEqual(set(a.profile.values()), {TA.PREFER_NOT})

    def test_taplog_rejects_anything_outside_the_fixed_lists(self):
        log, path = tmp_log()
        good = dict(session_id="abc123abc123", moment_id="m1", role="teacher", years_band="3-5", grade_band="mixed", teacher_decision="step_in",
                    teacher_type="hint", aria_decision="stay_quiet", aria_type="", agree="agree", timestamp="2026-10-24T10:00:00")
        self.assertTrue(log.write(good))
        for k, v in (("role", "Ms. Rivera"), ("teacher_type", "free text answer"), ("agree", "yes"), ("session_id", "x"), ("moment_id", "m 1; DROP"),
                     ("teacher_decision", "stay_quiet"), ("aria_type", "hint")):
            self.assertFalse(log.write({**good, k: v}), (k, v))
        self.assertEqual(len(list(csv.DictReader(open(path)))), 1)
        self.assertEqual(log.errors, 7)
        self.assertFalse(TA.TapLog(None).write(good))        # no path = no file
        bad_dir = TA.TapLog(Path(tempfile.mkdtemp()))        # a directory: write fails quietly
        self.assertFalse(bad_dir.write(good))


class Analysis(unittest.TestCase):
    def test_fleiss_kappa_matches_a_hand_computed_case(self):
        # 3 moments x 3 raters, 2 categories: P = (1, 1, 1/3), p = (5/9, 4/9) -> kappa = (7/9 - 41/81) / (40/81) = 0.55
        self.assertAlmostEqual(AN.fleiss_kappa([[3, 0], [0, 3], [2, 1]]), 0.55, places=9)
        self.assertAlmostEqual(AN.fleiss_kappa([[4, 0], [0, 4], [4, 0], [0, 4]]), 1.0)
        self.assertIsNone(AN.fleiss_kappa([[4, 0], [4, 0]]))             # everyone always says the same thing: undefined
        self.assertIsNone(AN.fleiss_kappa([[2, 1], [3, 1]]))             # unequal rater counts (3 vs 4)

    def rows(self, n_sessions, moments, step_in_for):
        out = []
        for s in range(n_sessions):
            for m in moments:
                dec = "step_in" if step_in_for(s, m) else "stay_quiet"
                out.append({"session_id": f"s{s:011d}", "moment_id": m["id"], "role": "teacher" if s % 2 else "other", "years_band": "6-10", "grade_band": "mixed",
                            "teacher_decision": dec, "teacher_type": "hint" if dec == "step_in" else "", "aria_decision": m["aria_decision"],
                            "aria_type": m["aria_type"], "agree": "agree", "timestamp": "t"})
        return out

    def test_nothing_is_reported_below_n15(self):
        res = AN.analyse(self.rows(14, MOMENTS, lambda s, m: True), BY_ID)
        for name in ("all", "math", "words_only"):
            self.assertTrue(res["sets"][name]["status"].startswith("not reported"), name)
            self.assertNotIn("fleiss_kappa", res["sets"][name])
        self.assertTrue(str(res["after_reveal"]).startswith("not reported"))
        self.assertTrue(all(g["status"].startswith("suppressed") for g in res["subgroups"]["role"].values()))

    def test_reported_at_n15_with_separate_sets_and_verified_wrong_excluded_from_the_aria_table(self):
        res = AN.analyse(self.rows(16, MOMENTS, lambda s, m: (s + len(m["id"]) + int(m["id"][1:])) % 3 != 0), BY_ID)
        for name in ("all", "math", "words_only", "verified_wrong"):
            self.assertEqual(res["sets"][name]["status"], "reported", name)
        self.assertEqual(set(res["sets"]["words_only"]["moments"]), {m["id"] for m in MOMENTS if m["words_only"]})
        self.assertTrue(set(res["sets"]["words_only"]["moments"]).isdisjoint(res["sets"]["math"]["moments"]))
        wrong = {m["id"] for m in MOMENTS if m["verified_wrong"]}
        self.assertTrue(wrong)
        for name in ("all", "math", "words_only"):
            self.assertTrue(wrong.isdisjoint(res["sets"][name]["moments_in_aria_comparison"]), name)
        vw = res["sets"]["verified_wrong"]
        self.assertIn("teacher_stay_quiet_rate_on_verified_wrong", vw)
        self.assertNotIn("confusion_majority_vs_aria", vw)
        conf = res["sets"]["all"]["confusion_majority_vs_aria"]
        self.assertEqual(sum(conf.values()) + len(res["sets"]["all"]["split_moments"]), len(res["sets"]["all"]["moments_in_aria_comparison"]))
        # 16 sessions: the single years band has 16 (reported); the two roles have 8 each (suppressed, plan section 4)
        self.assertIsNotNone(res["subgroups"]["years_band"]["6-10"]["step_in_share"])
        self.assertTrue(all(g["status"].startswith("suppressed") for g in res["subgroups"]["role"].values()))
        self.assertEqual({g["sessions"] for g in res["subgroups"]["role"].values()}, {"8"})

    def test_unanimous_teachers_who_all_agree_with_aria_score_full_agreement(self):
        res = AN.analyse(self.rows(15, MOMENTS, lambda s, m: m["aria_decision"] == "step_in"), BY_ID)
        self.assertEqual(res["sets"]["all"]["response_level_agreement_with_aria"], 1.0)
        self.assertEqual(res["sets"]["all"]["confusion_majority_vs_aria"]["majority=step_in|aria=stay_quiet"], 0)
        self.assertEqual(res["sets"]["all"]["confusion_majority_vs_aria"]["majority=stay_quiet|aria=step_in"], 0)


class UiWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("aria_demo_app_g", ROOT / "ui" / "demo_app.py")
        cls.ui = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ui)

    def visible(self, out):
        return [bool(o.get("visible")) if isinstance(o, dict) else None for o in out[1:6]]       # consent, decide, type, reveal, done

    def test_each_phase_shows_exactly_the_right_group(self):
        log, path = tmp_log()
        new, proceed, decide, choose_type, agree = self.ui.make_activity_handlers(MOMENTS, log)
        out = new()
        act = out[0]
        self.assertEqual(self.visible(out), [True, False, False, False, False])
        out = proceed(act, "teacher", "6-10", "mixed", True)
        self.assertEqual(self.visible(out), [False, True, False, False, False])
        self.assertIn("Moment 1 of 8", out[6]["value"])
        self.assertEqual(self.visible(decide(act, "step_in")), [False, False, True, False, False])
        out = choose_type(act, "hint")
        self.assertEqual(self.visible(out), [False, False, False, True, False])
        self.assertIn("ARIA would", out[7]["value"])
        self.assertEqual(self.visible(agree(act, "agree")), [False, True, False, False, False])
        self.assertEqual(act.rows_written, 1)
        skipped = proceed(Activity := self.ui.Activity(MOMENTS, log), "teacher", "6-10", "mixed", False)
        self.assertEqual(self.visible(skipped), [False, False, False, False, True])
        self.assertIn("Nothing was recorded", skipped[8]["value"])

    def test_html_is_escaped_and_the_reveal_never_precedes_the_answer(self):
        m = json.loads(json.dumps(MOMENTS))
        m[0]["message"] = "<script>alert(1)</script>"
        m[0]["aria_reply"] = "<img src=x onerror=alert(1)>"
        act = TA.Activity(m, tmp_log()[0])
        act.consent(True)
        card = self.ui.activity_card(act)
        self.assertNotIn("<script>", card)
        self.assertEqual(self.ui.activity_reveal(act), "")
        act.decide("step_in")
        act.choose_type("hint")
        self.assertNotIn("<img", self.ui.activity_reveal(act))

    def test_page_builds_with_and_without_moments(self):
        from agent.demo_bank import load_problems
        from agent.demo_tutor import DemoTutor
        tutor = DemoTutor(problems=load_problems(featured_only=True))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(json.dumps(MOMENTS))
            self.assertIsNotNone(self.ui.build_demo_ui(tutor, p, Path(d) / "t.csv"))
            self.assertIsNotNone(self.ui.build_demo_ui(tutor, Path(d) / "missing.json", Path(d) / "t.csv"))


if __name__ == "__main__":
    unittest.main()
