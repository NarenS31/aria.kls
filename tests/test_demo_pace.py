"""The demo's BKT tracker must reproduce aria_pace's own predictions, and read its numbers from the frozen files."""
import itertools
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import demo_pace as P  # noqa: E402
from agent import demo_tutor as PACE_T  # noqa: E402
from agent.demo_bank import load_problems  # noqa: E402
from agent.demo_tutor import DemoTutor  # noqa: E402
from agent.math_check import reveals_answer  # noqa: E402
from unittest import mock  # noqa: E402
from aria_pace import baselines as B  # noqa: E402
from aria_pace import policy as POL  # noqa: E402

CFG = P.load_config()
REF = B.BKT(min_seq=10).fit(None, kc_fits={None: (10 ** 6, dict(CFG.params))})      # aria_pace's own BKT, pooled parameters


def reference_risk(bits) -> float:
    """aria_pace.baselines.BKT.score + aria_pace.policy.platt_apply with the frozen calibration."""
    seq = SimpleNamespace(kc=None, correct=np.array(bits, dtype=np.int8))
    raw = REF.score(seq, len(bits))
    return float(POL.platt_apply(np.array([raw]), *CFG.platt)[0])


def mastered(bits) -> bool:
    return len(bits) >= 3 and all(b == 1 for b in bits[-3:])


class TrackerMatchesAriaPace(unittest.TestCase):
    def test_every_sequence_up_to_length_8(self):
        n = 0
        for L in range(1, 9):
            for bits in itertools.product((0, 1), repeat=L):
                if mastered(bits):
                    continue                                   # aria_pace's replay never scores mastered sequences
                self.assertAlmostEqual(P.calibrate(P.raw_risk(list(bits), CFG.params), CFG.platt), reference_risk(bits), places=12, msg=bits)
                n += 1
        self.assertGreater(n, 300)

    def test_at_and_after_the_horizon(self):
        for bits in ([0, 1, 0, 1, 0, 0, 1, 0, 1, 0], [0] * 9, [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0]):
            self.assertAlmostEqual(P.calibrate(P.raw_risk(bits, CFG.params), CFG.platt), reference_risk(bits), places=12)

    def test_fixed_status_sequence(self):
        """The turn-by-turn path a student takes: only verdicts change the tracker, and each risk equals aria_pace's score of that prefix."""
        script = [("none", None), ("wrong", 0), ("restate", None), ("step", 1), ("wrong", 0), ("none", None), ("partial", 1),
                  ("wrong", 0), ("correct", 1)]
        t, bits = P.PaceTracker(CFG), []
        self.assertIsNone(t.risk())
        for status, bit in script:
            updated = t.update(status)
            self.assertEqual(updated, bit is not None, status)
            if bit is not None:
                bits.append(bit)
            if bits and not mastered(bits):
                self.assertAlmostEqual(t.risk(), reference_risk(bits), places=12, msg=(status, bits))
        self.assertEqual(t.obs, [0, 1, 0, 1, 0, 1])
        # pin the numbers too, so a silent change of parameters or calibration cannot pass
        # (values printed by aria_pace.baselines.BKT.score + policy.platt_apply, not by the tracker)
        for bits, want in (([0], 0.282099), ([0, 0], 0.468123), ([0, 0, 0], 0.607844), ([1, 0], 0.174181), ([0, 1, 0, 1, 0, 1], 0.307192)):
            self.assertAlmostEqual(reference_risk(bits), want, places=5, msg=bits)
            tt = P.PaceTracker(CFG)
            tt.obs = list(bits)
            self.assertAlmostEqual(tt.risk(), want, places=5, msg=bits)

    def test_mastery_and_empty(self):
        t = P.PaceTracker(CFG)
        for s in ("correct", "step", "partial"):
            t.update(s)
        self.assertLess(t.risk(), 1e-3)                        # three right in a row: mastered
        t.reset()
        self.assertIsNone(t.risk())
        with self.assertRaises(ValueError):
            P.raw_risk([], CFG.params)

    def test_sessions_are_independent(self):
        a, b = P.PaceTracker(CFG), P.PaceTracker(CFG)
        a.update("wrong")
        self.assertIsNone(b.risk())


class RuleAndConfig(unittest.TestCase):
    def test_taus_weights_and_params_come_from_the_frozen_files(self):
        val = json.loads((P.RESULTS / "frozen_val.json").read_text())
        self.assertEqual(CFG.taus, {k: float(v) for k, v in val["configs"]["main"]["frozen"]["taus"]["bkt"].items()})
        self.assertEqual(CFG.weights, {k: tuple(v) for k, v in POL.PROFILES.items()})
        self.assertTrue(CFG.taus["quiet"] > CFG.taus["balanced"] > CFG.taus["supportive"] > 0)
        self.assertEqual([p["label"] for p in CFG.profiles()], ["Quiet", "Balanced", "Supportive"])

    def test_calibration_matches_aria_pace(self):
        for s in (0.0, 0.01, 0.2, 0.5, 0.9, 1.0):
            self.assertAlmostEqual(P.calibrate(s, CFG.platt), float(POL.platt_apply(np.array([s]), *CFG.platt)[0]), places=12)

    def test_window_comes_from_the_frozen_policy(self):
        self.assertEqual((CFG.k_min, CFG.k_max), (int(POL.DEC[0]), int(POL.DEC[-1])))
        self.assertEqual((CFG.k_min, CFG.k_max), (3, 9))

    def test_inside_the_window_speak_only_on_verified_wrong_with_risk_at_least_tau(self):
        for prof in P.PROFILE_ORDER:
            t = P.PaceTracker(CFG)
            t.update("step"); t.update("step")                      # attempts 1-2 were right
            d = t.decide("wrong", prof)                             # attempt 3
            self.assertTrue(d.verified_wrong and d.updated and d.in_window and d.n_obs == 3)
            self.assertEqual(d.speak, d.risk >= d.tau)
            t2 = P.PaceTracker(CFG)
            for s_ in ("step", "correct"):
                d2 = t2.decide(s_, prof)
                self.assertFalse(d2.speak or d2.verified_wrong)
            self.assertFalse(P.PaceTracker(CFG).decide("none", prof).speak)         # no verdict: never "a verified wrong step"
            self.assertFalse(P.PaceTracker(CFG).decide("restate", prof).speak)

    def test_attempts_1_and_2_keep_the_old_rule_in_every_profile(self):
        for prof in P.PROFILE_ORDER:
            t = P.PaceTracker(CFG)
            d1 = t.decide("wrong", prof)                            # attempt 1: risk 0.28 < Quiet's 0.57, still speaks
            d2 = t.decide("wrong", prof)                            # attempt 2
            for d, n in ((d1, 1), (d2, 2)):
                self.assertEqual((d.n_obs, d.in_window, d.verified_wrong, d.speak, d.reason), (n, False, True, True, "wrong_before_window"))
        t = P.PaceTracker(CFG)
        t.update("step"); t.update("step")
        self.assertEqual(t.decide("wrong", "quiet").reason, "wrong_risk_lt_tau")       # attempt 3: the gate starts

    def test_after_attempt_9_the_old_rule_holds_again(self):
        t = P.PaceTracker(CFG)
        t.obs = [1, 1] + [0] * 6                                    # eight answers so far
        d9 = t.decide("wrong", "quiet")                             # attempt 9: the last gated one
        self.assertEqual((d9.n_obs, d9.in_window, d9.speak), (9, True, d9.risk >= d9.tau))
        d10 = t.decide("wrong", "quiet")                            # attempt 10: old rule again, even if risk were below tau
        self.assertEqual((d10.n_obs, d10.in_window, d10.speak, d10.reason), (10, False, True, "wrong_after_window"))

    def test_wrong_step_below_tau_is_reported_not_dropped(self):
        t = P.PaceTracker(CFG)
        for s_ in ("correct", "step"):
            t.update(s_)
        d = t.decide("wrong", "quiet")
        self.assertTrue(d.verified_wrong)
        self.assertEqual(d.speak, d.risk >= d.tau)
        self.assertEqual(d.reason, "wrong_risk_ge_tau" if d.speak else "wrong_risk_lt_tau")

    def test_tampered_frozen_file_refuses_to_load(self):
        tmp = Path(tempfile.mkdtemp(prefix="pace_cfg_"))
        for f in ("frozen_v2.json", "frozen_v2.sha256", "frozen_val.json", "frozen_val.sha256", "baselines_val.json"):
            shutil.copy(P.RESULTS / f, tmp / f)
        P.load_config(tmp)                                     # untouched copy loads
        (tmp / "frozen_val.json").write_text((tmp / "frozen_val.json").read_text().replace("0.57", "0.10"))
        P.load_config.cache_clear()
        with self.assertRaises(RuntimeError):
            P.load_config(tmp)
        P.load_config.cache_clear()


FEATURED = load_problems(featured_only=True)
WRONG_ANSWERS = {p.id: p.msg_wrong() for p in FEATURED}


def session(profile, pid="alg-01", **kw):
    t = DemoTutor(problems=FEATURED, pace_profile=profile, **kw)
    return t, t.new_session(pid)


def get(pid):
    return next(x for x in FEATURED if x.id == pid)


def play(s, p, *msgs):
    return [s.turn(m) for m in msgs]


class DecisionRuleInTheTutor(unittest.TestCase):
    def test_attempts_1_and_2_speak_on_a_wrong_step_in_every_profile(self):
        for prof in P.PROFILE_ORDER:
            _, s = session(prof)
            for n in (1, 2):
                r = s.turn(WRONG_ANSWERS["alg-01"])
                self.assertEqual((r.status, r.intervened, r.meta["pace_reason"], r.meta["pace_n_obs"]), ("wrong", True, "wrong_before_window", n), prof)
                self.assertNotEqual(r.meta["intervention_type"], "none")

    def test_same_student_three_wrong_answers_follows_each_profile(self):
        # step, step (right), then the same wrong answer 3 times = attempts 3, 4, 5: risks 0.152, 0.254, 0.465
        want = {"quiet": [False, False, False], "balanced": [False, True, True], "supportive": [True, True, True]}
        for prof, expect in want.items():
            for p in FEATURED:
                _, s = session(prof, p.id)
                out = play(s, p, p.msg_step(), p.msg_step(), p.msg_wrong(), p.msg_wrong(), p.msg_wrong())
                self.assertEqual([r.status for r in out[:2]], [out[0].status] * 2)
                self.assertTrue(out[0].status in ("step", "partial"), (p.id, out[0].status))
                self.assertEqual([r.intervened for r in out[2:]], expect, (prof, p.id))
                self.assertEqual([round(r.meta["pace_risk"], 3) for r in out[2:]], [0.152, 0.254, 0.465], (prof, p.id))
                self.assertTrue(all(r.reply.strip() for r in out[2:]))

    def test_never_silent_on_a_verified_wrong_step(self):
        for prof in P.PROFILE_ORDER:
            for p in FEATURED:
                t = DemoTutor(problems=FEATURED, pace_profile=prof)
                s = t.new_session(p.id)
                for first in ([], [p.msg_step(), p.msg_step()]):
                    s.reset()
                    play(s, p, *first)
                    for _ in range(3):
                        r = s.turn(p.msg_wrong())
                        self.assertEqual(r.status, "wrong", p.id)
                        self.assertTrue(r.reply.strip(), (prof, p.id))                  # something is always said
                        self.assertFalse(reveals_answer(r.reply, p.truth, [p.prompt, p.msg_wrong()]).leaked)
                        if not r.intervened:                                           # flagged: no hint, no move text
                            self.assertEqual((r.meta["rule_id"], r.meta["template"]), ("pace.wrong_below_tau", ""))

    def test_higher_tau_never_speaks_when_a_lower_one_stays_quiet(self):
        for p in FEATURED:
            said = {}
            for prof in P.PROFILE_ORDER:
                _, s = session(prof, p.id)
                said[prof] = [r.meta["pace_reason"] for r in play(s, p, p.msg_step(), p.msg_step(), *[p.msg_wrong()] * 5)]
            for i in range(7):
                loud = lambda prof: said[prof][i] in ("wrong_risk_ge_tau", "wrong_before_window", "wrong_after_window")   # noqa: E731
                if said["quiet"][i].startswith("wrong") and loud("quiet"):
                    self.assertTrue(loud("balanced"))
                if said["balanced"][i].startswith("wrong") and loud("balanced"):
                    self.assertTrue(loud("supportive"))

    def test_quiet_speaks_exactly_when_the_frozen_risk_reaches_its_bar(self):
        _, s = session("quiet")
        play(s, None, FEATURED[0].msg_step(), FEATURED[0].msg_step())
        bits, seen = [1, 1], []
        for _ in range(7):                                          # attempts 3..9
            r = s.turn(WRONG_ANSWERS["alg-01"])
            bits.append(0)
            self.assertEqual(r.intervened, reference_risk(bits) >= CFG.taus["quiet"], bits)
            seen.append(r.intervened)
        self.assertEqual(seen[:3], [False, False, False])
        self.assertTrue(seen[-1])

    def test_only_verdicts_update_the_tracker(self):
        calc = get("calc-01")
        _, s = session("balanced", "calc-01")
        for msg in ("I don't know", "", "asdf qwer zxcv", calc.msg_restate()):
            r = s.turn(msg)
            self.assertFalse(r.meta["pace_updated"], msg)
        self.assertEqual(s.pace.obs, [])
        s.turn(calc.msg_step())
        s.turn(calc.msg_wrong())
        self.assertEqual(s.pace.obs, [1, 0])

    def test_reset_clears_the_tracker_and_keeps_the_profile(self):
        _, s = session("supportive")
        s.turn(WRONG_ANSWERS["alg-01"])
        self.assertEqual(len(s.pace.obs), 1)
        s.reset()
        self.assertEqual((s.pace.obs, s.profile), ([], "supportive"))

    def test_profile_switch_applies_from_the_next_step(self):
        p = get("alg-01")
        _, s = session("quiet")
        play(s, p, p.msg_step(), p.msg_step())
        self.assertFalse(s.turn(p.msg_wrong()).intervened)                           # attempt 3: 0.152 < 0.57
        s.set_profile("supportive")
        r = s.turn(p.msg_wrong())                                                    # attempt 4: 0.254 >= 0.09
        self.assertTrue(r.intervened)
        self.assertEqual(r.meta["pace_profile"], "supportive")
        with self.assertRaises(ValueError):
            s.set_profile("clinical")

    def test_sessions_do_not_share_trackers(self):
        t = DemoTutor(problems=FEATURED)
        a, b = t.new_session("alg-01"), t.new_session("alg-01")
        a.turn(WRONG_ANSWERS["alg-01"])
        self.assertEqual(b.pace.obs, [])

    def test_the_leak_check_still_screens_the_flagged_notice(self):
        p = get("alg-01")
        t, s = session("quiet")
        play(s, p, p.msg_step(), p.msg_step())
        with mock.patch.object(PACE_T.T, "WATCHING", {**PACE_T.T.WATCHING, "flagged": ["The answer is x = 17."]}):
            r = s.turn(p.msg_wrong())
        self.assertFalse(r.intervened)
        self.assertNotIn("17", r.reply)
        self.assertEqual(r.reply, "Watching.")

    def test_pace_off_restores_the_old_behaviour(self):
        t = DemoTutor(problems=FEATURED, pace=False)
        s = t.new_session("alg-01")
        s.turn("x - 12 = 5")
        r = s.turn(WRONG_ANSWERS["alg-01"])
        self.assertTrue(r.intervened)
        self.assertNotIn("pace_risk", r.meta)


class TurnLogHasPaceFieldsButNoText(unittest.TestCase):
    def test_risk_tau_profile_logged_and_no_student_text(self):
        path = Path(tempfile.mkdtemp(prefix="pace_log_")) / "turns.jsonl"
        t = DemoTutor(problems=FEATURED, log_path=path, pace_profile="quiet")
        s = t.new_session("alg-01")
        secret = "zebra-moonlight-4471"
        for msg in ("I don't know", "x - 12 = 5", "x = 7 " + secret, "ugh " + secret):
            s.turn(msg)
        raw = path.read_text()
        self.assertNotIn(secret, raw)
        rows = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(len(rows), 4)
        for r in rows:
            self.assertEqual((r["pace_profile"], r["pace_tau"]), ("quiet", CFG.taus["quiet"]))
        self.assertIsNone(rows[0]["pace_risk"])                                    # no verified step yet
        self.assertEqual([r["pace_updated"] for r in rows], [False, True, True, False])
        self.assertAlmostEqual(rows[2]["pace_risk"], 0.1742, places=3)
        self.assertEqual(rows[2]["pace_reason"], "wrong_before_window")          # attempt 2: the old rule, so it speaks
        self.assertEqual(rows[2]["pace_n_obs"], 2)
        self.assertTrue(rows[2]["intervened"])


if __name__ == "__main__":
    unittest.main()
