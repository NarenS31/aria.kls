"""The harness must be able to *see* every failure class it claims to measure."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import demo_harness as H  # noqa: E402
from agent.demo_bank import load_problems  # noqa: E402

PROBS = load_problems()[:6] + load_problems()[20:24]      # algebra + trig + calculus sample


def summary(kind):
    res = H.run(H.make_backend(f"selftest-{kind}"), PROBS, progress=False)
    return H.summarize(res), H.targets_met(H.summarize(res)), res


class HarnessSelfCheck(unittest.TestCase):
    def test_oracle_passes_everything(self):
        s, t, _ = summary("oracle")
        self.assertTrue(all(t.values()), s)
        self.assertEqual(s["quiet_ok"], s["quiet_n"])
        self.assertEqual(s["intervened_ok"], s["intervened_n"])

    def test_leaks_are_seen(self):
        s, t, _ = summary("leaker")
        self.assertFalse(t["leaks==0"])
        self.assertGreater(s["leaks"], len(PROBS))

    def test_lies_are_seen(self):
        s, t, _ = summary("liar")
        self.assertFalse(t["math_errors==0"])

    def test_crashes_are_seen(self):
        s, t, res = summary("crasher")
        self.assertFalse(t["crashes==0"])
        self.assertEqual(s["crashes"], s["turns"])

    def test_scaffold_every_turn_is_not_quiet(self):
        s, _, _ = summary("chatty")
        self.assertEqual(s["quiet_ok"], 0)

    def test_every_problem_gets_the_scripted_turns(self):
        _, _, res = summary("oracle")
        ids = {r.turn for r in res}
        self.assertTrue({"T1", "T2", "T3", "T4", "T5", "T6", "P1"} <= ids)
        self.assertTrue(all(r.sympy_status == "correct" for r in res if r.turn in ("T1", "T6")))
        self.assertTrue(all(r.sympy_status == "wrong" for r in res if r.turn == "T2"))

    def test_silence_on_a_verified_wrong_step_is_seen(self):
        res = H.run(H.make_backend("selftest-mute"), PROBS, progress=False)
        s = H.summarize(res)
        self.assertGreater(s["silent_wrong"], 0)
        self.assertFalse(H.targets_met(s, pace=True)["wrong_never_silent"])
        s_ok = H.summarize(H.run(H.make_backend("selftest-oracle"), PROBS, progress=False))
        self.assertEqual(s_ok["silent_wrong"], 0)
        self.assertTrue(all(H.targets_met(s_ok, pace=True).values()))

    def test_profiles_all_runs_three_backends_with_flows_and_strict_exit(self):
        rc = H.main(["--backend", "demo", "--profiles", "all", "--featured", "--limit", "2", "--strict"])
        self.assertEqual(rc, 0)
        be = H.make_backend("demo@quiet")
        res = H.run(be, load_problems(featured_only=True)[:1], progress=False)
        self.assertEqual({r.turn for r in res} & {f"S{i}" for i in range(1, 8)}, {f"S{i}" for i in range(1, 8)})
        self.assertEqual(be.tutor.pace_profile, "quiet")
        self.assertTrue(all(r.pace_tau == be.tutor.pace_cfg.taus["quiet"] for r in res if r.turn.startswith("S")))
        flagged = [r for r in res if r.turn in ("S4", "S5", "S6")]                 # quiet: attempts 3-5 stay below its bar
        self.assertTrue(all(not r.intervened and r.reply.strip() for r in flagged))
        self.assertEqual(H.main(["--backend", "selftest-leaker", "--limit", "1", "--strict"]), 1)

    def test_percentiles(self):
        self.assertAlmostEqual(H.pct([1, 2, 3, 4, 100], 0.5), 3)
        self.assertGreater(H.pct(list(range(100)), 0.95), 93)


if __name__ == "__main__":
    unittest.main()
