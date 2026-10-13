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

    def test_percentiles(self):
        self.assertAlmostEqual(H.pct([1, 2, 3, 4, 100], 0.5), 3)
        self.assertGreater(H.pct(list(range(100)), 0.95), 93)


if __name__ == "__main__":
    unittest.main()
