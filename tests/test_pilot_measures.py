"""scripts/pilot_measures.py on a synthetic log with hand-computed expected values (no real data)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_measures as PM  # noqa: E402


def row(sid, attempt, pid, turn, verdict, hint=False, level=0, t=1.0, itype=None, intervened=None):
    iv = hint if intervened is None else intervened
    return {"session_id": sid, "attempt": attempt, "problem_id": pid, "turn": turn, "correctness": verdict, "intervened": iv,
            "intervention_type": itype or ("hint" if hint else "none"), "hint_level": level, "t_s": t}


# A (calc-01): IDK+hint, partial, wrong+hint, wrong+hint, correct+reflection   B (alg-01): wrong+hint, wrong (flagged), wrong+hint   C (calc-01, other session): step, correct
SYNTH = [
    row("A", 1, "calc-01", 1, "none", True, 1, 2.0), row("A", 1, "calc-01", 2, "partial", False, 1, 5.0), row("A", 1, "calc-01", 3, "wrong", True, 2, 8.0),
    row("A", 1, "calc-01", 4, "wrong", True, 3, 12.0), row("A", 1, "calc-01", 5, "correct", False, 3, 15.0, itype="reflection", intervened=True),
    row("B", 1, "alg-01", 1, "wrong", True, 1, 4.0), row("B", 1, "alg-01", 2, "wrong", False, 1, 7.0), row("B", 1, "alg-01", 3, "wrong", True, 2, 9.0),
    row("C", 1, "calc-01", 1, "step", False, 0, 3.0), row("C", 1, "calc-01", 2, "correct", False, 0, 6.0, itype="reflection", intervened=True),
]
BANK = {"calc-01": {"concepts": ["calc.power_rule"]}, "alg-01": {"concepts": ["alg.linear_eq"]}}


class Measures(unittest.TestCase):
    def setUp(self):
        self.m = PM.measures(SYNTH, BANK)

    def test_counts(self):
        self.assertEqual((self.m["turns"], self.m["attempts"]), (10, 3))

    def test_a_correction_after_a_hint(self):
        a = self.m["a_correction_after_hint"]
        # hint turns: A1 (next scored: A2 partial = right), A3 (A4 wrong), A4 (A5 correct = right), B1 (B2 wrong), B3 (no later scored attempt)
        self.assertEqual((a["hints"], a["with_a_later_scored_attempt"], a["next_attempt_right"]), (5, 4, 2))
        self.assertAlmostEqual(a["share_right"], 0.5)
        f = a["first_hint_only"]                                              # A1 -> right, B1 -> wrong
        self.assertEqual((f["hints"], f["with_a_later_scored_attempt"], f["next_attempt_right"]), (2, 2, 1))

    def test_b_hints_and_hint_level(self):
        b = self.m["b_hints"]
        self.assertEqual(b["attempts_with_a_hint"], 2)
        self.assertAlmostEqual(b["hints_per_attempt_mean"], 5 / 3)
        self.assertAlmostEqual(b["max_hint_level_mean"], (3 + 2 + 0) / 3)
        self.assertEqual(b["max_hint_level_max"], 3)
        self.assertEqual(b["per_problem"]["calc-01"]["attempts"], 2)
        self.assertEqual(b["per_problem"]["calc-01"]["hints"], 3)
        self.assertIn("not a three-level engine", b["note"])

    def test_c_time_before_first_hint(self):
        c = self.m["c_time_before_first_hint_s"]
        self.assertEqual(c["attempts_with_a_hint"], 2)
        self.assertAlmostEqual(c["median"], 3.0)                              # first hints at t_s 2.0 and 4.0
        self.assertEqual(c["scored_attempts_before_first_hint_mean"], 0)
        self.assertIn("proxy", c["label"])

    def test_d_friction_index(self):
        d = self.m["d_friction_index"]
        self.assertEqual((d["attempts"], d["flagged_high_friction"]), (3, 2))   # A and B: >= 2 wrong, >= 1 hint, not secure; C: no wrong
        self.assertEqual(d["per_concept"]["calc.power_rule"], {"attempts": 2, "flagged": 1, "share": 0.5})
        self.assertIn("exploratory", d["label"])

    def test_secure_attempt_is_not_flagged(self):
        turns = [row("S", 1, "p", i, v, h) for i, (v, h) in enumerate([("wrong", True), ("wrong", True), ("step", False), ("step", False), ("correct", False)], 1)]
        self.assertFalse(PM.friction_flag(turns))                               # three right in a row at the end = secure

    def test_empty_and_unreadable_log(self):
        self.assertEqual(PM.measures([])["attempts"], 0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "turns.jsonl"
            p.write_text("not json\n\n" + json.dumps({"x": 1}) + "\n" + json.dumps(SYNTH[0]) + "\n")
            self.assertEqual(len(PM.load_rows(p)), 1)


class Script(unittest.TestCase):
    def test_prints_a_table_from_a_log_file_and_writes_json_only_on_request(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "turns.jsonl"
            log.write_text("\n".join(json.dumps(r) for r in SYNTH) + "\n")
            out = Path(d) / "m.json"
            res = subprocess.run([sys.executable, str(ROOT / "scripts" / "pilot_measures.py"), "--log", str(log), "--json", str(out)],
                                 capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(res.returncode, 0, res.stderr)
            for must in ("(a) Correction after a hint", "(b) Hints per problem", "NOT a three-level engine", "PROXY", "EXPLORATORY", "calc-01", "Solving linear equations"):
                self.assertIn(must, res.stdout)
            self.assertEqual(json.loads(out.read_text())["attempts"], 3)
            self.assertFalse(list(Path(d).glob("*.csv")))
            missing = subprocess.run([sys.executable, str(ROOT / "scripts" / "pilot_measures.py"), "--log", str(Path(d) / "nope.jsonl")], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(missing.returncode, 1)


if __name__ == "__main__":
    unittest.main()
