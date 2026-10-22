"""Learning graph v0: bank tags, BKT update with the frozen pooled parameters, state rules, no stored text, Reset, and read-only for the tutor."""
import importlib.util
import inspect
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from agent import demo_graph as G  # noqa: E402
from agent import demo_pace as PACE  # noqa: E402
from agent.demo_bank import load_problems  # noqa: E402
from agent.demo_summary import summarize  # noqa: E402
from agent.demo_tutor import DemoTutor  # noqa: E402

PROBS = load_problems()
BY_ID = {p.id: p for p in PROBS}
PAR = PACE.load_config().params


def load_ui():
    spec = importlib.util.spec_from_file_location("aria_demo_app_graph", ROOT / "ui" / "demo_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeProblem:
    def __init__(self, pid="calc-01"):
        self.id = pid


class Bank(unittest.TestCase):
    def test_every_problem_is_tagged_and_the_catalog_is_valid(self):
        cat = G.ConceptCatalog(PROBS)                                     # raises on inconsistent labels / unknown endpoints / cycles
        self.assertEqual(len(PROBS), 30)
        for p in PROBS:
            self.assertTrue(cat.tags[p.id], p.id)                         # featured ones are required; all 30 are tagged
        self.assertTrue(all(p.featured and cat.tags[p.id] for p in PROBS if p.featured))
        self.assertGreaterEqual(len(cat.edges), 15)

    def test_catalog_rejects_bad_data(self):
        class P:
            def __init__(self, pid, spec):
                self.id, self.spec = pid, spec
        ok = {"concepts": ["a"], "concept_labels": {"a": "A", "b": "B"}, "concept_edges": [["b", "a"]]}
        G.ConceptCatalog([P("x", ok)])
        with self.assertRaises(ValueError):
            G.ConceptCatalog([P("x", ok), P("y", {"concepts": ["a"], "concept_labels": {"a": "Other"}})])           # same id, two labels
        with self.assertRaises(ValueError):
            G.ConceptCatalog([P("x", {"concepts": ["a"], "concept_labels": {"a": "A"}, "concept_edges": [["z", "a"]]})])  # unknown endpoint
        with self.assertRaises(ValueError):
            G.ConceptCatalog([P("x", {"concepts": ["a"], "concept_labels": {"a": "A", "b": "B"}, "concept_edges": [["a", "b"], ["b", "a"]]})])  # cycle
        with self.assertRaises(ValueError):
            G.ConceptCatalog([P("x", {"concepts": ["q"], "concept_labels": {"a": "A"}})])                          # tagged but unlabelled

    def test_edges_are_hand_authored_not_learned(self):
        d = json.loads((ROOT / "data" / "demo_problems.json").read_text())
        self.assertTrue(all("concept_edges" in p and "concepts" in p for p in d))


class Update(unittest.TestCase):
    def setUp(self):
        self.g = G.LearningGraph(G.ConceptCatalog(PROBS), PAR)
        self.p = BY_ID["calc-01"]

    def test_posterior_equals_aria_pace_bkt_filter(self):
        from aria_pace import baselines as B
        for obs in ([1], [0], [1, 0, 1], [0, 0, 1, 1, 0], [1, 1, 1, 1]):
            post, _ = B.BKT._filter(np.array(obs, dtype=np.int8), PAR)
            self.assertAlmostEqual(G.posterior(obs, PAR), float(post[-1]), places=12)
        self.assertEqual(G.posterior([], PAR), PAR["prior"])

    def test_scored_attempts_update_every_tagged_concept_only(self):
        self.g.observe(self.p, "wrong", False)
        snap = {n["id"]: n for n in self.g.snapshot()["nodes"]}
        for cid in BY_ID["calc-01"].spec["concepts"]:
            self.assertEqual(snap[cid]["attempts"], 1)
        self.assertEqual(snap["calc.exponent_laws"]["state"], "unseen")             # a prerequisite that was never tagged on an attempted problem
        self.g.observe(BY_ID["alg-01"], "correct", False)
        self.assertNotIn("alg.linear_eq", {n["id"] for n in self.g.snapshot()["nodes"]} - {"alg.distribute", "alg.like_terms", "alg.inverse_ops", "alg.linear_eq"})

    def test_unscored_turns_do_not_update(self):
        for st in ("none", "restate"):
            self.g.observe(self.p, st, False)
        self.assertFalse(self.g.has_data())
        self.g.observe(self.p, "none", True)                                        # a hint after "I don't know" is on the node, not an attempt
        n = self.g.snapshot()["nodes"][0]
        self.assertEqual((n["attempts"], n["hints"], n["p_known"], n["state"]), (0, 1, None, "working"))

    def test_right_verdicts_are_correct_partial_step(self):
        for st in ("correct", "partial", "step"):
            self.g.observe(self.p, st, False)
        self.assertEqual({n["state"] for n in self.g.snapshot()["nodes"] if n["attempts"]}, {"secure"})   # three right in a row

    def test_state_rules(self):
        cid = "calc.power_rule"
        self.assertEqual(self.g.state_of(cid), "unseen")
        self.g.observe(self.p, "wrong", False)
        self.assertEqual(self.g.state_of(cid), "working")
        self.g.observe(self.p, "wrong", False)
        self.assertEqual(self.g.state_of(cid), "working")                           # two wrong but no hint yet
        self.g.observe(self.p, "wrong", True)
        self.assertEqual(self.g.state_of(cid), "high_friction")                     # >= 2 wrong and >= 1 hint, not secure
        for _ in range(3):
            self.g.observe(self.p, "step", False)
        self.assertEqual(self.g.state_of(cid), "secure")                            # secure beats friction once three in a row are right
        self.g.observe(self.p, "wrong", False)
        self.assertEqual(self.g.state_of(cid), "high_friction")

    def test_p_known_moves_the_right_way(self):
        self.g.observe(self.p, "wrong", False)
        low = {n["id"]: n["p_known"] for n in self.g.snapshot()["nodes"]}["calc.power_rule"]
        for _ in range(3):
            self.g.observe(self.p, "correct", False)
        high = {n["id"]: n["p_known"] for n in self.g.snapshot()["nodes"]}["calc.power_rule"]
        self.assertGreater(high, low)

    def test_thresholds_are_documented_constants(self):
        self.assertEqual((G.FRICTION_WRONG, G.FRICTION_HINTS, G.SECURE_RUN), (2, 1, 3))
        self.assertIn("exploratory, not validated", G.NOTE)


class NoStudentText(unittest.TestCase):
    SECRET = "zebra-moonlight-4471"

    def test_nothing_the_student_typed_is_stored(self):
        t = DemoTutor(problems=PROBS)
        s = t.new_session("alg-01")
        for m in (f"I don't know {self.SECRET}", f"x - 12 = 5 {self.SECRET}", f"x = 7 {self.SECRET}", f"ugh {self.SECRET} this is hard", "x = 17"):
            s.turn(m)
        self.assertTrue(s.graph.has_data())

        def walk(o, depth=0):
            if depth > 8:
                return
            if isinstance(o, (str, bytes)):
                yield str(o)
            elif isinstance(o, dict):
                for k, v in o.items():
                    yield from walk(k, depth + 1); yield from walk(v, depth + 1)
            elif isinstance(o, (list, tuple, set)) or o.__class__.__name__ == "deque":
                for v in o:
                    yield from walk(v, depth + 1)
            elif hasattr(o, "__dict__"):
                yield from walk(vars(o), depth + 1)
        blob = " | ".join(walk(s.graph))
        self.assertNotIn(self.SECRET, blob)
        self.assertNotIn(self.SECRET, json.dumps(s.graph.snapshot()))
        for word in ("x - 12", "x = 7", "don't know", "ugh"):
            self.assertNotIn(word, blob)
        ui = load_ui()
        self.assertNotIn(self.SECRET, ui.summary_html(summarize(s)))

    def test_only_numbers_ids_and_authored_labels_are_in_the_state(self):
        t = DemoTutor(problems=PROBS)
        s = t.new_session("calc-02")
        s.turn("My answer is 2x cos(2x) + 2x^2 sin(2x)")
        for rec in s.graph._node.values():
            self.assertEqual(set(rec), {"obs", "run", "wrong", "hints"})
            self.assertTrue(all(isinstance(x, int) for x in rec["obs"]) and isinstance(rec["wrong"], int))

    def test_module_has_no_file_or_network_code(self):
        src = inspect.getsource(G)
        code = re.sub(r'""".*?"""', "", src, flags=re.S)
        for bad in ("open(", "socket", "requests", "urllib", "pickle", "json.dump", "write_text", "sqlite", "http"):
            self.assertNotIn(bad, code, bad)

    def test_turn_log_fields_are_unchanged(self):
        t = DemoTutor(problems=PROBS)
        meta = t.new_session("alg-01").turn("x = 7").meta
        self.assertFalse([k for k in meta if "concept" in k or "graph" in k])


class ResetAndReadOnly(unittest.TestCase):
    def test_reset_clears_the_graph(self):
        t = DemoTutor(problems=PROBS)
        s = t.new_session("alg-01")
        s.turn("x = 7")
        self.assertTrue(s.graph.has_data())
        s.reset()
        self.assertFalse(s.graph.has_data())
        self.assertIsNone(summarize(s)["concept_map"])

    def test_graph_handed_over_when_switching_problem_and_cleared_by_reset(self):
        t = DemoTutor(problems=PROBS)
        a = t.new_session("calc-01")
        a.turn("My answer is 12x^3 - 10x - 9")
        b = t.new_session("calc-06", a.graph)
        self.assertIs(b.graph, a.graph)
        self.assertTrue(b.graph.has_data())
        b.reset()
        self.assertFalse(a.graph.has_data())
        self.assertIsNot(t.new_session("calc-01").graph, a.graph)                     # a fresh session without a hand-over starts empty

    def test_the_graph_cannot_change_what_the_tutor_says(self):
        msgs = ["I don't know", "12x^3 - 10x", "My answer is 12x^3 - 10x - 9", "My answer is 12x^3 - 10x - 9", "My answer is 12x^3 - 10x + 7"]

        def run(break_graph):
            t = DemoTutor(problems=PROBS, pace_profile="supportive")
            s = t.new_session("calc-01")
            if break_graph:
                def boom(*a, **k):
                    raise RuntimeError("graph failure")
                s.graph.observe = boom
            out = [(r.reply, r.intervened, r.state, r.status, r.meta["rule_id"], r.meta["pace_reason"]) for r in (s.turn(m) for m in msgs)]
            return out, s.graph.errors
        base, e0 = run(False)
        broken, e1 = run(True)
        self.assertEqual(base, broken)                                                # identical replies and decisions
        self.assertEqual((e0, e1), (0, len(msgs)))                                    # failures are counted, never raised

    def test_ui_start_keeps_the_graph_across_problem_switch_and_reset_clears_it(self):
        ui = load_ui()
        t = DemoTutor(problems=PROBS)
        start, send, reset = ui.make_handlers(t, list(t.problems.values()))
        session, hist, *_ = start(BY_ID["calc-01"].prompt)
        hist, session, *_ = send("My answer is 12x^3 - 10x - 9", hist, session)
        g = session.graph
        session2, *_ = start(BY_ID["calc-06"].prompt, None, session)
        self.assertIs(session2.graph, g)
        self.assertTrue(g.has_data())
        reset(session2)
        self.assertFalse(g.has_data())
        session3, *_ = start(BY_ID["calc-01"].prompt)                                 # no previous session: empty graph
        self.assertFalse(session3.graph.has_data())


class Rendering(unittest.TestCase):
    def test_map_appears_only_in_the_teacher_summary_with_the_exploratory_line(self):
        ui = load_ui()
        t = DemoTutor(problems=PROBS)
        s = t.new_session("alg-05")
        s.turn("I don't know")
        s.turn("(x - 2)(x - 3) = 0")
        page = ui.summary_html(summarize(s))
        self.assertIn("Concept map", page)
        self.assertIn("Estimated from this session only; exploratory, not validated.", page)
        self.assertIn("Factoring quadratics", page)
        self.assertNotIn("Concept map", ui.chat_html([{"role": "aria", "content": "hi"}]))
        self.assertNotIn("Concept map", ui.badge_html("idle", "x"))
        self.assertNotIn("Concept map", ui.summary_html(summarize(t.new_session("alg-01"))))        # nothing scored yet: no map


if __name__ == "__main__":
    unittest.main()
