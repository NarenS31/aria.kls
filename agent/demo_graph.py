"""Learning graph v0: a small, local, in-session concept map for the Teacher summary tab. EXPLORATORY, NOT VALIDATED.

* Concepts and prerequisite edges are hand-authored in data/demo_problems.json (per problem: `concepts`, `concept_edges`, `concept_labels`).
  Nothing is learned; the edges are for a person to review.
* One BKT estimate per concept, using the FROZEN pooled parameters that the live pace gate already loads (agent/demo_pace.py load_config().params;
  nothing is re-fitted). A concept is updated by every scored attempt (SymPy verdict correct / partial / step = right, wrong = wrong; restate and
  no-math turns are not scored) on a problem tagged with it. p_known is the filtered posterior after the last scored attempt.
* State of a concept (documented rule, evaluated in this order):
    unseen         no scored attempt and no hint on it yet
    secure         the last three scored attempts on it were all right (the project's frozen "three correct in a row" mastery rule)
    high_friction  not secure, and >= FRICTION_WRONG wrong attempts and >= FRICTION_HINTS hints on it
                   (hints = replies of type hint / task_chunking given on a problem tagged with the concept). Exploratory, not validated.
    working        anything else that has been touched
* Everything lives in memory for one session. No file, no network, and no student text is ever stored: only counts, 0/1 outcomes and concept ids.
  `clear()` (called by the session's Reset) empties it. The graph is read-only for the tutor: it never influences when ARIA speaks.
"""
from __future__ import annotations

from collections import deque

from agent import demo_pace as PACE

STATES = ("unseen", "working", "secure", "high_friction")
FRICTION_WRONG = 2
FRICTION_HINTS = 1
SECURE_RUN = 3
HINT_TYPES = ("hint", "task_chunking")
NOTE = "Estimated from this session only; exploratory, not validated."
RULE_TEXT = ("secure = last three scored attempts right; high-friction = not secure with at least two wrong attempts and one hint; "
             "working = touched but neither; unseen = no attempt or hint yet")


class ConceptCatalog:
    """Labels and prerequisite edges collected from the bank. Validated once at start-up (consistent labels, known endpoints, no cycles)."""

    def __init__(self, problems):
        self.labels: dict = {}
        self.edges: list = []
        self.tags: dict = {}                          # problem id -> [concept ids]
        seen = set()
        for p in problems:
            spec = getattr(p, "spec", {}) or {}
            tags = list(spec.get("concepts") or [])
            self.tags[p.id] = tags
            for cid, lab in (spec.get("concept_labels") or {}).items():
                if self.labels.setdefault(cid, lab) != lab:
                    raise ValueError(f"concept {cid!r} has two different labels in the bank")
            for a, b in spec.get("concept_edges") or []:
                if (a, b) not in seen:
                    seen.add((a, b))
                    self.edges.append((a, b))
        for cid in (c for t in self.tags.values() for c in t):
            if cid not in self.labels:
                raise ValueError(f"concept {cid!r} is tagged but has no label")
        for a, b in self.edges:
            if a not in self.labels or b not in self.labels:
                raise ValueError(f"edge {a!r} -> {b!r} has an endpoint without a label")
        self._check_acyclic()

    def _check_acyclic(self):
        succ: dict = {}
        for a, b in self.edges:
            succ.setdefault(a, []).append(b)
        state: dict = {}

        def visit(n):
            if state.get(n) == 1:
                raise ValueError(f"prerequisite cycle through {n!r}")
            if state.get(n) == 2:
                return
            state[n] = 1
            for m in succ.get(n, ()):
                visit(m)
            state[n] = 2
        for n in list(succ):
            visit(n)

    def prerequisites(self, cid: str) -> list:
        return [a for a, b in self.edges if b == cid]


def posterior(obs: list, par: dict) -> float:
    """BKT filtered P(known) after the observations (the same update as aria_pace.baselines.BKT._filter), frozen pooled parameters."""
    pk = par["prior"]
    for x in obs:
        e1 = (1 - par["slip"]) if x == 1 else par["slip"]
        e0 = par["guess"] if x == 1 else (1 - par["guess"])
        pk = pk * e1 / (pk * e1 + (1 - pk) * e0)
        post = pk
        pk = pk + (1 - pk) * par["learn"]
    return post if obs else par["prior"]


class LearningGraph:
    def __init__(self, catalog: ConceptCatalog, params: dict = None):
        self.catalog = catalog
        self.params = params if params is not None else PACE.load_config().params
        self.errors = 0                                # observe() failures swallowed by the session (never raised into a turn)
        self.clear()

    def clear(self) -> None:
        self._node: dict = {}                          # concept id -> {"obs": [0/1], "run": deque(maxlen=3), "wrong": int, "hints": int}

    def has_data(self) -> bool:
        return bool(self._node)

    def _rec(self, cid: str) -> dict:
        return self._node.setdefault(cid, {"obs": [], "run": deque(maxlen=SECURE_RUN), "wrong": 0, "hints": 0})

    def observe(self, problem, status: str, hint_given: bool) -> None:
        """One tutor turn on `problem`. status = SymPy verdict string; hint_given = the tutor replied with a hint this turn."""
        tags = self.catalog.tags.get(problem.id) or []
        right, wrong = status in PACE.RIGHT, status in PACE.WRONG
        for cid in tags:
            if not (right or wrong or hint_given):
                continue
            r = self._rec(cid)
            if right or wrong:
                r["obs"].append(1 if right else 0)
                r["run"].append(1 if right else 0)
                r["wrong"] += int(wrong)
            r["hints"] += int(bool(hint_given))

    def state_of(self, cid: str) -> str:
        r = self._node.get(cid)
        if r is None:
            return "unseen"
        if len(r["run"]) == SECURE_RUN and all(r["run"]):
            return "secure"
        if r["wrong"] >= FRICTION_WRONG and r["hints"] >= FRICTION_HINTS:
            return "high_friction"
        return "working"

    def snapshot(self) -> dict:
        """Concept ids, authored labels, states and counts only. Touched concepts plus their direct prerequisites, with the edges between them."""
        touched = [c for c in self.catalog.labels if c in self._node]
        shown = list(dict.fromkeys(touched + [a for c in touched for a in self.catalog.prerequisites(c)]))
        nodes = []
        for cid in shown:
            r = self._node.get(cid)
            nodes.append({"id": cid, "label": self.catalog.labels[cid], "state": self.state_of(cid),
                          "p_known": None if r is None or not r["obs"] else round(posterior(r["obs"], self.params), 3),
                          "attempts": 0 if r is None else len(r["obs"]), "wrong": 0 if r is None else r["wrong"],
                          "hints": 0 if r is None else r["hints"]})
        edges = [[a, b] for a, b in self.catalog.edges if a in shown and b in shown]
        return {"nodes": nodes, "edges": edges, "note": NOTE, "rule": RULE_TEXT}
