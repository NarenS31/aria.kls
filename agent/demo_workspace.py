"""Session-only workspace events. Revisions replay an active branch, never rewrite research logs."""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass

from agent.demo_input import sanitize_input
from agent.demo_summary import summarize


@dataclass
class WorkEvent:
    kind: str
    text: str
    profile: str
    result: object


class Workspace:
    def __init__(self, tutor, problem_id=None):
        self.tutor = tutor
        self.session = tutor.new_session(problem_id or next(iter(tutor.problems)))
        self.events = []
        self.branches = []
        self.archive = []
        self.revision = 0
        self.notice = ""
        self.reflection = ""
        self.base_graph = copy.deepcopy(self.session.graph)
        self.lock = threading.RLock()

    def __deepcopy__(self, memo):
        # Gradio creates this object on load, not as a shared initial State value.
        return self

    @property
    def solved(self):
        return any(e.kind == "step" and e.result.accepted for e in self.events)

    def _replay(self):
        session = self.tutor.new_session(self.session.problem.id, graph=copy.deepcopy(self.base_graph))
        session.replaying = True
        self.events = copy.deepcopy(self.events)
        for e in self.events:
            session.set_profile(e.profile)
            e.result = session.request_hint() if e.kind == "hint" else session.turn(e.text)
            if e.result.meta.get("fallback"):
                raise RuntimeError("Could not replay the active work")
        session.set_profile(self.session.profile)
        session.replaying = False
        self.session = session

    def _checkpoint(self):
        session = copy.copy(self.session)
        for key, value in self.session.__dict__.items():
            if key not in ("tutor", "problem"):
                setattr(session, key, copy.deepcopy(value))
        return session

    def submit(self, text, expected_revision, edit=None):
        with self.lock:
            if expected_revision != self.revision:
                return False
            clean, flags = sanitize_input(text)
            if not clean.strip():
                self.notice = "Write a math step before checking it."
                return False
            if flags:
                self.notice = "That input contains unsupported characters or is too long. Please shorten or revise it."
                return False
            if self.solved and edit is None:
                self.notice = "This problem is complete. Choose the next problem or revise an earlier step."
                return False
            original = list(self.events)
            checkpoint = self._checkpoint()
            if edit is not None:
                if edit < 0 or edit >= len(self.events) or self.events[edit].kind != "step":
                    self.notice = "Choose a submitted step to revise."
                    return False
                self.events = self.events[:edit]
                try:
                    self._replay()
                except RuntimeError:
                    self.events, self.session = original, checkpoint
                    self.notice = "The revision could not be checked. Your input is still here; try again."
                    return False
            result = self.session.turn(clean)
            if result.meta.get("fallback"):
                self.events = original
                self.session = checkpoint
                self.notice = "The math check did not finish. Your input is still here; try again."
                return False
            if edit is not None:
                self.branches.append({"replaced_at": edit, "events": original, "revision": self.revision})
                self.reflection = ""
            self.events.append(WorkEvent("step", clean, self.session.profile, result))
            self.revision += 1
            self.notice = "Earlier work was revised; later steps were removed from this branch." if edit is not None else ""
            return True

    def hint(self, expected_revision):
        with self.lock:
            if expected_revision != self.revision or self.solved:
                return False
            move = self.session._next_move()
            idx = next((i for i, m in enumerate(self.session.problem.moves) if m is move), -1)
            available = 1 + len((move or {}).get("alternatives", []))
            if move is None or self.session.move_hints.get(idx, 0) >= available:
                self.notice = "There are no more authored hints for this step. Try a revised step or ask a teacher."
                return False
            checkpoint = self._checkpoint()
            result = self.session.request_hint()
            if result.meta.get("fallback"):
                self.session = checkpoint
                self.notice = "The hint could not be prepared. Please try again."
                return False
            self.events.append(WorkEvent("hint", "", self.session.profile, result))
            self.revision += 1
            self.notice = ""
            return True

    def change_problem(self, problem_id, retry=False):
        with self.lock:
            if self.events:
                self.archive.append({"problem": self.session.problem.id, "events": self.events,
                                     "branches": self.branches, "summary": summarize(self.session), "reflection": self.reflection})
            graph = self.session.graph
            profile = self.session.profile
            self.session = self.tutor.new_session(problem_id, graph=graph)
            self.session.set_profile(profile)
            self.base_graph = copy.deepcopy(graph)
            self.events, self.branches = [], []
            self.revision += 1
            self.reflection = ""
            self.notice = "A fresh attempt is ready. Your earlier work remains in session history." if retry else ""

    def clear(self):
        with self.lock:
            pid, profile = self.session.problem.id, self.session.profile
            self.session = self.tutor.new_session(pid)
            self.session.set_profile(profile)
            self.events, self.branches, self.archive = [], [], []
            self.base_graph = copy.deepcopy(self.session.graph)
            self.revision += 1
            self.reflection = ""
            self.notice = "Session cleared."

    def summary(self):
        result = summarize(self.session)
        result["revision_count"] = len(self.branches)
        result["previous_attempts"] = len(self.archive)
        return result
