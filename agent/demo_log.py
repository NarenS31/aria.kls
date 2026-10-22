"""Local per-turn log for the summit demo. One JSON line per tutor turn; NO student text, NO tutor text.

Only whitelisted fields are written, numbers as numbers, and every string must be a short identifier
(letters, digits, `_ . : -`, at most 48 characters), so free text cannot reach the file even by mistake.
Writing never raises: a logging failure must not cost the student a reply.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_logs" / "turns.jsonl"
MAX_BYTES = 5_000_000

_IDENT = re.compile(r"^[A-Za-z0-9_.:\-]{0,48}$")
FIELDS = {
    "session_id": str, "attempt": int, "problem_id": str, "turn": int, "latency_ms": float, "correctness": str,
    "derived_state": str, "kind": str, "analyzer_state": str, "analyzer_confidence": float, "rule_id": str,
    "intervention_type": str, "template": str, "intervened": bool, "llm_used": bool, "llm_outcome": str,
    "fallback": str, "move_index": int, "t_s": float, "hint_level": int, "student_chars": int, "reply_chars": int, "input_flags": list,
    # ARIA-Pace (numbers and identifiers only): calibrated BKT risk, the profile's threshold, the profile key
    "pace_profile": str, "pace_tau": float, "pace_risk": float, "pace_n_obs": int, "pace_updated": bool, "pace_reason": str,
}


def _clean(key: str, val):
    typ = FIELDS[key]
    if val is None:
        return None
    if typ is bool:
        return bool(val)
    if typ is int:
        return int(val)
    if typ is float:
        return round(float(val), 3)
    if typ is list:
        return [v for v in val if isinstance(v, str) and _IDENT.match(v)][:12]
    return val if isinstance(val, str) and _IDENT.match(val) else "invalid"


class TurnLog:
    def __init__(self, path: Optional[Path | str] = DEFAULT_PATH):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.errors = 0

    def write(self, record: dict) -> bool:
        if self.path is None:
            return False
        try:
            row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            row.update({k: _clean(k, v) for k, v in record.items() if k in FIELDS})
            line = json.dumps(row, ensure_ascii=True)
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists() and self.path.stat().st_size > MAX_BYTES:
                    self.path.replace(self.path.with_suffix(".jsonl.1"))
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            return True
        except Exception:  # noqa: BLE001
            self.errors += 1
            return False
