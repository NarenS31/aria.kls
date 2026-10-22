"""Loader for the ASSISTments 2012-13 file (SPEC Amendment 3, A3.3). Only rows of students in `keep_ids` are ever kept in memory, so the locked test
students are never loaded unless the caller passes them (only the single C5 evaluation does).

Sequence = (user_id, assignment_id), MasterySection rows only, ordered by (start_time, problem_log_id); first-attempt correct = (correct >= 1);
KC = skill_id of the first opportunity as an integer string ("277.0" -> "277"), missing -> "UNK"; affect = the four detector columns, raw.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import first_mastery
from .loader import Sequence

AFF = ["Average_confidence(FRUSTRATED)", "Average_confidence(CONFUSED)", "Average_confidence(CONCENTRATING)", "Average_confidence(BORED)"]
USECOLS = ["problem_log_id", "user_id", "assignment_id", "start_time", "correct", "first_action", "type", "skill_id"] + AFF


@dataclass
class Seq12(Sequence):
    affect: np.ndarray = None          # (n, 4) raw detector outputs, order = AFF
    aff_rank: np.ndarray = None        # (n, 4) rank-transformed (set by affect.RankTransform)


def norm_kc(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "UNK"
    s = str(x)
    return s[:-2] if s.endswith(".0") else s


def load_2012(path, keep_ids, chunksize=1_000_000):
    keep = set(int(i) for i in keep_ids)
    parts = []
    for c in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize, low_memory=False):
        c = c[(c.type == "MasterySection") & c.user_id.isin(keep)]
        if len(c):
            parts.append(c)
    d = pd.concat(parts, ignore_index=True)
    assert set(d.user_id.unique().tolist()) <= keep, "a student outside keep_ids was loaded"
    d["_t"] = pd.to_datetime(d.start_time, errors="coerce")
    d = d.sort_values(["user_id", "assignment_id", "_t", "problem_log_id"], kind="stable").reset_index(drop=True)
    u, a = d.user_id.to_numpy(), d.assignment_id.to_numpy()
    starts = np.flatnonzero(np.r_[True, (u[1:] != u[:-1]) | (a[1:] != a[:-1])])
    ends = np.r_[starts[1:], len(d)]
    cb = (d.correct.to_numpy() >= 1).astype(np.int8)
    fa = d.first_action.to_numpy().astype(np.int8)
    aff = d[AFF].to_numpy(dtype=np.float32)
    sk = d.skill_id.to_numpy()
    seqs = []
    for s, e in zip(starts, ends):
        c = cb[s:e]
        seqs.append(Seq12(int(u[s]), int(a[s]), norm_kc(sk[s]), c, int(e - s), first_mastery(c), fa[s:e], aff[s:e]))
    return seqs
