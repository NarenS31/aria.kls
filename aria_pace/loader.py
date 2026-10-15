"""Load the ASSISTments 2009-10 skill-builder file and build per-student problem-set sequences (SPEC 1)."""
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import EXPECTED_BYTES, EXPECTED_SHA256, RAW_CSV
from .labels import first_mastery

USECOLS = ["user_id", "assignment_id", "order_id", "skill_id", "correct", "first_action", "hint_count"]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path=RAW_CSV, verify_checksum=True):
    """Return (clean rows, data_report). Cleaning order is fixed by SPEC 1 and every step is counted."""
    if verify_checksum:
        size = path.stat().st_size
        if size != EXPECTED_BYTES or sha256_file(path) != EXPECTED_SHA256:
            raise RuntimeError(f"{path} does not match the checksum in SPEC.md; see docs/aria_pace/DATA.md")
    df = pd.read_csv(path, encoding="latin-1", usecols=USECOLS, low_memory=False)
    rep = {"rows_in_file": int(len(df)), "students_in_file": int(df.user_id.nunique())}
    null_skill = df.skill_id.isna()
    rep["rows_dropped_null_skill"] = int(null_skill.sum())
    rep["pct_rows_dropped_null_skill"] = round(100 * null_skill.mean(), 2)
    df = df.loc[~null_skill].copy()
    if df.duplicated(["user_id", "order_id"]).any():
        raise RuntimeError("(user_id, order_id) not unique")
    if not set(df.correct.unique()) <= {0, 1}:
        raise RuntimeError("correct has values outside {0,1}")
    df["skill_id"] = df.skill_id.astype(str)
    df = df.sort_values(["user_id", "order_id"], kind="stable").reset_index(drop=True)
    rep["rows_after_cleaning"] = int(len(df))
    rep["students_after_cleaning"] = int(df.user_id.nunique())
    # Descriptive only (first_action meanings are inferred, not documented; never used by a model).
    err = df.correct == 0
    rep["first_attempt_wrong_rows"] = int(err.sum())
    rep["first_attempt_wrong_by_first_action_code"] = {
        str(k): int(v) for k, v in df.loc[err, "first_action"].value_counts().sort_index().items()
    }
    return df, rep


@dataclass
class Sequence:
    user_id: int
    assignment_id: int
    kc: str            # skill_id string of the first opportunity
    correct: np.ndarray  # int8, ordered by order_id; index 0 is opportunity 1
    n: int
    m: float           # first mastery opportunity (1-based), np.inf if none
    # Inferred (undocumented) codes: 0 answer attempt, 1 hint request, 2 other. Used ONLY to define the
    # primary vs hint-first-inclusive verified error (SPEC A1.3); never by a risk model.
    first_action: np.ndarray = None


def build_sequences(df):
    """One Sequence per (user_id, assignment_id), opportunities recomputed as rank by order_id."""
    seqs = []
    for (u, a), g in df.groupby(["user_id", "assignment_id"], sort=False):
        g = g.sort_values("order_id", kind="stable")
        c = g.correct.to_numpy(dtype=np.int8)
        seqs.append(Sequence(int(u), int(a), str(g.skill_id.iloc[0]), c, len(c), first_mastery(c),
                             g.first_action.to_numpy(dtype=np.int8)))
    return seqs
