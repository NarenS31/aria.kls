"""E7: Quiet misses (CV held-out) split by in-system help before try 10 (exploratory; ANALYSIS_PLAN.md Addendum 1)."""
import time

import numpy as np

import common as C
import cv_heldout as CVH
from aria_pace import stage5 as S5
from e5_logged_hints import hint_arrays

TAU = C.FROZEN_TAUS["quiet"]
SEED = 20261116


def in_system_help(H, harr):
    """(N,) bool: hint_count > 0 or first_action == 1 at any try 1-9."""
    out = np.zeros(len(H.seqs), bool)
    hc_any = np.zeros(len(H.seqs), bool)
    hf_any = np.zeros(len(H.seqs), bool)
    for r, q in enumerate(H.seqs):
        h = harr[(q.user_id, q.assignment_id)]
        assert np.array_equal(h["correct"], q.correct), "hint columns misaligned"
        k = min(9, q.n)
        hc_any[r] = (h["hint_count"][:k] > 0).any()
        hf_any[r] = (q.first_action[:k] == 1).any()
    out = hc_any | hf_any
    return out, hc_any, hf_any


def main():
    t0 = time.time()
    log = C.log_fn()
    df, parts, train, val, _ = C.load_pool(log)
    d = CVH.get(train, val, log)
    H = d["H"]
    C.assert_no_test(H.sid, parts)
    err = H.err["primary"]
    help_, hc_any, hf_any = in_system_help(H, hint_arrays(df))
    Sx = np.nan_to_num(d["cal"]["bkt"], nan=-1.0)
    fired = ((Sx >= TAU) & H.valid & err).any(1)
    opp = (H.valid & err).any(1)
    stuck = (H.y == 1) & opp
    missed = stuck & ~fired
    caught = stuck & fired
    vec = np.column_stack([stuck, missed, missed & ~help_, stuck & ~help_]).astype(float)
    agg = S5.Agg(H.sid)
    A = agg.by_student(vec)
    cnt = np.random.RandomState(SEED).multinomial(agg.S, np.ones(agg.S) / agg.S, size=C.N_BOOT).astype(float)
    tb = cnt @ A
    t = A.sum(0)

    def split(mask):
        return {"n": int(mask.sum()), "with_in_system_help": int((mask & help_).sum()), "without": int((mask & ~help_).sum()),
                "with_hint_count_gt0": int((mask & hc_any).sum()), "with_hint_first_attempt": int((mask & hf_any).sum()),
                "share_with_help": float((mask & help_).sum() / mask.sum())}
    out = {"analysis": "E7 misses split by in-system help", "operating_point": f"BKT, frozen Quiet tau {TAU}, per-fold CAL Platt (pooled CV held-out)",
           "in_system_help_definition": "hint_count > 0 or first_action == 1 at any try 1-9",
           "bootstrap": {"draws": C.N_BOOT, "seed": SEED, "unit": "student"},
           "misses": split(missed), "caught": split(caught), "stuck_with_help_opportunity": split(stuck),
           "no_help_from_anyone_missed_rate": {
               "definition": "misses with no in-system help at tries 1-9 / all stuck sequences with a help opportunity",
               "rate": float(t[2] / t[0]), "ci95": C.ci(tb[:, 2] / tb[:, 0]), "numerator": int(t[2]), "denominator": int(t[0])},
           "miss_rate_within_no_help_stuck": {
               "definition": "misses with no in-system help / stuck sequences with a help opportunity and no in-system help",
               "rate": float(t[2] / t[3]), "ci95": C.ci(tb[:, 2] / tb[:, 3]), "numerator": int(t[2]), "denominator": int(t[3])},
           "overall_missed_rate": {"rate": float(t[1] / t[0]), "ci95": C.ci(tb[:, 1] / tb[:, 0])}}
    log(f"wrote {C.write('e7_misses_by_help.json', out, t0)}")


if __name__ == "__main__":
    main()
