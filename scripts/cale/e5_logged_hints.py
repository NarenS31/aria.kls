"""E5: logged-hint check (exploratory; ANALYSIS_PLAN.md E5). Train counts and validation re-evaluation with frozen BKT, Platt and taus.

Indicator of a logged hint at a try: hint_count > 0 (hints the student requested from ASSISTments on that problem).
"""
import time

import numpy as np
import pandas as pd

import common as C
from aria_pace import baselines as B
from aria_pace import policy as P
from aria_pace import stage5 as S5


def hint_arrays(df):
    """(user_id, assignment_id) -> hint_count / extra columns ordered by order_id, built exactly as loader.build_sequences groups rows."""
    extra = pd.read_csv(C.RAW_CSV, encoding="latin-1", usecols=["user_id", "order_id", "skill_id", "hint_total", "bottom_hint"], low_memory=False)
    extra = extra.loc[~extra.skill_id.isna(), ["user_id", "order_id", "hint_total", "bottom_hint"]]
    m = df.merge(extra, on=["user_id", "order_id"], how="left", validate="one_to_one")
    assert len(m) == len(df)
    out = {}
    for (u, a), g in m.groupby(["user_id", "assignment_id"], sort=False):
        g = g.sort_values("order_id", kind="stable")
        out[(int(u), int(a))] = {"hint_count": g.hint_count.to_numpy(), "hint_total": g.hint_total.to_numpy(),
                                 "bottom_hint": g.bottom_hint.to_numpy(), "correct": g.correct.to_numpy(dtype=np.int8)}
    return out


def hint_matrix(tab, H):
    """(N, 9) bool: logged hint at tries 1..9 (False beyond n)."""
    M = np.zeros((len(tab.seqs), 9), bool)
    for r, q in enumerate(tab.seqs):
        h = H[(q.user_id, q.assignment_id)]
        assert np.array_equal(h["correct"], q.correct), "hint columns misaligned"
        k = min(9, q.n)
        M[r, :k] = h["hint_count"][:k] > 0
    return M


def counts_for(tab, HM, label):
    dec = HM[:, 2:9]                                  # hint at try i, i = 3..9
    upto = np.cumsum(HM, 1)[:, 2:9] > 0               # hint at some try t <= i
    err = tab.err["primary"]
    v = tab.valid
    any19 = HM.any(1)
    return {"split": label, "in_scope_sequences": int(len(tab.y)), "students": int(len(set(tab.sid.tolist()))),
            "sequences_with_logged_hint_tries_1_9": int(any19.sum()), "share_sequences": float(any19.mean()),
            "stuck_sequences": int((tab.y == 1).sum()), "stuck_with_logged_hint": int((any19 & (tab.y == 1)).sum()),
            "share_stuck_with_logged_hint": float(any19[tab.y == 1].mean()),
            "nonstuck_with_logged_hint": int((any19 & (tab.y == 0)).sum()), "share_nonstuck_with_logged_hint": float(any19[tab.y == 0].mean()),
            "decision_cells": int(v.sum()), "decision_cells_with_hint_at_that_try": int((v & dec).sum()),
            "decision_cells_at_or_after_first_hint": int((v & upto).sum()),
            "verified_error_cells": int((v & err).sum()), "verified_error_cells_with_hint_at_that_try": int((v & err & dec).sum()),
            "verified_error_cells_at_or_after_first_hint": int((v & err & upto).sum())}


def evaluate(tab, cal, taus, seed):
    err = tab.err["primary"]
    agg = S5.Agg(tab.sid)
    counts = np.random.RandomState(seed).multinomial(agg.S, np.ones(agg.S) / agg.S, size=C.N_BOOT).astype(float)
    Sx = np.nan_to_num(cal, nan=-1.0)
    al = agg.by_student(S5.seq_vec(tab, err, tab.valid & err))
    out = {"n_sequences": int(len(tab.y)), "n_students": int(agg.S), "n_stuck": int(tab.y.sum()),
           "always_help_hints_per_student": float(al.sum(0)[0] / agg.S)}
    for prof, tau in taus.items():
        A = agg.by_student(S5.seq_vec(tab, err, Sx >= tau))
        t, a = A.sum(0), al.sum(0)
        tb, ab = counts @ A, counts @ al
        with np.errstate(divide="ignore", invalid="ignore"):
            out[prof] = {"tau": tau, "hints_per_student": float(t[0] / agg.S), "hints_ci95": C.ci(tb[:, 0] / agg.S),
                         "reduction_vs_always_help": float(1 - t[0] / a[0]), "reduction_ci95": C.ci(1 - tb[:, 0] / ab[:, 0]),
                         "missed_help_rate": float(t[3] / t[2]) if t[2] else None, "missed_ci95": C.ci(tb[:, 3] / tb[:, 2]),
                         "missed": int(t[3]), "stuck_with_help_opportunity": int(t[2]), "interventions": int(t[0]),
                         "always_help_interventions": int(a[0])}
    return out


def main():
    t0 = time.time()
    log = C.log_fn()
    df, parts, train, val, _ = C.load_pool(log)
    frozen = C.read_result("frozen_val.json")
    taus = frozen["configs"]["main"]["frozen"]["taus"]["bkt"]
    a, b = frozen["platt"]["bkt"]["intercept"], frozen["platt"]["bkt"]["slope"]
    H = hint_arrays(df)
    tr, va = P.Table(train, include_leavers=False), P.Table(val, include_leavers=False)
    C.assert_no_test(np.concatenate([tr.sid, va.sid]), parts)
    HM_tr, HM_va = hint_matrix(tr, H), hint_matrix(va, H)
    # cross-tab of the indicator with first_action and bottom_hint (rows of in-scope validation sequences, tries 1-9)
    rows = []
    for q in va.seqs:
        h = H[(q.user_id, q.assignment_id)]
        k = min(9, q.n)
        rows.append(pd.DataFrame({"hint": h["hint_count"][:k] > 0, "first_action": q.first_action[:k], "correct": q.correct[:k],
                                  "bottom_hint": h["bottom_hint"][:k]}))
    xt = pd.concat(rows)
    xtab = {"rows_tries_1_9": int(len(xt)), "hint_rows": int(xt.hint.sum()),
            "hint_rows_by_first_action": {str(k): int(v) for k, v in xt[xt.hint].first_action.value_counts().sort_index().items()},
            "hint_rows_correct_1": int((xt.hint & (xt.correct == 1)).sum()),
            "wrong_rows": int((xt.correct == 0).sum()), "wrong_rows_with_hint": int((xt.hint & (xt.correct == 0)).sum()),
            "verified_error_rows (correct 0, first_action 0)": int(((xt.correct == 0) & (xt.first_action == 0)).sum()),
            "verified_error_rows_with_hint": int(((xt.correct == 0) & (xt.first_action == 0) & xt.hint).sum()),
            "hint_rows_reaching_bottom_hint": int((xt.hint & (xt.bottom_hint == 1)).sum())}
    bkt = B.BKT(frozen["models"]["bkt_min_seq"]).fit(train)
    cal = P.platt_apply(P.score_matrix(va, bkt.score), a, b)
    seed = C.SEEDS["e5_bootstrap"]
    main_ = evaluate(va, cal, taus, seed)
    keep = ~HM_va.any(1)
    va_a = va.subset(keep)
    ex_a = evaluate(va_a, cal[keep], taus, seed)
    va_b = va.subset(np.ones(len(va.y), bool))
    va_b.valid = va.valid & ~(np.cumsum(HM_va, 1)[:, 2:9] > 0)
    ex_b = evaluate(va_b, cal, taus, seed)
    # the step-1 check: main_ must equal val_results.json main/frozen bkt
    vr = C.read_result("val_results.json")["main"]["variants"]["frozen"]["methods"]["bkt"]
    chk = {p: (vr[p]["missed_help"] == main_[p]["missed"] and abs(vr[p]["interventions_per_student"] - main_[p]["hints_per_student"]) < 1e-12)
           for p in taus}
    assert all(chk.values()), chk
    out = {"analysis": "E5 logged-hint check",
           "columns_used": {"indicator": "hint_count > 0 (number of hints the student requested on that problem in ASSISTments)",
                            "descriptive_only": ["first_action (inferred codes: 1 = hint as first action)", "bottom_hint", "hint_total (hints available, not received; not used as an indicator)"]},
           "frozen_values_used": {"bkt_min_seq": frozen["models"]["bkt_min_seq"], "platt": [a, b], "taus": taus},
           "counts": {"train": counts_for(tr, HM_tr, "train"), "validation": counts_for(va, HM_va, "validation")},
           "indicator_crosstab_validation": xtab,
           "validation_main": main_, "validation_main_matches_val_results_json": chk,
           "a_exclude_sequences_with_hint_tries_1_9": ex_a,
           "b_drop_decision_points_at_or_after_first_hint": ex_b,
           "definitions": {"a": "sequences with hint_count > 0 at any try 1-9 removed; everything else unchanged",
                           "b": "decision point i dropped if hint_count > 0 at any try t <= i (the hint at try i is received before decision point i); a stuck sequence with no remaining verified-error decision point has no help opportunity and leaves the missed-help denominator"},
           "caveat": "Exclusion on hints is a selection on a post-decision variable; this checks how much the replay result depends on sequences where ASSISTments itself gave help. It is not a causal estimate. Nothing is re-chosen; frozen tau, Platt and BKT are used."}
    log(f"wrote {C.write('e5_logged_hints.json', out, t0)}")


if __name__ == "__main__":
    main()
