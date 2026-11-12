"""E3: what distinguishes missed from caught stuck sequences at the Quiet threshold (CV held-out; descriptive; ANALYSIS_PLAN.md E3)."""
import time
from collections import Counter

import numpy as np
import pandas as pd

import common as C
import cv_heldout as CVH
from aria_pace import policy as P
from aria_pace import stage5 as S5

TAU = C.FROZEN_TAUS["quiet"]


def skill_names():
    df = pd.read_csv(C.RAW_CSV, encoding="latin-1", usecols=["skill_id", "skill_name"], low_memory=False)
    df = df.loc[~df.skill_id.isna()]
    df["skill_id"] = df.skill_id.astype(str)
    return df.groupby("skill_id").skill_name.agg(lambda s: s.mode().iloc[0] if s.notna().any() else "").to_dict()


def seq_features(q, cal_row, valid_row, err_row):
    c = q.correct[:9].astype(int)
    w = 1 - c
    longest = run = 0
    for v in w:
        run = run + 1 if v else 0
        longest = max(longest, run)
    switches = int(np.sum(c[1:] != c[:-1])) if len(c) > 1 else 0
    first_err = int(np.argmax(w == 1)) if w.any() else len(c)
    fa = q.first_action[:9]
    vals = cal_row[valid_row]
    ev = cal_row[valid_row & err_row]
    return {"n": int(q.n), "wrong_1_9": int(w.sum()), "verified_errors_at_decision_points": int((valid_row & err_row).sum()),
            "wrong_not_verified_1_9": int(((c == 0) & (fa != 0)).sum()), "hint_first_wrong_1_9": int(((c == 0) & (fa == 1)).sum()),
            "longest_wrong_run_1_9": longest, "switches_1_9": switches, "has_2plus_consecutive_wrong": bool(longest >= 2),
            "correct_before_first_error": first_err, "max_risk_all_decision_points": float(vals.max()) if len(vals) else np.nan,
            "max_risk_at_verified_errors": float(ev.max()) if len(ev) else np.nan, "pattern_1_9": "".join(map(str, c))}


def describe(rows):
    df = pd.DataFrame(rows)
    out = {"n_sequences": int(len(df))}
    for col in df.columns:
        if col == "pattern_1_9":
            continue
        v = df[col].astype(float)
        if df[col].dtype == bool:
            out[col] = {"share": float(v.mean())}
        else:
            v = v.dropna()
            out[col] = {"mean": float(v.mean()), "median": float(v.median()), "q25": float(v.quantile(.25)), "q75": float(v.quantile(.75)),
                        "min": float(v.min()), "max": float(v.max())}
    return out


def main():
    t0 = time.time()
    log = C.log_fn()
    _, parts, train, val, _ = C.load_pool(log)
    d = CVH.get(train, val, log)
    H = d["H"]
    C.assert_no_test(H.sid, parts)
    err = H.err["primary"]
    cal = d["cal"]["bkt"]
    Sx = np.nan_to_num(cal, nan=-1.0)
    fire = (Sx >= TAU) & H.valid & err
    fired = fire.any(1)
    opp = (H.valid & err).any(1)
    stuck = (H.y == 1) & opp
    missed_idx = np.where(stuck & ~fired)[0]
    caught_idx = np.where(stuck & fired)[0]
    names = skill_names()
    feats = {r: seq_features(H.seqs[r], cal[r], H.valid[r], err[r]) for r in np.concatenate([missed_idx, caught_idx])}
    groups = {"missed": [feats[r] for r in missed_idx], "caught": [feats[r] for r in caught_idx]}
    pat_m = Counter(f["pattern_1_9"] for f in groups["missed"])
    pat_c = Counter(f["pattern_1_9"] for f in groups["caught"])
    top = [{"pattern_1_9": p, "missed": n, "caught": pat_c.get(p, 0), "share_of_misses": n / len(missed_idx),
            "wrong_count": p.count("0")} for p, n in pat_m.most_common(5)]
    kc_m = Counter(H.seqs[r].kc for r in missed_idx)
    kc_all = Counter(H.seqs[r].kc for r in np.where(stuck)[0])
    skills = [{"skill_id": k, "skill_name": names.get(k, ""), "missed": n, "stuck_with_help_opportunity": kc_all[k], "miss_rate": n / kc_all[k]}
              for k, n in kc_m.most_common(10)]
    # Missed sequences: where in the risk distribution did they sit relative to tau?
    mr = np.array([f["max_risk_at_verified_errors"] for f in groups["missed"]])
    out = {"analysis": "E3 error analysis of missed help", "operating_point": f"BKT, frozen Quiet tau {TAU}, per-fold CAL Platt (pooled CV held-out)",
           "population": "stuck (y=1) sequences with a help opportunity (a primary verified error at a decision point 3-9)",
           "n_stuck_with_help_opportunity": int(stuck.sum()), "n_missed": int(len(missed_idx)), "n_caught": int(len(caught_idx)),
           "missed_help_rate": float(len(missed_idx) / stuck.sum()),
           "describe": {g: describe(v) for g, v in groups.items()}, "top5_patterns_among_misses": top,
           "top_skills_among_misses": skills, "n_distinct_skills_among_misses": len(kc_m),
           "missed_max_risk_at_errors_quantiles": {q: float(np.nanquantile(mr, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "missed_share_with_max_risk_within_0.10_of_tau": float(np.mean(mr >= TAU - 0.10)),
           "missed_where_risk_reached_tau_at_some_decision_point": int(np.sum([f["max_risk_all_decision_points"] >= TAU for f in groups["missed"]])),
           "missed_where_risk_reached_tau_note": "risk crossed tau at a decision point, but no primary verified error (correct 0 and first_action 0) occurred at or after it; the later wrong answers were hint-first or code-2 rows, so the policy could not fire",
           "test_misses": "NOT DONE: the locked test run saved only aggregates (aria_pace/results/test_results.json); no per-sequence test outputs exist on disk, and the test set was not re-run.",
           "pattern_legend": "string over tries 1-9 (shorter if the sequence ended), 1 = correct first attempt, 0 = not"}
    p = C.write("e3_errors.json", out, t0)
    write_summary(out)
    log(f"wrote {p}")


def write_summary(o):
    dm, dc = o["describe"]["missed"], o["describe"]["caught"]

    def row(k, lab, stat="median"):
        if "share" in dm[k]:
            return f"| {lab} | {100 * dm[k]['share']:.1f}% | {100 * dc[k]['share']:.1f}% |"
        return (f"| {lab} | {dm[k]['median']:.2f} (IQR {dm[k]['q25']:.2f}-{dm[k]['q75']:.2f}; mean {dm[k]['mean']:.2f}) | "
                f"{dc[k]['median']:.2f} (IQR {dc[k]['q25']:.2f}-{dc[k]['q75']:.2f}; mean {dc[k]['mean']:.2f}) |")
    lines = ["# E3 summary: missed vs caught stuck sequences (exploratory, CV held-out)", "",
             f"Operating point: {o['operating_point']}. {o['n_missed']} missed and {o['n_caught']} caught of {o['n_stuck_with_help_opportunity']} "
             f"stuck sequences with a help opportunity (missed-help rate {100 * o['missed_help_rate']:.1f}%).", "",
             "| Feature | Missed | Caught |", "|---|---|---|",
             row("n", "Sequence length n"), row("wrong_1_9", "Wrong answers, tries 1-9"),
             row("verified_errors_at_decision_points", "Verified errors at decision points 3-9"),
             row("wrong_not_verified_1_9", "Wrong but not a verified error (hint-first/other code), tries 1-9"),
             row("longest_wrong_run_1_9", "Longest wrong run, tries 1-9"), row("switches_1_9", "Right/wrong switches, tries 1-9"),
             row("has_2plus_consecutive_wrong", "Has 2+ consecutive wrongs"), row("correct_before_first_error", "Correct answers before first error"),
             row("max_risk_at_verified_errors", "Max calibrated risk at a verified error"), "",
             "Five most common patterns among misses (tries 1-9, 1 = correct):", "",
             "| Pattern | Missed | Caught with same pattern |", "|---|---|---|"]
    lines += [f"| `{t['pattern_1_9']}` | {t['missed']} ({100 * t['share_of_misses']:.1f}%) | {t['caught']} |" for t in o["top5_patterns_among_misses"]]
    lines += ["", f"Test misses: {o['test_misses']}", "", "Descriptive only; replay on logs; no causal reading."]
    (C.OUT / "e3_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
