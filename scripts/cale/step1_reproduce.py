"""Step 1: recompute the validation numbers the frozen thresholds were chosen from and compare with the saved files.

Repeats aria_pace/freeze_val.py in memory (nothing under aria_pace/results/ is written) and compares with
frozen_val.json (Platt within 1e-9, taus and n exactly) and val_results.json (every value). Exit code 1 on any mismatch.
"""
import json
import sys
import time

import numpy as np

import common as C
from aria_pace import policy as P
from aria_pace import stage5 as S5


def diff_json(a, b, path="", out=None, tol=1e-9):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append({"path": f"{path}/{k}", "issue": "missing in " + ("recomputed" if k not in a else "saved")})
            else:
                diff_json(a[k], b[k], f"{path}/{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": path, "issue": f"length {len(a)} vs {len(b)}"})
        for i, (x, y) in enumerate(zip(a, b)):
            diff_json(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (abs(a - b) <= tol * max(1.0, abs(b))):
            out.append({"path": path, "recomputed": a, "saved": b, "abs_diff": abs(a - b)})
    elif a != b:
        out.append({"path": path, "recomputed": a, "saved": b})
    return out


def main():
    t0 = time.time()
    log = C.log_fn()
    df, parts, train, val, rep = C.load_pool(log)
    frozen_saved = C.read_result("frozen_val.json")
    val_saved = C.read_result("val_results.json")
    sel = C.read_result("baselines_val.json")["selection"]["chosen"]
    head_C = C.read_result("riskhead_val.json")["chosen_C"]
    models = S5.fit_models(train, sel["pfa_C"], sel["bkt_min_seq"], head_C)
    log("models fitted on train")

    full = P.Table(val, include_leavers=True)
    raw = S5.raw_scores(full, models)
    prim_mask = ~full.leaver
    prim = full.subset(prim_mask)
    platt = {}
    for m in S5.RISK_METHODS:
        rows_y = np.repeat(prim.y[:, None], P.NJ, axis=1)[prim.valid]
        a, b = P.platt_fit(rows_y, raw[m][prim_mask][prim.valid])
        platt[m] = {"intercept": a, "slope": b}
    cal = {m: P.platt_apply(raw[m], platt[m]["intercept"], platt[m]["slope"]) for m in S5.RISK_METHODS}
    chosen = {name: S5.choose_values(tab, kind, S) for name, (tab, kind, S) in S5.split_configs(full, cal).items()}
    frozen_re = {"models": {"pfa_C": sel["pfa_C"], "bkt_min_seq": sel["bkt_min_seq"], "head_C": head_C},
                 "primary_risk_method": "bkt", "platt": platt,
                 "configs": {"main": {"frozen": chosen["main"]},
                             "leavers": {"carried": chosen["main"], "rechosen": chosen["leavers"]},
                             "hintfirst": {"carried": chosen["main"], "rechosen": chosen["hintfirst"]}}}
    saved_cmp = {k: v for k, v in frozen_saved.items() if k != "note"}
    d_frozen = diff_json(json.loads(json.dumps(frozen_re)), saved_cmp, tol=1e-9)
    log(f"frozen_val.json differences: {len(d_frozen)}")

    # Evaluate validation with the SAVED frozen values (as freeze_val.py does) and compare with val_results.json.
    val_re = S5.evaluate_split(val, frozen_saved, models, extra_val_stability=True)
    val_re["_note"] = val_saved.get("_note")
    d_val = diff_json(json.loads(json.dumps(val_re)), val_saved, tol=1e-9)
    log(f"val_results.json differences: {len(d_val)}")

    mv = val_saved["main"]["variants"]["frozen"]["methods"]
    key = {p: {"bkt_interventions_per_student": mv["bkt"][p]["interventions_per_student"],
               "always_help_interventions_per_student": mv["always_help"][p]["interventions_per_student"],
               "reduction_vs_always_help": 1 - mv["bkt"][p]["interventions"] / mv["always_help"][p]["interventions"],
               "missed_help_share": mv["bkt"][p]["missed_help_share"], "missed_help": mv["bkt"][p]["missed_help"],
               "n_y1_with_help_opportunity": mv["bkt"][p]["n_y1_with_help_opportunity"]} for p in P.PROFILES}
    ok = not d_frozen and not d_val
    C.assert_no_test([q.user_id for q in train + val], parts)
    C.write("step1_reproduction.json", {
        "reproduces": ok, "frozen_val_differences": d_frozen[:200], "n_frozen_val_differences": len(d_frozen),
        "val_results_differences": d_val[:200], "n_val_results_differences": len(d_val),
        "tolerance": "relative 1e-9 (absolute below 1)", "recomputed_platt": platt,
        "recomputed_taus_main": chosen["main"], "main_validation_key_numbers_saved": key,
        "data_report": rep,
        "note": "Validation numbers are optimistic (calibrator and tau were chosen on the same split), as in the original."}, t0)
    log("REPRODUCES" if ok else "MISMATCH: stop and report")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
