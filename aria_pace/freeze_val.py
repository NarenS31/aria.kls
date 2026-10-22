"""Stage 5a: fit on train, choose everything on VALIDATION, freeze it, and report validation numbers.

python -m aria_pace.freeze_val   (from the repo root). Test students are discarded right after the split.
Writes results/frozen_val.json (+ .sha256) and results/val_results.json.
"""
import hashlib
import json
import time

import numpy as np

from . import policy as P
from . import stage5 as S5
from .constants import RAW_CSV, RESULTS
from .loader import build_sequences, load_raw
from .split import make_split


def main():
    t0 = time.time()
    df, _ = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique())
    test_ids = set(parts["test"])
    seqs = [q for q in build_sequences(df) if q.user_id not in test_ids]
    train_ids = set(parts["train"])
    train = [q for q in seqs if q.user_id in train_ids]
    val = [q for q in seqs if q.user_id not in train_ids]
    assert not {q.user_id for q in seqs} & test_ids

    sel = json.load(open(RESULTS / "baselines_val.json"))["selection"]["chosen"]
    head_C = json.load(open(RESULTS / "riskhead_val.json"))["chosen_C"]
    models = S5.fit_models(train, sel["pfa_C"], sel["bkt_min_seq"], head_C)
    print(f"models fitted t={time.time() - t0:.0f}s", flush=True)

    full = P.Table(val, include_leavers=True)
    raw = S5.raw_scores(full, models)
    prim_mask = ~full.leaver
    prim = full.subset(prim_mask)
    platt = {}
    for m in S5.RISK_METHODS:
        mask = prim.valid
        rows_y = np.repeat(prim.y[:, None], P.NJ, axis=1)[mask]
        a, b = P.platt_fit(rows_y, raw[m][prim_mask][mask])
        platt[m] = {"intercept": a, "slope": b}
    cal = {m: P.platt_apply(raw[m], platt[m]["intercept"], platt[m]["slope"]) for m in S5.RISK_METHODS}
    configs = S5.split_configs(full, cal)

    chosen = {name: S5.choose_values(tab, kind, S) for name, (tab, kind, S) in configs.items()}
    frozen = {
        "note": "Chosen on the validation split only (SPEC A2.2-A2.3). Do not edit.",
        "models": {"pfa_C": sel["pfa_C"], "bkt_min_seq": sel["bkt_min_seq"], "head_C": head_C},
        "primary_risk_method": "bkt", "platt": platt,
        "configs": {
            "main": {"frozen": chosen["main"]},
            "leavers": {"carried": chosen["main"], "rechosen": chosen["leavers"]},
            "hintfirst": {"carried": chosen["main"], "rechosen": chosen["hintfirst"]},
        },
    }
    text = json.dumps(frozen, indent=2, sort_keys=True)
    (RESULTS / "frozen_val.json").write_text(text)
    (RESULTS / "frozen_val.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
    print(f"frozen t={time.time() - t0:.0f}s", flush=True)

    val_res = S5.evaluate_split(val, frozen, models, extra_val_stability=True)
    val_res["_note"] = "VALIDATION numbers are optimistic: calibrator, tau and n were chosen on this same split."
    with open(RESULTS / "val_results.json", "w") as f:
        json.dump(val_res, f, indent=2, sort_keys=True)
    print(f"wrote val_results t={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
