"""Stage 3 entry point: python -m aria_pace.run_baselines  (from the repo root).

Writes results/{data_report,split,label_counts,baselines_val}.json. Validation only: the test split is
frozen (SPEC 4) and its labels are never read or counted here.
"""
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn

from . import baselines as B
from . import metrics as M
from .constants import BOOTSTRAP_SEED, KS, RAW_CSV, RESULTS, VARIANTS
from .labels import eligibility, label
from .loader import build_sequences, load_raw
from .split import make_split, split_record

N_BOOT = 1000
PFA_C_GRID = (0.01, 0.1, 1.0, 10.0)
BKT_MIN_SEQ_GRID = (10, 30)


def dump(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def count_scope(seqs, k, variant, read_labels=True):
    """SPEC 3 counts. With read_labels=False only label-free counts are produced (used for the frozen test split)."""
    c = {"sequences": len(seqs), "students": len({q.user_id for q in seqs}), "n_lt_k": 0, "mastered_by_k": 0, "eligible": 0}
    scored, pos, leavers = [], 0, 0
    for q in seqs:
        e = eligibility(q.n, q.m, k)
        if e == "short":
            c["n_lt_k"] += 1
        elif e == "mastered":
            c["mastered_by_k"] += 1
        else:
            c["eligible"] += 1
            if read_labels:
                y = label(q.n, q.m, variant)
                if y is None:
                    leavers += 1
                else:
                    scored.append(q); pos += y
    if read_labels:
        c.update({"eligible_unlabelled_leavers": leavers, "scored_sequences": len(scored),
                  "scored_students": len({q.user_id for q in scored}), "positives": pos,
                  "label_rate": (pos / len(scored)) if scored else None})
    return c


def scored_set(seqs, k, variant):
    out = []
    for q in seqs:
        if eligibility(q.n, q.m, k) == "eligible":
            y = label(q.n, q.m, variant)
            if y is not None:
                out.append((q, y))
    return out


def main():
    t0 = time.time()
    df, rep = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique())
    seqs = build_sequences(df)
    rep["sequences_total"] = len(seqs)
    rep["env"] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                  "scipy": scipy.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()}
    sets = {name: set(ids) for name, ids in parts.items()}
    by_split = {name: [q for q in seqs if q.user_id in s] for name, s in sets.items()}
    assert sum(len(v) for v in by_split.values()) == len(seqs)
    rep["split_sizes"] = {n: {"students": len(sets[n]), "sequences": len(by_split[n])} for n in by_split}
    dump("data_report.json", rep)
    dump("split.json", split_record(parts))

    # ---- label counts: train and val only; test gets label-free counts
    lc = {}
    for variant in VARIANTS:
        for k in KS:
            lc[f"variant{variant}_k{k}"] = {
                "train": count_scope(by_split["train"], k, variant),
                "val": count_scope(by_split["val"], k, variant),
                "test_label_free": count_scope(by_split["test"], k, variant, read_labels=False),
            }
    dump("label_counts.json", lc)

    # ---- fit on train, choose hyperparameters on val by next-response log-loss (never the WS label)
    train, val = by_split["train"], by_split["val"]
    sel = {"pfa_C": {}, "bkt_min_seq": {}}
    pfa_models = {C: B.PFA(C).fit(train) for C in PFA_C_GRID}
    for C, m in pfa_models.items():
        sel["pfa_C"][str(C)] = m.next_response_logloss(val)
    best_C = min(PFA_C_GRID, key=lambda C: sel["pfa_C"][str(C)])
    pfa = pfa_models[best_C]

    kc_fits = B.fit_all_kcs(train)
    bkt_models = {ms: B.BKT(ms).fit(train, kc_fits) for ms in BKT_MIN_SEQ_GRID}
    for ms, m in bkt_models.items():
        sel["bkt_min_seq"][str(ms)] = m.next_response_logloss(val)
    best_ms = min(BKT_MIN_SEQ_GRID, key=lambda ms: sel["bkt_min_seq"][str(ms)])
    bkt = bkt_models[best_ms]
    pooled = kc_fits[None][1]
    sel["chosen"] = {"pfa_C": best_C, "bkt_min_seq": best_ms}
    sel["bkt_kcs_with_own_fit"] = int(sum(1 for kc, (n, p) in kc_fits.items() if kc is not None and p is not None and n >= best_ms))
    sel["bkt_kcs_total_in_train"] = int(len(kc_fits) - 1)
    sel["bkt_pooled_params"] = pooled
    sel["pfa_global_beta_gamma_rho"] = [float(x) for x in pfa.glob]

    # ---- validation scores and metrics
    models = [B.AlwaysHelp(), B.ThreeWrongInARow(), pfa, bkt]
    out = {"selection": sel, "n_bootstrap": N_BOOT, "bootstrap_seed": BOOTSTRAP_SEED, "split": "val", "results": {}}
    for variant in VARIANTS:
        for k in KS:
            pairs = scored_set(val, k, variant)
            y = np.array([p[1] for p in pairs]); sid = [p[0].user_id for p in pairs]
            for mdl in models:
                s = np.array([mdl.score(q, k) for q, _ in pairs])
                key = f"variant{variant}_k{k}"
                out["results"].setdefault(key, {})[mdl.name] = M.summarize(y, s, sid, N_BOOT, BOOTSTRAP_SEED)
            print(f"done variant {variant} k={k} N={len(pairs)}  t={time.time() - t0:.0f}s", flush=True)
    dump("baselines_val.json", out)
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
