"""Stage 6 / C4: does rank-transformed detector affect add information to the risk head? (b-P1, SPEC A3.4). 2012 TRAIN and VALIDATION only.

    python -m aria_pace.run_affect       (from the repo root)

The 2012 locked-test students are never loaded (the loader keeps only train+val ids, and this script asserts it). Writes results/affect_val.json.
Model: the A1.8 windowed logistic head, fitted on every decision row i = 3..9 (n >= i, m > i, labelled, variant A); C in {0.01,0.1,1,10} chosen by
2012-validation log-loss on those rows. Scored at k = 3, 5, 8 on eligible, labelled sequences. Variant B scored with the variant-A models.
"""
import json
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from . import affect as A
from . import paired as P
from .constants import KS, RESULTS
from .labels import eligibility, label
from .loader2012 import load_2012
from .riskhead import FEATURES, RiskHead, features
from .split_2012 import CSV_2012

SEED_B = 20261026
N_BOOT = 1000
C_GRID = (0.01, 0.1, 1.0, 10.0)
LAG_NAMES = {0: "primary_rows<=k", 1: "lagged_rows<=k-1"}


def rehearsal_running():
    return bool(subprocess.run(["pgrep", "-f", r"ui\.demo_app|demo_rehearsal|demo_screens|demo_harness"], capture_output=True, text=True).stdout.split())


def rows(seqs, variant="A", ks=range(3, 10)):
    """Yield (seq, i, y) for decision rows."""
    for q in seqs:
        for i in ks:
            if eligibility(q.n, q.m, i) == "eligible":
                y = label(q.n, q.m, variant)
                if y is not None:
                    yield q, i, y


def design(seqs, ks, variant="A"):
    base, aff0, aff1, y, sid, ii = [], [], [], [], [], []
    for q, i, lab in rows(seqs, variant, ks):
        cm, cx = A.cumulative(q.aff_rank)
        base.append(features(q.correct, i))
        aff0.append(A.affect_features(q.aff_rank, cm, cx, i, 0))
        aff1.append(A.affect_features(q.aff_rank, cm, cx, i, 1))
        y.append(lab); sid.append(q.user_id); ii.append(i)
    return {k: np.array(v, dtype=float) for k, v in dict(base=base, aff0=aff0, aff1=aff1, y=y, sid=sid, i=ii).items()}


def fit_select(Xtr, ytr, Xva, yva):
    best = None
    for C in C_GRID:
        m = RiskHead(C).fit(Xtr, ytr)
        ll = log_loss(yva, m.predict_rows(Xva), labels=[0, 1])
        if best is None or ll < best[0]:
            best = (ll, C, m)
    return best


def main():
    if rehearsal_running():
        sys.exit("A demo server / rehearsal / harness is running: not training now (SPEC A3.9).")
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731
    sp = json.load(open(RESULTS / "split_2012.json"))
    tr_ids, va_ids, te_ids = (set(sp["students"][k]) for k in ("train", "val", "test"))
    seqs = load_2012(CSV_2012, tr_ids | va_ids)
    assert not ({q.user_id for q in seqs} & te_ids), "locked 2012 test student loaded"
    train = [q for q in seqs if q.user_id in tr_ids]; val = [q for q in seqs if q.user_id in va_ids]
    log(f"2012 train sequences {len(train)}, validation {len(val)}; test students loaded: 0")

    rt = A.RankTransform().fit(np.concatenate([q.affect for q in train]))
    for q in seqs:
        q.aff_rank = rt.transform(q.affect)
    log(f"rank transform fitted on {sum(q.n for q in train)} train rows")

    D_tr, D_va = design(train, range(3, 10)), design(val, range(3, 10))
    log(f"decision rows: train {len(D_tr['y'])}, val {len(D_va['y'])}; positive rate train {D_tr['y'].mean():.3f}")

    rng = np.random.RandomState(20261027)
    def shuffled(D):                                             # negative control: affect vectors permuted across rows with the same i
        a0, a1 = D["aff0"].copy(), D["aff1"].copy()
        for i in np.unique(D["i"]):
            ix = np.where(D["i"] == i)[0]; p = rng.permutation(ix)
            a0[ix], a1[ix] = D["aff0"][p], D["aff1"][p]
        return a0, a1
    sh_tr, sh_va = shuffled(D_tr), shuffled(D_va)

    specs = {"without": (D_tr["base"], D_va["base"]),
             "with_primary": (np.c_[D_tr["base"], D_tr["aff0"]], np.c_[D_va["base"], D_va["aff0"]]),
             "with_lagged": (np.c_[D_tr["base"], D_tr["aff1"]], np.c_[D_va["base"], D_va["aff1"]]),
             "shuffled_primary_control": (np.c_[D_tr["base"], sh_tr[0]], np.c_[D_va["base"], sh_va[0]])}
    fitted, sel = {}, {}
    for name, (Xt, Xv) in specs.items():
        ll, C, m = fit_select(Xt, D_tr["y"], Xv, D_va["y"])
        fitted[name] = m; sel[name] = {"C": C, "val_decision_row_logloss": float(ll), "n_features": int(Xt.shape[1])}
        log(f"{name}: C={C} val log-loss {ll:.5f}")

    out = {"selection": sel, "n_rows": {"train": int(len(D_tr["y"])), "val": int(len(D_va["y"]))}, "results": {}, "paired_with_minus_without": {},
           "mean_over_k": {}, "seeds": {"split": sp["seed"], "bootstrap": SEED_B, "shuffle_control": 20261027}}
    for variant in ("A", "B"):
        per_k = {}
        for k in KS:
            Dk = design(val, [k], variant)
            Xs = {"without": Dk["base"], "with_primary": np.c_[Dk["base"], Dk["aff0"]], "with_lagged": np.c_[Dk["base"], Dk["aff1"]]}
            # shuffled control for evaluation: permute affect block among the k-rows (fixed seed)
            p = np.random.RandomState(20261027 + k).permutation(len(Dk["y"]))
            Xs["shuffled_primary_control"] = np.c_[Dk["base"], Dk["aff0"][p]]
            sc = {n: fitted[n].predict_rows(Xs[n]) for n in Xs}
            y, sid = Dk["y"].astype(int), Dk["sid"]
            per_k[k] = (y, sid, sc)
            out["results"][f"variant{variant}_k{k}"] = {"N": int(len(y)), "positives": int(y.sum()), "students": int(len(set(sid))),
                                                       "auc": {n: float(roc_auc_score(y, s)) for n, s in sc.items()}}
            for n in ("with_primary", "with_lagged", "shuffled_primary_control"):
                out["paired_with_minus_without"][f"{n}|variant{variant}_k{k}"] = P.auc_diff(y, sid, sc[n], sc["without"], N_BOOT, SEED_B)
            r = out["results"][f"variant{variant}_k{k}"]["auc"]
            d = out["paired_with_minus_without"]
            log(f"{variant} k={k} N={len(y)} AUC without {r['without']:.4f} | primary {r['with_primary']:.4f} ({d[f'with_primary|variant{variant}_k{k}']['diff']:+.4f} "
                f"[{d[f'with_primary|variant{variant}_k{k}']['ci95'][0]:+.4f},{d[f'with_primary|variant{variant}_k{k}']['ci95'][1]:+.4f}]) | lagged {r['with_lagged']:.4f} "
                f"({d[f'with_lagged|variant{variant}_k{k}']['diff']:+.4f}) | shuffled {r['shuffled_primary_control']:.4f}")
        for n in ("with_primary", "with_lagged", "shuffled_primary_control"):
            out["mean_over_k"][f"{n}|variant{variant}"] = P.mean_auc_diff({k: (v[0], v[1], v[2][n], v[2]["without"]) for k, v in per_k.items()}, N_BOOT, SEED_B)
            m = out["mean_over_k"][f"{n}|variant{variant}"]
            log(f"{variant} mean over k=3,5,8  {n} - without = {m['mean_diff']:+.4f} [{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]")
    wp = fitted["with_primary"].coefficients()
    names = FEATURES + A.AFFECT_FEATURE_NAMES
    out["standardised_coefficients_with_primary"] = {n: float(c) for n, c in zip(names, fitted["with_primary"].clf.coef_[0])}
    out["env"] = {"seconds": round(time.time() - t0), "test_students_loaded": 0}
    json.dump(out, open(RESULTS / "affect_val.json", "w"), indent=2, sort_keys=True)
    log("wrote results/affect_val.json")


if __name__ == "__main__":
    main()
