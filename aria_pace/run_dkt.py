"""Stage 6 / C3: train DKT on 2009 train, select on 2009 validation, compare with BKT (SPEC Amendment 3, A3.5).

    python -m aria_pace.run_dkt          (from the repo root; 2009 train and validation only, the 2009 test split is dropped on load)

Writes results/dkt_val.json, results/dkt_2009.pt (weights) and results/dkt_2009_meta.json (vocabulary, hidden size).
The promotion decision is computed mechanically from the validation numbers by the pre-registered rule; nothing is chosen by hand.
"""
import json
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from . import baselines as B
from . import dkt as D
from . import metrics as M
from .constants import BOOTSTRAP_SEED, KS, RAW_CSV, RESULTS
from .labels import eligibility, label
from .loader import build_sequences, load_raw
from .split import make_split

N_BOOT = 1000
PAIRED_SEED = 20261024
MARGIN = 0.01
KS_DECIDE = (3, 5)


def rehearsal_running() -> bool:
    out = subprocess.run(["pgrep", "-f", r"ui\.demo_app|demo_rehearsal|demo_screens|demo_harness"], capture_output=True, text=True).stdout.split()
    return bool(out)


def scored(seqs, k, variant):
    out = []
    for q in seqs:
        if eligibility(q.n, q.m, k) == "eligible":
            y = label(q.n, q.m, variant)
            if y is not None:
                out.append((q, y))
    return out


def paired_diff(y, sid, s_new, s_old, n_boot=N_BOOT, seed=PAIRED_SEED):
    """AUC(new) - AUC(old): point estimate and 95% percentile interval, resampling students; the same resample is used for both models."""
    y, sid = np.asarray(y), np.asarray(sid)
    uniq, inv = np.unique(sid, return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([by[i] for i in rng.randint(0, len(uniq), len(uniq))])
        if len(set(y[idx].tolist())) < 2:
            continue
        d.append(roc_auc_score(y[idx], s_new[idx]) - roc_auc_score(y[idx], s_old[idx]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(roc_auc_score(y, s_new) - roc_auc_score(y, s_old)), "ci95": [float(lo), float(hi)], "draws": len(d)}


def main():
    if rehearsal_running():
        sys.exit("A demo server / rehearsal / harness is running on this machine: not training now (SPEC A3.9).")
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731
    df, _ = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique())
    test_ids = set(parts["test"])
    seqs = [q for q in build_sequences(df) if q.user_id not in test_ids]            # the 2009 test split is never used
    train_ids = set(parts["train"])
    train = [q for q in seqs if q.user_id in train_ids]
    val = [q for q in seqs if q.user_id not in train_ids]
    assert not ({q.user_id for q in train} & {q.user_id for q in val}) and not ({q.user_id for q in seqs} & test_ids)
    log(f"2009 train sequences {len(train)}, validation {len(val)}")

    chosen = json.load(open(RESULTS / "baselines_val.json"))["selection"]["chosen"]
    bkt = B.BKT(chosen["bkt_min_seq"]).fit(train)
    log("BKT fitted")

    vocab = D.build_vocab(train)
    log(f"vocabulary: {len(vocab)} KCs with >= {D.MIN_KC} training sequences (+ UNK)")
    runs = []
    for h in D.HIDDEN_GRID:
        t1 = time.time()
        model, info = D.train_one(train, val, vocab, h, log=None)
        info["train_seconds"] = round(time.time() - t1, 1)
        runs.append((model, info))
        log(f"hidden={h}: best epoch {info['best_epoch']} of {info['epochs_run']}, validation next-response log-loss {info['val_next_response_logloss']:.5f} ({info['train_seconds']}s)")
    model, sel = min(runs, key=lambda r: r[1]["val_next_response_logloss"])
    log(f"selected hidden={sel['hidden']} (by validation next-response log-loss)")
    torch.save(model.state_dict(), RESULTS / "dkt_2009.pt")
    json.dump({"vocab": vocab, "hidden": sel["hidden"], "n_kc": len(vocab) + 1, "min_kc": D.MIN_KC, "max_len": D.MAX_LEN, "seed": D.SEED, "torch": torch.__version__},
              open(RESULTS / "dkt_2009_meta.json", "w"), indent=1, sort_keys=True)

    out = {"selection": {"grid": [r[1] for r in runs], "chosen_hidden": sel["hidden"], "criterion": "validation next-response log-loss",
                         "bkt_val_next_response_logloss": bkt.next_response_logloss(val)},
           "results": {}, "paired_dkt_minus_bkt": {}}
    for variant in ("A", "B"):
        for k in KS:
            pairs = scored(val, k, variant)
            qs = [p[0] for p in pairs]
            y = np.array([p[1] for p in pairs]); sid = np.array([p[0].user_id for p in pairs])
            s_bkt = np.array([bkt.score(q, k) for q in qs])
            s_dkt = D.risk_scores(model, vocab, qs, k)
            key = f"variant{variant}_k{k}"
            out["results"][key] = {"bkt": M.summarize(y, s_bkt, sid, N_BOOT, BOOTSTRAP_SEED), "dkt": M.summarize(y, s_dkt, sid, N_BOOT, BOOTSTRAP_SEED),
                                   "unknown_kc_share_val": float(np.mean([vocab.get(q.kc, 0) == 0 for q in qs]))}
            out["paired_dkt_minus_bkt"][key] = paired_diff(y, sid, s_dkt, s_bkt)
            log(f"{key}: N={len(y)}  AUC BKT {out['results'][key]['bkt']['auc']:.4f}  DKT {out['results'][key]['dkt']['auc']:.4f}  diff {out['paired_dkt_minus_bkt'][key]['diff']:+.4f} "
                f"CI [{out['paired_dkt_minus_bkt'][key]['ci95'][0]:+.4f}, {out['paired_dkt_minus_bkt'][key]['ci95'][1]:+.4f}]")

    # pre-registered rule (A3.5), variant A only, decided at k = 3 and k = 5; k = 8 is reported but does not decide
    rule = {}
    for k in KS_DECIDE:
        p = out["paired_dkt_minus_bkt"][f"variantA_k{k}"]
        rule[f"k{k}"] = {"diff": p["diff"], "ci_low": p["ci95"][0], "diff_ge_0.01": p["diff"] >= MARGIN, "ci_excludes_0_above": p["ci95"][0] > 0}
    promote = all(v["diff_ge_0.01"] and v["ci_excludes_0_above"] for v in rule.values())
    mean_diff = float(np.mean([out["paired_dkt_minus_bkt"][f"variantA_k{k}"]["diff"] for k in KS_DECIDE]))
    out["decision"] = {"rule": f"DKT - BKT AUC >= {MARGIN} AND paired 95% CI lower bound > 0, at BOTH k=3 and k=5 (variant A); k=8 reported, not deciding",
                       "per_k": rule, "promote_dkt": bool(promote), "primary_risk_score": "dkt" if promote else "bkt",
                       "mean_diff_k3_k5": mean_diff, "dkt_close_so_train_sakt": bool((not promote) and mean_diff >= -MARGIN),
                       "k8_reported_not_deciding": out["paired_dkt_minus_bkt"]["variantA_k8"]}
    out["env"] = {"torch": torch.__version__, "threads": D.THREADS, "seed": D.SEED, "paired_seed": PAIRED_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "seconds": round(time.time() - t0)}
    with open(RESULTS / "dkt_val.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    log(f"DECISION: {'PROMOTE DKT' if promote else 'KEEP BKT'} | mean diff k3,k5 = {mean_diff:+.4f} | SAKT needed: {out['decision']['dkt_close_so_train_sakt']}")


if __name__ == "__main__":
    main()
