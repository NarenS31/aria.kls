"""Amendment A5: multi-skill DKT and SAKT on 2009 TRAIN / VALIDATION only (SPEC.md A5).

    python -m aria_pace.run_multiskill          (from the repo root)

The promotion decision (A5.4) is computed mechanically from the validation AUC differences; if a model is promoted the run STOPS (A5.5): no calibrator,
no tau, no test number. Writes results/multiskill_val.json only (and the promoted model's weights, if any, to results/multiskill_PROMOTED_*.pt).
"""
import json
import sys
import time

import numpy as np
import torch

from . import baselines as B
from . import dkt as D
from . import metrics as M
from . import multiskill as MS
from .constants import BOOTSTRAP_SEED, KS, RAW_CSV, RESULTS
from .loader import build_sequences, load_raw
from .run_dkt import KS_DECIDE, MARGIN, N_BOOT, PAIRED_SEED, paired_diff, rehearsal_running, scored
from .split import make_split

STATUS = "validation only; promotion rule pre-registered (A5.4), everything else exploratory"
EPS = 1e-6


def matched_logloss(preds, val, streams, bkt):
    """Per-student sums over the matched set: every non-first opportunity of every validation sequence that the model predicts (A5.6)."""
    per = {}
    for q in val:
        pos = streams.pos[(q.user_id, q.assignment_id)]
        par = bkt.params(q.kc)
        _, prior = B.BKT._filter(q.correct, par)
        for i in range(1, q.n):
            key = (q.user_id, int(pos[i]))
            if key not in preds:
                continue
            y = int(q.correct[i])
            pm = float(np.clip(preds[key], EPS, 1 - EPS))
            pb = float(np.clip(prior[i] * (1 - par["slip"]) + (1 - prior[i]) * par["guess"], EPS, 1 - EPS))
            r = per.setdefault(q.user_id, [0.0, 0.0, 0])
            r[0] += -(y * np.log(pm) + (1 - y) * np.log(1 - pm))
            r[1] += -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
            r[2] += 1
    return per


def ll_ci(per_models, n_boot=N_BOOT, seed=PAIRED_SEED):
    """per: {student: [sum model, sum bkt, n]} -> point values and paired student-bootstrap interval of mean(model) - mean(bkt)."""
    ids = sorted(per_models)
    A = np.array([per_models[i] for i in ids])
    n = A[:, 2].sum()
    point = {"model": float(A[:, 0].sum() / n), "bkt": float(A[:, 1].sum() / n), "n_steps": int(n), "n_students": len(ids)}
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n_boot):
        s = A[rng.randint(0, len(ids), len(ids))].sum(0)
        d.append((s[0] - s[1]) / s[2])
    lo, hi = np.percentile(d, [2.5, 97.5])
    point["diff_model_minus_bkt"] = point["model"] - point["bkt"]
    point["diff_ci95"] = [float(lo), float(hi)]
    return point


def main():
    if rehearsal_running():
        sys.exit("A demo server / rehearsal / harness is running on this machine: not training now (SPEC A3.9).")
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731
    df, _ = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique())
    test_ids = set(parts["test"])
    df = df[~df.user_id.isin(test_ids)].reset_index(drop=True)                      # the 2009 test split is never used
    train_ids, val_ids = set(parts["train"]), sorted(set(df.user_id.unique()) - set(parts["train"]))
    assert not set(df.user_id.unique()) & test_ids
    seqs = build_sequences(df)
    train = [q for q in seqs if q.user_id in train_ids]
    val = [q for q in seqs if q.user_id not in train_ids]
    vocab = MS.build_vocab(df, train_ids)
    streams = MS.Streams(df, vocab)
    tr_wins = MS.train_windows(streams, sorted(train_ids))
    va_wins = MS.val_windows(streams, val_ids)
    log(f"train students {len(train_ids)} windows {len(tr_wins)}; validation students {len(val_ids)} windows {len(va_wins)}; vocabulary {len(vocab)} skills (+UNK)")

    sel = json.load(open(RESULTS / "baselines_val.json"))["selection"]["chosen"]
    bkt = B.BKT(sel["bkt_min_seq"]).fit(train)
    out = {"status": STATUS, "spec": "aria_pace/SPEC.md Amendment 5", "vocabulary_size": len(vocab), "models": {}, "decision": {}}
    models = {}
    for kind in ("dkt", "sakt"):
        runs = []
        for size in D.HIDDEN_GRID:
            t1 = time.time()
            m, info = MS.train_one(kind, streams, tr_wins, streams, va_wins, vocab, size)
            info["train_seconds"] = round(time.time() - t1, 1)
            runs.append((m, info))
            log(f"{kind} size={size}: best epoch {info['best_epoch']} of {info['epochs_run']}, validation next-response log-loss {info['val_next_response_logloss']:.5f} ({info['train_seconds']}s)")
        model, sel_info = min(runs, key=lambda r: r[1]["val_next_response_logloss"])
        models[kind] = model
        out["models"][kind] = {"grid": [r[1] for r in runs], "chosen_size": sel_info["size"], "criterion": "validation next-response log-loss (all non-first window rows)"}
        log(f"{kind}: selected size {sel_info['size']}")

    preds = {k: MS.predict_windows(k, models[k], streams, va_wins) for k in models}
    for kind in models:
        out["models"][kind]["matched_next_response_logloss"] = ll_ci(matched_logloss(preds[kind], val, streams, bkt))
        o = out["models"][kind]["matched_next_response_logloss"]
        log(f"{kind}: matched log-loss {o['model']:.5f} vs BKT {o['bkt']:.5f} (diff {o['diff_model_minus_bkt']:+.5f} CI {o['diff_ci95']}) over {o['n_steps']} steps")

    out["auc_variantA_val"], out["paired_minus_bkt_variantA"] = {}, {}
    for k in KS:
        pairs = scored(val, k, "A")
        qs = [p[0] for p in pairs]
        y = np.array([p[1] for p in pairs]); sid = np.array([q.user_id for q in qs])
        s_b = np.array([bkt.score(q, k) for q in qs])
        out["auc_variantA_val"][f"k{k}"] = {"bkt": M.summarize(y, s_b, sid, N_BOOT, BOOTSTRAP_SEED)}
        for kind in models:
            t1 = time.time()
            s_m = MS.risk_scores(kind, models[kind], streams, vocab, qs, k)
            out["auc_variantA_val"][f"k{k}"][f"ms_{kind}"] = M.summarize(y, s_m, sid, N_BOOT, BOOTSTRAP_SEED)
            out["paired_minus_bkt_variantA"].setdefault(f"ms_{kind}", {})[f"k{k}"] = paired_diff(y, sid, s_m, s_b)
            p = out["paired_minus_bkt_variantA"][f"ms_{kind}"][f"k{k}"]
            log(f"k={k} {kind}: N={len(y)} AUC BKT {out['auc_variantA_val'][f'k{k}']['bkt']['auc']:.4f} MS {out['auc_variantA_val'][f'k{k}'][f'ms_{kind}']['auc']:.4f} diff {p['diff']:+.4f} CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] ({time.time() - t1:.0f}s)")

    # pre-registered rule (A5.4)
    promoted = []
    for kind in models:
        rule = {}
        for k in KS_DECIDE:
            p = out["paired_minus_bkt_variantA"][f"ms_{kind}"][f"k{k}"]
            rule[f"k{k}"] = {"diff": p["diff"], "ci_low": p["ci95"][0], "diff_ge_0.01": p["diff"] >= MARGIN, "ci_excludes_0_above": p["ci95"][0] > 0}
        ok = all(v["diff_ge_0.01"] and v["ci_excludes_0_above"] for v in rule.values())
        out["decision"][f"ms_{kind}"] = {"per_k": rule, "promote": bool(ok),
                                        "mean_diff_k3_k5": float(np.mean([out["paired_minus_bkt_variantA"][f"ms_{kind}"][f"k{k}"]["diff"] for k in KS_DECIDE])),
                                        "k8_reported_not_deciding": out["paired_minus_bkt_variantA"][f"ms_{kind}"]["k8"]}
        if ok:
            promoted.append(kind)
    out["decision"]["rule"] = f"AUC(model) - AUC(BKT) >= {MARGIN} AND paired 95% CI lower bound > 0, at BOTH k=3 and k=5 (variant A); k=8 reported, not deciding"
    out["decision"]["promoted"] = promoted
    out["decision"]["primary_risk_score"] = "STOP: see A5.5" if promoted else "bkt"
    out["env"] = {"torch": torch.__version__, "threads": D.THREADS, "seed": D.SEED, "paired_seed": PAIRED_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "seconds": round(time.time() - t0)}
    with open(RESULTS / "multiskill_val.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    if promoted:
        for kind in promoted:
            torch.save(models[kind].state_dict(), RESULTS / f"multiskill_PROMOTED_{kind}.pt")
        log(f"DECISION: PROMOTED {promoted}. STOP (A5.5): a fresh holdout must be declared before any test number is produced.")
    else:
        log("DECISION: KEEP BKT (no multi-skill model promoted)")


if __name__ == "__main__":
    main()
