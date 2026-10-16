"""Stage 6 / C3 (SAKT branch of A3.5, architecture fixed in Amendment 3a): train SAKT on 2009 train, select on 2009 validation, compare with BKT.

    python -m aria_pace.run_sakt         (2009 train and validation only; the 2009 test split is dropped on load)

Writes results/sakt_val.json, results/sakt_2009.pt, results/sakt_2009_meta.json. The overall promotion decision (DKT, SAKT or neither) is
computed mechanically from dkt_val.json and this run by the pre-registered rule.
"""
import json
import sys
import time

import numpy as np
import torch

from . import baselines as B
from . import dkt as D
from . import metrics as M
from . import sakt as S
from .constants import BOOTSTRAP_SEED, KS, RAW_CSV, RESULTS
from .loader import build_sequences, load_raw
from .run_dkt import KS_DECIDE, MARGIN, N_BOOT, paired_diff, rehearsal_running, scored
from .split import make_split


def main():
    if rehearsal_running():
        sys.exit("A demo server / rehearsal / harness is running on this machine: not training now (SPEC A3.9).")
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731
    df, _ = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique())
    test_ids = set(parts["test"])
    seqs = [q for q in build_sequences(df) if q.user_id not in test_ids]
    train_ids = set(parts["train"])
    train = [q for q in seqs if q.user_id in train_ids]
    val = [q for q in seqs if q.user_id not in train_ids]
    assert not ({q.user_id for q in seqs} & test_ids)
    chosen = json.load(open(RESULTS / "baselines_val.json"))["selection"]["chosen"]
    bkt = B.BKT(chosen["bkt_min_seq"]).fit(train)
    vocab = D.build_vocab(train)
    log(f"train {len(train)}, validation {len(val)}, vocabulary {len(vocab)} KCs")

    runs = []
    for d in S.HIDDEN_GRID:
        t1 = time.time()
        model, info = S.train_one(train, val, vocab, d)
        info["train_seconds"] = round(time.time() - t1, 1)
        runs.append((model, info))
        log(f"d={d}: best epoch {info['best_epoch']} of {info['epochs_run']}, validation next-response log-loss {info['val_next_response_logloss']:.5f} ({info['train_seconds']}s)")
    model, sel = min(runs, key=lambda r: r[1]["val_next_response_logloss"])
    log(f"selected d={sel['hidden']} (validation next-response log-loss)")
    torch.save(model.state_dict(), RESULTS / "sakt_2009.pt")
    json.dump({"vocab": vocab, "d": sel["hidden"], "heads": S.HEADS, "n_kc": len(vocab) + 1, "max_len": D.MAX_LEN, "seed": D.SEED, "torch": torch.__version__},
              open(RESULTS / "sakt_2009_meta.json", "w"), indent=1, sort_keys=True)

    out = {"selection": {"grid": [r[1] for r in runs], "chosen_d": sel["hidden"], "criterion": "validation next-response log-loss",
                         "bkt_val_next_response_logloss": bkt.next_response_logloss(val)}, "results": {}, "paired_sakt_minus_bkt": {}}
    for variant in ("A", "B"):
        for k in KS:
            pairs = scored(val, k, variant)
            qs = [p[0] for p in pairs]
            y = np.array([p[1] for p in pairs]); sid = np.array([p[0].user_id for p in pairs])
            s_bkt = np.array([bkt.score(q, k) for q in qs])
            s_sakt = S.risk_scores(model, vocab, qs, k)
            key = f"variant{variant}_k{k}"
            out["results"][key] = {"bkt": M.summarize(y, s_bkt, sid, N_BOOT, BOOTSTRAP_SEED), "sakt": M.summarize(y, s_sakt, sid, N_BOOT, BOOTSTRAP_SEED)}
            out["paired_sakt_minus_bkt"][key] = paired_diff(y, sid, s_sakt, s_bkt)
            p = out["paired_sakt_minus_bkt"][key]
            log(f"{key}: N={len(y)}  AUC BKT {out['results'][key]['bkt']['auc']:.4f}  SAKT {out['results'][key]['sakt']['auc']:.4f}  diff {p['diff']:+.4f} CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]")

    rule = {}
    for k in KS_DECIDE:
        p = out["paired_sakt_minus_bkt"][f"variantA_k{k}"]
        rule[f"k{k}"] = {"diff": p["diff"], "ci_low": p["ci95"][0], "diff_ge_0.01": p["diff"] >= MARGIN, "ci_excludes_0_above": p["ci95"][0] > 0}
    sakt_ok = all(v["diff_ge_0.01"] and v["ci_excludes_0_above"] for v in rule.values())
    dkt = json.load(open(RESULTS / "dkt_val.json"))
    dkt_ok = bool(dkt["decision"]["promote_dkt"])
    mean_s = float(np.mean([out["paired_sakt_minus_bkt"][f"variantA_k{k}"]["diff"] for k in KS_DECIDE]))
    mean_d = float(dkt["decision"]["mean_diff_k3_k5"])
    winner = "bkt"
    if sakt_ok and dkt_ok:
        winner = "sakt" if mean_s > mean_d else "dkt"
    elif sakt_ok:
        winner = "sakt"
    elif dkt_ok:
        winner = "dkt"
    out["decision"] = {"rule": dkt["decision"]["rule"], "sakt_per_k": rule, "promote_sakt": bool(sakt_ok), "promote_dkt": dkt_ok,
                       "mean_diff_k3_k5": {"sakt": mean_s, "dkt": mean_d}, "primary_risk_score": winner,
                       "k8_reported_not_deciding": out["paired_sakt_minus_bkt"]["variantA_k8"]}
    out["env"] = {"torch": torch.__version__, "threads": D.THREADS, "seed": D.SEED, "paired_seed": 20261024, "seconds": round(time.time() - t0)}
    with open(RESULTS / "sakt_val.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    log(f"DECISION: primary risk score = {winner.upper()} | SAKT promoted: {sakt_ok} | mean diff vs BKT (k3,k5): SAKT {mean_s:+.4f}, DKT {mean_d:+.4f}")


if __name__ == "__main__":
    main()
