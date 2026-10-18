"""C5: the ONE-TIME evaluation on the locked 2012 test portion (SPEC A3.7). Everything comes from results/frozen_v2.json.

    python -m aria_pace.run_test2012     refuses if results/TEST_LOCK_2012 exists. The lock file ("started") is written BEFORE the test rows are loaded.
Claim (a): 2009-trained models on the 2012 test. Claim (b): with vs without affect, trained on 2012-train, chosen on 2012-val, evaluated here once.
"""
import datetime
import hashlib
import json
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from . import affect as A
from . import baselines as B
from . import dkt as D
from . import metrics as M
from . import paired as P
from . import policy as POL
from . import sakt as SK
from . import stage5 as S5
from .constants import KS, RAW_CSV, RESULTS
from .loader import build_sequences, load_raw
from .loader2012 import load_2012
from .riskhead import RiskHead, features
from .run_affect import design, rehearsal_running
from .run_dkt import scored
from .split import make_split
from .split_2012 import CSV_2012

N_BOOT = 1000
SEED_A, SEED_B = 20261025, 20261026
AFF_MODELS = ("without", "with_primary", "with_lagged")


def aff_vec(q, i, name):
    cm, cx = A.cumulative(q.aff_rank)
    return A.affect_features(q.aff_rank, cm, cx, i, 1 if name == "with_lagged" else 0)


def load_torch_models():
    dm = json.load(open(RESULTS / "dkt_2009_meta.json")); sm = json.load(open(RESULTS / "sakt_2009_meta.json"))
    dk = D.DKT(dm["n_kc"], dm["hidden"]); dk.load_state_dict(torch.load(RESULTS / "dkt_2009.pt")); dk.eval()
    sk = SK.SAKT(sm["n_kc"], sm["d"]); sk.load_state_dict(torch.load(RESULTS / "sakt_2009.pt")); sk.eval()
    assert dm["vocab"] == sm["vocab"] if "vocab" in sm else True
    return dk, dm["vocab"], sk, sm.get("vocab", dm["vocab"])


def claim_a_risk(test, models, dk, dvocab, sk, svocab, log):
    out = {"results": {}, "paired_vs_bkt": {}}
    for k in KS:
        pairsA = scored(test, k, "A")
        qs = [p[0] for p in pairsA]; yA = np.array([p[1] for p in pairsA]); sid = np.array([q.user_id for q in qs])
        sc = {"bkt": np.array([models["bkt"].score(q, k) for q in qs]), "pfa": np.array([models["pfa"].score(q, k) for q in qs]),
              "risk_head": np.array([models["risk_head"].score(q, k) for q in qs]),
              "dkt_style": D.risk_scores(dk, dvocab, qs, k), "sakt_style": SK.risk_scores(sk, svocab, qs, k)}
        log(f"(a) k={k}: scored {len(qs)} sequences with 5 models")
        inB = np.array([label_B(q) is not None for q in qs])
        for variant, mask in (("A", np.ones(len(qs), bool)), ("B", inB)):
            y = np.array([label_of(q, variant) for q, m in zip(qs, mask) if m]); s_ = sid[mask]
            key = f"variant{variant}_k{k}"
            out["results"][key] = {m: M.summarize(y, s[mask], s_, N_BOOT, SEED_A) for m, s in sc.items()}
            out["results"][key]["unknown_kc_share_2009_vocab"] = float(np.mean([dvocab.get(q.kc, 0) == 0 for q, m in zip(qs, mask) if m]))
            out["paired_vs_bkt"][key] = {m: P.auc_diff(y, s_, sc[m][mask], sc["bkt"][mask], N_BOOT, SEED_A) for m in sc if m != "bkt"}
            r = out["results"][key]
            log(f"   {key}: N={len(y)} AUC " + " ".join(f"{m}={r[m]['auc']:.4f}" for m in sc))
    return out


def label_of(q, variant):
    from .labels import label
    return label(q.n, q.m, variant)


def label_B(q):
    return label_of(q, "B")


def table_scores(table, name, model, rt_ready=True):
    def fn(q, i):
        x = np.array([features(q.correct, i) + ([] if name == "without" else list(aff_vec(q, i, name)))])
        return float(model.predict_rows(x)[0])
    return POL.score_matrix(table, fn)


def claim_b(train, val, test, log):
    sel = json.load(open(RESULTS / "frozen_v2.json"))["within_dataset_b"]["head_C"]
    Dtr, Dva = design(train, range(3, 10)), design(val, range(3, 10))
    Xtr = {"without": Dtr["base"], "with_primary": np.c_[Dtr["base"], Dtr["aff0"]], "with_lagged": np.c_[Dtr["base"], Dtr["aff1"]]}
    models = {n: RiskHead(sel[n]).fit(Xtr[n], Dtr["y"]) for n in AFF_MODELS}
    # sanity: the validation numbers of C4 must be reproduced exactly before anything is read on test
    prev = json.load(open(RESULTS / "affect_val.json"))
    Dk = design(val, [3], "A")
    Xv = {"without": Dk["base"], "with_primary": np.c_[Dk["base"], Dk["aff0"]], "with_lagged": np.c_[Dk["base"], Dk["aff1"]]}
    for n in AFF_MODELS:
        got = roc_auc_score(Dk["y"], models[n].predict_rows(Xv[n])); want = prev["results"]["variantA_k3"]["auc"][n]
        assert abs(got - want) < 1e-9, (n, got, want)
    log("(b) refit reproduces the C4 validation AUCs exactly")
    out = {"results": {}, "paired_with_minus_without": {}, "mean_over_k": {}, "head_C": sel}
    per_k = {}
    for variant in ("A", "B"):
        per_k = {}
        for k in KS:
            Dt = design(test, [k], variant)
            Xt = {"without": Dt["base"], "with_primary": np.c_[Dt["base"], Dt["aff0"]], "with_lagged": np.c_[Dt["base"], Dt["aff1"]]}
            sc = {n: models[n].predict_rows(Xt[n]) for n in AFF_MODELS}
            y, sid = Dt["y"].astype(int), Dt["sid"]
            per_k[k] = (y, sid, sc)
            key = f"variant{variant}_k{k}"
            out["results"][key] = {"N": int(len(y)), "positives": int(y.sum()), "students": int(len(set(sid.tolist()))),
                                   "auc": {n: float(roc_auc_score(y, s)) for n, s in sc.items()},
                                   "metrics": {n: M.summarize(y, s, sid, 200, SEED_B) for n, s in sc.items()} if variant == "A" else None}
            for n in ("with_primary", "with_lagged"):
                out["paired_with_minus_without"][f"{n}|{key}"] = P.auc_diff(y, sid, sc[n], sc["without"], N_BOOT, SEED_B)
            log(f"   (b) {key} N={len(y)}: " + " ".join(f"{n}={out['results'][key]['auc'][n]:.4f}" for n in AFF_MODELS))
        for n in ("with_primary", "with_lagged"):
            out["mean_over_k"][f"{n}|variant{variant}"] = P.mean_auc_diff({k: (v[0], v[1], v[2][n], v[2]["without"]) for k, v in per_k.items()}, N_BOOT, SEED_B)
            m = out["mean_over_k"][f"{n}|variant{variant}"]
            log(f"   (b) variant {variant} mean over k {n}-without = {m['mean_diff']:+.4f} [{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]")
    pm, lm = out["mean_over_k"]["with_primary|variantA"], out["mean_over_k"]["with_lagged|variantA"]
    rule = lambda m: bool(m["ci95"][0] > 0 and m["mean_diff"] >= 0.005)  # noqa: E731
    out["decision"] = {"primary_passes": rule(pm), "lagged_passes": rule(lm),
                       "verdict": ("affect adds information" if rule(pm) and rule(lm) else "not robust (primary/lagged disagree)" if rule(pm) != rule(lm) else "no evidence affect adds information at the pre-registered threshold")}
    # secondary (descriptive): policy cost J per profile; tau re-chosen on 2012-val per model
    tab_v = POL.Table(val, include_leavers=False); tab_t = POL.Table(test, include_leavers=False)
    agg = S5.Agg(tab_t.sid); counts = agg.draws(N_BOOT, SEED_B)
    sec = {"models": {}, "j_diff": {}}; Jb = {}
    for n in AFF_MODELS:
        raw_v, raw_t = table_scores(tab_v, n, models[n]), table_scores(tab_t, n, models[n])
        mask = tab_v.valid
        a, b = POL.platt_fit(np.repeat(tab_v.y[:, None], POL.NJ, 1)[mask], raw_v[mask])
        cal_v, cal_t = POL.platt_apply(raw_v, a, b), POL.platt_apply(raw_t, a, b)
        tot_v = S5.sweep_totals(tab_v, tab_v.err["primary"], cal_v)
        sec["models"][n] = {}
        for prof, (wI, wD) in POL.PROFILES.items():
            tau = S5.pick(wI * tot_v[:, 1] + wD * tot_v[:, 4], POL.TAU_GRID)
            vec = S5.seq_vec(tab_t, tab_t.err["primary"], np.nan_to_num(cal_t, nan=-1.0) >= tau)
            Aagg = agg.by_student(vec)
            Jb[(n, prof)] = (wI * (counts @ Aagg[:, 1]) + wD * (counts @ Aagg[:, 4])) / agg.S
            sec["models"][n][prof] = {"tau_chosen_on_2012_val": tau, **S5.point_metrics(vec, agg.S, wI, wD)}
    for n in ("with_primary", "with_lagged"):
        for prof, (wI, wD) in POL.PROFILES.items():
            d = Jb[(n, prof)] - Jb[("without", prof)]
            point = sec["models"][n][prof]["cost_J_per_student"] - sec["models"]["without"][prof]["cost_J_per_student"]
            sec["j_diff"][f"{n}|{prof}"] = {"diff_with_minus_without": point, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    out["secondary_policy_J_descriptive"] = sec
    return out


def main(argv=()):
    dry = "--dry" in argv        # smoke test: VALIDATION students stand in for test, no lock, output to /tmp-like path given by --out
    lock = RESULTS / "TEST_LOCK_2012"
    if lock.exists() and not dry:
        sys.exit("TEST_LOCK_2012 exists: the 2012 test portion was already evaluated.")
    if rehearsal_running():
        sys.exit("A demo/rehearsal is running: not running now (SPEC A3.9).")
    text = (RESULTS / "frozen_v2.json").read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != (RESULTS / "frozen_v2.sha256").read_text().strip():
        sys.exit("frozen_v2.json does not match frozen_v2.sha256")
    frozen_val_text = (RESULTS / "frozen_val.json").read_text()
    assert hashlib.sha256(frozen_val_text.encode()).hexdigest() == json.loads(text)["cross_dataset_a"]["frozen_val_sha256"]
    if not dry:
        lock.write_text(json.dumps({"started_utc": datetime.datetime.utcnow().isoformat(), "frozen_v2_sha256": sha, "status": "started"}, indent=2))
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731

    # ---- 2009-trained models (train split only)
    df, _ = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique()); tr9 = set(parts["train"])
    train9 = [q for q in build_sequences(df) if q.user_id in tr9]
    frozen_val = json.loads(frozen_val_text); m = frozen_val["models"]
    models = S5.fit_models(train9, m["pfa_C"], m["bkt_min_seq"], m["head_C"])
    dk, dvocab, sk, svocab = load_torch_models()
    log(f"2009 models fitted/loaded on {len(train9)} train sequences")

    # ---- 2012 data
    sp = json.load(open(RESULTS / "split_2012.json"))
    ids = {k: set(sp["students"][k]) for k in ("train", "val", "test")}
    if dry:
        ids["test"] = set(ids["val"])
    seqs = load_2012(CSV_2012, ids["train"] | ids["val"] | (set() if dry else ids["test"]))
    train = [q for q in seqs if q.user_id in ids["train"]]; val = [q for q in seqs if q.user_id in ids["val"]]; test = [q for q in seqs if q.user_id in ids["test"]]
    assert dry or not (ids["test"] & {q.user_id for q in train + val})
    rt = A.RankTransform().fit(np.concatenate([q.affect for q in train]))
    for q in seqs:
        q.aff_rank = rt.transform(q.affect)
    log(f"2012 sequences train {len(train)} val {len(val)} test {len(test)} (test students {len({q.user_id for q in test})})")

    res = {"_frozen_v2_sha256": sha, "_note": "Single evaluation. Nothing was chosen on the test portion."}
    res["a_cross_dataset"] = {"risk": claim_a_risk(test, models, dk, dvocab, sk, svocab, log)}
    S5.BOOT_SEED = SEED_A
    res["a_cross_dataset"]["policy"] = S5.evaluate_split(test, frozen_val, models)
    log("(a) policy replay done")
    res["b_within_dataset_affect"] = claim_b(train, val, test, log)
    res["scope_test"] = {"students": len(ids["test"]), "sequences": len(test)}
    outp = argv[argv.index("--out") + 1] if dry else RESULTS / "test_results_2012.json"
    with open(outp, "w") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=float)
    if dry:
        log(f"DRY RUN finished (validation stood in for test): {outp}"); return
    lock.write_text(json.dumps({"started_utc": json.loads(lock.read_text())["started_utc"], "completed_utc": datetime.datetime.utcnow().isoformat(),
                                "frozen_v2_sha256": sha, "status": "completed"}, indent=2))
    log("wrote results/test_results_2012.json; TEST_LOCK_2012 completed")


if __name__ == "__main__":
    main(sys.argv[1:])
