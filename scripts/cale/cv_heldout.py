"""Pooled cross-validation held-out predictions for E1-E4 and E6 (no test student).

Folds: aria_pace/cv_tradeoff.py make_folds / fit_cal_split (the A7 folds, seed 20261022). Per fold:
  models = stage5.fit_models(FIT, frozen hyperparameters)   (BKT min_seq 10, PFA C 0.1, risk head C 10)
  Platt per method on CAL primary-scope decision rows        (policy.platt_fit, as freeze_val.py / cv_tradeoff.calibrate_and_choose)
  taus and n chosen on CAL                                    (stage5.choose_values: argmin J, ties to larger)
  risk on the held-out fold only, calibrated with that fold's CAL Platt.
The held-out tables of the five folds are concatenated. Results are cached in results_cale/_cache (gitignored).
"""
from __future__ import annotations

import hashlib
import pickle

import numpy as np

import common as C
from aria_pace import cv_tradeoff as CV
from aria_pace import policy as P
from aria_pace import stage5 as S5

CACHE_VERSION = "cv_heldout v1"


def concat_tables(tabs):
    t = P.Table.__new__(P.Table)
    t.items = [x for tb in tabs for x in tb.items]
    t.seqs = [x[0] for x in t.items]
    for a in ("y", "leaver", "sid", "valid", "err_primary", "err_all"):
        setattr(t, a, np.concatenate([getattr(tb, a) for tb in tabs]))
    t.err = {"primary": t.err_primary, "incl_hint_first": t.err_all}
    return t


def _build(train, val, log):
    pool = train + val
    sel = C.read_result("baselines_val.json")["selection"]["chosen"]
    head_C = C.read_result("riskhead_val.json")["chosen_C"]
    ids = sorted({q.user_id for q in pool})
    folds = CV.make_folds(ids)
    by_user = {}
    for q in pool:
        by_user.setdefault(q.user_id, []).append(q)
    held_tabs, raw_parts, cal_parts, fold_rows, fold_info = [], {m: [] for m in S5.RISK_METHODS}, {m: [] for m in S5.RISK_METHODS}, [], []
    for f, held_ids in enumerate(folds):
        held = set(held_ids)
        rest = [u for u in ids if u not in held]
        fit_ids, cal_ids = CV.fit_cal_split(rest, f)
        assert not (set(fit_ids) | set(cal_ids)) & held and not set(fit_ids) & set(cal_ids)
        fit_seqs = [q for u in fit_ids for q in by_user[u]]
        cal_seqs = [q for u in cal_ids for q in by_user[u]]
        held_seqs = [q for u in held_ids for q in by_user[u]]
        models = S5.fit_models(fit_seqs, sel["pfa_C"], sel["bkt_min_seq"], head_C)
        cal_tab = P.Table(cal_seqs, include_leavers=False)
        cal_raw = S5.raw_scores(cal_tab, models)
        platt = {}
        for m in S5.RISK_METHODS:
            y = np.repeat(cal_tab.y[:, None], P.NJ, axis=1)[cal_tab.valid]
            platt[m] = P.platt_fit(y, cal_raw[m][cal_tab.valid])
        cal_cal = {m: P.platt_apply(cal_raw[m], *platt[m]) for m in S5.RISK_METHODS}
        chosen = S5.choose_values(cal_tab, "primary", cal_cal)
        h = P.Table(held_seqs, include_leavers=False)
        h_raw = S5.raw_scores(h, models)
        held_tabs.append(h)
        for m in S5.RISK_METHODS:
            raw_parts[m].append(h_raw[m])
            cal_parts[m].append(P.platt_apply(h_raw[m], *platt[m]))
        fold_rows.append(np.full(len(h.y), f))
        fold_info.append({"fold": f, "n_held_students": len(held_ids), "n_fit_students": len(fit_ids), "n_cal_students": len(cal_ids),
                          "n_held_sequences_in_scope": int(len(h.y)), "n_cal_sequences_in_scope": int(len(cal_tab.y)),
                          "platt": {m: list(platt[m]) for m in S5.RISK_METHODS}, "cal_chosen": chosen,
                          "bkt_pooled_params": models["bkt"].kc_fits[None][1]})
        log(f"fold {f}: held {len(h.y)} seqs, BKT platt {platt['bkt'][0]:.3f},{platt['bkt'][1]:.3f}, CAL taus bkt {chosen['taus']['bkt']}, n {chosen['n']}")
    H = concat_tables(held_tabs)
    assert len(set(H.sid.tolist())) == len({u for t in held_tabs for u in t.sid.tolist()})
    return {"H": H, "raw": {m: np.concatenate(v) for m, v in raw_parts.items()}, "cal": {m: np.concatenate(v) for m, v in cal_parts.items()},
            "fold": np.concatenate(fold_rows), "fold_info": fold_info, "fold_sizes_students": [len(x) for x in folds],
            "hyper": {"pfa_C": sel["pfa_C"], "bkt_min_seq": sel["bkt_min_seq"], "head_C": head_C}}


def get(train, val, log=print):
    key = hashlib.sha256((CACHE_VERSION + C.sha256_file(C.RAW_CSV) + C.sha256_file(C.RESULTS / "baselines_val.json")
                          + C.sha256_file(C.RESULTS / "riskhead_val.json")).encode()).hexdigest()[:16]
    path = C.CACHE / f"cv_heldout_{key}.pkl"
    if path.exists():
        log(f"using cached CV held-out predictions {path.name}")
        C.read_result("baselines_val.json"); C.read_result("riskhead_val.json")
        return pickle.loads(path.read_bytes())
    d = _build(train, val, log)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(d))
    return d


def chosen_fire(d, method, prof):
    """Fire matrix at each fold's CAL-chosen tau (the A7 procedure)."""
    taus = np.array([d["fold_info"][f]["cal_chosen"]["taus"][method][prof] for f in d["fold"]])
    return np.nan_to_num(d["cal"][method], nan=-1.0) >= taus[:, None]
