"""Amendment A7: cross-validated saving-versus-missed-help trade-off curves per mastery definition (exploratory, validation only; SPEC.md A7).

Pool = 2009 train + validation students (the test students are dropped by the caller, exactly as run_mastery_sens.py does). Five student-level folds;
for each held-out fold the remaining students are split into FIT (BKT) and CAL (Platt + tau), and the chosen tau / calibrator is applied to the
held-out fold only. Reuses the A4 machinery in mastery_sens.py unchanged (definitions, relabelling, exact BKT risk, tau choice, student aggregates).
"""
from __future__ import annotations

import numpy as np

from . import baselines as B
from . import mastery_sens as MS
from . import policy as P
from . import stage5 as S5

N_FOLDS = 5
FOLD_SEED = 20261022
BOOT_SEED = 20261023
BOOT_DRAWS = 1000
FIT_FRACTION = 0.75
STATUS = "exploratory, validation only (A7)"


def make_folds(student_ids, n_folds=N_FOLDS, seed=FOLD_SEED):
    """Seeded permutation of the sorted ids, consecutive fifths. Returns a list of sorted id lists (disjoint, covering)."""
    ids = np.array(sorted(set(int(u) for u in student_ids)))
    perm = np.random.RandomState(seed).permutation(len(ids))
    return [sorted(ids[chunk].tolist()) for chunk in np.array_split(perm, n_folds)]


def fit_cal_split(rest_ids, fold_index, seed=FOLD_SEED, fit_fraction=FIT_FRACTION):
    """FIT (3/4) and CAL (1/4) by student, RandomState(seed + fold_index)."""
    ids = np.array(sorted(set(int(u) for u in rest_ids)))
    perm = np.random.RandomState(seed + fold_index).permutation(len(ids))
    n_fit = int(round(fit_fraction * len(ids)))
    return sorted(ids[perm[:n_fit]].tolist()), sorted(ids[perm[n_fit:]].tolist())


def _prim_table(seqs, d):
    full = MS.SensTable(MS.relabel(seqs, d), include_leavers=True, T=d.T)
    return full.subset(~full.leaver)


def calibrate_and_choose(cal_seqs, bkt, d):
    """Platt + tau per profile from CAL only (the A4 `run_definition` steps up to the choice of tau)."""
    prim = _prim_table(cal_seqs, d)
    raw = MS.raw_matrix(prim, bkt, d)
    m = prim.valid
    y = np.repeat(prim.y[:, None], P.NJ, axis=1)[m]
    a, b = P.platt_fit(y, raw[m])
    cal = P.platt_apply(raw, a, b)
    taus, _ = MS.choose_taus(prim, prim.err["primary"], cal, d.T)
    return {"platt": (a, b), "taus": taus, "n_cal_sequences": int(len(prim.y))}


def heldout_student_vectors(held_seqs, bkt, d, platt, taus):
    """Per held-out student: the 8 policy totals at every tau of the grid, at the always-help policy, and at each profile's chosen tau.
    Nothing about the held-out fold was used to choose `platt` or `taus`."""
    prim = _prim_table(held_seqs, d)
    raw = MS.raw_matrix(prim, bkt, d)
    cal = P.platt_apply(raw, *platt)
    err = prim.err["primary"]
    Sx = np.nan_to_num(cal, nan=-1.0)
    agg = S5.Agg(prim.sid)
    grid = np.stack([agg.by_student(MS.seq_vec_T(prim, err, Sx >= tau, d.T)) for tau in P.TAU_GRID])       # (n_tau, S, 8)
    always = agg.by_student(MS.seq_vec_T(prim, err, prim.valid & err, d.T))
    chosen = {prof: agg.by_student(MS.seq_vec_T(prim, err, Sx >= taus[prof], d.T)) for prof in P.PROFILES}
    return {"sid": agg.uniq, "grid": np.transpose(grid, (1, 0, 2)), "always": always, "chosen": chosen}


def pool(parts):
    sid = np.concatenate([p["sid"] for p in parts])
    assert len(set(sid.tolist())) == len(sid), "a student appears in two held-out folds"
    return {"sid": sid, "grid": np.concatenate([p["grid"] for p in parts]), "always": np.concatenate([p["always"] for p in parts]),
            "chosen": {prof: np.concatenate([p["chosen"][prof] for p in parts]) for prof in P.PROFILES}}


def _saving_missed(tot_policy, tot_always):
    with np.errstate(divide="ignore", invalid="ignore"):
        saving = 1.0 - tot_policy[..., 0] / tot_always[..., 0, None] if tot_policy.ndim == 3 else 1.0 - tot_policy[..., 0] / tot_always[..., 0]
        missed = np.where(tot_policy[..., 2] > 0, tot_policy[..., 3] / np.where(tot_policy[..., 2] > 0, tot_policy[..., 2], 1), np.nan)
    return saving, missed


def curves_and_points(pooled, n_boot=BOOT_DRAWS, seed=BOOT_SEED):
    S = len(pooled["sid"])
    counts = np.random.RandomState(seed).multinomial(S, np.ones(S) / S, size=n_boot).astype(float)          # (B, S)
    always_b = counts @ pooled["always"]                                                                     # (B, 8)
    grid_b = np.einsum("bs,stf->btf", counts, pooled["grid"])                                                # (B, n_tau, 8)
    sv_b, ms_b = _saving_missed(grid_b, always_b)
    always_pt = pooled["always"].sum(0)
    grid_pt = pooled["grid"].sum(0)
    sv_pt, ms_pt = _saving_missed(grid_pt[None], always_pt[None])
    sv_pt, ms_pt = sv_pt[0], ms_pt[0]

    def ci(a):
        return np.nanpercentile(a, [2.5, 97.5], axis=0)
    sv_ci, ms_ci = ci(sv_b), ci(ms_b)
    out = {"tau_grid": P.TAU_GRID.tolist(), "n_students": int(S),
           "saving": sv_pt.tolist(), "saving_ci95": [sv_ci[0].tolist(), sv_ci[1].tolist()],
           "missed": [None if np.isnan(x) else float(x) for x in ms_pt], "missed_ci95": [[None if np.isnan(x) else float(x) for x in r] for r in ms_ci],
           "always_help_interventions_per_student": float(always_pt[0] / S)}
    pts = {}
    for prof, A in pooled["chosen"].items():
        tot_b = counts @ A
        s_b, m_b = _saving_missed(tot_b, always_b)
        tot = A.sum(0)
        s0, m0 = _saving_missed(tot[None], always_pt[None])
        pts[prof] = {"saving": float(s0[0]), "saving_ci95": np.nanpercentile(s_b, [2.5, 97.5]).tolist(),
                     "missed": None if np.isnan(m0[0]) else float(m0[0]), "missed_ci95": np.nanpercentile(m_b, [2.5, 97.5]).tolist(),
                     "interventions_per_student": float(tot[0] / S)}
    out["chosen_tau_pooled_result"] = pts
    return out


def tau_movement(per_fold_taus, d_name, frozen_tau, control_taus=None):
    """per_fold_taus: list (fold) of {profile: tau}. Range, SD, shift against the frozen tau and, if given, against D3 in the same fold."""
    out = {}
    for prof in P.PROFILES:
        v = np.array([t[prof] for t in per_fold_taus], float)
        row = {"fold_taus": v.tolist(), "min": float(v.min()), "max": float(v.max()), "range": float(v.max() - v.min()), "sd": float(v.std(ddof=1)),
               "shift_vs_frozen": (v - frozen_tau[prof]).tolist(), "mean_shift_vs_frozen": float((v - frozen_tau[prof]).mean())}
        if control_taus is not None:
            c = np.array([t[prof] for t in control_taus], float)
            sh = v - c
            row["shift_vs_D3_same_fold"] = sh.tolist()
            row["n_folds_material_shift_vs_D3"] = int((np.abs(sh) > 0.10 + 1e-12).sum())
        out[prof] = row
    return out


def run(pool_seqs, min_seq, frozen_tau, defs=MS.DEFS, log=print):
    ids = sorted({q.user_id for q in pool_seqs})
    folds = make_folds(ids)
    by_user = {}
    for q in pool_seqs:
        by_user.setdefault(q.user_id, []).append(q)
    per_def = {d.name: {"parts": [], "taus": [], "platt": [], "n_cal": []} for d in defs}
    for f, held_ids in enumerate(folds):
        held = set(held_ids)
        rest = [u for u in ids if u not in held]
        fit_ids, cal_ids = fit_cal_split(rest, f)
        assert not (set(fit_ids) | set(cal_ids)) & held and not set(fit_ids) & set(cal_ids)
        fit_seqs = [q for u in fit_ids for q in by_user[u]]
        cal_seqs = [q for u in cal_ids for q in by_user[u]]
        held_seqs = [q for u in held_ids for q in by_user[u]]
        bkt = B.BKT(min_seq).fit(fit_seqs)
        for d in defs:
            ch = calibrate_and_choose(cal_seqs, bkt, d)
            per_def[d.name]["parts"].append(heldout_student_vectors(held_seqs, bkt, d, ch["platt"], ch["taus"]))
            per_def[d.name]["taus"].append(ch["taus"])
            per_def[d.name]["platt"].append(list(ch["platt"]))
            per_def[d.name]["n_cal"].append(ch["n_cal_sequences"])
            log(f"fold {f} {d.name}: taus {ch['taus']} platt ({ch['platt'][0]:.3f}, {ch['platt'][1]:.3f}) cal sequences {ch['n_cal_sequences']}")
    out = {"status": STATUS, "spec": "aria_pace/SPEC.md Amendment 7", "folds": N_FOLDS, "fold_sizes_students": [len(x) for x in folds],
           "fold_seed": FOLD_SEED, "bootstrap": {"draws": BOOT_DRAWS, "seed": BOOT_SEED}, "bkt_min_seq": min_seq,
           "risk_scores": "BKT only (PFA and the risk head were not re-run: A7.5)", "definitions": {}}
    control = per_def[defs[0].name]["taus"]
    for d in defs:
        pooled = pool(per_def[d.name]["parts"])
        res = curves_and_points(pooled)
        res["definition"] = d.text
        res["fold_platt"] = per_def[d.name]["platt"]
        res["fold_cal_sequences"] = per_def[d.name]["n_cal"]
        res["tau_movement"] = tau_movement(per_def[d.name]["taus"], d.name, frozen_tau, None if d.name == defs[0].name else control)
        out["definitions"][d.name] = res
    return out
