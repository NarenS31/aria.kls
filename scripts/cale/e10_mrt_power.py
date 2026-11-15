"""E10: micro-randomized trial planning for next-answer correctness (train + validation; exploratory; ANALYSIS_PLAN.md Addendum 2).

Decision point (DP): try i in 3..9, i <= n, not mastered through try i, verified error at try i (correct 0, first_action 0).
Proximal outcome: correct at try i + 1. Session = one sequence (no session id or timestamp in the 2009-10 file).
Two answers per effect size: the Liao et al. (2016) formula and a nonparametric simulation that replays real students.
"""
import time

import numpy as np
from scipy import optimize, stats

import common as C
from aria_pace.constants import MASTERY_RUN

SEED_BOOT = 20261119
SEED_SIM = 20261120
RHO, ALPHA, POWER = 0.5, 0.05, 0.80
Q, P_ = 1, 1                       # control and effect parameters in the working model
EFFECTS = (0.03, 0.05, 0.08)
GRID = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6)
N_GRID_SIMS, N_CONFIRM, N_CAL_SIMS, N_CAL_STUDENTS, CAL_ITERS = 1000, 2000, 200, 2000, 3
SCENARIOS = ("obs", 1, 5, 10)
TRIES = 10                         # DP up to try 9, outcome up to try 10
MAX_ROWS = 3_000_000               # rows per replay batch


# ---------------------------------------------------------------- data

def arrays(seqs):
    S = len(seqs)
    c = np.full((S, TRIES), -1, np.int8)
    fa = np.full((S, TRIES), -1, np.int8)
    n = np.array([q.n for q in seqs])
    for r, q in enumerate(seqs):
        k = min(TRIES, q.n)
        c[r, :k] = q.correct[:k]
        fa[r, :k] = q.first_action[:k]
    return c, fa, n


def replay(c, fa, n, delta, rng):
    """Replays sequences row-wise; randomizes at each DP and applies the flip model. Returns per-DP arrays.

    c, fa are modified in place (pass copies). Returns (row, try, has_outcome, A, Y0, Y)."""
    S = len(n)
    run = np.zeros(S, np.int16)
    mastered = np.zeros(S, bool)
    out = []
    for i in range(1, TRIES):                       # 1-based try
        x = c[:, i - 1]
        run = np.where(x == 1, run + 1, 0)
        mastered |= run >= MASTERY_RUN
        if i < 3:
            continue
        dp = (i <= n) & ~mastered & (x == 0) & (fa[:, i - 1] == 0)
        rows = np.flatnonzero(dp)
        has = n[rows] > i
        A = rng.random(len(rows)) < RHO
        y0 = c[rows, i].copy()
        flip = has & A & (y0 == 0) & (rng.random(len(rows)) < delta)
        c[rows[flip], i] = 1
        fa[rows[flip], i] = 0
        out.append((rows, np.full(len(rows), i), has, A, y0, c[rows, i].copy()))
    return [np.concatenate(z) for z in zip(*out)]


def observed_dps(seqs):
    """Deterministic DP list on the logged data, by direct loops (cross-check for the vectorized replay)."""
    res = []
    for r, q in enumerate(seqs):
        for i in range(3, 10):
            if i <= q.n and q.m > i and q.correct[i - 1] == 0 and q.first_action[i - 1] == 0:
                res.append((r, i, q.n > i, int(q.correct[i]) if q.n > i else -1))
    return np.array(res)


# ---------------------------------------------------------------- Liao et al. (2016)

def liao_power(N, d, tau):
    df2 = N - Q - P_
    lam = N * RHO * (1 - RHO) * d * d * tau
    crit = stats.f.ppf(1 - ALPHA, P_, df2)
    return float(stats.ncf.sf(crit, P_, df2, lam))


def liao_n(d, tau):
    lo, hi = Q + P_ + 2, 10
    while liao_power(hi, d, tau) < POWER:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if liao_power(mid, d, tau) >= POWER:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------- simulation

class Pool:
    def __init__(self, seqs, sid):
        order = np.argsort(sid, kind="stable")
        self.c, self.fa, self.n = arrays([seqs[k] for k in order])
        s_sorted = sid[order]
        self.users, self.start, self.count = np.unique(s_sorted, return_index=True, return_counts=True)
        self.U = len(self.users)

    def draw(self, rng, B, N, K):
        """Rows for B trials of N students. Returns (row index into pool, cluster id = trial * N + slot)."""
        stu = rng.integers(0, self.U, size=(B, N)).ravel()
        cl = np.arange(B * N)
        if K == "obs":
            cnt = self.count[stu]
            cl_rows = np.repeat(cl, cnt)
            off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            rows = np.repeat(self.start[stu], cnt) + off
        else:
            cl_rows = np.repeat(cl, K)
            s_rep = np.repeat(stu, K)
            rows = self.start[s_rep] + (rng.random(len(s_rep)) * self.count[s_rep]).astype(int)
        return rows, cl_rows

    def rows_per_student(self, K):
        return float(self.count.mean()) if K == "obs" else float(K)


def simulate(pool, K, N, delta, B, rng):
    """B simulated trials. Returns dict of per-trial arrays: reject, beta, achieved effect parts, DPs."""
    rows, cl = pool.draw(rng, B, N, K)
    c, fa, n = pool.c[rows].copy(), pool.fa[rows].copy(), pool.n[rows]
    r, _, has, A, y0, y = replay(c, fa, n, delta, rng)
    r, A, y0, y = r[has], A[has].astype(float), y0[has].astype(float), y[has].astype(float)
    clu = cl[r]
    tr = clu // N
    x = A - RHO
    bc = lambda w: np.bincount(tr, weights=w, minlength=B)
    s1, sx, sxx, sy, sxy = bc(np.ones_like(x)), bc(x), bc(x * x), bc(y), bc(x * y)
    det = s1 * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = (sxx * sy - sx * sxy) / det
        b1 = (s1 * sxy - sx * sy) / det
        e = y - b0[tr] - b1[tr] * x
        g0 = np.bincount(clu, weights=e, minlength=B * N)
        g1 = np.bincount(clu, weights=x * e, minlength=B * N)
        ct = np.arange(B * N) // N
        m00 = np.bincount(ct, weights=g0 * g0, minlength=B)
        m01 = np.bincount(ct, weights=g0 * g1, minlength=B)
        m11 = np.bincount(ct, weights=g1 * g1, minlength=B)
        # V = Binv M Binv, Binv = [[sxx, -sx], [-sx, s1]] / det; element (1,1):
        v11 = (sx * sx * m00 - 2 * sx * s1 * m01 + s1 * s1 * m11) / det ** 2 * N / (N - Q - P_)
        tstat = b1 / np.sqrt(v11)
    crit = stats.t.ppf(1 - ALPHA / 2, N - Q - P_)
    rej = np.abs(np.nan_to_num(tstat)) > crit
    return {"reject": rej, "beta": b1, "dps": s1, "room": bc(1 - y0), "n_dp": s1}


def run_trials(pool, K, N, delta, total, rng):
    per = max(1, int(MAX_ROWS / (N * pool.rows_per_student(K))))
    acc = {"reject": [], "beta": [], "dps": [], "room": [], "n_dp": []}
    done = 0
    while done < total:
        b = min(per, total - done)
        o = simulate(pool, K, N, delta, b, rng)
        for k in acc:
            acc[k].append(o[k])
        done += b
    return {k: np.concatenate(v) for k, v in acc.items()}


def calibrate_delta(pool, K, effect, rng):
    """delta such that the average proximal effect over the trial's DPs, delta * E[1 - Y0], equals the effect."""
    delta, hist = effect / 0.5, []
    for _ in range(CAL_ITERS):
        o = run_trials(pool, K, N_CAL_STUDENTS, min(delta, 1.0), N_CAL_SIMS, rng)
        room = o["room"].sum() / o["n_dp"].sum()
        delta = effect / room
        hist.append({"delta": delta, "share_next_wrong_under_trial": room})
    assert delta <= 1.0, "effect larger than the room for improvement"
    return delta, hist


def probit_fit(Ns, k, m):
    x = np.log(Ns)

    def nll(th):
        p = np.clip(stats.norm.cdf(th[0] + th[1] * x), 1e-12, 1 - 1e-12)
        return -(k * np.log(p) + (m - k) * np.log(1 - p)).sum()
    th = optimize.minimize(nll, [-5.0, 1.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000}).x
    return th, float(np.exp((stats.norm.ppf(POWER) - th[0]) / th[1]))


# ---------------------------------------------------------------- descriptives

def boot_ratio(cnt, num, den):
    return C.ci((cnt @ num) / (cnt @ den))


def icc_anova(n_i, y_i):
    """Method-of-moments ICC for a binary outcome with unbalanced clusters (one-way ANOVA)."""
    k, Nt = len(n_i), n_i.sum()
    p_i = y_i / n_i
    p = y_i.sum() / Nt
    msb = (n_i * (p_i - p) ** 2).sum() / (k - 1)
    msw = (n_i * p_i * (1 - p_i)).sum() / (Nt - k)
    n0 = (Nt - (n_i ** 2).sum() / Nt) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def describe(seqs, sid, kc, dp, log):
    users, uidx = np.unique(sid, return_inverse=True)
    U = len(users)
    seq_stu = uidx
    r, i, has, y = dp[:, 0], dp[:, 1], dp[:, 2].astype(bool), dp[:, 3]
    S = len(seqs)
    dps_seq = np.bincount(r, minlength=S)
    dpo_seq = np.bincount(r[has], minlength=S)
    seq_per_stu = np.bincount(seq_stu, minlength=U)
    dps_stu = np.bincount(seq_stu, weights=dps_seq, minlength=U)
    dpo_stu = np.bincount(seq_stu, weights=dpo_seq, minlength=U)
    per_session_mean = dpo_stu / seq_per_stu
    rng = np.random.RandomState(SEED_BOOT)
    cnt = rng.multinomial(U, np.ones(U) / U, size=C.N_BOOT).astype(float)
    ro, yo = r[has], y[has].astype(float)
    so = seq_stu[ro]
    ys = np.bincount(so, weights=yo, minlength=U)
    ns = np.bincount(so, minlength=U).astype(float)
    p0 = float(ys.sum() / ns.sum())

    def dist(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "median": float(np.median(v)),
                "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)), "p90": float(np.percentile(v, 90)),
                "max": float(v.max()), "share_zero": float((v == 0).mean())}

    has_dp = dpo_stu > 0
    by_try = {}
    for t in range(3, 10):
        m = i[has] == t
        num = np.bincount(so[m], weights=yo[m], minlength=U)
        den = np.bincount(so[m], minlength=U).astype(float)
        by_try[t] = {"dps_with_outcome": int(m.sum()), "p_next_correct": float(yo[m].mean()), "ci95": boot_ratio(cnt, num, den)}
    five = ns >= 5
    rates5 = ys[five] / ns[five]
    binom_sd = float(np.sqrt(np.mean(rates5.mean() * (1 - rates5.mean()) / ns[five])))
    icc = icc_anova(ns[ns > 0], ys[ns > 0])
    icc_b = []
    for b in range(C.N_BOOT):
        w = cnt[b].astype(int)
        keep = (w > 0) & (ns > 0)
        icc_b.append(icc_anova(np.repeat(ns[keep], w[keep]), np.repeat(ys[keep], w[keep])))
    kc_o = np.array(kc)[ro]
    skills = {}
    for k in np.unique(kc_o):
        m = kc_o == k
        if m.sum() >= 100:
            skills[k] = (int(m.sum()), float(yo[m].mean()))
    sr = np.array([v[1] for v in skills.values()])
    # consecutive DPs within a student (order: sequence order within user, then try)
    order = np.lexsort((i[has], ro, so))
    s_o, y_o, r_o = so[order], yo[order], ro[order]
    same_stu = s_o[1:] == s_o[:-1]
    same_seq = same_stu & (r_o[1:] == r_o[:-1])
    corr = lambda m: {"pairs": int(m.sum()), "r": float(np.corrcoef(y_o[:-1][m], y_o[1:][m])[0, 1])}
    log(f"p0 {p0:.4f}; DPs with outcome {int(has.sum())} of {len(r)}; ICC {icc:.3f}")
    return {
        "students": U, "sessions": S, "sessions_per_student": dist(seq_per_stu),
        "decision_points": {"total": int(len(r)), "with_outcome": int(has.sum()), "without_outcome_left_assignment": int((~has).sum())},
        "dps_per_session_with_outcome": {**dist(dpo_seq), "share_sessions_with_any": float((dpo_seq > 0).mean())},
        "dps_per_session_incl_no_outcome": dist(dps_seq),
        "dps_per_session_with_outcome_sessions_n_ge_3": dist(dpo_seq[np.array([q.n for q in seqs]) >= 3]),
        "dps_per_student_total_with_outcome": {**dist(dpo_stu), "ci95_mean": C.ci(cnt @ dpo_stu / U)},
        "dps_per_student_total_with_outcome_students_with_any": dist(dpo_stu[has_dp]),
        "dps_per_student_total_incl_no_outcome": dist(dps_stu),
        "students_per_session_mean_dps_with_outcome": {**dist(per_session_mean), "ci95_mean": C.ci(cnt @ per_session_mean / U)},
        "students_with_any_dp": int(has_dp.sum()),
        "share_of_dps_with_outcome_from_top_students": {f"top_{q}pct": float(np.sort(dpo_stu)[::-1][:U * q // 100].sum() / dpo_stu.sum())
                                                       for q in (10, 20)},
        "base_rate_next_correct": {"p0": p0, "ci95": boot_ratio(cnt, ys, ns), "sd_bernoulli": float(np.sqrt(p0 * (1 - p0)))},
        "by_try": by_try,
        "between_student": {"students_with_ge5_dps": int(five.sum()), "sd_of_student_rates": float(rates5.std(ddof=1)),
                            "sd_expected_from_binomial_noise_alone": binom_sd,
                            "student_rate_p10_p50_p90": [float(np.percentile(rates5, q)) for q in (10, 50, 90)],
                            "icc_anova": icc, "icc_ci95": C.ci(np.array(icc_b))},
        "between_skill": {"skills_with_ge100_dps": len(skills), "dps_covered": int(sum(v[0] for v in skills.values())),
                          "median": float(np.median(sr)), "iqr": [float(np.percentile(sr, 25)), float(np.percentile(sr, 75))],
                          "min": float(sr.min()), "max": float(sr.max())},
        "lag1_correlation": {"consecutive_dps_same_student": corr(same_stu), "consecutive_dps_same_session": corr(same_seq)},
        "_arrays": {"per_session_mean": per_session_mean, "dpo_stu": dpo_stu},
    }


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    log = C.log_fn()
    _, parts, train, val, _ = C.load_pool(log)
    seqs = train + val
    sid = np.array([q.user_id for q in seqs])
    C.assert_no_test(sid, parts)
    kc = [q.kc for q in seqs]
    dp = observed_dps(seqs)

    # cross-check: vectorized replay with delta = 0 finds exactly the same DPs and outcomes
    c, fa, n = arrays(seqs)
    rr, ii, hh, _, y0, yy = replay(c.copy(), fa.copy(), n, 0.0, np.random.default_rng(0))
    o1 = np.lexsort((ii, rr))
    o2 = np.lexsort((dp[:, 1], dp[:, 0]))
    assert np.array_equal(rr[o1], dp[o2, 0]) and np.array_equal(ii[o1], dp[o2, 1]) and np.array_equal(hh[o1], dp[o2, 2].astype(bool))
    assert np.array_equal(np.where(hh, yy, -1)[o1], dp[o2, 3]) and np.array_equal(y0, yy)

    desc = describe(seqs, sid, kc, dp, log)
    arr = desc.pop("_arrays")
    p0 = desc["base_rate_next_correct"]["p0"]
    sigma = np.sqrt(p0 * (1 - p0))
    tau = {"obs": float(arr["dpo_stu"].mean())}
    for K in SCENARIOS[1:]:
        tau[K] = float(K * arr["per_session_mean"].mean())

    pool = Pool(seqs, sid)
    rng = np.random.default_rng(SEED_SIM)
    results = {}
    for K in SCENARIOS:
        for eff in EFFECTS:
            key = f"S_{K}|delta_{int(round(eff * 100))}pt"
            d = eff / sigma
            nL = liao_n(d, tau[K])
            delta, hist = calibrate_delta(pool, K, eff, rng)
            Ns = np.unique(np.maximum(10, np.round(nL * np.array(GRID)).astype(int)))
            grid = []
            for N in Ns:
                o = run_trials(pool, K, int(N), delta, N_GRID_SIMS, rng)
                grid.append({"N": int(N), "power": float(o["reject"].mean()), "liao_power": liao_power(int(N), d, tau[K]),
                             "mean_dps_with_outcome_per_student": float(o["n_dp"].mean() / N),
                             "mean_beta_hat": float(np.nanmean(o["beta"]))})
            k = np.array([g["power"] * N_GRID_SIMS for g in grid])
            th, n_star = probit_fit(np.array([g["N"] for g in grid], float), k, np.full(len(grid), N_GRID_SIMS))
            nS = int(np.ceil(n_star))
            conf = run_trials(pool, K, nS, delta, N_CONFIRM, rng)
            pw = float(conf["reject"].mean())
            at_liao = next(g for g in grid if g["N"] == nL)
            liao_ok = 0.77 <= at_liao["power"] <= 0.83
            results[key] = {
                "scenario": K, "effect": eff, "standardized_effect_d": float(d), "tau_bar": tau[K],
                "liao": {"N": nL, "power_formula": liao_power(nL, d, tau[K]), "simulated_power_at_N": at_liao["power"],
                         "simulated_power_mc_se": float(np.sqrt(at_liao["power"] * (1 - at_liao["power"]) / N_GRID_SIMS))},
                "simulation": {"delta_flip_probability": delta, "calibration": hist,
                               "grid": grid, "probit": [float(th[0]), float(th[1])], "N": nS,
                               "confirm_power": pw, "confirm_mc_se": float(np.sqrt(pw * (1 - pw) / N_CONFIRM)),
                               "confirm_mean_beta_hat": float(np.nanmean(conf["beta"])),
                               "achieved_effect_in_confirm_trials": float(delta * conf["room"].sum() / conf["n_dp"].sum()),
                               "confirm_dps_with_outcome_per_student": float(conf["n_dp"].mean() / nS)},
                "decision": "liao" if liao_ok else "simulation",
                "answer_N": nL if liao_ok else nS,
            }
            log(f"{key}: Liao N {nL} (sim power there {at_liao['power']:.3f}); sim N {nS} (confirm {pw:.3f}); delta {delta:.3f}")

    # Type I error check
    n5 = results["S_obs|delta_5pt"]["simulation"]["N"]
    o = run_trials(pool, "obs", n5, 0.0, N_CONFIRM, rng)
    t1 = float(o["reject"].mean())
    log(f"type I at N {n5}: {t1:.4f}")

    out = {"analysis": "E10 micro-randomized trial planning: proximal effect of a hint after a wrong step on next-answer correctness",
           "data": "train + validation students (no test student), all sequences, no policy-scope filter",
           "session_definition": "one sequence (one student's rows in one assignment); the file has no session id or timestamp",
           "decision_point": "try i in 3..9, i <= n, not mastered through try i (MASTERY_RUN consecutive correct), correct == 0 and first_action == 0 at try i",
           "proximal_outcome": "correct at try i + 1 (first answer on the next problem); DPs without try i + 1 have no outcome",
           "design": {"randomization_probability": RHO, "alpha_two_sided": ALPHA, "target_power": POWER, "effects_risk_difference": list(EFFECTS)},
           "liao_method": {"reference": "Liao, Klasnja, Tewari, Murphy (2016), Statistics in Medicine 35(12):1944-1971",
                           "working_model": "Y = a0 + (A - rho) b0 + e; constant effect (p = 1), intercept-only control (q = 1)",
                           "standardized_effect": "d = effect / sqrt(p0 (1 - p0))",
                           "noncentrality": "N rho (1 - rho) d^2 tau_bar, tau_bar = expected DPs with outcome per student",
                           "test": "noncentral F(1, N - q - p)", "tau_bar_by_scenario": tau},
           "simulation_method": {"resampling": "real train + validation students, uniformly with replacement; S_obs keeps all their sessions, S_K draws K of their sessions with replacement",
                                 "effect_model": "under A = 1 a logged wrong next answer becomes correct with probability delta; later tries as logged; DPs recomputed after each change",
                                 "analysis": "WCLS: OLS of Y on (1, A - 0.5), student-clustered sandwich x N/(N - 2), two-sided t(N - 2)",
                                 "grid_multipliers_of_liao_N": list(GRID), "sims_per_grid_point": N_GRID_SIMS,
                                 "confirm_sims": N_CONFIRM, "calibration": f"{CAL_ITERS} iterations x {N_CAL_SIMS} trials of {N_CAL_STUDENTS} students", "seed": SEED_SIM},
           "estimates": desc, "bootstrap": {"draws": C.N_BOOT, "seed": SEED_BOOT, "unit": "student"},
           "results": results,
           "type_I_check": {"scenario": "obs", "N": n5, "trials": N_CONFIRM, "rejection_rate": t1,
                            "mc_se": float(np.sqrt(t1 * (1 - t1) / N_CONFIRM))}}
    log(f"wrote {C.write('e10_mrt_power.json', out, t0)}")


if __name__ == "__main__":
    main()
