"""Figure 1 (CALE-AI paper): the entanglement timeline for one real validation sequence. No test data; no new analysis.

Risk = the frozen pipeline's BKT stuck risk (BKT fitted on train, min_seq from frozen_val.json) with the frozen Platt
calibrator, i.e. exactly the quantity the validation evaluation and CALE E2/E5 use. Logged ASSISTments hints = hint_count > 0
(the E5 indicator), aligned by the E5 helper.

Selection rule (fixed in this file before any candidate was looked at; see paper/cale/FIGURES.md):
the FIRST sequence, in the order of policy.Table(validation, include_leavers=False), such that
  (1) it is stuck (y = 1);
  (2) it has >= 3 verified errors at decision points (tries 3-9);
  (3) calibrated risk is below the Quiet threshold 0.57 at its first decision point and reaches >= 0.57 at a later one;
  (4) the Quiet policy fires (a verified error with risk >= 0.57) at some try i*;
  (5) ASSISTments logged a hint (hint_count > 0) at some try in 3..i*, i.e. in the same window, at or before ARIA's hint.
Writes paper/cale/figures/fig1_entanglement.{pdf,png} and paper/cale/figures/fig1_selection.json.
"""
import hashlib
import json

import numpy as np

import common as C
import figstyle as FS
from aria_pace import baselines as B
from aria_pace import policy as P
from e5_logged_hints import hint_arrays

import matplotlib.pyplot as plt  # noqa: E402  (figstyle sets the backend and rcParams)
from matplotlib.lines import Line2D  # noqa: E402

TAU = C.FROZEN_TAUS["quiet"]
OUTDIR = C.REPO / "paper" / "cale" / "figures"
T_SHOW = 10


def select(tab, cal, H):
    for r, q in enumerate(tab.seqs):
        if tab.y[r] != 1:
            continue
        v, e = tab.valid[r], tab.err_primary[r]
        if (v & e).sum() < 3:
            continue
        s = cal[r]
        js = np.flatnonzero(v)
        if s[js[0]] >= TAU or not (s[js[1:]] >= TAU).any():
            continue
        fire = np.flatnonzero(v & e & (s >= TAU))
        if not len(fire):
            continue
        i_star = int(P.DEC[fire[0]])
        h = H[(q.user_id, q.assignment_id)]
        assert np.array_equal(h["correct"], q.correct), "hint columns misaligned"
        if not (h["hint_count"][2:i_star] > 0).any():
            continue
        return r, i_star, h
    raise RuntimeError("no sequence meets the rule")


def main():
    log = C.log_fn()
    df, parts, train, val, _ = C.load_pool(log)
    frozen = C.read_result("frozen_val.json")
    a, b = frozen["platt"]["bkt"]["intercept"], frozen["platt"]["bkt"]["slope"]
    bkt = B.BKT(frozen["models"]["bkt_min_seq"]).fit(train)
    tab = P.Table(val, include_leavers=False)
    C.assert_no_test(tab.sid, parts)
    cal = P.platt_apply(P.score_matrix(tab, bkt.score), a, b)
    H = hint_arrays(df)
    r, i_star, h = select(tab, cal, H)
    q = tab.seqs[r]
    import pandas as pd
    names = pd.read_csv(C.RAW_CSV, encoding="latin-1", usecols=["skill_id", "skill_name"], low_memory=False).dropna()
    skill = str(names.loc[names.skill_id.astype(int).astype(str) == str(int(float(q.kc))), "skill_name"].iloc[0])
    n_show = min(T_SHOW, q.n)
    tries = np.arange(1, n_show + 1)
    corr = q.correct[:n_show]
    fa = q.first_action[:n_show]
    hint = h["hint_count"][:n_show] > 0
    v, e, s = tab.valid[r], tab.err_primary[r], cal[r]
    dp_try = P.DEC[v]
    dp_risk = s[v]
    ve_try = P.DEC[v & e]

    fig, (ax, st) = plt.subplots(2, 1, figsize=(FS.WIDTH, 2.35), sharex=True, gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.08})
    q_col = FS.SETTING["quiet"]
    ax.axvspan(2.5, 9.5, color="#F2F2F2", zorder=0, lw=0)
    ax.axhline(TAU, color=FS.OI["gray"], ls="--", lw=0.9, zorder=1)
    ax.text(1.0, TAU + 0.03, r"Quiet threshold $\tau$ = 0.57", color=FS.OI["gray"], va="bottom", ha="left")
    ax.plot(dp_try, dp_risk, color=q_col, zorder=3)
    ax.scatter(dp_try, dp_risk, s=14, facecolor="white", edgecolor=q_col, lw=0.9, zorder=4, marker=FS.SETTING_MARKER["quiet"])
    ax.scatter(ve_try, s[np.isin(P.DEC, ve_try)], s=16, facecolor=q_col, edgecolor=q_col, zorder=5, marker=FS.SETTING_MARKER["quiet"])
    ax.annotate("ARIA (Quiet)\nfirst hint", xy=(i_star, s[list(P.DEC).index(i_star)]), xytext=(9.45, 0.2),
                ha="right", va="center", arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": "black"})
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Stuck risk")
    ax.set_yticks([0, 0.5, 1])
    ax.text(2.65, 1.03, "tries 3-9", ha="left", va="top", color=FS.OI["gray"])

    y_ans, y_hint = 1.0, 0.0
    for t, c, f in zip(tries, corr, fa):
        if c == 1:
            st.scatter(t, y_ans, marker="o", s=16, facecolor="black", edgecolor="black", zorder=3)
        elif f == 0:
            st.scatter(t, y_ans, marker="x", s=18, color=FS.OI["vermillion"], lw=1.2, zorder=3)
        else:
            st.scatter(t, y_ans, marker="x", s=18, color=FS.OI["vermillion"], lw=1.2, zorder=3)
            st.scatter(t, y_ans, marker="o", s=34, facecolor="none", edgecolor=FS.OI["vermillion"], lw=0.7, zorder=3)
    st.scatter(tries[hint], np.full(hint.sum(), y_hint), marker="D", s=16, color=FS.OI["orange"], zorder=3)
    st.set_ylim(-0.6, 1.6)
    st.set_yticks([y_ans, y_hint])
    st.set_yticklabels(["Answer", "ASSISTments\nhint"])
    st.tick_params(axis="y", length=0)
    st.spines["left"].set_visible(False)
    st.set_xlim(0.5, n_show + 0.5)
    st.set_xticks(tries)
    st.set_xlabel("Try (problem in the assignment)")
    handles = [Line2D([], [], color=q_col, marker="o", mfc=q_col, ms=4, label="Risk at a verified error"),
               Line2D([], [], color=q_col, marker="o", mfc="white", ms=4, label="Risk, other try"),
               Line2D([], [], ls="", marker="o", color="black", ms=4, label="Correct"),
               Line2D([], [], ls="", marker="x", color=FS.OI["vermillion"], ms=5, label="Wrong"),
               Line2D([], [], ls="", marker="D", color=FS.OI["orange"], ms=4, label="Logged hint")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=1.4,
               columnspacing=0.9, handletextpad=0.4)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.98, bottom=0.36)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "fig1_entanglement.pdf")
    fig.savefig(OUTDIR / "fig1_entanglement.png", dpi=300)
    plt.close(fig)

    sel = {"rule": __doc__.split("FIGURES.md):")[1].split("Writes")[0].strip(),
           "position_in_validation_table_order": r, "sequences_scanned_before_it": r,
           "sequence_hash": hashlib.sha256(f"{q.user_id}:{q.assignment_id}".encode()).hexdigest()[:12],
           "skill_name": skill, "n_opportunities": int(q.n), "label_stuck": int(tab.y[r]),
           "correct_tries_1_10": corr.tolist(), "first_action_tries_1_10": fa.tolist(),
           "hint_count_tries_1_10": h["hint_count"][:n_show].tolist(),
           "decision_tries": dp_try.tolist(), "risk_at_decision_tries": [round(float(x), 3) for x in dp_risk],
           "verified_error_tries": ve_try.tolist(), "quiet_first_hint_try": i_star,
           "logged_hint_tries": tries[hint].tolist(),
           "note": "user_id and assignment_id are not stored; sequence_hash = sha256('user_id:assignment_id')[:12]. Validation split only."}
    (OUTDIR / "fig1_selection.json").write_text(json.dumps(sel, indent=1))
    log(f"picked validation sequence #{r}; quiet fires at try {i_star}; hints at {sel['logged_hint_tries']}")


if __name__ == "__main__":
    main()
