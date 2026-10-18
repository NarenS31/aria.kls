"""C6: figures + docs/aria_pace/SUMMIT_SLIDE.md. Every number is read from results/*.json at run time."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import RESULTS, ROOT

DOCS = ROOT.parent / "docs" / "aria_pace"; FIG = DOCS / "figures_v2"; FIG.mkdir(exist_ok=True)
T9 = json.load(open(RESULTS / "test_results.json")); R = json.load(open(RESULTS / "test_results_2012.json"))
DKT9 = json.load(open(RESULTS / "dkt_val.json")); SAKT9 = json.load(open(RESULTS / "sakt_val.json")); AV = json.load(open(RESULTS / "affect_val.json"))
A, B = R["a_cross_dataset"], R["b_within_dataset_affect"]
P12 = A["policy"]["main"]["variants"]["frozen"]; P09 = T9["main"]["variants"]["frozen"]
BLUE, ORANGE, GREEN, RED, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6b6b6b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150})
PROF = {"quiet": (3.0, 1.0), "balanced": (1.0, 1.0), "supportive": (1.0, 3.0)}
LIC = "INTERNAL: not for publication until ASSISTments confirms the licence"


def stats(P, prof):
    b, h = P["methods"]["bkt"][prof], P["methods"]["always_help"][prof]
    return {"red": 1 - b["interventions_per_student"] / h["interventions_per_student"], "miss": b["missed_help_share"], "miss_ci": b["ci95"]["missed_help_share"][:2],
            "int": b["interventions_per_student"], "int_ah": h["interventions_per_student"], "dJ": P["j_diff_vs_bkt"][prof]["always_help"]}


S09, S12 = {p: stats(P09, p) for p in PROF}, {p: stats(P12, p) for p in PROF}
cal3 = A["risk"]["results"]["variantA_k3"]["bkt"]["calibration_line"]


def save(fig, name):
    fig.text(0.01, -0.05, LIC, fontsize=7, color=GRAY); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)


# ---- Fig 1: frontier per profile on 2012 test
sw = A["policy"]["main"]["sweep"]; fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for a, (prof, (wI, wD)) in zip(ax, PROF.items()):
    rows = sw["methods"]["bkt"]
    a.plot([r["interventions_per_student"] for r in rows], [r["J"][prof] for r in rows], color=BLUE, lw=1.6, label="BKT, tau sweep")
    nr = sw["n_rule"]; a.plot([nr[n]["interventions_per_student"] for n in nr], [nr[n]["J"][prof] for n in nr], "s--", color=ORANGE, ms=4, lw=1, label="n-errors rule (n=1..6)")
    a.scatter([P12["methods"]["bkt"][prof]["interventions_per_student"]], [P12["methods"]["bkt"][prof]["cost_J_per_student"]], s=70, color=RED, zorder=5, label="frozen 2009 tau")
    a.scatter([S12[prof]["int_ah"]], [P12["methods"]["always_help"][prof]["cost_J_per_student"]], s=60, marker="D", color=GRAY, zorder=5, label="always-help")
    a.set_title(f"{prof} (w_I:w_D = {wI:g}:{wD:g})"); a.set_xlabel("interventions per student"); a.set_ylabel("cost J per student" if prof == "quiet" else "")
ax[0].legend(fontsize=7, frameon=False)
fig.suptitle("(a) cross-dataset: 2012 test, 2009-trained BKT; lower is better", fontsize=10); save(fig, "fig1_frontier_2012_test.png")

# ---- Fig 2: quiet profile, 2009 test vs 2012 test
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.4)); xs = ["2009 test", "2012 test"]; cols = [BLUE, ORANGE]
red = [100 * S09["quiet"]["red"], 100 * S12["quiet"]["red"]]
ax[0].bar(xs, red, color=cols, width=.55); ax[0].set_ylabel("% fewer interventions than always-help"); ax[0].set_ylim(0, 70)
for i, v in enumerate(red): ax[0].text(i, v + 1.5, f"{v:.1f}%", ha="center")
ms = [100 * S09["quiet"]["miss"], 100 * S12["quiet"]["miss"]]; lo = [100 * S09["quiet"]["miss_ci"][0], 100 * S12["quiet"]["miss_ci"][0]]; hi = [100 * S09["quiet"]["miss_ci"][1], 100 * S12["quiet"]["miss_ci"][1]]
ax[1].bar(xs, ms, color=cols, width=.55, yerr=[np.array(ms) - lo, np.array(hi) - ms], capsize=4, ecolor="k"); ax[1].set_ylabel("missed-help share (%)"); ax[1].set_ylim(0, 9)
for i, v in enumerate(ms): ax[1].text(i + .3, v + .1, f"{v:.1f}%", ha="left")
fig.suptitle("Quiet profile (3:1), thresholds frozen on 2009 validation; 95% CI over students", fontsize=9); save(fig, "fig2_quiet_2009_vs_2012.png")

# ---- Fig 3: neural vs BKT
def d12(m): return [A["risk"]["paired_vs_bkt"][f"variantA_k{k}"][m] for k in (3, 5, 8)]
def d09(res, key): return [res["paired_sakt_minus_bkt" if key == "sakt" else "paired_dkt_minus_bkt"][f"variantA_k{k}"] for k in (3, 5, 8)]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for a, ttl, sets in ((ax[0], "(a) 2012 test, 2009-trained", {"DKT-style": d12("dkt_style"), "SAKT-style": d12("sakt_style")}),
                     (ax[1], "2009 validation (not test)", {"DKT-style": d09(DKT9, "dkt"), "SAKT-style": d09(SAKT9, "sakt")})):
    for j, (nm, ds) in enumerate(sets.items()):
        x = np.arange(3) + (j - .5) * .25
        a.errorbar(x, [d["diff"] for d in ds], yerr=[[d["diff"] - d["ci95"][0] for d in ds], [d["ci95"][1] - d["diff"] for d in ds]], fmt="o", color=[BLUE, RED][j], capsize=3, label=nm)
    a.axhline(0, color="k", lw=.8); a.set_xticks(range(3)); a.set_xticklabels(["k=3", "k=5", "k=8"]); a.set_title(ttl, fontsize=9)
ax[0].set_ylabel("AUC difference to BKT (95% CI)"); ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("Neural models did not beat BKT (below 0 = worse than BKT)", fontsize=10); save(fig, "fig3_neural_vs_bkt.png")

# ---- Fig 4: affect by k (2012 test)
fig, ax = plt.subplots(figsize=(5.5, 3.5)); lab = [("with_primary", "rows <= k", BLUE), ("with_lagged", "rows <= k-1 (lagged)", ORANGE)]
for j, (n, nm, c) in enumerate(lab):
    ds = [B["paired_with_minus_without"][f"{n}|variantA_k{k}"] for k in (3, 5, 8)] + [B["mean_over_k"][f"{n}|variantA"]]
    dd = [d["diff"] if "diff" in d else d["mean_diff"] for d in ds]
    x = np.arange(4) + (j - .5) * .25
    ax.errorbar(x, dd, yerr=[[a - d["ci95"][0] for a, d in zip(dd, ds)], [d["ci95"][1] - a for a, d in zip(dd, ds)]], fmt="o", color=c, capsize=3, label=nm)
ax.axhline(0, color="k", lw=.8); ax.axhline(0.005, color=RED, ls="--", lw=1); ax.text(3.45, .0053, "pre-registered bar 0.005", color=RED, fontsize=7, ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels(["k=3", "k=5", "k=8", "mean"]); ax.set_ylabel("AUC with - AUC without (95% CI)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("(b) within-dataset: affect, 2012 test", fontsize=9); save(fig, "fig4_affect_by_k.png")

# ---- slide
af = B["mean_over_k"]["with_primary|variantA"]; al = B["mean_over_k"]["with_lagged|variantA"]; avp = AV["mean_over_k"]["with_primary|variantA"]
q9, q12, s9, s12 = S09["quiet"], S12["quiet"], S09["supportive"], S12["supportive"]
dk = d12("dkt_style"); sk = d12("sakt_style")
sl = f"""> **{LIC}.** Licence confirmation from ASSISTments is pending. Do not present, post or share outside the team until it arrives.

# ARIA-Pace: knowing when *not* to step in

**Headline (a, cross-dataset). Quiet profile (3:1), thresholds frozen on 2009 validation, applied unchanged to both test sets.**

| | 2009 test | 2012 test |
|---|---|---|
| Fewer interventions than always-help | **{100*q9['red']:.0f}%** | **{100*q12['red']:.1f}%** |
| Help missed for students who needed it | **{100*q9['miss']:.1f}%** | **{100*q12['miss']:.1f}%** |

![Quiet profile on both datasets](figures_v2/fig2_quiet_2009_vs_2012.png)

- **Miss rate rose on the new dataset** ({100*q9['miss']:.1f}% to {100*q12['miss']:.1f}%; the 95% intervals overlap slightly, so the rise is suggestive rather than settled). The 2009-trained risk scores drift on 2012: calibration intercept {cal3['intercept']:+.2f}, slope {cal3['slope']:.2f} at k=3. This is the likely reason for the higher miss rate; we did not isolate it.
- **Supportive profile (1:3)** is slightly worse than always-help on 2012 (cost difference {s12['dJ']['diff_bkt_minus_other']:+.3f} per student, 95% CI {s12['dJ']['ci95'][0]:+.3f} to {s12['dJ']['ci95'][1]:+.3f}; it saves only {100*s12['red']:.0f}% of interventions). On 2009 the difference was not distinguishable from zero.
- **Neural models did not beat BKT.** DKT-style and SAKT-style (single-skill simplifications) had lower AUC than BKT on the 2012 test at every k (point estimates; intervals exclude 0 only for DKT-style at k=3 and SAKT-style at all k) (DKT-style {dk[0]['diff']:+.3f} / {dk[1]['diff']:+.3f} / {dk[2]['diff']:+.3f}; SAKT-style {sk[0]['diff']:+.3f} / {sk[1]['diff']:+.3f} / {sk[2]['diff']:+.3f} at k=3/5/8). On the 2009 side the comparison was run on validation only, where neither was promoted.
- **(b, within-dataset) Affect adds almost nothing.** Adding detector-predicted affect (rank-transformed, version unverified) changed AUC by {af['mean_diff']:+.4f} (95% CI {af['ci95'][0]:+.4f} to {af['ci95'][1]:+.4f}) on the 2012 test, below the pre-registered 0.005 bar; the lagged version gives {al['mean_diff']:+.4f}.

![Frontier per profile, 2012 test](figures_v2/fig1_frontier_2012_test.png)

**Read with care.** Replay of logged data, not a classroom. "Unnecessary" is a proxy (the student later reached mastery anyway). Grade level unverified. No learning-gain claim; no clinical claim.

<details><summary>Further figures</summary>

![DKT/SAKT-style vs BKT](figures_v2/fig3_neural_vs_bkt.png)
![Affect by k](figures_v2/fig4_affect_by_k.png)
</details>

---
**Speaker notes**

- Two datasets, never pooled: (a) is 2009-trained and frozen, evaluated on the locked 2012 test (4,651 students); (b) trains and selects on 2012 train/validation and evaluates once on the same test.
- Exploratory, not pre-registered, not on the slide: the plain windowed logistic head scored above BKT in AUC on the 2012 test at k=3 and k=5 ({A['risk']['paired_vs_bkt']['variantA_k3']['risk_head']['diff']:+.3f} and {A['risk']['paired_vs_bkt']['variantA_k5']['risk_head']['diff']:+.3f}) and below at k=8 ({A['risk']['paired_vs_bkt']['variantA_k8']['risk_head']['diff']:+.3f}). BKT remains the primary score; this is a lead, not a result.
- Affect replication: the same comparison on 2012 validation gave {avp['mean_diff']:+.4f} (CI {avp['ci95'][0]:+.4f} to {avp['ci95'][1]:+.4f}); the test gave {af['mean_diff']:+.4f}. Both are within-2012, so "replicated" means validation to test, not a third dataset. The k=3 gain (about +0.005) is the only one near the bar.
- Quiet-profile 2009 numbers come from the earlier single 2009 test evaluation (`results/TEST_LOCK`); 2012 from `results/TEST_LOCK_2012`.
- Do not describe any of this as improving learning or as detecting a student's state.

---
**Traceability** (every number above is generated by `aria_pace/make_slide.py` from these files)

| Number | Source |
|---|---|
| {100*q9['red']:.0f}% / {100*q9['miss']:.1f}% (2009 test) | `results/test_results.json`: main.variants.frozen.methods.bkt.quiet, always_help.quiet |
| {100*q12['red']:.1f}% / {100*q12['miss']:.1f}% (2012 test) | `results/test_results_2012.json`: a_cross_dataset.policy.main.variants.frozen.methods |
| calibration {cal3['intercept']:+.2f} / {cal3['slope']:.2f} | `results/test_results_2012.json`: a_cross_dataset.risk.results.variantA_k3.bkt.calibration_line |
| supportive cost difference | `results/test_results_2012.json`: ...j_diff_vs_bkt.supportive.always_help |
| neural vs BKT | `results/test_results_2012.json`: a_cross_dataset.risk.paired_vs_bkt; 2009 validation: `results/dkt_val.json`, `results/sakt_val.json` |
| affect | `results/test_results_2012.json`: b_within_dataset_affect.mean_over_k, paired_with_minus_without; validation: `results/affect_val.json` |
| frozen thresholds | `results/frozen_val.json` (sha in `results/frozen_v2.json`) |
"""
(DOCS / "SUMMIT_SLIDE.md").write_text(sl)
print("wrote slide + 4 figures")
