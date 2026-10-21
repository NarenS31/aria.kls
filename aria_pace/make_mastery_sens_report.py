"""Table + figure for Amendment A4 from results/mastery_sensitivity_val.json (no recomputation)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import RESULTS, ROOT

PROFS = ("quiet", "balanced", "supportive")
DOCS = ROOT.parent / "docs" / "aria_pace"


def main():
    d = json.load(open(RESULTS / "mastery_sensitivity_val.json"))
    defs = d["definitions"]
    names = list(defs)
    ctrl = defs["D3"]["taus"]
    f = lambda x, n=3: "—" if x is None else f"{x:.{n}f}"  # noqa: E731
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    L = ["# Mastery-definition sensitivity (Amendment A4)", "",
         "**Status: exploratory sensitivity (validation only; not pre-registered).** The control check (D3 reproduces `frozen_val.json`) is the only pre-registered part and it passed "
         f"(Platt equal to 1e-9: {d['control_check_A4_3']['platt_equal_1e-9']}; taus equal: {d['control_check_A4_3']['taus_equal']}). "
         "Source: `aria_pace/results/mastery_sensitivity_val.json`, produced by `python -m aria_pace.run_mastery_sens`. All numbers are on the 2009 **validation** split and are optimistic: "
         "the Platt calibrator and `tau` are chosen on the same split. No locked split was read. **Nothing here changes the frozen thresholds, the demo, or any published number.**", "",
         "| Definition | In-scope sequences / students | Label rate (y=1) | BKT AUC k=3 | k=5 | k=8 |", "|---|---|---|---|---|---|"]
    for n in names:
        r = defs[n]; s = r["scope"]; a = r["bkt_auc_val"]
        cell = lambda k: f"{a[k]['auc']:.3f} [{a[k]['auc_ci95_student_bootstrap'][0]:.3f}, {a[k]['auc_ci95_student_bootstrap'][1]:.3f}]"  # noqa: E731
        L.append(f"| **{n}** {r['definition']} | {s['in_scope_labelled_sequences']} / {s['students']} | {pct(s['label_rate'])} | {cell('k3')} | {cell('k5')} | {cell('k8')} |")
    L += ["", "AUC cells: point estimate [95% student-bootstrap interval].", "",
          "| Definition | Profile | Chosen tau (bootstrap median, 10th-90th pct) | Shift vs D3 | Interventions/student (always-help) | Reduction vs always-help [95% CI] | Missed-help share [95% CI] |",
          "|---|---|---|---|---|---|---|"]
    for n in names:
        for p in PROFS:
            r = defs[n]; t = r["taus"][p]; b = r["tau_bootstrap"][p]; po = r["policy_at_chosen_tau"][p]
            shift = t - ctrl[p]
            flag = " **material**" if abs(shift) > 0.10 + 1e-9 else ""
            L.append(f"| {n} | {p} | {t:.2f} ({b['boot_median']:.2f}, {b['boot_p10_p90'][0]:.2f}-{b['boot_p10_p90'][1]:.2f}) | {shift:+.2f}{flag} | "
                     f"{po['interventions_per_student']:.2f} ({po['always_help_interventions_per_student']:.2f}) | {pct(po['reduction_vs_always_help'])} [{pct(po['reduction_ci95'][0])}, {pct(po['reduction_ci95'][1])}] | "
                     f"{pct(po['missed_help_share'])} [{pct(po['missed_help_ci95'][0])}, {pct(po['missed_help_ci95'][1])}] |")
    L += ["", "\"Material\" means the chosen `tau` differs from the D3 value by more than 0.10 (pre-declared in A4.5; the window used in `stage5.py:193`).", "",
          "![tau by mastery definition](../../aria_pace/figures/mastery_sensitivity_tau.png)", "",
          "Figure: chosen `tau` per profile under each definition (points) with the 10th-90th percentile of the student-bootstrap distribution of the chosen `tau` (bars); the grey band is +/-0.10 around the D3 control.", ""]
    (DOCS / "MASTERY_SENSITIVITY.md").write_text("\n".join(L))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, p in zip(axes, PROFS):
        xs = range(len(names))
        ys = [defs[n]["taus"][p] for n in names]
        lo = [defs[n]["taus"][p] - defs[n]["tau_bootstrap"][p]["boot_p10_p90"][0] for n in names]
        hi = [defs[n]["tau_bootstrap"][p]["boot_p10_p90"][1] - defs[n]["taus"][p] for n in names]
        ax.axhspan(ctrl[p] - 0.10, ctrl[p] + 0.10, color="0.88", zorder=0)
        ax.axhline(ctrl[p], color="0.5", lw=0.8, ls="--", zorder=1)
        ax.errorbar(xs, ys, yerr=[[max(0, v) for v in lo], [max(0, v) for v in hi]], fmt="o", color="C0", capsize=3, zorder=3)
        ax.set_xticks(list(xs)); ax.set_xticklabels(names, rotation=30)
        ax.set_title(f"{p}  (D3 tau {ctrl[p]:.2f})"); ax.set_ylim(-0.02, 0.85); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("chosen tau (validation)")
    fig.suptitle("Exploratory sensitivity, validation only: how the chosen tau moves with the mastery definition", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "mastery_sensitivity_tau.png", dpi=150)
    print("wrote docs/aria_pace/MASTERY_SENSITIVITY.md and aria_pace/figures/mastery_sensitivity_tau.png")


if __name__ == "__main__":
    main()
