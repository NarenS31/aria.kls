"""docs/aria_pace/CLOSED_LOOP_SENSITIVITY.md and figure from results/closed_loop_sensitivity.json (no recomputation). ASSUMPTION-DEPENDENT, NOT EVIDENCE."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import RESULTS, ROOT

PROFS = ("quiet", "balanced", "supportive")
SYM = {"beats": "B", "flips": "F", "indistinguishable": "."}
DOCS = ROOT.parent / "docs" / "aria_pace"
LABEL = "Assumption-dependent sensitivity analysis, not evidence of effect"


def breakeven(ds, xs):
    """x where delta crosses 0 by linear interpolation between grid values (first crossing from negative to positive); None if no crossing."""
    for i in range(len(xs) - 1):
        if ds[i] < 0 <= ds[i + 1]:
            return xs[i] + (xs[i + 1] - xs[i]) * (-ds[i]) / (ds[i + 1] - ds[i])
    return None


def main():
    d = json.load(open(RESULTS / "closed_loop_sensitivity.json"))
    G = d["grid"]
    idx = {(p["p_help_effect"], p["p_interrupt_cost"], p["p_stuck_cost"]): p for p in d["points"]}
    L = [f"# Parametric closed-loop sensitivity (Amendment A6)", "", f"**{LABEL}.** Status in the results file: `{d['status']}`. Source: `aria_pace/results/closed_loop_sensitivity.json`, "
         "produced by `python -m aria_pace.run_closed_loop`. No LLM, no real data, no locked split; frozen pooled BKT parameters, Platt calibrator and `tau` are read, never re-tuned. "
         f"{d['n_students_per_point']:,} simulated students per grid point, seed {d['seed']}, common random numbers across policies.", "",
         "**What it is:** a BKT-style student (latent known / unknown) who can be helped, interrupted, or left stuck, with three **assumed** parameters: `p_help_effect` (help raises every later learning chance), "
         "`p_interrupt_cost` (an interrupted student slips more on the next response), `p_stuck_cost` (a student who is wrong at a decision point and is not helped abandons with this probability). "
         "The policy's risk model equals the generator, a deliberately favourable assumption. **What it is not:** any measurement of what real students do. The numbers say how the *replay* conclusion depends on assumptions the replay cannot test.", "",
         "Cost: `J = w_I * unnecessary + w_D * delay` per in-scope sequence, evaluated on the simulated (realised) trajectories; `Delta J = J(profile policy) - J(always-help)` under that profile's weights; "
         "negative = the policy is cheaper. 95% intervals are normal approximations of the paired per-sequence difference and reflect **simulation noise only**.", "",
         "## Pre-registered checks (A6.6)", "",
         "All three pass in `aria_pace/tests/test_closed_loop.py`: (i) with all three parameters at 0 the simulator's per-sequence unnecessary / delay / missed / fired equal the replay machinery's exactly; "
         "(ii) the first-response correctness rate matches `prior*(1-slip)+(1-prior)*guess` within Monte Carlo error; (iii) the risk table equals `BKT.score`.", "",
         "## Where the policy beats always-help, and where it flips", "",
         "| Profile (frozen tau) | beats | indistinguishable | flips (worse) | of 100 grid points |", "|---|---|---|---|---|"]
    for prof in PROFS:
        c = Counter(p["delta_J_policy_minus_always_help"][prof]["region"] for p in d["points"])
        L.append(f"| {prof} ({d['frozen_inputs']['taus_bkt'][prof]:.2f}) | {c['beats']} | {c['indistinguishable']} | {c['flips']} | 100 |")
    p0 = idx[(0.0, 0.0, 0.0)]
    L += ["", f"At the all-zero point (no effect of help, no interruption cost, no stuck cost) the quiet policy is cheaper than always-help (`Delta J` {p0['delta_J_policy_minus_always_help']['quiet']['delta']:+.3f} per sequence) and the balanced one slightly cheaper "
          f"({p0['delta_J_policy_minus_always_help']['balanced']['delta']:+.3f}); the supportive policy intervenes exactly when always-help does (`Delta J` = {p0['delta_J_policy_minus_always_help']['supportive']['delta']:+.3f}): this reproduces the direction of the replay result. "
          "The simulator then shows how that advantage changes as the assumed cost of being left stuck grows.", "",
         "Region maps (`B` beats, `F` flips, `.` indistinguishable); rows = `p_help_effect`, columns = `p_stuck_cost` " + str(G["p_stuck_cost"]) + ". The sign pattern for the quiet and balanced policies is the same at every `p_interrupt_cost` except in a few border cells, so the extremes are shown; every point is in the JSON.", ""]
    for prof in ("quiet", "balanced"):
        for pi in (G["p_interrupt_cost"][0], G["p_interrupt_cost"][-1]):
            L += [f"{prof}, p_interrupt_cost = {pi}:", "```"]
            for ph in G["p_help_effect"]:
                L.append(f"p_help_effect {ph:<4}  " + "  ".join(SYM[idx[(ph, pi, ps)]["delta_J_policy_minus_always_help"][prof]["region"]] for ps in G["p_stuck_cost"]))
            L += ["```", ""]
    L += ["## Approximate flip point in `p_stuck_cost` (linear interpolation of `Delta J` between grid values)", "",
          "| Profile | p_interrupt_cost | p_help_effect | `p_stuck_cost` at which `Delta J` crosses 0 |", "|---|---|---|---|"]
    xs = G["p_stuck_cost"]
    for prof in ("quiet", "balanced"):
        for pi in (G["p_interrupt_cost"][0], G["p_interrupt_cost"][-1]):
            for ph in (G["p_help_effect"][0], G["p_help_effect"][-1]):
                ds = [idx[(ph, pi, ps)]["delta_J_policy_minus_always_help"][prof]["delta"] for ps in xs]
                b = breakeven(ds, xs)
                L.append(f"| {prof} | {pi} | {ph} | {'~' + format(b, '.3f') if b is not None else 'none in the grid'} |")
    # narrative computed from the numbers
    qb = [breakeven([idx[(ph, pi, ps)]["delta_J_policy_minus_always_help"]["quiet"]["delta"] for ps in xs], xs) for ph in G["p_help_effect"] for pi in G["p_interrupt_cost"]]
    bb = [breakeven([idx[(ph, pi, ps)]["delta_J_policy_minus_always_help"]["balanced"]["delta"] for ps in xs], xs) for ph in G["p_help_effect"] for pi in G["p_interrupt_cost"]]
    L += ["", "## Reading (conditional on the assumptions)", "",
          f"- The sign of the policy's advantage is decided almost entirely by `p_stuck_cost`, the assumed harm of leaving a stuck student unaided. For the quiet profile `Delta J` changes sign at about {min(b for b in qb if b):.3f} to {max(b for b in qb if b):.3f} per unaided wrong step; "
          f"for the balanced profile at about {min(b for b in bb if b):.3f} to {max(b for b in bb if b):.3f}. Below that the policy is cheaper than always-help, above it always-help is cheaper.",
          "- `p_help_effect` and `p_interrupt_cost` move the numbers only a little in this model and do not change the sign pattern except in border cells (the quiet policy goes from worse to indistinguishable at `p_stuck_cost` = 0.1 when help is strong): help changes learning only for students who are still unknown, and an interruption costs one response.",
          "- The supportive profile (`tau` = 0.09) makes **exactly the same decisions as always-help** at every grid point (interventions per sequence differ by 0.0, `Delta J` is 0.0 everywhere): every wrong response at a decision point carries a calibrated risk above 0.09 in this simulator, so the policy never stays quiet. (On the real 2009 test split it saved 3% of interventions, `RESULTS.md`; the simulator is not meant to reproduce that.)",
          "- The replay result (fewer interventions with few missed students) therefore does not transfer to any real claim without a measurement of how costly it is to leave a student unaided for one more step; replay cannot supply that, a study with students could.", "",
          f"![closed-loop regions](../../aria_pace/figures/closed_loop_regions.png)", "", f"Figure: `Delta J` (policy minus always-help) over (`p_stuck_cost`, `p_help_effect`) for each profile and `p_interrupt_cost`; blue = policy cheaper, red = always-help cheaper; letters as above. **{LABEL}.**", ""]
    (DOCS / "CLOSED_LOOP_SENSITIVITY.md").write_text("\n".join(L))

    fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharex=True, sharey=True)
    vmax = 1.1
    for r, prof in enumerate(PROFS):
        for c, pi in enumerate(G["p_interrupt_cost"]):
            ax = axes[r, c]
            M = np.array([[idx[(ph, pi, ps)]["delta_J_policy_minus_always_help"][prof]["delta"] for ps in xs] for ph in G["p_help_effect"]])
            im = ax.imshow(M, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            for i, ph in enumerate(G["p_help_effect"]):
                for j, ps in enumerate(xs):
                    ax.text(j, i, SYM[idx[(ph, pi, ps)]["delta_J_policy_minus_always_help"][prof]["region"]], ha="center", va="center", fontsize=9, color="k")
            ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=7); ax.set_yticks(range(len(G["p_help_effect"]))); ax.set_yticklabels(G["p_help_effect"], fontsize=7)
            if r == 0:
                ax.set_title(f"p_interrupt_cost = {pi}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{prof}\np_help_effect", fontsize=9)
            if r == 2:
                ax.set_xlabel("p_stuck_cost", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Delta J = J(policy) - J(always-help)  (negative: policy cheaper)")
    fig.suptitle(LABEL + " (B beats, F flips, . indistinguishable)", fontsize=10)
    fig.savefig(ROOT / "figures" / "closed_loop_regions.png", dpi=140, bbox_inches="tight")
    print("wrote docs/aria_pace/CLOSED_LOOP_SENSITIVITY.md and aria_pace/figures/closed_loop_regions.png")


if __name__ == "__main__":
    main()
