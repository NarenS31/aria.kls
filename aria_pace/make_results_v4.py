"""Builds aria_pace/RESULTS_v4.md from the result files (every number is read from them; nothing is recomputed)."""
import json

from .constants import RESULTS, ROOT

pct = lambda x, n=1: "n/a" if x is None else f"{100 * x:.{n}f}%"  # noqa: E731
R = lambda n: json.load(open(RESULTS / n))  # noqa: E731
stress = json.load(open(ROOT.parent / "results" / "stress_50k_summit_pace_mac.json"))
cv, pdrift, hz = R("cv_tradeoff_val.json"), R("param_drift_val.json"), R("hazard_val.json")
out = []
w = out.append
w("# RESULTS v4 (branch `aria-pace-v4`)")
w("")
w("**Nothing here changes a published number, a frozen file, the tagged demo (`summit-2026-10-24-pace`) or `eval/`. No locked test split was read (2009 test students are dropped right after the split; the 2012 file is never loaded) and no new test evaluation was made.** A7, A8 and A9 are numbered amendments in `aria_pace/SPEC.md`, written and committed before their code and results (SPEC commit `1179a31`, code `0951ab7`). All three are **exploratory, validation only**. The 2009 data are ASSISTments skill-builder math logs, grade level not stated.")
w("")
w("## Step 1. Stress test of the demo build (read-only): targets met")
L = stress["latency_ms"]
w(f"{stress['turns']:,} simulated turns through the real `DemoSession`, 30 problems x 3 profiles, fuzzed input (`docs/aria_pace/STRESS_50K.md`, `scripts/stress_50k.py`, `results/stress_50k_summit_pace_mac.json`; demo commit `{stress['repo_head']}` untouched): **{stress['leaks']} leaks, {stress['math_errors']} math errors, {stress['false_claims']} false claims, {stress['crashes']} crashes, {stress['silent_wrong_steps']} silent wrong steps (of {stress['verified_wrong_turns']:,} verified-wrong turns), {stress['fallbacks']} SymPy fallbacks**; latency p50 {L['p50']:.1f} / p95 {L['p95']:.1f} / p99 {L['p99']:.1f} ms, max {L['max']:.0f} ms, none over 5 s. Caveat: the extreme-number inputs were turned away quickly, so the SymPy time-budget path was not exercised by this run (covered by the existing robustness tests).")
w("")
w("## Step 2. A7: cross-validated trade-off per mastery definition (validation only, BKT only)")
w("Five student-level folds over train + validation; per fold, tau and the Platt calibrator are chosen on a separate calibration part and applied to the held-out fold only (so, unlike A4, the numbers below are **not optimistic**). Figures: `docs/aria_pace/figures_v4/cv_tradeoff_{quiet,balanced,supportive}.png`. Pooled held-out result at the per-fold chosen tau (95% student bootstrap):")
w("")
w("| Definition | Profile | Saving vs always-help | Missed-help share | Fold taus (range, SD) | Mean shift vs frozen tau | Folds with a material (>0.10) shift vs D3 |")
w("|---|---|---|---|---|---|---|")
for n, r in cv["definitions"].items():
    for p in ("quiet", "balanced", "supportive"):
        pt, tm = r["chosen_tau_pooled_result"][p], r["tau_movement"][p]
        w(f"| {n} | {p} | {pct(pt['saving'])} [{pct(pt['saving_ci95'][0])}, {pct(pt['saving_ci95'][1])}] | {pct(pt['missed'])} [{pct(pt['missed_ci95'][0])}, {pct(pt['missed_ci95'][1])}] | "
          f"{', '.join(f'{t:.2f}' for t in tm['fold_taus'])} ({tm['range']:.2f}, {tm['sd']:.3f}) | {tm['mean_shift_vs_frozen']:+.3f} | {tm.get('n_folds_material_shift_vs_D3', '-')} |")
q3 = cv["definitions"]["D3"]["chosen_tau_pooled_result"]["quiet"]
w("")
w(f"- **Under the frozen definition (D3) the cross-validated Quiet result is a saving of {pct(q3['saving'])} with {pct(q3['missed'])} missed help** (A4's optimistic single-split figure was 49.8% and 4.8%): the saving holds, the missed-help share is higher once tau is not chosen on the evaluated students.")
w("- **The saving is fragile to the definition**, as in A4, now without the selection optimism: Quiet saves " + ", ".join(f"{n} {pct(r['chosen_tau_pooled_result']['quiet']['saving'])}" for n, r in cv['definitions'].items()) + ".")
w("- **How much tau moves:** within D3 the five fold taus stay within 0.09 for Quiet (0.51-0.60) and shift little from the frozen 0.57; under other definitions the chosen tau moves by more than 0.10 in several folds (table, last column), most for D5T20 (Quiet mean shift " + f"{cv['definitions']['D5T20']['tau_movement']['quiet']['mean_shift_vs_frozen']:+.2f}" + ") and the EMA rule. So a threshold tuned for 3-in-a-row would not carry over to another definition.")
w("- **Limitation:** BKT only. PFA and the risk head were not re-run (they need a definition-specific rewrite of their scoring; not cheap; A7.5).")
w("")
w("## Step 3. A8: how much do BKT parameters drift across skills? (validation only)")
sp, dec = pdrift["spread"], pdrift["decisions_validation"]
w(f"**This estimates transfer risk across ASSISTments skills. It is NOT a calculus calibration.** Per-skill BKT fitted on 2009 train only: {sp['n_skills_fitted']} skills have enough sequences (covering {pct(sp['share_of_train_sequences_covered'])} of train sequences). Pooled: " + ", ".join(f"{k} {v:.2f}" for k, v in sp["pooled"].items()) + ".")
w("")
w("| Parameter | Median | IQR (25th-75th) | Min - max | Pooled | Skills at lower / upper bound |")
w("|---|---|---|---|---|---|")
for p, r in sp["parameters"].items():
    w(f"| {p} | {r['median']:.2f} | {r['q25']:.2f}-{r['q75']:.2f} | {r['min']:.2f}-{r['max']:.2f} | {r['pooled']:.2f} | {r['n_at_lower_bound']} / {r['n_at_upper_bound']} |")
w("")
w("Groups by skill name (rule fixed in A8.3; names are a rough label): " + "; ".join(f"{g}: {v['n_skills']} skills" for g, v in sp["by_group"].items()) + ". Group medians differ (e.g. learning rate " + ", ".join(f"{g} {v['parameters']['learn']['median']:.2f}" for g, v in sp["by_group"].items() if isinstance(v['parameters'], dict)) + ").")
w("")
w(f"Decisions on validation ({dec['decision_cells_with_verified_error']:,} cells with a verified wrong step), frozen Platt and tau, **own-skill parameters (the frozen research pipeline) versus pooled parameters (what the demo does)**:")
w("")
w("| Profile | Cells that flip | Reduction: own-skill | Reduction: pooled | Missed help: own-skill | Missed help: pooled |")
w("|---|---|---|---|---|---|")
for p, r in dec["profiles"].items():
    o, q = r["own_skill_params"], r["pooled_params"]
    w(f"| {p} | {pct(r['flip_share_of_cells'])} ({r['flips']}) | {pct(o['reduction_vs_always_help'])} [{pct(o['reduction_ci95'][0])}, {pct(o['reduction_ci95'][1])}] | {pct(q['reduction_vs_always_help'])} [{pct(q['reduction_ci95'][0])}, {pct(q['reduction_ci95'][1])}] | {pct(o['missed_help_share'])} | {pct(q['missed_help_share'])} |")
w("")
qr = dec["profiles"]["quiet"]
w("- Pooled parameters make the policy **speak more**: in the Quiet profile " + f"{qr['flag_to_speak(own flags, pooled speaks)']} cells switch from flag to speak and {qr['speak_to_flag(own speaks, pooled flags)']} the other way; in the Supportive profile the pooled version is exactly always-help (0.0% saving). Quiet flips by group: " + ", ".join(f"{g} {pct(v['share'])}" for g, v in qr["flip_share_by_group"].items() if v["share"] is not None) + ".")
w("- Reading: parameters differ widely between skills (many sit at the fitting bounds), and applying one pooled set costs about 6 points of Quiet saving on these skills while lowering measured missed help, so the demo's numbers are not the research pipeline's. How this transfers to calculus is **not estimated** here.")
w("")
w("## Step 4. Calculus data availability (research, nothing downloaded)")
w("`docs/aria_pace/CALCULUS_DATA.md`. **Public student-level step logs for calculus: not found.** DataShop's catalog could not be browsed by the fetcher (HTTP 404), so a calculus dataset there is not ruled out; account requirements for public data are described inconsistently by two DataShop pages; several other sources appeared only in search results and are marked unverified. Verified: the MATH benchmark (12,500 competition problems with step-by-step solutions) is a problem/solution dataset with no student attempts or logs.")
w("")
w("## Step 5. A9: per-turn model with response time (validation only)")
w("`ms_first_response` exists in the raw file but `aria_pace/loader.py:12` does not load it; usable after a rank transform fitted on train. Per-turn logistic model (BKT risk, wrong in last 3, cumulative hints, rank-transformed log response time), k = 1..9:")
w("")
w("| k | In frozen window | n val | AUC BKT | AUC model | Diff vs BKT [95% CI] | Diff without response time | Response time only (AUC) |")
w("|---|---|---|---|---|---|---|---|")
for k, r in hz["per_k"].items():
    m = r["models"]
    w(f"| {k} | {'yes' if r['in_frozen_policy_window'] else 'no'} | {r['n_val']} | {r['auc_bkt_raw']:.3f} | {m['full']['auc']:.3f} | {m['full']['auc_minus_bkt']:+.4f} [{m['full']['ci95'][0]:+.4f}, {m['full']['ci95'][1]:+.4f}] | {m['no_rt']['auc_minus_bkt']:+.4f} [{m['no_rt']['ci95'][0]:+.4f}, {m['no_rt']['ci95'][1]:+.4f}] | {m['rt_only']['auc']:.3f} |")
w("")
w(f"- **Not promoted** (rule: >= +0.01 with the interval above zero at both k = 3 and k = 5): the gain is {hz['per_k']['3']['models']['full']['auc_minus_bkt']:+.4f} at k = 3 and {hz['per_k']['5']['models']['full']['auc_minus_bkt']:+.4f} at k = 5. BKT stays. This is a negative result.")
w("- Response time adds little: removing it changes the difference by at most a few thousandths at k = 3-5 (the gain at k = 3 is mostly the error-count and hint features), and response time alone barely beats chance (AUC about 0.5-0.6; negative sign of its coefficient: slower first responses go with lower risk). The only k whose point estimate reaches +0.01 is k = 1 (+0.0103, interval +0.0003 to +0.0201), which is **outside the frozen policy window** (decision points 3-9, A1.2) and so cannot change the policy.")
w("- The live demo already rescored the BKT risk after every verified attempt and applies the gate at attempts 3-9 (`agent/demo_pace.py:204-205`, window from `aria_pace/policy.py` `DEC`, `agent/demo_pace.py:67`); k = 1 and 2 use the old always-speak rule (`wrong_before_window`, `:208`). The demo was not touched.")
w("")
w("## Step 6. Teacher activity v2 (design + branch `activity-v2`; not in the demo)")
w("Branch `activity-v2` from `summit-activity`: optional categorical reason when stepping in (off by default; the free-text note is kept in memory only and never written), per-profile and reason analyses, draft amendment in `docs/TEACHER_ACTIVITY.md` and `docs/teacher_timing_analysis.md`. Tests: `tests/test_teacher_activity.py` 33 passed. **Finding (negative):** ARIA's decision on all 8 current moments is identical under Quiet, Balanced and Supportive (the two wrong-step moments are before attempt 3; the rest have no wrong step), so per-profile alignment cannot differ with these moments. Not run with real teachers; any collection needs organizer sign-off and adult-reviewed consent wording first. Not merged into the demo build.")
w("")
w("## Step 7. Pilot design fix (docs only)")
w("`docs/aria_pace/PILOT_PROTOCOL.md` section 10 and `docs/aria_pace/LIMITS.md`: the cost of delayed help (w_D) and the cost of interruption (w_I) need separate comparisons; recovery latency can inform but cannot derive the weights, which stay a stated value judgment. Draft; needs ethics review, a school sponsor, parental consent and an adult responsible for the data.")
w("")
w("## Step 8. Annotation kappa readiness")
w("`docs/aria_pace/KAPPA_READINESS.md`. **No script computing the main 150-item (seven-category) kappa was found**; `scripts/annotation_kappa.py` (new, outside `eval/`) does it and passes 4 tests on synthetic annotators labelled \"code test only\" (generated in a temp directory, not saved). The real run needs two independent annotators and the private key (gitignored, local only). Humans are needed; no annotation was created.")
w("")
w("## What could not be done or was not verified")
w("- A7 covers BKT only (PFA and the risk head not re-run). A8 and A9 use validation students only and are exploratory; none is a calibration for calculus.")
w("- The stress test's oracle is the demo's own SymPy checker, and the time-budget fallback was not triggered by any input (0 fallbacks).")
w("- DataShop's catalog was not browsed (no account, no download); other dataset claims are unverified search leads.")
w("- `activity-v2` was tested through its handlers and a Gradio build, not in a browser; no Safari check.")
w("- Several earlier items remain open: the website's production deployment of PR #10 was not visible, and `docs/summit/QA.md` / `BRIEF.md` are on other branches.")
(ROOT / "RESULTS_v4.md").write_text("\n".join(out) + "\n")
print("wrote aria_pace/RESULTS_v4.md")
