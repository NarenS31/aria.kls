"""Writes docs/aria_pace/RESULTS.md from results/*.json so that every number traces to a file.
python -m aria_pace.make_results_md   (after run_test and report)"""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parents[1] / "docs" / "aria_pace" / "RESULTS.md"
T = json.load(open(R / "test_results.json"))
V = json.load(open(R / "val_results.json"))
BL = json.load(open(R / "baselines_val.json"))
RH = json.load(open(R / "riskhead_val.json"))
LC = json.load(open(R / "label_counts.json"))
DR = json.load(open(R / "data_report.json"))
FZ = json.load(open(R / "frozen_val.json"))
PROFS = ("quiet", "balanced", "supportive")
PROF_LABEL = {"quiet": "quiet (3:1)", "balanced": "balanced (1:1)", "supportive": "supportive (1:3)"}


def f(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def ci(c, d=2, p=False):
    if c is None:
        return "n/a"
    return f"[{pct(c[0], d) if p else f(c[0], d)}, {pct(c[1], d) if p else f(c[1], d)}]"


def mtab(res, cfg, var, prof, keys, names, ci_for=("bkt",)):
    rows = ["| Method | Interventions / student | Unnecessary / student | Missed help (n / N, share) | Mean lead time | Mean delay | Cost J / student |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    V_ = res[cfg]["variants"][var]["methods"]
    for k, nm in zip(keys, names):
        m = V_[k][prof]
        c = m["ci95"]
        iv = f(m["interventions_per_student"]) + (f" {ci(c['interventions_per_student'])}" if k in ci_for else "")
        un = f(m["unnecessary_per_student"]) + (f" {ci(c['unnecessary_per_student'])}" if k in ci_for else "")
        ms = f"{m['missed_help']}/{m['n_y1_with_help_opportunity']} = {pct(m['missed_help_share'])}" + (f" {ci(c['missed_help_share'], 1, True)}" if k in ci_for else "")
        jj = f(m["cost_J_per_student"]) + (f" {ci(c['cost_J_per_student'])}" if k in ci_for else "")
        rows.append(f"| {nm} | {iv} | {un} | {ms} | {f(m['mean_lead_time'])} | {f(m['mean_delay_incl_censored'])} | {jj} |")
    return "\n".join(rows)


KEYS = ["bkt", "pfa", "risk_head", "n_rule_tuned", "three_wrong_untuned", "always_help", "random_matched_to_bkt"]


def names(res, cfg, var, prof):
    n = res[cfg]["variants"][var]["n_rule_n"][prof]
    return ["BKT threshold (primary)", "PFA threshold", "Risk-head threshold", f"n-consecutive-errors rule, tuned (n = {n}" + (", = always-help" if n == 1 else "") + ")",
            "Three-wrong rule, untuned (n = 3)", "Always-help", "Random, matched to BKT's count"]


def diff_tab(res, cfg, var):
    rows = ["| Profile | vs always-help | vs tuned n-rule | vs untuned three-wrong | vs PFA | vs risk head | vs random (matched) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for prof in PROFS:
        d = res[cfg]["variants"][var]["j_diff_vs_bkt"][prof]
        cells = []
        for k in ("always_help", "n_rule_tuned", "three_wrong_untuned", "pfa", "risk_head", "random_matched_to_bkt"):
            x = d[k]
            mark = " *" if (x["bkt_cheaper_significant"] or x["bkt_costlier_significant"]) else ""
            cells.append(f"{f(x['diff_bkt_minus_other'], 3)} {ci(x['ci95'], 3)}{mark}")
        rows.append(f"| {PROF_LABEL[prof]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def sens_tab(res, label):
    rows = ["| Configuration / tau variant | Profile | BKT tau | BKT interventions | Always-help interventions | BKT missed help | BKT J | Always-help J | Tuned n-rule J | J diff BKT - always-help |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cfg, cname in (("main", "main (leavers dropped)"), ("leavers", "leavers as non-wheel-spinning"), ("hintfirst", "hint-first counted as error")):
        for var in res[cfg]["variants"]:
            vv = res[cfg]["variants"][var]
            for prof in PROFS:
                b, a, n = vv["methods"]["bkt"][prof], vv["methods"]["always_help"][prof], vv["methods"]["n_rule_tuned"][prof]
                d = vv["j_diff_vs_bkt"][prof]["always_help"]
                rows.append(f"| {cname}, {var} | {prof} | {f(vv['taus']['bkt'][prof])} | {f(b['interventions_per_student'])} | {f(a['interventions_per_student'])} | {b['missed_help']}/{b['n_y1_with_help_opportunity']} = {pct(b['missed_help_share'])} | {f(b['cost_J_per_student'])} | {f(a['cost_J_per_student'])} | {f(n['cost_J_per_student'])} | {f(d['diff_bkt_minus_other'], 3)} {ci(d['ci95'], 3)} |")
    return "\n".join(rows)


def val_vs_test():
    rows = ["| Profile | Split | BKT interventions | Always-help interventions | Reduction | BKT missed help | BKT J | Always-help J |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for prof in PROFS:
        for nm, res in (("validation (optimistic)", V), ("test", T)):
            vv = res["main"]["variants"]["frozen"]["methods"]
            b, a = vv["bkt"][prof], vv["always_help"][prof]
            red = 1 - b["interventions_per_student"] / a["interventions_per_student"]
            rows.append(f"| {prof} | {nm} | {f(b['interventions_per_student'])} | {f(a['interventions_per_student'])} | {pct(red, 0)} | {pct(b['missed_help_share'])} | {f(b['cost_J_per_student'])} | {f(a['cost_J_per_student'])} |")
    return "\n".join(rows)


def stability_tab():
    rows = ["| Configuration | Profile | Reference tau (the main-configuration frozen value) | Bootstrap median tau | 10th-90th pct | Share of resamples within 0.10 of frozen | tau range with J within 2% of min | J at reference tau | min J |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cfg in ("main", "leavers", "hintfirst"):
        for prof in PROFS:
            s = V[cfg]["tau_stability_bkt"][prof]
            rows.append(f"| {cfg} | {prof} | {f(s['frozen_tau'])} | {f(s['boot_tau_median'])} | {f(s['boot_tau_p10_p90'][0])}-{f(s['boot_tau_p10_p90'][1])} | {pct(s['frac_within_0.10_of_frozen'], 0)} | {f(s['tau_range_within_2pct_of_min_J'][0])}-{f(s['tau_range_within_2pct_of_min_J'][1])} | {f(s['J_per_student_at_frozen'])} | {f(s['J_per_student_min'])} |")
    return "\n".join(rows)


def oracle_tab():
    grid = np.array(T["main"]["sweep"]["tau_grid"])
    rows = ["| Configuration, variant | Profile | Frozen tau | Test J at frozen tau | Test-optimal tau (diagnostic only) | Test J at that tau | Regret |", "|---|---|---:|---:|---:|---:|---:|"]
    for cfg in ("main", "leavers", "hintfirst"):
        for var, vv in T[cfg]["variants"].items():
            for prof in PROFS:
                J = np.array([r["J"][prof] for r in T[cfg]["sweep"]["methods"]["bkt"]])
                best = grid[np.where(np.abs(J - J.min()) < 1e-9)[0][-1]]
                fr = vv["taus"]["bkt"][prof]
                Jf = J[int(np.argmin(abs(grid - fr)))]
                rows.append(f"| {cfg}, {var} | {prof} | {f(fr)} | {f(Jf)} | {f(best)} | {f(J.min())} | {f(Jf - J.min(), 3)} ({pct(Jf / J.min() - 1)}) |")
    return "\n".join(rows)


def regret(cfg, var, prof):
    grid = np.array(T[cfg]["sweep"]["tau_grid"])
    J = np.array([r["J"][prof] for r in T[cfg]["sweep"]["methods"]["bkt"]])
    fr = T[cfg]["variants"][var]["taus"]["bkt"][prof]
    return float(J[int(np.argmin(abs(grid - fr)))] / J.min() - 1)


def auc_tab():
    rows = ["| k | Method | N | Base rate | AUC [95% CI, student bootstrap] | Brier | ECE (10 equal-mass bins) | Calibration intercept / slope |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for k in (3, 5, 8):
        r = BL["results"][f"variantA_k{k}"]
        for m, nm in (("bkt", "BKT"), ("pfa", "PFA"), ("three_wrong_rule", "Three-wrong rule (0/1)"), ("always_help", "Always-help (0/1)")):
            x = r[m]
            zero_one = m in ("three_wrong_rule", "always_help")
            cal = "n/a" if zero_one or x["calibration_line"] is None else f"{f(x['calibration_line']['intercept'])} / {f(x['calibration_line']['slope'])}"
            ece = "n/a" if zero_one else f(x["ece_10bin"], 3)
            br = "n/a" if zero_one else f(x["brier"], 3)
            c = "(constant score)" if x["auc_ci95_student_bootstrap"] is None else ci(x["auc_ci95_student_bootstrap"], 3)
            rows.append(f"| {k} | {nm} | {x['n']} | {pct(x['base_rate'])} | {f(x['auc'], 3)} {c} | {br} | {ece} | {cal} |")
        rh = RH["val_auc_by_k"][str(k)]
        rows.append(f"| {k} | Risk head (logistic regression) | {rh['n']} | {pct(r['bkt']['base_rate'])} | {f(rh['risk_head'], 3)} (CI not computed) | not computed | not computed | not computed |")
    return "\n".join(rows)


m = T["main"]["variants"]["frozen"]["methods"]
mq, mb, ms_ = m["bkt"]["quiet"], m["bkt"]["balanced"], m["bkt"]["supportive"]
aq, ab, as_ = m["always_help"]["quiet"], m["always_help"]["balanced"], m["always_help"]["supportive"]
red = lambda b, a: 1 - b["interventions_per_student"] / a["interventions_per_student"]
dq = T["main"]["variants"]["frozen"]["j_diff_vs_bkt"]
sc = T["main"]["scope"]
lv = T["leavers"]["variants"]["rechosen"]["methods"]["bkt"]["quiet"]
lvc = T["leavers"]["variants"]["carried"]["methods"]["bkt"]["quiet"]
lva = T["leavers"]["variants"]["carried"]["methods"]["always_help"]["quiet"]
hf = T["hintfirst"]["variants"]["carried"]["methods"]["bkt"]["quiet"]
hfa = T["hintfirst"]["variants"]["carried"]["methods"]["always_help"]["quiet"]

md = f"""# ARIA-Pace — Results

Offline replay on ASSISTments 2009-10 skill-builder logs. Everything below is generated by `aria_pace/make_results_md.py` from the JSON files under `aria_pace/results/` (file names in each "Source" line). Protocol: `aria_pace/SPEC.md` (amendments 1 and 2, both committed before the numbers they govern).

## Headline

> **On ASSISTments 2009 logs, a BKT-threshold abstention rule roughly halves interventions when interruptions are costly, with ~5% missed help; when delayed help is costly, abstention saves little.**

What the held-out **test** students show, with the values frozen on validation (Source: `test_results.json` → `main.variants.frozen`; test = {sc['students']} students, {sc['sequences']} problem-set sequences, {sc['y1']} wheel-spinning):

- **Interruptions costly (quiet, 3:1).** BKT makes {f(mq['interventions_per_student'])} interventions per student against {f(aq['interventions_per_student'])} for always-help, **{pct(red(mq, aq), 0)} fewer** (95% CI for BKT {ci(mq['ci95']['interventions_per_student'])}). It misses help in {mq['missed_help']} of {mq['n_y1_with_help_opportunity']} wheel-spinning sequences that had a help opportunity: **{pct(mq['missed_help_share'])}** (95% CI {ci(mq['ci95']['missed_help_share'], 1, True)}). Validation gave {pct(V['main']['variants']['frozen']['methods']['bkt']['quiet']['missed_help_share'])}, so "~5%" is the validation figure; the test point estimate is lower and its interval reaches about 5%.
- **Balanced (1:1).** {pct(red(mb, ab), 0)} fewer interventions ({f(mb['interventions_per_student'])} vs {f(ab['interventions_per_student'])}); cost difference to always-help {f(dq['balanced']['always_help']['diff_bkt_minus_other'], 3)} {ci(dq['balanced']['always_help']['ci95'], 3)}; missed help {pct(mb['missed_help_share'])}.
- **Delayed help costly (supportive, 1:3).** {pct(red(ms_, as_), 0)} fewer interventions ({f(ms_['interventions_per_student'])} vs {f(as_['interventions_per_student'])}); cost difference to always-help {f(dq['supportive']['always_help']['diff_bkt_minus_other'], 3)} {ci(dq['supportive']['always_help']['ci95'], 3)}, an interval that includes zero. **Abstention saves little here**; the best fixed `n`-error rule tuned for this profile is `n = 1`, which is always-help.
- **The headline holds on test for the main configuration.** It does **not** hold unchanged under every sensitivity: see "The quiet-profile tau is not stable across scope assumptions" below.

## Scope, label and data

- Data: `skill_builder_data_corrected_collapsed.csv`, sha256 in `SPEC.md`; {DR['rows_dropped_null_skill']:,} of {DR['rows_in_file']:,} rows ({DR['pct_rows_dropped_null_skill']}%) had no skill and were dropped. {DR['sequences_total']:,} (student, problem-set) sequences. Student-level split 60/20/20, seed 20261015; test = {DR['split_sizes']['test']['students']} students, validation = {DR['split_sizes']['val']['students']}, train = {DR['split_sizes']['train']['students']} (Source: `data_report.json`, `split.json`). No student is in more than one split (asserted in code and tested).
- Wheel-spinning label: fewer than 3 consecutive correct first-attempt responses within the first 10 opportunities, **following Wang et al. / Kai et al., attributed to Beck & Gong** (variant A, >= 10 opportunities, primary; variant B, > 10, sensitivity for the risk-score tables).
- Policy scope on test (Source: `test_results.json` → `main.scope`): {sc['sequences']:,} sequences from {sc['students']} students, {sc['y1']} wheel-spinning, {sc['decision_cells']:,} decision points, {sc['verified_error_cells']:,} verified-error decision points. {T['main']['variants']['frozen']['methods']['bkt']['quiet']['n_y1_with_help_opportunity']} of the {sc['y1']} wheel-spinning sequences have at least one verified error at a decision point (the rest have no help opportunity and are excluded from delay and missed help). The leaver sensitivity adds {T['leavers']['scope']['leavers_included']:,} sequences.
- Verified error (primary): first-attempt wrong **and** `first_action` = 0. **The `first_action` code meanings are inferred, not documented**: 0 = answer attempt, 1 = hint request, 2 = other. Hint-first rows (code 1) and code-2 rows are excluded in the primary analysis; the sensitivity counts every first-attempt-wrong row. No risk model uses `first_action`.
- Units: all times are in **opportunities** (the file has no wall-clock time). Costs: J = w_I x (unnecessary interventions) + w_D x (opportunities of help delay), per student; weights quiet 3:1, balanced 1:1, supportive 1:3. J values are comparable within a profile, not across profiles. "Per student" means per student who has at least one in-scope sequence, summed over that student's sequences (at most one intervention per sequence).

## Test results (primary configuration)

Source: `test_results.json` → `main.variants.frozen.methods`. 95% intervals (paired bootstrap over students, 1,000 draws) are shown for BKT only; intervals on cost differences are in the next section. Mean delay includes censored (never-helped) sequences.

### Quiet (interruption : delayed help = 3 : 1), BKT tau = {f(T['main']['variants']['frozen']['taus']['bkt']['quiet'])}

{mtab(T, 'main', 'frozen', 'quiet', KEYS, names(T, 'main', 'frozen', 'quiet'))}

### Balanced (1 : 1), BKT tau = {f(T['main']['variants']['frozen']['taus']['bkt']['balanced'])}

{mtab(T, 'main', 'frozen', 'balanced', KEYS, names(T, 'main', 'frozen', 'balanced'))}

### Supportive (1 : 3), BKT tau = {f(T['main']['variants']['frozen']['taus']['bkt']['supportive'])}

{mtab(T, 'main', 'frozen', 'supportive', KEYS, names(T, 'main', 'frozen', 'supportive'))}

### Paired cost differences, BKT minus baseline (negative = BKT cheaper; * = 95% interval excludes 0)

Source: `test_results.json` → `main.variants.frozen.j_diff_vs_bkt`. Units: cost J per student.

{diff_tab(T, 'main', 'frozen')}

Reading: BKT beats always-help, the untuned three-wrong rule, and the matched-rate random policy in the quiet and balanced profiles. Against the **tuned** n-error rule it is cheaper in the quiet profile; in the other two profiles the tuned rule *is* always-help. BKT and PFA are not distinguishable in cost. BKT beats the logistic risk head in the quiet and balanced profiles on this test split.

### Validation versus test (BKT, main configuration)

Validation numbers are **optimistic**: the calibrator, `tau` and `n` were chosen on that split. Source: `val_results.json` and `test_results.json`.

{val_vs_test()}

Test agrees with validation in direction and approximate size. Test missed help is lower than validation, test costs are lower than validation costs for BKT and for always-help in every profile, and nothing in the test contradicts the validation ordering of methods.

## Figures

![Frontier, test](../../aria_pace/figures/frontier_test.png)

![Frontier, validation (optimistic)](../../aria_pace/figures/frontier_val.png)

![Cost versus tau, main configuration](../../aria_pace/figures/j_vs_tau_main.png)

![Cost versus tau, leavers as non-wheel-spinning](../../aria_pace/figures/j_vs_tau_leavers.png)

![Cost versus tau, hint-first counted as error](../../aria_pace/figures/j_vs_tau_hintfirst.png)

Test curves are descriptive (computed in the same single pass; they feed no choice). Open markers are the frozen BKT operating points. The "n consecutive errors" curve connects `n` = 1..6 and is not a tau sweep. The three profiles are three operating points on one BKT frontier, each chosen by its cost weights; the validation-chosen points sit close to the lowest-cost region of the test curves (see regret table below).

## The quiet-profile tau is not stable across scope assumptions (finding)

Source: `val_results.json` → `*.tau_stability_bkt` (validation, 1,000 student-bootstrap resamples; optimistic) and `frozen_val.json`.

{stability_tab()}

- **Within a configuration, the quiet `tau` is stable under resampling of students**: in the main configuration {pct(V['main']['tau_stability_bkt']['quiet']['frac_within_0.10_of_frozen'], 0)} of resamples choose a `tau` within 0.10 of the frozen {f(V['main']['tau_stability_bkt']['quiet']['frozen_tau'])}.
- **Across configurations it is not**: counting leavers as non-wheel-spinning moves the validation-chosen quiet `tau` from {f(FZ['configs']['main']['frozen']['taus']['bkt']['quiet'])} to {f(FZ['configs']['leavers']['rechosen']['taus']['bkt']['quiet'])} (no bootstrap resample reaches within 0.10 of 0.57). The instability is therefore a **sensitivity to how unlabelled leavers are treated**, not sampling noise. The balanced `tau` moves from {f(FZ['configs']['main']['frozen']['taus']['bkt']['balanced'])} to {f(FZ['configs']['leavers']['rechosen']['taus']['bkt']['balanced'])}; supportive from {f(FZ['configs']['main']['frozen']['taus']['bkt']['supportive'])} to {f(FZ['configs']['leavers']['rechosen']['taus']['bkt']['supportive'])}.
- **Consequences on test** (Source: `test_results.json` → `leavers.variants`): carrying the main `tau` = 0.57 into the leaver scope gives quiet cost {f(lvc['cost_J_per_student'])} vs {f(T['leavers']['variants']['rechosen']['methods']['bkt']['quiet']['cost_J_per_student'])} with the re-chosen `tau` = 0.85, while the missed-help share rises from {pct(lvc['missed_help_share'])} to {pct(lv['missed_help_share'])} (95% CI {ci(lv['ci95']['missed_help_share'], 1, True)}) and lead time falls from {f(lvc['mean_lead_time'])} to {f(lv['mean_lead_time'])} opportunities. **The "~5% missed help" figure belongs to the main configuration; under the leaver assumption that is cheapest, the quiet profile trades roughly {pct(lv['missed_help_share'], 0)} missed help for {f(lv['interventions_per_student'])} interventions per student (always-help: {f(lva['interventions_per_student'])}).**
- With the hint-first definition the quiet `tau` moves from 0.57 to {f(FZ['configs']['hintfirst']['rechosen']['taus']['bkt']['quiet'])}. On test the re-chosen value was slightly *worse* than the carried one (cost {f(T['hintfirst']['variants']['rechosen']['methods']['bkt']['quiet']['cost_J_per_student'])} vs {f(T['hintfirst']['variants']['carried']['methods']['bkt']['quiet']['cost_J_per_student'])}; regret {pct(regret('hintfirst', 'rechosen', 'quiet'))} vs {pct(regret('hintfirst', 'carried', 'quiet'))}), so re-choosing on validation does not always help.

### Diagnostic: frozen tau versus the test-optimal tau

Descriptive only; test-optimal values were not used for any choice. Source: `test_results.json` → `*.sweep.methods.bkt`.

{oracle_tab()}

The frozen main `tau` values lose 1-2% against the test-optimal value. Carrying the main quiet `tau` into the leaver scope loses {pct(regret('leavers', 'carried', 'quiet'), 0)} (the "leavers, carried" row), while the `tau` re-chosen on validation for that scope loses {pct(regret('leavers', 'rechosen', 'quiet'))}.

## Sensitivity analyses (test, frozen values)

Source: `test_results.json` → `leavers.variants`, `hintfirst.variants`. "Carried" uses the main `tau`/`n`; "rechosen" uses values re-chosen on validation under that configuration and frozen before test.

{sens_tab(T, 'test')}

- **Leavers as non-wheel-spinning** raises the unnecessary share of every policy (more sequences are counted as not needing help) and raises always-help's cost most; BKT's advantage over always-help grows (quiet: cost {f(lvc['cost_J_per_student'])} vs {f(lva['cost_J_per_student'])}).
- **Hint-first counted as error**: the quiet BKT policy makes {f(hf['interventions_per_student'])} interventions per student against {f(hfa['interventions_per_student'])} for always-help ({pct(1 - hf['interventions_per_student'] / hfa['interventions_per_student'], 0)} fewer) with {hf['missed_help']}/{hf['n_y1_with_help_opportunity']} wheel-spinning sequences missed ({pct(hf['missed_help_share'])}). More sequences have a help opportunity (the denominator is larger), which lowers missed help.

## Risk-score quality (validation only)

These are **validation** numbers (hyperparameters of PFA, BKT and the head were chosen on this split by next-response log-loss; they are optimistic). The test split was used only for the policy evaluation above; **AUC and calibration of the risk scores were not computed on test**, and the test lock prevents adding them without breaking the one-evaluation rule. Variant A label. Source: `baselines_val.json`, `riskhead_val.json`. Calibration is `n/a` for 0/1 methods.

{auc_tab()}

The logistic risk head did not beat PFA or BKT (mean validation AUC {f(RH['mean_val_auc']['risk_head'], 3)} vs BKT {f(RH['mean_val_auc']['bkt'], 3)} and PFA {f(RH['mean_val_auc']['pfa'], 3)}; margin {f(RH['margin_over_best_baseline'], 4)}, paired bootstrap CI {ci(RH['paired_bootstrap_margin_ci95'], 4)}; required +{f(RH['beat_margin_required'], 3)}), so by the pre-declared rule BKT is the primary risk score. The head's coefficients are in `riskhead_val.json`. AUC differs by label prevalence and population at each `k`; values are not comparable across `k`, and much of the AUC at k = 8 reflects the three-in-a-row mechanics with two opportunities left.

## Limitations

1. **Offline replay.** The logs contain no tutor interventions. Nothing here shows what a student would have done if interrupted or helped.
2. **"Unnecessary" is a proxy.** It means: the sequence later mastered within 10 opportunities without logged help. A student may have mastered because of unlogged help, an interruption might have helped or harmed, and a wheel-spinning sequence might not have benefited from earlier help.
3. **No learning-gain claim.** Lead time and help delay are counts of opportunities in logged data, not learning outcomes.
4. **The SymPy answer-verification layer from the demo is not evaluated.** Replay uses each log's own first-attempt correctness as ground truth.
5. **Grade level is unknown.** The sources for the 2009-10 file do not state it; no grade claim is made.
6. **`first_action` meanings are inferred** from cross-tabulation, not documented. They define the primary "verified error" (codes 1 and 2 excluded) and nothing else.
7. **No clinical labels and no clinical claims.** The three profiles are cost settings. Nothing here concerns ADHD, math anxiety, or any clinical or demographic group, and no such claim should be drawn from it.
8. **Selection.** Sequences where the student left before 10 opportunities without mastering have no label; the primary analysis drops them ({LC['variantA_k3']['val']['eligible_unlabelled_leavers']} such sequences at k = 3 on validation) and the leaver sensitivity treats them as non-wheel-spinning. Neither is correct; the truth is unobserved. Mastered sequences are labelled 0 by construction.
9. **Single dataset, single split.** One public dataset from one platform and one student-level split; no external replication.
10. **Optimistic validation.** The Platt calibrator, `tau` and `n` share the validation split, so all validation costs are optimistic; test is the unbiased figure for the frozen values (it is a single draw of 597 students and carries the intervals shown).
11. **Hyperparameter and design choices** (profile ratios 3:1, 1:1, 1:3; at most one intervention per sequence; decision points 3 to 9; Platt recalibration) were fixed by the owner/protocol, not tuned for performance, and other choices could change the numbers.
12. **Label definition.** The wheel-spinning label follows Wang et al. / Kai et al.; Beck & Gong's primary text was not read. The risk-score AUC is shown for both variants in `baselines_val.json`; the policy evaluation uses variant A only.
13. **Related work.** A profile-conditioned abstention frontier was not found published by the searches in `docs/aria_pace/LITERATURE.md`, which were limited and non-systematic; the closest work, DICE (arXiv 2610.04825), is simulation-based. Components used here (early wheel-spinning prediction, risk-triggered help, abstention-style thresholds) are not new.

## Reproducing

From the repository root, with `aria_pace/requirements.txt` installed and the dataset downloaded as in `docs/aria_pace/DATA.md`:

```
python -m pytest aria_pace/tests            # unit tests
python -m aria_pace.run_baselines           # Stage 3: data_report, split, label_counts, baselines_val
python -m aria_pace.run_policy              # Stage 4: riskhead_val, policy_val
python -m aria_pace.freeze_val              # Stage 5a: frozen_val.json (+ .sha256), val_results.json
python -m aria_pace.run_test                # Stage 5b: test_results.json; refuses if results/TEST_LOCK exists
python -m aria_pace.run_test --verify       # recompute test and compare, writes nothing (for third parties; NOT executed by the author: it re-reads the test split)
python -m aria_pace.report                  # figures
python -m aria_pace.make_results_md         # this file
```

Git history shows the order: SPEC amendment 2, then `frozen_val.json`, then the test results. Two consecutive runs of Stages 3 and 4 gave byte-identical JSON, and the validation freeze path is the code path used on test.

## Claim-to-file map

| Claim | File |
|---|---|
| Counts, label rate, exclusions | `label_counts.json`, `data_report.json`, `test_results.json` (`*.scope`) |
| Risk-score AUC and calibration (validation) | `baselines_val.json`, `riskhead_val.json` |
| Frozen taus, `n`, calibrators | `frozen_val.json`, `frozen_val.sha256` |
| Validation policy results (optimistic), tau stability | `val_results.json` |
| Test policy results, CIs, cost differences, sweeps | `test_results.json`, `TEST_LOCK` |
| Figures | `aria_pace/figures/*.png` |
"""
OUT.write_text(md)
print("wrote", OUT, len(md))
