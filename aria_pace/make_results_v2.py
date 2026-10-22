"""Builds docs/aria_pace/RESULTS_v2.md from results/test_results_2012.json (every number is read from that file)."""
import json

from .constants import RESULTS, ROOT

R = json.load(open(RESULTS / "test_results_2012.json"))
LOCK = json.load(open(RESULTS / "TEST_LOCK_2012"))
A, B = R["a_cross_dataset"], R["b_within_dataset_affect"]
f4 = lambda x: f"{x:.4f}"  # noqa: E731
sg = lambda x: f"{x:+.4f}"  # noqa: E731
ci = lambda c, g=f4: f"[{g(c[0])}, {g(c[1])}]"  # noqa: E731
out = []
w = out.append

w("""# ARIA-Pace RESULTS v2: the locked ASSISTments 2012-13 test portion

> **INTERNAL: not for publication.** Licence confirmation from ASSISTments is PENDING (docs/aria_pace/DATA_v2.md). Nothing here is to be published, posted or shown outside the team until it arrives.

Evaluated once on the locked 2012 test portion (seed 20261016; `results/split_2012.json`). Frozen values: `results/frozen_v2.json` (sha256 %s..., committed in c458650 before any test row was read). Lock file `results/TEST_LOCK_2012` (started %s, completed %s). Raw numbers: `results/test_results_2012.json`; this file is generated from it by `aria_pace/make_results_v2.py`.

## Caveats (they apply to every number below)

1. **"DKT-style" and "SAKT-style"** are simplified models: one skill per sequence (the first opportunity's KC), a small vocabulary, no multi-skill input. They are not reproductions of Piech et al. or Pandey & Karypis. They are reported, not promoted (BKT stays the primary risk score; promotion decided on 2009 validation).
2. **Affect** = outputs of ASSISTments' published affect detectors, **of unverified version**, not human-observed labels and not probabilities. They are rank-transformed (per-column mid-rank ECDF fitted on 2012-train rows). No human BROMP labels exist in this file.
3. **"Unnecessary" is a proxy**: an intervention on a sequence that later reaches mastery within 10 opportunities. It is not a measure of harm or of learner experience.
4. **Grade level is unverified** for both datasets.
5. **This is a replay of logged data, not a classroom.** No intervention was delivered to anyone; "delay" and "unnecessary" are counted against logged outcomes.
6. **Licence confirmation is pending** (see banner).
7. **No learning-gain claim and no clinical claim** is made or implied anywhere in this document.
8. The first_action codes (0/1/2) are inferred, not documented; "verified error" uses first_action == 0. 2012 KCs missing or unseen in 2009 use pooled parameters (BKT/PFA) or the UNK id (DKT/SAKT-style); the share of scored sequences with a KC outside the 2009 vocabulary is in table A1.
9. 2012 sequences are ordered by (start_time, problem_log_id); mastery in 2012 skill builders ends the sequence far more often than in 2009 (DATA_v2.md), so the two datasets differ in structure as well as in population.

Claims are never pooled: **(a)** cross-dataset (2009-trained models, no 2012 data in training) and **(b)** within-dataset affect question (2012-train, chosen on 2012-val).
""" % (LOCK["frozen_v2_sha256"][:12], LOCK["started_utc"], LOCK["completed_utc"]))

# ---- pre-registered
w("## Pre-registered endpoints\n")
w("### (a-P1) AUC of the primary risk score (BKT, 2009-trained), variant A, 95% CI over students\n")
w("| k | N | positives | AUC [95% CI] | ECE | calibration intercept / slope | Brier |\n|---|---|---|---|---|---|---|")
for k in (3, 5, 8):
    r = A["risk"]["results"][f"variantA_k{k}"]["bkt"]; cl = r["calibration_line"]
    w(f"| {k} | {r['n']:,} | {r['positives']:,} | {f4(r['auc'])} {ci(r['auc_ci95_student_bootstrap'])} | {r['ece_10bin']:.3f} | {cl['intercept']:+.2f} / {cl['slope']:.2f} | {r['brier']:.3f} |")
w("\nScores are the raw (uncalibrated) 2009-trained BKT risk. A negative intercept means the 2009-trained scores over-estimate risk on 2012 on average, and a slope below 1 means they are too spread out; the frozen 2009 Platt calibrator is applied unchanged in the policy replay.\n")

p = A["policy"]["main"]; v = p["variants"]["frozen"]
w("### (a-P2) and (a-P3) Policy replay on the 2012 test, BKT with the frozen 2009 taus (main configuration)\n")
w(f"Scope: {p['scope']['students']:,} students, {p['scope']['sequences']:,} sequences, {p['scope']['y1']:,} wheel-spinning, leavers excluded. Frozen taus (BKT): " +
  ", ".join(f"{k} {t}" for k, t in v["taus"]["bkt"].items()) + ". Bootstrap over students, 1,000 draws, seed 20261025.\n")
w("| profile | interventions/student BKT [CI] | always-help | reduction vs always-help | missed-help share BKT [CI] | J/student BKT | J/student always-help | J(BKT) - J(always-help) [CI] |\n|---|---|---|---|---|---|---|---|")
for prof in ("quiet", "balanced", "supportive"):
    b, h = v["methods"]["bkt"][prof], v["methods"]["always_help"][prof]
    d = v["j_diff_vs_bkt"][prof]["always_help"]
    red = 1 - b["interventions_per_student"] / h["interventions_per_student"]
    w(f"| {prof} | {b['interventions_per_student']:.3f} {ci(b['ci95']['interventions_per_student'][:2], lambda x: f'{x:.3f}')} | {h['interventions_per_student']:.3f} | {100*red:.1f}% | "
      f"{100*b['missed_help_share']:.2f}% {ci([100*c for c in b['ci95']['missed_help_share'][:2]], lambda x: f'{x:.2f}%')} | {b['cost_J_per_student']:.3f} | {h['cost_J_per_student']:.3f} | "
      f"{d['diff_bkt_minus_other']:+.3f} {ci(d['ci95'], lambda x: f'{x:+.3f}')} |")
w("\nNegative J difference = BKT policy cheaper than always-help under that profile's weights (quiet 3:1, balanced 1:1, supportive 1:3 as w_I:w_D).\n")

bm = B["mean_over_k"]
w("### (b-P1) Does affect add information? 2012 test, variant A, mean over k = 3, 5, 8 of AUC(with) - AUC(without)\n")
w("| variant of the question | mean difference [95% CI] |\n|---|---|")
w(f"| rows <= k (primary) | {sg(bm['with_primary|variantA']['mean_diff'])} {ci(bm['with_primary|variantA']['ci95'], lambda x: f'{x:+.4f}')} |")
w(f"| rows <= k-1 (lagged sensitivity) | {sg(bm['with_lagged|variantA']['mean_diff'])} {ci(bm['with_lagged|variantA']['ci95'], lambda x: f'{x:+.4f}')} |")
w(f"\nPre-registered rule: \"affect adds information\" only if the CI lower bound is above 0 AND the point estimate is at least 0.005. Primary passes: {B['decision']['primary_passes']}; lagged passes: {B['decision']['lagged_passes']}. "
  f"**Verdict: {B['decision']['verdict']}.** The lower bounds are above 0, so a small positive difference is detectable, but it is well below the pre-registered 0.005 threshold.\n")
w("| k | N | AUC without | AUC with (rows <= k) | difference [CI] | AUC with (rows <= k-1) | difference [CI] |\n|---|---|---|---|---|---|---|")
for k in (3, 5, 8):
    r = B["results"][f"variantA_k{k}"]; a = r["auc"]
    d1, d2 = B["paired_with_minus_without"][f"with_primary|variantA_k{k}"], B["paired_with_minus_without"][f"with_lagged|variantA_k{k}"]
    w(f"| {k} | {r['N']:,} | {f4(a['without'])} | {f4(a['with_primary'])} | {sg(d1['diff'])} {ci(d1['ci95'], lambda x: f'{x:+.4f}')} | {f4(a['with_lagged'])} | {sg(d2['diff'])} {ci(d2['ci95'], lambda x: f'{x:+.4f}')} |")
w("\nThe gain is concentrated at k = 3 (about +0.005 to +0.006) and is near zero at k = 8. Head logistic model, C = 10 for all (the edge of the pre-registered grid), selected on 2012 validation.\n")

# ---- exploratory
w("## Exploratory (everything below was not a pre-registered endpoint)\n")
w("### A1. All risk models on the 2012 test (claim a), variant A: AUC [95% CI] and paired difference to BKT\n")
w("| k | BKT | PFA | windowed head | DKT-style | SAKT-style | scored with KC outside the 2009 vocabulary |\n|---|---|---|---|---|---|---|")
for k in (3, 5, 8):
    r = A["risk"]["results"][f"variantA_k{k}"]; d = A["risk"]["paired_vs_bkt"][f"variantA_k{k}"]
    cells = [f"{f4(r['bkt']['auc'])} {ci(r['bkt']['auc_ci95_student_bootstrap'])}"]
    for m in ("pfa", "risk_head", "dkt_style", "sakt_style"):
        cells.append(f"{f4(r[m]['auc'])} ({sg(d[m]['diff'])} {ci(d[m]['ci95'], lambda x: f'{x:+.4f}')})")
    w(f"| {k} | " + " | ".join(cells) + f" | {100*r['unknown_kc_share_2009_vocab']:.1f}% |")
w("\nDKT-style and SAKT-style are reported, not promoted. Neither beats BKT on 2012, consistent with the 2009 validation decision. "
  "**The windowed logistic head scores higher than BKT at k = 3 and 5 on 2012**; BKT was the pre-registered primary on 2009 validation evidence, so this is reported as an exploratory observation, not a promotion. The decision of which score is primary is not revisited.\n")
w("### A2. Variant B (n > 10) AUC, same models\n")
w("| k | BKT | PFA | windowed head | DKT-style | SAKT-style |\n|---|---|---|---|---|---|")
for k in (3, 5, 8):
    r = A["risk"]["results"][f"variantB_k{k}"]
    w(f"| {k} | " + " | ".join(f4(r[m]["auc"]) for m in ("bkt", "pfa", "risk_head", "dkt_style", "sakt_style")) + " |")
w("\n### A3. Policy comparators, main configuration (frozen 2009 values)\n")
w("| profile | policy | interventions/student | missed-help share | J/student | J(BKT) - J(policy) [CI] (negative = BKT cheaper) |\n|---|---|---|---|---|---|")
names = {"n_rule_tuned": "n-errors rule (n tuned on 2009 val)", "three_wrong_untuned": "three-wrong rule (untuned)", "pfa": "PFA policy", "risk_head": "head policy", "random_matched_to_bkt": "random, matched to BKT"}
for prof in ("quiet", "balanced", "supportive"):
    for key, nm in names.items():
        m = v["methods"][key][prof]; d = v["j_diff_vs_bkt"][prof][key]
        mh = "n/a" if m["missed_help_share"] is None else f"{100*m['missed_help_share']:.2f}%"
        w(f"| {prof} | {nm} | {m['interventions_per_student']:.3f} | {mh} | {m['cost_J_per_student']:.3f} | {d['diff_bkt_minus_other']:+.3f} {ci(d['ci95'], lambda x: f'{x:+.3f}')} |")
w("\n### A4. Other configurations (BKT, carried 2009 taus vs taus re-chosen on 2009 validation for that configuration)\n")
w("| configuration | scope | quiet: interventions/student | quiet: missed-help | quiet: J(BKT) - J(always-help) [CI] |\n|---|---|---|---|---|")
for cfg in ("leavers", "hintfirst"):
    for vv in ("carried", "rechosen"):
        x = A["policy"][cfg]["variants"][vv]; b = x["methods"]["bkt"]["quiet"]; d = x["j_diff_vs_bkt"]["quiet"]["always_help"]
        mh = "n/a" if b["missed_help_share"] is None else f"{100*b['missed_help_share']:.2f}%"
        w(f"| {cfg} / {vv} (tau quiet {x['taus']['bkt']['quiet']}) | {A['policy'][cfg]['scope']['sequences']:,} sequences | {b['interventions_per_student']:.3f} | {mh} | {d['diff_bkt_minus_other']:+.3f} {ci(d['ci95'], lambda x: f'{x:+.3f}')} |")
w("\n### B2. Affect, variant B (n > 10), mean over k of the difference\n")
w(f"- rows <= k: {sg(bm['with_primary|variantB']['mean_diff'])} {ci(bm['with_primary|variantB']['ci95'], lambda x: f'{x:+.4f}')}\n- rows <= k-1: {sg(bm['with_lagged|variantB']['mean_diff'])} {ci(bm['with_lagged|variantB']['ci95'], lambda x: f'{x:+.4f}')}\n")
w("### B3. Affect and the policy (descriptive; tau re-chosen on 2012 validation per model, main configuration)\n")
w("| profile | model | tau | interventions/student | J/student | J(with) - J(without) [CI] |\n|---|---|---|---|---|---|")
sec = B["secondary_policy_J_descriptive"]
for prof in ("quiet", "balanced", "supportive"):
    for n in ("without", "with_primary", "with_lagged"):
        m = sec["models"][n][prof]
        dd = sec["j_diff"].get(f"{n}|{prof}")
        w(f"| {prof} | {n} | {m['tau_chosen_on_2012_val']} | {m['interventions_per_student']:.3f} | {m['cost_J_per_student']:.4f} | " + (f"{dd['diff_with_minus_without']:+.4f} {ci(dd['ci95'], lambda x: f'{x:+.4f}')}" if dd else "(reference)") + " |")
w("\nEvery interval for the policy cost difference spans zero or is exactly zero (identical decisions), so affect changes no policy outcome measurably here.\n")
w("## Reproduction\n\n`python -m aria_pace.run_test2012` refuses to run again (TEST_LOCK_2012). Figures for the summit slide are produced in C6 from `results/test_results_2012.json`.\n")
(ROOT.parent / "docs" / "aria_pace" / "RESULTS_v2.md").write_text("\n".join(out))
print("wrote RESULTS_v2.md")
