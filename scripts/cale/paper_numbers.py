"""Single source of numbers for the CALE-AI workshop paper.

Reads only existing result files (aria_pace/results/*.json, aria_pace/results_cale/*.json); runs no analysis.
Writes paper/cale/numbers.json, NUMBERS.md, numbers.tex, tab_main.tex, tab_robust.tex.
Every value is read from a file by a field path ("/"-separated keys, "[i]" list index) or derived from such values by a stated formula.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "cale"
SRC = {"test": "aria_pace/results/test_results.json", "val": "aria_pace/results/val_results.json",
       "a7": "aria_pace/results/cv_tradeoff_val.json", "a6": "aria_pace/results/closed_loop_sensitivity.json",
       "dkt": "aria_pace/results/dkt_val.json", "sakt": "aria_pace/results/sakt_val.json",
       "ms": "aria_pace/results/multiskill_val.json",
       **{f"e{i}": f"aria_pace/results_cale/{n}.json" for i, n in [
           (1, "e1_tradeoff"), (2, "e2_calibration"), (3, "e3_errors"), (4, "e4_paired"), (5, "e5_logged_hints"),
           (6, "e6_skills"), (7, "e7_misses_by_help"), (8, "e8_hintfirst_variant"), (9, "e9_riskscore_paired"),
           (10, "e10_mrt_power")]}}
LABELS = ("locked-test", "CV", "validation", "exploratory")
_cache: dict = {}


def load(src):
    if src not in _cache:
        _cache[src] = json.loads((REPO / SRC[src]).read_text())
    return _cache[src]


def get(src, path):
    o = load(src)
    for part in path.split("/"):
        m = re.fullmatch(r"(.*?)((?:\[\d+\])*)", part)
        if m.group(1):
            o = o[m.group(1)]
        for idx in re.findall(r"\[(\d+)\]", m.group(2)):
            o = o[int(idx)]
    return o


def status(src):
    o = load(src)
    return o.get("status") or o.get("_note") or o.get("decision", {}).get("rule") if isinstance(o, dict) else None


ENTRIES: list = []


def add(key, macro, src, path, label, fmt, desc, ci=None):
    assert label in LABELS and re.fullmatch(r"[A-Za-z]+", macro), (key, macro)
    v = get(src, path)
    c = get(src, ci) if ci else None
    if c is not None:
        c = [c[0], c[1]]
    ENTRIES.append({"key": key, "macro": macro, "value": v, "ci95": c, "format": fmt, "label": label, "description": desc,
                    "source": {"file": SRC[src], "field": path, "ci_field": ci}, "source_status": status(src)})


def derive(key, macro, value, label, fmt, desc, formula, inputs, ci=None):
    assert label in LABELS and re.fullmatch(r"[A-Za-z]+", macro), (key, macro)
    ENTRIES.append({"key": key, "macro": macro, "value": value, "ci95": ci, "format": fmt, "label": label, "description": desc,
                    "source": {"derived": formula, "inputs": inputs}, "source_status": None})


# ------------------------------------------------------------------ main-table rows
ROWS = [  # (row id, macro stem, test method, test profile, CV source spec)
    ("quiet", "Quiet", "bkt", "quiet", ("e1", "frozen_tau_markers/quiet", None)),
    ("balanced", "Balanced", "bkt", "balanced", ("e1", "frozen_tau_markers/balanced", None)),
    ("supportive", "Supportive", "bkt", "supportive", ("e1", "frozen_tau_markers/supportive", None)),
    ("always_help", "AlwaysHelp", "always_help", "quiet", ("e1", "always_help", None)),
    ("two_wrong", "TwoWrong", "n_rule_tuned", "quiet", ("e1", "n_rule/2", None)),
    ("three_wrong", "ThreeWrong", "three_wrong_untuned", "quiet", ("e1", "n_rule/3", None)),
    ("random", "Random", "random_matched_to_bkt", "quiet", ("e1", "random_matched/0.57", None)),
    ("pfa", "PFA", "pfa", "quiet", ("e1", "curves/pfa", 0.46)),
    ("risk_head", "RiskHead", "risk_head", "quiet", ("e1", "curves/risk_head", 0.55)),
]
ROW_LABEL = {"quiet": r"BKT Quiet ($\tau$ 0.57)", "balanced": r"BKT Balanced ($\tau$ 0.19)", "supportive": r"BKT Supportive ($\tau$ 0.09)",
             "always_help": "Always-help", "two_wrong": "Two-wrong rule (tuned)", "three_wrong": "Three-wrong rule (untuned)",
             "random": "Random, matched to Quiet", "pfa": r"PFA ($\tau$ 0.46)", "risk_head": r"Risk head ($\tau$ 0.55)"}


def build_main_rows():
    T = "main/variants/frozen/methods"
    ah = get("test", f"{T}/always_help/quiet/interventions_per_student")
    tau_grid = get("e1", "tau_grid")
    for rid, stem, m, prof, (cs, cp, tau) in ROWS:
        b = f"{T}/{m}/{prof}"
        add(f"test.{rid}.hints", f"{stem}TestHints", "test", f"{b}/interventions_per_student", "locked-test", "f2",
            f"{rid}: first hints (interventions) per student, locked 2009 test", ci=f"{b}/ci95/interventions_per_student")
        v = get("test", f"{b}/interventions_per_student")
        derive(f"test.{rid}.reduction", f"{stem}TestReduction", 1 - v / ah, "locked-test", "pct1",
               f"{rid}: reduction in first hints vs always-help, locked test (no interval in the file)",
               "1 - interventions_per_student / always_help.interventions_per_student",
               [f"{SRC['test']}:{b}/interventions_per_student", f"{SRC['test']}:{T}/always_help/quiet/interventions_per_student"])
        add(f"test.{rid}.missed", f"{stem}TestMissed", "test", f"{b}/missed_help_share", "locked-test", "pct1",
            f"{rid}: missed-help share, locked test", ci=f"{b}/ci95/missed_help_share")
        add(f"test.{rid}.missed_n", f"{stem}TestMissedN", "test", f"{b}/missed_help", "locked-test", "int",
            f"{rid}: missed-help count (of 518), locked test")
        if tau is None:
            fields = {"hints": ("hints_per_student", "hints_ci95"), "reduction": ("reduction_vs_always_help", "reduction_ci95"),
                      "missed": ("missed_help_rate", "missed_ci95"), "missed_n": ("missed_help", None)}
            for f, (vp, cip) in fields.items():
                add(f"cv.{rid}.{f}", f"{stem}CV{f.title().replace('_', '')}", cs, f"{cp}/{vp}", "CV",
                    "int" if f == "missed_n" else ("f2" if f == "hints" else "pct1"),
                    f"{rid}: {f}, pooled CV held-out (CALE E1)", ci=f"{cp}/{cip}" if cip else None)
        else:
            j = tau_grid.index(tau)
            for f, (vp, cip) in {"hints": ("hints_per_student", "hints_ci95"), "reduction": ("reduction_vs_always_help", "reduction_ci95"),
                                 "missed": ("missed_help_rate", "missed_ci95"), "missed_n": ("missed_help", None)}.items():
                add(f"cv.{rid}.{f}", f"{stem}CV{f.title().replace('_', '')}", cs, f"{cp}/{vp}[{j}]", "CV",
                    "int" if f == "missed_n" else ("f2" if f == "hints" else "pct1"),
                    f"{rid}: {f} at its frozen validation tau {tau}, pooled CV held-out (CALE E1 sweep, index {j} of tau_grid)",
                    ci=f"{cp}/{cip}[{j}]" if cip else None)


# ------------------------------------------------------------------ everything else
def build_other():
    T = "main/variants/frozen"
    add("test.n_students", "TestStudents", "test", f"{T}/n_students", "locked-test", "int", "in-scope test students")
    add("test.n_sequences", "TestSequences", "test", f"{T}/n_sequences", "locked-test", "int", "in-scope test sequences")
    add("test.n_stuck", "TestStuck", "test", f"{T}/n_y1", "locked-test", "int", "stuck (y = 1) test sequences")
    add("test.n_stuck_opp", "TestStuckOpp", "test", f"{T}/methods/bkt/quiet/n_y1_with_help_opportunity", "locked-test", "int",
        "stuck test sequences with a help opportunity")
    for p, s in (("quiet", "Quiet"), ("balanced", "Balanced"), ("supportive", "Supportive")):
        add(f"test.{p}.jdiff_ah", f"{s}TestJDiffAH", "test", f"{T}/j_diff_vs_bkt/{p}/always_help/diff_bkt_minus_other", "locked-test", "f3",
            f"{p}: cost J per student, BKT minus always-help (paired bootstrap)", ci=f"{T}/j_diff_vs_bkt/{p}/always_help/ci95")
    add("test.quiet.jdiff_two_wrong", "QuietTestJDiffTwoWrong", "test", f"{T}/j_diff_vs_bkt/quiet/n_rule_tuned/diff_bkt_minus_other",
        "locked-test", "f3", "quiet: cost J, BKT minus tuned two-wrong rule", ci=f"{T}/j_diff_vs_bkt/quiet/n_rule_tuned/ci95")

    V = "main/variants/frozen/methods"
    add("val.quiet.hints", "QuietValHints", "val", f"{V}/bkt/quiet/interventions_per_student", "validation", "f2", "BKT Quiet hints per student, validation (optimistic)")
    add("val.always_help.hints", "AlwaysHelpValHints", "val", f"{V}/always_help/quiet/interventions_per_student", "validation", "f2", "always-help hints per student, validation")
    add("val.quiet.missed", "QuietValMissed", "val", f"{V}/bkt/quiet/missed_help_share", "validation", "pct1", "BKT Quiet missed help, validation (optimistic)")
    derive("val.quiet.reduction", "QuietValReduction",
           1 - get("val", f"{V}/bkt/quiet/interventions_per_student") / get("val", f"{V}/always_help/quiet/interventions_per_student"),
           "validation", "pct1", "BKT Quiet reduction vs always-help, validation", "1 - bkt/always_help interventions_per_student",
           [f"{SRC['val']}:{V}/bkt/quiet/interventions_per_student", f"{SRC['val']}:{V}/always_help/quiet/interventions_per_student"])
    for k in (3, 5, 8):
        add(f"val.bkt_auc_k{k}", f"BKTValAUC{'ABC'[[3,5,8].index(k)]}", "ms", f"auc_variantA_val/k{k}/bkt/auc", "validation", "f3",
            f"BKT stuck-risk AUC at k = {k}, validation (variant A); macro suffix A/B/C = k 3/5/8", ci=f"auc_variantA_val/k{k}/bkt/auc_ci95_student_bootstrap")
    for mod, src, pfx, path in (("DKT", "dkt", "paired_dkt_minus_bkt", None), ("SAKT", "sakt", "paired_sakt_minus_bkt", None)):
        for k, suf in ((3, "A"), (5, "B")):
            add(f"val.{mod.lower()}_minus_bkt_k{k}", f"{mod}MinusBKTAUC{suf}", src, f"{pfx}/variantA_k{k}/diff", "validation", "f3",
                f"{mod} minus BKT AUC at k = {k} (paired; promotion needs >= 0.01 and CI above 0 at k 3 and 5)", ci=f"{pfx}/variantA_k{k}/ci95")
    add("val.dkt_promoted", "DKTPromoted", "dkt", "decision/promote_dkt", "validation", "bool", "DKT promoted (pre-registered rule A3.5)")
    add("val.sakt_promoted", "SAKTPromoted", "sakt", "decision/promote_sakt", "validation", "bool", "SAKT promoted (A3a)")
    for mod, suf in (("ms_dkt", "MSDKT"), ("ms_sakt", "MSSAKT")):
        for k, ks in ((3, "A"), (5, "B")):
            add(f"val.{mod}_minus_bkt_k{k}", f"{suf}MinusBKTAUC{ks}", "ms", f"paired_minus_bkt_variantA/{mod}/k{k}/diff", "validation", "f3",
                f"{mod} minus BKT AUC at k = {k} (A5)", ci=f"paired_minus_bkt_variantA/{mod}/k{k}/ci95")
        add(f"val.{mod}_promoted", f"{suf}Promoted", "ms", f"decision/{mod}/promote", "validation", "bool", f"{mod} promoted (A5.4)")

    # A7: CV trade-off per mastery definition (BKT, each fold's CAL-chosen tau)
    add("cv.n_students", "CVStudents", "e1", "n_students", "CV", "int", "in-scope held-out students pooled over the 5 CV folds")
    add("cv.n_sequences", "CVSequences", "e1", "n_sequences", "CV", "int", "in-scope held-out sequences pooled over the CV folds")
    add("cv.n_stuck_opp", "CVStuckOpp", "e1", "frozen_tau_markers/quiet/stuck_with_help_opportunity", "CV", "int", "stuck CV sequences with a help opportunity")
    for d in ("D3", "D4", "D5T15", "D5T20", "DEMA"):
        dm = d.replace("5T15", "FiveTFifteen").replace("5T20", "FiveTTwenty").replace("D3", "DThree").replace("D4", "DFour").replace("DEMA", "DEma")
        b = f"definitions/{d}/chosen_tau_pooled_result/quiet"
        add(f"a7.{d}.quiet.reduction", f"AsevenQuiet{dm}Reduction", "a7", f"{b}/saving", "CV", "pct1",
            f"A7 {d}: Quiet reduction vs always-help, CAL-chosen tau per fold", ci=f"{b}/saving_ci95")
        add(f"a7.{d}.quiet.missed", f"AsevenQuiet{dm}Missed", "a7", f"{b}/missed", "CV", "pct1",
            f"A7 {d}: Quiet missed help, CAL-chosen tau per fold", ci=f"{b}/missed_ci95")
    add("a7.D3.quiet.tau_min", "AsevenQuietTauMin", "a7", "definitions/D3/tau_movement/quiet/min", "CV", "f2", "A7 D3: smallest fold-chosen Quiet tau")
    add("a7.D3.quiet.tau_max", "AsevenQuietTauMax", "a7", "definitions/D3/tau_movement/quiet/max", "CV", "f2", "A7 D3: largest fold-chosen Quiet tau")

    # A6: closed-loop simulation, Quiet vs always-help
    pts = get("a6", "points")
    for p, s in (("quiet", "Quiet"), ("balanced", "Balanced"), ("supportive", "Supportive")):
        for reg, rs in (("beats", "Beats"), ("indistinguishable", "Ties"), ("flips", "Flips")):
            derive(f"a6.{p}.{reg}", f"AsixSim{s}{rs}", sum(q["delta_J_policy_minus_always_help"][p]["region"] == reg for q in pts),
                   "exploratory", "int", f"A6: grid points (of {len(pts)}) where the {p} policy {reg} always-help",
                   f"count of points with delta_J_policy_minus_always_help/{p}/region == '{reg}'", [f"{SRC['a6']}:points"])
    zero = next(q for q in pts if q["p_help_effect"] == q["p_interrupt_cost"] == q["p_stuck_cost"] == 0)
    i0 = pts.index(zero)
    for p, s in (("quiet", "Quiet"), ("balanced", "Balanced")):
        add(f"a6.{p}.delta_zero", f"AsixSim{s}DeltaZero", "a6", f"points[{i0}]/delta_J_policy_minus_always_help/{p}/delta", "exploratory", "f3",
            f"A6: delta J ({p} minus always-help) per sequence at the all-zero point", ci=f"points[{i0}]/delta_J_policy_minus_always_help/{p}/ci95_simulation_noise_only")
        fl = flip_points(pts, p)
        derive(f"a6.{p}.flip_min", f"AsixSim{s}FlipMin", min(fl), "exploratory", "f3", f"A6: smallest p_stuck_cost at which {p} delta J crosses 0 (over all help/interrupt settings)",
               "linear interpolation of delta J between adjacent p_stuck_cost grid values, per (p_help_effect, p_interrupt_cost)", [f"{SRC['a6']}:points"])
        derive(f"a6.{p}.flip_max", f"AsixSim{s}FlipMax", max(fl), "exploratory", "f3", f"A6: largest p_stuck_cost at which {p} delta J crosses 0",
               "as flip_min", [f"{SRC['a6']}:points"])

    # CALE E2-E10 (exploratory)
    X = "exploratory"
    add("e2.ece_raw", "EtwoECERaw", "e2", "validation_all_tries/before_platt/ece_10bin", X, "f3", "E2: ECE of raw BKT risk, validation tries 3-9")
    add("e2.ece_platt", "EtwoECEPlatt", "e2", "validation_all_tries/after_platt/ece_10bin", X, "f3", "E2: ECE after frozen Platt (in-sample)")
    add("e2.brier", "EtwoBrier", "e2", "validation_all_tries/after_platt/brier", X, "f3", "E2: Brier after Platt", ci="validation_all_tries/after_platt/brier_ci95")
    add("e2.slope_raw", "EtwoSlopeRaw", "e2", "validation_all_tries/before_platt/calibration_line/slope", X, "f3", "E2: calibration slope, raw BKT")
    for t, w in ((3, "Three"), (5, "Five"), (7, "Seven")):
        add(f"e2.ece_raw_try{t}", f"EtwoECERawTry{w}", "e2", f"validation_per_try/{t}/before_platt/ece_10bin", X, "f3", f"E2: raw ECE at try {t}")
    add("e3.n_missed", "EthreeMissed", "e3", "n_missed", X, "int", "E3: Quiet misses (CV)")
    add("e3.n_caught", "EthreeCaught", "e3", "n_caught", X, "int", "E3: Quiet caught (CV)")
    add("e3.risk_reached", "EthreeRiskReached", "e3", "missed_where_risk_reached_tau_at_some_decision_point", X, "int",
        "E3: misses whose risk reached 0.57 but had no verified error at or after it")
    derive("e3.risk_reached_share", "EthreeRiskReachedShare", get("e3", "missed_where_risk_reached_tau_at_some_decision_point") / get("e3", "n_missed"),
           X, "pct0", "E3: share of misses where risk reached tau", "risk_reached / n_missed", [f"{SRC['e3']}:missed_where_risk_reached_tau_at_some_decision_point", f"{SRC['e3']}:n_missed"])
    P = "primary_frozen_tau_0.57"
    add("e4.diff_missed", "EfourDiffMissed", "e4", f"{P}/diff_missed_help_rate_bkt_minus_rule", X, "pp1", "E4: missed help, BKT Quiet minus two-wrong rule (points)", ci=f"{P}/diff_missed_ci95")
    add("e4.diff_hints", "EfourDiffHints", "e4", f"{P}/diff_hints_per_student_bkt_minus_rule", X, "f3", "E4: hints per student, BKT minus rule", ci=f"{P}/diff_hints_ci95")
    add("e5.stuck_hint_share", "EfiveStuckHintShare", "e5", "counts/validation/share_stuck_with_logged_hint", X, "pct0", "E5: stuck validation sequences with a logged ASSISTments hint")
    add("e5.nonstuck_hint_share", "EfiveNonstuckHintShare", "e5", "counts/validation/share_nonstuck_with_logged_hint", X, "pct0", "E5: non-stuck validation sequences with a logged hint")
    add("e5.b.missed", "EfiveRuleBMissed", "e5", "b_drop_decision_points_at_or_after_first_hint/quiet/missed_help_rate", X, "pct1",
        "E5(b): Quiet missed help without decision points at or after the first logged hint (validation)", ci="b_drop_decision_points_at_or_after_first_hint/quiet/missed_ci95")
    add("e5.main.missed", "EfiveMainMissed", "e5", "validation_main/quiet/missed_help_rate", X, "pct1", "E5 reference: validation Quiet missed help", ci="validation_main/quiet/missed_ci95")
    add("e6.n_skills", "EsixSkills", "e6", "n_skills_included", X, "int", "E6: skills with >= 30 stuck sequences with a help opportunity")
    add("e6.median_reduction", "EsixMedianReduction", "e6", "reduction_distribution/median", X, "pct1", "E6: median per-skill Quiet reduction")
    add("e6.q25_reduction", "EsixQLowReduction", "e6", "reduction_distribution/q25", X, "pct1", "E6: 25th percentile per-skill reduction")
    add("e6.q75_reduction", "EsixQHighReduction", "e6", "reduction_distribution/q75", X, "pct1", "E6: 75th percentile per-skill reduction")
    add("e6.median_missed", "EsixMedianMissed", "e6", "missed_help_distribution/median", X, "pct1", "E6: median per-skill missed help")
    add("e7.misses_with_help", "EsevenMissesWithHelp", "e7", "misses/share_with_help", X, "pct1", "E7: Quiet misses that received an ASSISTments hint before try 10")
    add("e7.no_help_rate", "EsevenNoHelpRate", "e7", "no_help_from_anyone_missed_rate/rate", X, "pct1", "E7: stuck sequences that got no help from anyone", ci="no_help_from_anyone_missed_rate/ci95")
    add("e8.quiet.missed", "EeightQuietMissed", "e8", "variant/quiet/missed_help_rate", X, "pct1", "E8: Quiet missed help when hint-first attempts count as errors", ci="variant/quiet/missed_ci95")
    add("e8.quiet.hints", "EeightQuietHints", "e8", "variant/quiet/hints_per_student", X, "f2", "E8: Quiet hints per student (variant)")
    add("e8.two_wrong.missed", "EeightTwoWrongMissed", "e8", "variant/two_wrong_rule/missed_help_rate", X, "pct1", "E8: two-wrong rule missed help (variant)")
    add("e9.diff_head", "EnineDiffHead", "e9", "differences/bkt_minus_risk_head/diff_missed_help_rate", X, "pp1", "E9: missed help, BKT minus risk head at matched hint rate (points)", ci="differences/bkt_minus_risk_head/ci95")
    add("e9.diff_pfa", "EnineDiffPFA", "e9", "differences/bkt_minus_pfa/diff_missed_help_rate", X, "pp1", "E9: missed help, BKT minus PFA at matched hint rate (points)", ci="differences/bkt_minus_pfa/ci95")
    E = "estimates"
    add("e10.p0", "EtenBaseRate", "e10", f"{E}/base_rate_next_correct/p0", X, "pct1", "E10: P(next answer correct | wrong step at tries 3-9)", ci=f"{E}/base_rate_next_correct/ci95")
    add("e10.icc", "EtenICC", "e10", f"{E}/between_student/icc_anova", X, "f3", "E10: within-student ICC of the next-answer outcome", ci=f"{E}/between_student/icc_ci95")
    add("e10.dp_per_session", "EtenDPPerSession", "e10", f"{E}/dps_per_session_with_outcome/mean", X, "f2", "E10: decision points (with outcome) per session")
    add("e10.sessions_none", "EtenSessionsNone", "e10", f"{E}/dps_per_session_with_outcome/share_zero", X, "pct0", "E10: sessions with no decision point")
    add("e10.dp_per_student", "EtenDPPerStudent", "e10", f"{E}/dps_per_student_total_with_outcome/mean", X, "f1", "E10: decision points per student, full record",
        ci=f"{E}/dps_per_student_total_with_outcome/ci95_mean")
    add("e10.top20_share", "EtenTopTwentyShare", "e10", f"{E}/share_of_dps_with_outcome_from_top_students/top_20pct", X, "pct0", "E10: share of decision points from the top 20% of students")
    for sc, sm in (("obs", "Obs"), ("1", "One"), ("5", "Five"), ("10", "Ten")):
        for eff, em in ((3, "Three"), (5, "Five"), (8, "Eight")):
            add(f"e10.N.S{sc}.{eff}pt", f"EtenN{sm}Sess{em}pt".replace("pt", "Pt"), "e10", f"results/S_{sc}|delta_{eff}pt/answer_N", X, "int",
                f"E10: students for 80% power, {eff}-point effect, scenario S_{sc} (pre-registered Liao/simulation rule)")
    add("e10.type1", "EtenTypeOne", "e10", "type_I_check/rejection_rate", X, "pct1", "E10: simulated Type I error at alpha 0.05")


def flip_points(pts, p):
    out = []
    grid = sorted({q["p_stuck_cost"] for q in pts})
    for he in sorted({q["p_help_effect"] for q in pts}):
        for ic in sorted({q["p_interrupt_cost"] for q in pts}):
            d = {q["p_stuck_cost"]: q["delta_J_policy_minus_always_help"][p]["delta"] for q in pts
                 if q["p_help_effect"] == he and q["p_interrupt_cost"] == ic}
            for a, b in zip(grid, grid[1:]):
                if d[a] < 0 <= d[b]:
                    out.append(a + (b - a) * (-d[a]) / (d[b] - d[a]))
                    break
    return out


# ------------------------------------------------------------------ formatting
def fmt(v, f):
    if f == "bool":
        return "yes" if v else "no"
    if f == "int":
        return f"{int(v):,}".replace(",", "{,}")
    if f.startswith("pct"):
        return f"{100 * v:.{int(f[3:])}f}"
    if f.startswith("pp"):
        return f"{100 * v:+.{int(f[2:])}f}"
    return f"{v:.{int(f[1:])}f}"


def fmt_ci(c, f):
    if c is None:
        return None
    f2 = f.replace("pp", "pct") if f.startswith("pp") else f
    return f"[{fmt(c[0], f2)}, {fmt(c[1], f2)}]"


def write_numbers():
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [e["key"] for e in ENTRIES]
    macros = [e["macro"] for e in ENTRIES]
    assert len(set(keys)) == len(keys) and len(set(macros)) == len(macros)
    for e in ENTRIES:
        e["text"] = fmt(e["value"], e["format"])
        e["ci_text"] = fmt_ci(e["ci95"], e["format"])
    src_hash = {v: hashlib.sha256((REPO / v).read_bytes()).hexdigest() for v in sorted(set(SRC.values()))}
    doc = {"note": "Generated by scripts/cale/paper_numbers.py from existing result files only. Do not edit by hand.",
           "formats": {"pct1": "percent, 1 decimal (macro holds the number without %)", "pct0": "percent, 0 decimals",
                       "pp1": "difference in percentage points, signed", "f2": "decimal, 2 places", "f3": "decimal, 3 places",
                       "int": "integer", "bool": "yes/no"},
           "labels": {"locked-test": "the single locked 2009 test evaluation (test_results.json)",
                      "validation": "2009 validation split; calibrators and thresholds chosen there, so optimistic",
                      "CV": "5-fold cross-validation over train + validation students (A7 folds); CALE E1 for the main-table CV column",
                      "exploratory": "post hoc analyses (CALE E2-E10) and the A6 assumption-dependent simulation"},
           "source_sha256": src_hash, "entries": ENTRIES}
    (OUT / "numbers.json").write_text(json.dumps(doc, indent=1))
    tex = ["% Generated by scripts/cale/paper_numbers.py from paper/cale/numbers.json. Do not edit by hand.",
           "% Percent macros hold the number only; write \\QuietCVReduction\\% in the text. <Name>CI holds the 95% interval."]
    for e in ENTRIES:
        tex.append(f"\\newcommand{{\\{e['macro']}}}{{{e['text']}}}% {e['label']}: {e['key']}")
        if e["ci_text"]:
            tex.append(f"\\newcommand{{\\{e['macro']}CI}}{{{e['ci_text']}}}")
    (OUT / "numbers.tex").write_text("\n".join(tex) + "\n")
    md = ["# Numbers used in the CALE-AI paper", "",
          "Generated by `scripts/cale/paper_numbers.py` from existing result files; **do not edit by hand** (re-run the script).",
          "Every value is read from the named file and field (`/`-separated keys, `[i]` = list index) or derived by the stated formula.",
          "Percent values are shown as percent; the LaTeX macro holds the number without `%`. Intervals are 95% (student bootstrap unless the source says otherwise).",
          "", "Labels: **locked-test** = the single locked 2009 test run; **validation** = 2009 validation (thresholds and calibrators chosen there, optimistic); "
          "**CV** = 5-fold cross-validation over train + validation students; **exploratory** = post hoc CALE analyses E2-E10 and the A6 simulation. "
          "All CALE files (E1-E10) carry the status \"exploratory, post hoc\"; the E1 CV operating points are labelled CV because they are the CV column of the main table.", ""]
    groups = {}
    for e in ENTRIES:
        groups.setdefault(e["label"], []).append(e)
    for lab in LABELS:
        md += [f"## {lab}", "", "| Macro | Value | 95% CI | Description | Source |", "|---|---|---|---|---|"]
        for e in groups.get(lab, []):
            s = e["source"]
            src = f"`{s['file']}` → `{s['field']}`" + (f" (CI: `{s['ci_field']}`)" if s.get("ci_field") else "") if "file" in s else f"derived: {s['derived']}; from {', '.join('`' + x + '`' for x in s['inputs'])}"
            unit = "%" if e["format"].startswith("pct") else (" pts" if e["format"].startswith("pp") else "")
            md.append(f"| `\\{e['macro']}` | {e['text'].replace('{,}', ',')}{unit} | {(e['ci_text'] or '').replace('{,}', ',')} | {e['description']} | {src} |")
        md.append("")
    md += ["## Source file hashes (sha256)", ""] + [f"- `{k}`: `{v}`" for k, v in src_hash.items()]
    (OUT / "NUMBERS.md").write_text("\n".join(md) + "\n")


# ------------------------------------------------------------------ tables (built from numbers.json)
def table_main():
    doc = json.loads((OUT / "numbers.json").read_text())
    by = {e["key"]: e for e in doc["entries"]}

    def cell(k, ci=True):
        e = by[k]
        s = f"\\{e['macro']}"
        if ci and e["ci95"] is not None:
            s += f"{{\\scriptsize\\,\\{e['macro']}CI}}"
        return s
    lines = [r"% Generated by scripts/cale/paper_numbers.py from numbers.json (values via numbers.tex macros). Do not edit by hand.",
             r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Locked 2009 test (\TestStudents{} students)} & \multicolumn{3}{c}{5-fold CV, held out (\CVStudents{} students)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Policy & Hints/student & Reduction (\%) & Missed help (\%) & Hints/student & Reduction (\%) & Missed help (\%) \\", r"\midrule"]
    for rid, *_ in ROWS:
        if rid == "always_help":
            lines.append(r"\midrule")
        lines.append(f"{ROW_LABEL[rid]} & {cell(f'test.{rid}.hints')} & {cell(f'test.{rid}.reduction')} & {cell(f'test.{rid}.missed')} & "
                     f"{cell(f'cv.{rid}.hints')} & {cell(f'cv.{rid}.reduction')} & {cell(f'cv.{rid}.missed')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Main results. Test: the single locked evaluation with thresholds frozen on validation; reductions on test have no interval in the result file. "
              r"CV: the same frozen thresholds applied to per-fold calibrated risk on held-out folds (post hoc, CALE E1). Missed help = stuck sequences with a help "
              r"opportunity that received no hint before try 10 (test $N=\TestStuckOpp$, CV $N=\CVStuckOpp$). Baselines, PFA and the risk head are at the Quiet "
              r"setting; the tuned rule for Balanced and Supportive is always-help. Brackets: 95\% student-bootstrap intervals.}",
              r"\label{tab:main}", r"\end{table*}"]
    (OUT / "tab_main.tex").write_text("\n".join(lines) + "\n")


def table_robust():
    pts = get("a6", "points")
    sc = sorted({q["p_stuck_cost"] for q in pts})
    lines = [r"% Generated by scripts/cale/paper_numbers.py from aria_pace/results/closed_loop_sensitivity.json (A6). Do not edit by hand.",
             r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Assumed & \multicolumn{3}{c}{Grid points (of 20)} & $\Delta J$ per sequence \\", r"\cmidrule(lr){2-4}",
             r"$p_{\mathrm{stuck}}$ & Quiet & Tie & Always & (Quiet $-$ always) \\", r"\midrule"]
    for s in sc:
        sub = [q for q in pts if q["p_stuck_cost"] == s]
        reg = [q["delta_J_policy_minus_always_help"]["quiet"]["region"] for q in sub]
        dl = [q["delta_J_policy_minus_always_help"]["quiet"]["delta"] for q in sub]
        lines.append(f"{s:g} & {reg.count('beats')} & {reg.count('indistinguishable')} & {reg.count('flips')} & "
                     f"${min(dl):+.2f}$ to ${max(dl):+.2f}$ \\\\")
    he = sorted({q["p_help_effect"] for q in pts})
    ic = sorted({q["p_interrupt_cost"] for q in pts})
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Closed-loop simulation (A6; assumed simulated students, not evidence of effect): when does the Quiet policy ($\tau=0.57$) "
              r"beat always-help? Rows: the assumed probability that a student left unaided after a wrong step abandons, the assumption that drives the sign. "
              f"Each row pools {len(he)} help-effect values ({he[0]:g}--{he[-1]:g}) $\\times$ {len(ic)} interruption-cost values ({ic[0]:g}--{ic[-1]:g}). "
              r"Quiet / Always: that policy has the lower cost $J$; tie: the 95\% interval of simulation noise includes 0. Quiet's advantage flips at $p_{\mathrm{stuck}}\approx\AsixSimQuietFlipMin$--\AsixSimQuietFlipMax. "
              r"Supportive ($\tau=0.09$) makes the same decisions as always-help at every point.}",
              r"\label{tab:robust}", r"\end{table}"]
    (OUT / "tab_robust.tex").write_text("\n".join(lines) + "\n")


def main():
    build_main_rows()
    build_other()
    write_numbers()
    table_main()
    table_robust()
    print(f"{len(ENTRIES)} numbers -> {OUT.relative_to(REPO)}/numbers.json, NUMBERS.md, numbers.tex, tab_main.tex, tab_robust.tex")


if __name__ == "__main__":
    main()
