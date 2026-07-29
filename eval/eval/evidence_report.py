#!/usr/bin/env python3.11
"""
Honest evidence + limitations reporting for ARIA.

Reads the actual result JSONs and produces two files:

  data/eval/EVIDENCE.md    — every ARIA claim mapped to its evidence tier:
      | Claim | Evidence | Tier | n | Limitation |
    Tiers (strictly applied):
      A — validated against real human labels
      B — validated against real behavior (proxy / measured labels)
      C — validated across independent generators
      D — synthetic only, circular
      E — asserted, not validated

  data/eval/LIMITATIONS.md — the limitations reviewers should see, with the
    real generator-overfitting gap number filled in.

No result numbers are hard-coded; everything is pulled from:
  data/eval/eval_100.json                    (metacognition eval; Part 0)
  data/eval/cross_generator_results.json     (Part 1)
  data/eval/external_validation_results.json (Part 4)

Sources that are absent are reported as "pending", never fabricated.

    python3.11 eval/evidence_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULTS_DIR = os.path.join(REPO_ROOT, "data", "eval")
METACOG_CANDIDATES = [
    os.path.join(RESULTS_DIR, "eval_100.json"),
    os.path.join(RESULTS_DIR, "metacognition_eval_results.json"),
]
CROSSGEN_PATH = os.path.join(RESULTS_DIR, "cross_generator_results.json")
EXTVAL_PATH = os.path.join(RESULTS_DIR, "external_validation_results.json")
EVIDENCE_MD = os.path.join(RESULTS_DIR, "EVIDENCE.md")
LIMITATIONS_MD = os.path.join(RESULTS_DIR, "LIMITATIONS.md")


def _load(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _load_metacog():
    for p in METACOG_CANDIDATES:
        d = _load(p)
        if d is not None:
            return d, p
    return None, None


def _fmt(v, fmt="{:.3f}"):
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return "—"


def _find_exp(extval, letter):
    if not extval:
        return None
    for e in extval.get("experiments", []):
        if e.get("experiment") == letter:
            return e
    return None


# ==================================================================
# build the claim rows
# ==================================================================

def build_rows(metacog, crossgen, extval) -> list[dict]:
    rows: list[dict] = []

    gap = None
    gap_interp = None
    if crossgen:
        gap = crossgen.get("generalization_gap", {}).get("mean_points")
        gap_interp = crossgen.get("generalization_gap", {}).get("interpretation")

    def state_detection_tier():
        if gap is None:
            return "D", ("cross-generator validation pending; on the synthetic test "
                         "split only, this is circular (classifier tuned on the same "
                         "generator).")
        if gap < 5:
            return "C", (f"generalizes across generators (mean gap {gap} pts < 5); "
                         "still no real human-labelled think-aloud test (would be A).")
        if gap <= 15:
            return "C", (f"partial generator overfitting (mean gap {gap} pts); some "
                         "signal is genuine, some is llama3.1-specific.")
        return "D", (f"largely detects llama3.1's fingerprint (mean gap {gap} pts > "
                     "15); accuracy does not transfer to other generators.")

    # 1. cognitive-state detection
    if metacog:
        sd = metacog.get("state_detection", {})
        tier, lim = state_detection_tier()
        rows.append({
            "claim": f"Cognitive-state detection from think-aloud text "
                     f"(accuracy {_fmt(sd.get('accuracy'), '{:.1%}')}, "
                     f"macro-F1 {_fmt(sd.get('macro_f1'))})",
            "evidence": "metacognition_eval + cross_generator_eval",
            "tier": tier, "n": sd.get("n"), "limitation": lim,
        })

    # 2. cross-generator robustness (the gap itself)
    if crossgen:
        per = crossgen.get("generalization_gap", {}).get("per_generator_points", {})
        n_total = sum((crossgen.get("per_generator", {}).get(t, {}) or {}).get("n", 0)
                      for t in per)
        rows.append({
            "claim": f"Classifier accuracy transfers to unseen generators "
                     f"(mean gap {gap} pts: {per})",
            "evidence": "cross_generator_eval (mistral/gemma2/phi3 vs llama3.1)",
            "tier": "C", "n": n_total or None,
            "limitation": gap_interp or "—",
        })
    else:
        rows.append({
            "claim": "Classifier accuracy transfers to unseen generators",
            "evidence": "cross_generator_eval", "tier": "D", "n": None,
            "limitation": "cross-generator run not yet complete.",
        })

    # 3. intervention appropriateness (LLM judge)
    if metacog:
        ia = metacog.get("intervention_appropriateness", {})
        per_state = ia.get("per_state", {})
        n_rated = sum(v.get("n_rated", 0) for v in per_state.values())

        def _ms(st):
            return _fmt((per_state.get(st, {}) or {}).get("mean_appropriateness"), "{:.2f}")

        rows.append({
            "claim": f"Interventions are state-appropriate "
                     f"(mean {_fmt(ia.get('overall_mean'), '{:.2f}')}/2 by LLM judge; "
                     f"FRUSTRATED {_ms('FRUSTRATED')}, RUSHING {_ms('RUSHING')})",
            "evidence": "metacognition_eval Metric 2 (llama3.1 judge)",
            "tier": "E", "n": n_rated,
            "limitation": "LLM-as-judge with NO human agreement study; the judge "
                          "shares a model family with the data generator.",
        })

    # 4. behavioral proxy vs real gaze-measured confusion (Exp D)
    expd = _find_exp(extval, "D")
    if expd and expd.get("status") == "ran":
        rows.append({
            "claim": f"Behavioral incorrectness aligns with real (gaze-measured) "
                     f"confusion (agreement {_fmt(expd.get('accuracy'))}, "
                     f"macro-F1 {_fmt(expd.get('macro_f1'))})",
            "evidence": "external_validation Exp D (EduAgent310, real gaze labels)",
            "tier": "B", "n": expd.get("n_labeled"),
            "limitation": "EduAgent310 has no think-aloud text, so ARIA's TEXT "
                          "classifier is not tested; only FLOW/CONFUSED are "
                          "recoverable and the classes are highly imbalanced.",
        })
    else:
        rows.append({
            "claim": "Behavioral signals align with real cognitive measures",
            "evidence": "external_validation Exp D (EduAgent310)",
            "tier": "B" if expd else "E", "n": None,
            "limitation": (expd.get("reason") if expd else "EduAgent310 not present."),
        })

    # 5. non-LLM text robustness (NCTE)
    expb = _find_exp(extval, "B")
    if expb and expb.get("status") == "ran":
        rows.append({
            "claim": f"Classifier does not collapse on real classroom speech "
                     f"(degenerate={expb.get('degenerate')}, "
                     f"conf drop {_fmt(expb.get('confidence_drop'))})",
            "evidence": "external_validation Exp B (NCTE real utterances)",
            "tier": "E", "n": expb.get("n_utterances"),
            "limitation": "No ARIA ground-truth labels on NCTE; distributional "
                          "evidence only (manual review sample provided).",
        })
    else:
        rows.append({
            "claim": "Classifier survives contact with real human (non-LLM) text",
            "evidence": "external_validation Exp B (NCTE)", "tier": "E", "n": None,
            "limitation": "NCTE transcripts require a data-request form; run pending.",
        })

    # 6. transfer / calibration / timing (synthetic)
    if metacog:
        tf = metacog.get("transfer_detection", {})
        rows.append({
            "claim": f"Self-initiated-metacognition (transfer) detection "
                     f"(F1 {_fmt(tf.get('f1'))})",
            "evidence": "metacognition_eval Metric 4",
            "tier": "D", "n": tf.get("n") or tf.get("support"),
            "limitation": "Only PLANNING/INSIGHT are separable in the synthetic "
                          "corpus; labels are deterministic by construction "
                          "(circular).",
        })
        cal = metacog.get("calibration_validity", {})
        rows.append({
            "claim": f"Calibration measurement is valid "
                     f"(sim error {_fmt(cal.get('mean_calibration_error') or cal.get('calibration_error') or cal.get('ece'))})",
            "evidence": "metacognition_eval Metric 5", "tier": "D", "n": cal.get("n"),
            "limitation": "Confidence ratings are simulated, not from real students.",
        })
        tim = metacog.get("timing_validity", {})
        rows.append({
            "claim": f"Optimal-intervention-timing detection is valid "
                     f"(match rate "
                     f"{_fmt(tim.get('match_rate') or tim.get('optimal_match_rate'), '{:.0%}')})",
            "evidence": "metacognition_eval Metric 6",
            "tier": "D", "n": tim.get("n") or tim.get("n_scenarios"),
            "limitation": "Timing optima are hypotheses tested on simulated "
                          "scenarios, not measured on real recovery outcomes.",
        })

    # 7. longitudinal (single real user)
    rows.append({
        "claim": "Longitudinal metacognitive-growth tracking (real usage)",
        "evidence": "data/metacognition/longitudinal_naren.json",
        "tier": "B", "n": 1,
        "limitation": "n=1 real user; no statistical power, not generalizable.",
    })
    return rows


# ==================================================================
# emit EVIDENCE.md
# ==================================================================

TIER_LEGEND = [
    ("A", "validated against real human labels"),
    ("B", "validated against real behavior (proxy / measured labels)"),
    ("C", "validated across independent generators"),
    ("D", "synthetic only, circular"),
    ("E", "asserted, not validated"),
]


def write_evidence(rows, stamp) -> None:
    lines = ["# ARIA — Evidence Table\n",
             f"_Generated {stamp} from the result JSONs. Every row's tier is applied "
             "strictly; nothing here is aspirational._\n",
             "## Tier legend\n"]
    for t, desc in TIER_LEGEND:
        lines.append(f"- **{t}** — {desc}")
    lines.append("")
    dist = Counter(r["tier"] for r in rows)
    lines.append("**Tier distribution:** "
                 + ", ".join(f"{t}={dist.get(t, 0)}" for t, _ in TIER_LEGEND) + "\n")
    lines.append("| Claim | Evidence | Tier | n | Limitation |")
    lines.append("|---|---|:--:|--:|---|")
    for r in rows:
        n = r["n"] if r["n"] is not None else "—"
        claim = str(r["claim"]).replace("|", "\\|")
        ev = str(r["evidence"]).replace("|", "\\|")
        lim = str(r["limitation"]).replace("|", "\\|")
        lines.append(f"| {claim} | {ev} | {r['tier']} | {n} | {lim} |")
    lines.append("")
    lines.append("### How to read this\n")
    lines.append("The strongest ARIA claims are Tier B/C. **No claim is Tier A** "
                 "(real human think-aloud labels) until the EDM 2024 think-aloud "
                 "dataset is obtained (see `datasets/REQUEST_EMAIL.md`). Anything "
                 "still at Tier D is validated only against the generator that "
                 "produced its training data; Tier E is asserted from design "
                 "rationale, not measured.\n")
    with open(EVIDENCE_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ==================================================================
# emit LIMITATIONS.md
# ==================================================================

def write_limitations(metacog, crossgen, extval, stamp) -> None:
    gap = (crossgen or {}).get("generalization_gap", {}).get("mean_points")
    gap_interp = (crossgen or {}).get("generalization_gap", {}).get("interpretation")
    per_gap = (crossgen or {}).get("generalization_gap", {}).get("per_generator_points", {})
    lexical = (crossgen or {}).get("lexical_overlap", {}).get("mean_jaccard_across_generators")
    expd = _find_exp(extval, "D")
    expb = _find_exp(extval, "B")

    L = ["# ARIA — Limitations\n",
         f"_Generated {stamp}. These are stated plainly and quantified where "
         "possible; reviewers reward honesty over polish._\n"]

    L.append("## 1. Circular evaluation (and what cross-generator testing showed)\n")
    L.append("ARIA's think-aloud corpus was generated by a single model "
             "(llama3.1:8b) and the classifier was built against it. Accuracy on the "
             "held-out split of that same corpus is therefore partly circular.")
    if gap is not None:
        L.append(f"\nThe cross-generator experiment (Part 1) regenerates a held-out "
                 f"set with mistral:7b, gemma2:9b and phi3:medium and re-runs the "
                 f"UNCHANGED classifier. **Mean generalization gap: {gap} points** "
                 f"(per generator: {per_gap}). {gap_interp}")
        if lexical is not None:
            L.append(f"\nMean Jaccard overlap of the top-20 discriminative tokens per "
                     f"state across generators is {lexical}; low overlap on a state "
                     f"means its detection rests on generator-specific words, not "
                     f"shared semantics.")
    else:
        L.append("\nThe cross-generator run is not yet complete, so the size of any "
                 "generator-specific overfitting is not yet quantified here.")

    L.append("\n## 2. LLM-as-judge without a human agreement study\n")
    if metacog:
        ia = metacog.get("intervention_appropriateness", {})
        L.append(f"Intervention appropriateness (overall "
                 f"{_fmt(ia.get('overall_mean'), '{:.2f}')}/2) is scored by an LLM "
                 f"judge ({ia.get('judge_model', 'llama3.1:8b')}), NOT by humans. "
                 "There is no inter-rater agreement study against expert educators, "
                 "and the judge shares a model family with the data generator, which "
                 "can inflate agreement. Treat these as Tier-E (asserted) numbers.")
    else:
        L.append("Intervention appropriateness is scored by an LLM judge with no "
                 "human agreement study.")

    L.append("\n## 3. Single real user (n = 1) for longitudinal metrics\n")
    L.append("Longitudinal metacognitive-growth tracking is based on one real user "
             "(`longitudinal_naren.json`). n = 1 has no statistical power and does "
             "not generalize; it is a case study, not evidence of effect.")

    L.append("\n## 4. Proxy-derived labels are not ground truth\n")
    L.append("For behavioral datasets (ASSISTments, Eedi), ARIA states are DERIVED "
             "from observable behavior (response time, attempts, hints, outcome) via "
             "documented, confidence-tiered rules. These proxies are not human "
             "cognitive labels and must never be reported as such. Every derived "
             "record is tagged with its `proxy_method` and `proxy_confidence`.")

    L.append("\n## 5. INSIGHT is undetectable from behavioral data\n")
    L.append("A moment of insight cannot be read from response-time / attempt / hint "
             "logs — it requires the student's words. The ASSISTments proxy therefore "
             "never emits INSIGHT, and EduAgent310's gaze measures only recover "
             "CONFUSED and FLOW. INSIGHT (and PLANNING, RUSHING, FRUSTRATED, STUCK) "
             "have no behavioral-only signal in these corpora.")
    if expd and expd.get("status") == "ran":
        L.append(f"\nExperiment D (EduAgent310, real gaze-derived labels, "
                 f"n={expd.get('n_labeled')}) recovered only "
                 f"{expd.get('present_classes')}, and a correctness-based behavioral "
                 f"proxy agreed with the real labels at only "
                 f"{_fmt(expd.get('accuracy'))} accuracy "
                 f"(macro-F1 {_fmt(expd.get('macro_f1'))}) — behavioral incorrectness "
                 "is a weak stand-in for measured confusion.")

    L.append("\n## 6. Non-commercial licence constraints (Eedi)\n")
    L.append("The Eedi / NeurIPS 2020 dataset is CC BY-NC-ND 4.0 — NON-COMMERCIAL, no "
             "derivatives. `commercial_use_allowed=False` is propagated into every "
             "Eedi-derived record, and any report touching Eedi prints a "
             "non-commercial banner. EduAgent and NCTE are likewise research-use only. "
             "ARIA cannot be commercialized on top of these datasets.")

    L.append("\n## 7. Non-LLM text transfer\n")
    if expb and expb.get("status") == "ran":
        L.append(f"On real classroom speech (NCTE, n={expb.get('n_utterances')}), the "
                 f"classifier's mean confidence changed by "
                 f"{_fmt(expb.get('confidence_drop'))} vs synthetic text and the "
                 f"prediction distribution was "
                 f"{'degenerate' if expb.get('degenerate') else 'non-degenerate'}. "
                 "There are no ground-truth ARIA labels for NCTE, so this is "
                 "distributional evidence only.")
    else:
        L.append("Robustness to real human (non-LLM) text is not yet measured — the "
                 "NCTE transcripts require a data-request form. A synthetic-to-real "
                 "confidence drop or a degenerate prediction distribution would be a "
                 "publishable negative finding.")

    L.append("\n## 8. No real human think-aloud labels yet (no Tier-A evidence)\n")
    L.append("The only dataset that directly matches ARIA's input modality (real "
             "think-aloud text with SRL labels) is the EDM 2024 dataset, which is "
             "access-controlled. Until it is obtained (request email prepared at "
             "`datasets/REQUEST_EMAIL.md`), ARIA has NO Tier-A validation of its core "
             "claim — text-based cognitive-state detection against real human labels.")

    with open(LIMITATIONS_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", default=None,
                    help="Override the generation timestamp (for reproducible output).")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    metacog, metacog_path = _load_metacog()
    crossgen = _load(CROSSGEN_PATH)
    extval = _load(EXTVAL_PATH)
    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")

    print("evidence_report inputs:")
    print(f"  metacog            : {metacog_path or 'MISSING'}")
    print(f"  cross_generator    : {CROSSGEN_PATH if crossgen else 'MISSING'}")
    print(f"  external_validation: {EXTVAL_PATH if extval else 'MISSING'}")

    rows = build_rows(metacog, crossgen, extval)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_evidence(rows, stamp)
    write_limitations(metacog, crossgen, extval, stamp)

    dist = Counter(r["tier"] for r in rows)
    print(f"\nEVIDENCE.md    -> {EVIDENCE_MD}  ({len(rows)} claims; "
          f"tiers {dict(sorted(dist.items()))})")
    print(f"LIMITATIONS.md -> {LIMITATIONS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
