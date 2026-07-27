#!/usr/bin/env python3.11
"""
External validation of ARIA against real education datasets.

Four experiments, each tagged by the strength of evidence it can produce:

  DIRECT        — compared against real human/measured labels
  PROXY         — compared against labels derived from behavior (not ground truth)
  DISTRIBUTIONAL— no labels; only distribution / shape comparisons

  EXPERIMENT A (ASSISTments) — behavioral-proxy agreement. Not a text-
      classification validation (there is no text). Tests whether ARIA's state
      model predicts real behavior: state-prevalence chi-square, STUCK
      turns-to-recovery vs timing.py's predicted optimum, and an empirical
      state-transition matrix vs ARIA's transition matrix on synthetic data
      (Frobenius distance).  Tier: PROXY.

  EXPERIMENT B (NCTE) — non-LLM text robustness. Runs the UNCHANGED classifier on
      real classroom utterances (no ARIA ground truth) and reports the
      prediction distribution (is it degenerate?), mean confidence vs synthetic,
      and dumps 30 (utterance, predicted_state, evidence) triples for eyeballing.
      Tier: DISTRIBUTIONAL.

  EXPERIMENT C (EduAgent705 vs ARIA corpus) — cross-corpus synthetic comparison.
      Tier: DISTRIBUTIONAL.

  EXPERIMENT D (EduAgent310) — real cognitive labels. EduAgent310 carries real
      gaze-derived confusion/inattention. Compares an independent behavioral
      proxy (answer correctness) against those real cognitive labels
      (accuracy / macro-F1 / confusion). Tier: DIRECT.

Every experiment runs only if its data is present, and skips with a clear
message otherwise. Results are written to
data/eval/external_validation_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from scipy.stats import chi2 as _chi2_dist
except Exception:  # pragma: no cover
    _chi2_dist = None

from metacognition.analyzer import (  # noqa: E402
    CognitiveStateAnalyzer,
    COGNITIVE_STATES,
    compute_classification_metrics,
    _join_think_aloud,
    _load_jsonl,
)
from metacognition.timing import DEFAULT_OPTIMAL  # noqa: E402
from datasets.registry import get_spec  # noqa: E402
from datasets.adapters import DatasetNotAvailableError  # noqa: E402
from datasets.adapters.eedi import NON_COMMERCIAL_BANNER  # noqa: E402
from datasets.adapters.assistments import AssistmentsAdapter  # noqa: E402

# STATE_RANK for measuring "recovery" (higher == better cognitive state).
from metacognition.interventions import STATE_RANK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, "data", "synthetic_thinkaloud")
TEST_PATH = os.path.join(OUT_DIR, "test.jsonl")
RESULTS_DIR = os.path.join(REPO_ROOT, "data", "eval")
RESULTS_PATH = os.path.join(RESULTS_DIR, "external_validation_results.json")
NCTE_REVIEW_PATH = os.path.join(RESULTS_DIR, "ncte_review_sample.txt")
SESSION_HISTORY_CANDIDATES = [
    os.path.join(REPO_ROOT, "data", "synthetic_history.jsonl"),
    os.path.join(os.path.dirname(REPO_ROOT), "data", "synthetic_history.jsonl"),
    os.path.join(REPO_ROOT, "data", "metacognition", "synthetic_history.jsonl"),
]


# ==================================================================
# shared helpers
# ==================================================================

def load_records(name: str, limit: int = 0):
    """Return (records, spec, error). records is None if unavailable."""
    spec = get_spec(name)
    adapter_cls = spec.resolve_adapter()
    if adapter_cls is None:
        return None, spec, f"{name}: no adapter implemented"
    adapter = adapter_cls(spec)
    try:
        recs = []
        for r in adapter.to_aria_schema():
            recs.append(r)
            if limit and len(recs) >= limit:
                break
        return recs, spec, None
    except DatasetNotAvailableError as e:
        return None, spec, str(e)


def chi_square_gof(observed: dict[str, int], expected_props: dict[str, float]) -> dict:
    """Goodness-of-fit of observed state counts vs expected proportions."""
    states = [s for s in observed if observed[s] > 0]
    obs = np.array([observed[s] for s in states], dtype=float)
    total = obs.sum()
    exp = np.array([expected_props.get(s, 0.0) for s in states], dtype=float)
    if exp.sum() == 0:
        return {"error": "no expected mass over observed states"}
    exp = exp / exp.sum() * total  # scale expected to observed total
    stat = float(np.sum((obs - exp) ** 2 / np.where(exp == 0, np.nan, exp)))
    dof = max(1, len(states) - 1)
    if _chi2_dist is not None and np.isfinite(stat):
        p = float(_chi2_dist.sf(stat, dof))
    else:
        p = None
    return {"chi2": round(stat, 3), "dof": dof,
            "p_value": (round(p, 5) if p is not None else None),
            "states": states,
            "observed": {s: int(observed[s]) for s in states},
            "expected": {s: round(float(e), 2) for s, e in zip(states, exp)}}


def synthetic_state_distribution() -> dict[str, float]:
    """ARIA's assumed state prevalence = the synthetic corpus distribution."""
    if not os.path.exists(TEST_PATH):
        return {s: 1.0 / len(COGNITIVE_STATES) for s in COGNITIVE_STATES}
    counts = Counter()
    for rec in _load_jsonl(TEST_PATH):
        st = rec.get("cognitive_state")
        if st in COGNITIVE_STATES:
            counts[st] += 1
    total = sum(counts.values()) or 1
    return {s: counts.get(s, 0) / total for s in COGNITIVE_STATES}


def frobenius(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _row_normalize(mat: np.ndarray) -> np.ndarray:
    out = mat.astype(float).copy()
    for i in range(out.shape[0]):
        s = out[i].sum()
        if s > 0:
            out[i] /= s
    return out


def aria_reference_transition_matrix(analyzer: CognitiveStateAnalyzer,
                                     max_sessions: int = 40) -> np.ndarray | None:
    """Build ARIA's state-transition matrix by running the classifier over the
    student turns of synthetic sessions. Only used by Experiment A."""
    path = next((p for p in SESSION_HISTORY_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        return None
    idx = {s: i for i, s in enumerate(COGNITIVE_STATES)}
    counts = np.zeros((len(COGNITIVE_STATES), len(COGNITIVE_STATES)))
    sessions = 0
    for rec in _load_jsonl(path):
        turns = rec.get("turns", [])
        student_texts = [t.get("content", "") for t in turns if t.get("role") == "user"]
        states = [analyzer.analyze(t)["state"] for t in student_texts if t.strip()]
        for a, b in zip(states, states[1:]):
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += 1
        sessions += 1
        if sessions >= max_sessions:
            break
    return _row_normalize(counts)


# ==================================================================
# EXPERIMENT A — ASSISTments behavioral-proxy agreement (PROXY)
# ==================================================================

def experiment_a(limit: int, use_llm: bool) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT A — Behavioral proxy agreement (ASSISTments)   [PROXY]")
    print("=" * 70)
    recs, spec, err = load_records("assistments2009", limit=limit)
    result = {"experiment": "A", "tier": "PROXY", "dataset": "assistments2009"}
    if recs is None:
        print(f"  SKIPPED: {err}")
        result.update({"status": "skipped", "reason": err})
        return result

    # 1) distribution check ------------------------------------------------
    proxy_counts = Counter(r["aria_state_proxy"] for r in recs
                           if r["aria_state_proxy"] is not None)
    expected = synthetic_state_distribution()
    dist = chi_square_gof(proxy_counts, expected)
    print(f"  proxy state counts: {dict(proxy_counts)}")
    print(f"  chi-square vs ARIA assumed prevalence: "
          f"chi2={dist.get('chi2')} dof={dist.get('dof')} p={dist.get('p_value')}")

    # 2) timing check (STUCK turns-to-recovery) ----------------------------
    timing = _assistments_timing(spec, limit)
    if timing.get("n_stuck_episodes"):
        print(f"  STUCK episodes: n={timing['n_stuck_episodes']} "
              f"mean turns-to-recovery={timing['mean_turns_to_recovery']} "
              f"vs ARIA optimal turn={timing['aria_optimal_turn']} "
              f"-> {timing['verdict']}")
    else:
        print(f"  timing check: {timing.get('note')}")

    # 3) sequence check (transition matrix Frobenius) ----------------------
    seq = _assistments_sequence(spec, limit, use_llm)
    if seq.get("frobenius_distance") is not None:
        print(f"  transition matrix Frobenius distance (real vs ARIA-synthetic): "
              f"{seq['frobenius_distance']}")
    else:
        print(f"  sequence check: {seq.get('note')}")

    result.update({"status": "ran", "distribution_check": dist,
                   "timing_check": timing, "sequence_check": seq,
                   "n_records": len(recs),
                   "commercial_use_allowed": spec.commercial_use_allowed})
    return result


def _assistments_sequences(spec, limit):
    """Rebuild per-user ordered proxy sequences from the raw ASSISTments CSV."""
    adapter = AssistmentsAdapter(spec)
    rows = list(adapter.load_raw())
    if limit:
        rows = rows[: limit * 4]  # a bit more than `limit` since many map to None
    pct = adapter._percentiles(rows)
    by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in rows:
        uid = r.get("user_id") or r.get("student_id")
        order = r.get("order_id")
        try:
            order_i = int(float(order))
        except (TypeError, ValueError):
            order_i = len(by_user[uid])
        from datasets.adapters.assistments import _to_bool, _to_int
        state, _conf = adapter.classify(
            _to_bool(r.get(adapter.COL_CORRECT)),
            _to_int(r.get(adapter.COL_ATTEMPT)),
            _to_int(r.get(adapter.COL_HINT)),
            _to_int(r.get(adapter.COL_RT)),
            pct.get(r.get(adapter.COL_PROBLEM)))
        if uid is not None and state is not None:
            by_user[uid].append((order_i, state))
    for uid in by_user:
        by_user[uid].sort(key=lambda t: t[0])
    return {uid: [s for _o, s in seq] for uid, seq in by_user.items()}


def _assistments_timing(spec, limit) -> dict:
    try:
        seqs = _assistments_sequences(spec, limit)
    except DatasetNotAvailableError as e:
        return {"note": str(e)}
    recoveries = []
    for states in seqs.values():
        i = 0
        while i < len(states):
            if states[i] == "STUCK":
                j = i + 1
                while j < len(states) and STATE_RANK.get(states[j], 0) <= STATE_RANK["STUCK"]:
                    j += 1
                if j < len(states):
                    recoveries.append(j - i)  # turns until a better state
                i = j
            else:
                i += 1
    if not recoveries:
        return {"note": "no STUCK episodes with a subsequent recovery found",
                "n_stuck_episodes": 0}
    mean_rec = float(np.mean(recoveries))
    optimal = DEFAULT_OPTIMAL.get("STUCK")
    verdict = ("match" if abs(mean_rec - optimal) <= 1.0 else "mismatch")
    return {"n_stuck_episodes": len(recoveries),
            "mean_turns_to_recovery": round(mean_rec, 2),
            "aria_optimal_turn": optimal, "verdict": verdict}


def _assistments_sequence(spec, limit, use_llm) -> dict:
    try:
        seqs = _assistments_sequences(spec, limit)
    except DatasetNotAvailableError as e:
        return {"note": str(e)}
    idx = {s: i for i, s in enumerate(COGNITIVE_STATES)}
    counts = np.zeros((len(COGNITIVE_STATES), len(COGNITIVE_STATES)))
    for states in seqs.values():
        for a, b in zip(states, states[1:]):
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += 1
    real_mat = _row_normalize(counts)
    analyzer = CognitiveStateAnalyzer(use_llm=use_llm)
    aria_mat = aria_reference_transition_matrix(analyzer)
    if aria_mat is None:
        return {"note": "ARIA reference sessions unavailable; matrix not compared",
                "real_transition_matrix": real_mat.round(3).tolist()}
    return {"frobenius_distance": round(frobenius(real_mat, aria_mat), 4),
            "states": COGNITIVE_STATES,
            "real_transition_matrix": real_mat.round(3).tolist(),
            "aria_transition_matrix": aria_mat.round(3).tolist()}


# ==================================================================
# EXPERIMENT B — NCTE non-LLM text robustness (DISTRIBUTIONAL)
# ==================================================================

def experiment_b(limit: int, use_llm: bool) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT B — Non-LLM text robustness (NCTE)   [DISTRIBUTIONAL]")
    print("=" * 70)
    result = {"experiment": "B", "tier": "DISTRIBUTIONAL", "dataset": "ncte"}
    recs, spec, err = load_records("ncte", limit=(limit or 0))
    if recs is None:
        print(f"  SKIPPED: {err}")
        result.update({"status": "skipped", "reason": err})
        return result

    analyzer = CognitiveStateAnalyzer(use_llm=use_llm)
    preds, confs, triples = [], [], []
    for i, r in enumerate(recs):
        text = r.get("text") or ""
        if not text.strip():
            continue
        res = analyzer.analyze(text)
        preds.append(res["state"])
        confs.append(float(res.get("confidence", 0.0)))
        triples.append((text, res["state"], res.get("evidence", "")))
        if (i + 1) % 100 == 0:
            print(f"    ...classified {i + 1}")

    if not preds:
        print("  SKIPPED: no usable NCTE utterances")
        result.update({"status": "skipped", "reason": "no utterances"})
        return result

    pred_dist = Counter(preds)
    n = len(preds)
    top_state, top_n = pred_dist.most_common(1)[0]
    degenerate = (top_n / n) >= 0.8 or len(pred_dist) <= 2
    mean_conf_real = round(float(np.mean(confs)), 4)
    mean_conf_syn = _synthetic_mean_confidence(analyzer, limit=min(100, limit or 100))

    # dump review sample
    _dump_ncte_review(triples)

    print(f"  n utterances: {n}")
    print(f"  prediction distribution: {dict(pred_dist)}")
    print(f"  degenerate collapse? {'YES' if degenerate else 'no'} "
          f"(top={top_state} {top_n}/{n})")
    print(f"  mean confidence NCTE={mean_conf_real}  vs synthetic={mean_conf_syn}  "
          f"(drop={round((mean_conf_syn or 0) - mean_conf_real, 4)})")
    print(f"  review sample -> {NCTE_REVIEW_PATH}")

    result.update({
        "status": "ran", "n_utterances": n,
        "prediction_distribution": dict(pred_dist),
        "degenerate": degenerate, "top_state": top_state,
        "mean_confidence_real": mean_conf_real,
        "mean_confidence_synthetic": mean_conf_syn,
        "confidence_drop": (round((mean_conf_syn or 0) - mean_conf_real, 4)
                            if mean_conf_syn is not None else None),
        "review_sample_path": NCTE_REVIEW_PATH,
        "commercial_use_allowed": spec.commercial_use_allowed,
    })
    return result


def _synthetic_mean_confidence(analyzer: CognitiveStateAnalyzer, limit: int) -> float | None:
    if not os.path.exists(TEST_PATH):
        return None
    recs = _load_jsonl(TEST_PATH)[:limit]
    confs = [float(analyzer.analyze(_join_think_aloud(r.get("think_aloud", "")))
                   .get("confidence", 0.0)) for r in recs]
    return round(float(np.mean(confs)), 4) if confs else None


def _dump_ncte_review(triples: list, n: int = 30) -> None:
    os.makedirs(os.path.dirname(NCTE_REVIEW_PATH), exist_ok=True)
    # deterministic sample (every k-th) so the file is stable across runs
    if len(triples) > n:
        step = max(1, len(triples) // n)
        sample = triples[::step][:n]
    else:
        sample = triples
    with open(NCTE_REVIEW_PATH, "w", encoding="utf-8") as fh:
        fh.write("NCTE classifier review sample — (utterance | predicted_state | evidence)\n")
        fh.write("No ARIA ground-truth labels exist for these; eyeball for plausibility.\n")
        fh.write("=" * 78 + "\n\n")
        for i, (text, state, evidence) in enumerate(sample, 1):
            fh.write(f"[{i:02d}] STATE={state}\n")
            fh.write(f"     utterance: {text}\n")
            fh.write(f"     evidence : {evidence}\n\n")


# ==================================================================
# EXPERIMENT C — EduAgent705 cross-corpus synthetic (DISTRIBUTIONAL)
# ==================================================================

def experiment_c(limit: int) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT C — Cross-corpus synthetic (EduAgent705 vs ARIA)   [DISTRIBUTIONAL]")
    print("=" * 70)
    result = {"experiment": "C", "tier": "DISTRIBUTIONAL", "dataset": "eduagent705"}
    recs, spec, err = load_records("eduagent705", limit=(limit or 0))
    if recs is None:
        print(f"  SKIPPED: {err}")
        result.update({"status": "skipped", "reason": err})
        return result

    has_text = any(r.get("text") for r in recs)
    proxy_counts = Counter(r["aria_state_proxy"] for r in recs
                           if r["aria_state_proxy"] is not None)
    aria_dist = synthetic_state_distribution()

    if not has_text and not proxy_counts:
        note = ("EduAgent705 (student_demo_generated.csv) provides synthetic agent "
                "attributes but no think-aloud text and no cognitive-state labels, "
                "so a lexical / classifier-confidence cross-corpus comparison is not "
                "possible. The realized with-text cross-corpus comparison is the "
                "cross-generator experiment (Part 1: mistral/gemma2/phi3 vs llama3.1).")
        print(f"  LIMITED: {note}")
        result.update({"status": "limited", "reason": note,
                       "n_records": len(recs), "has_text": False,
                       "aria_state_distribution": aria_dist})
        return result

    # If a future release has text/states, compare distributions.
    dist = chi_square_gof(proxy_counts, aria_dist) if proxy_counts else None
    print(f"  EduAgent705 proxy state counts: {dict(proxy_counts)}")
    result.update({"status": "ran", "n_records": len(recs), "has_text": has_text,
                   "eduagent705_state_counts": dict(proxy_counts),
                   "aria_state_distribution": aria_dist,
                   "distribution_check": dist})
    return result


# ==================================================================
# EXPERIMENT D — EduAgent310 real cognitive labels (DIRECT)
# ==================================================================

def experiment_d(limit: int) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT D — Real cognitive labels (EduAgent310)   [DIRECT]")
    print("=" * 70)
    result = {"experiment": "D", "tier": "DIRECT", "dataset": "eduagent310"}
    recs, spec, err = load_records("eduagent310", limit=(limit or 0))
    if recs is None:
        print(f"  SKIPPED: {err}")
        result.update({"status": "skipped", "reason": err})
        return result

    # Ground truth = crosswalked real gaze-derived label (aria_state_proxy).
    # Prediction = INDEPENDENT behavioral proxy from answer correctness:
    #   correct -> FLOW, incorrect -> CONFUSED. (No text, so ARIA's text
    #   classifier cannot be applied here — this validates the core behavioral
    #   assumption "incorrectness co-occurs with measured confusion".)
    y_true, y_pred = [], []
    n_labeled = 0
    for r in recs:
        gt = r["aria_state_proxy"]
        if gt is None:
            continue
        correct = r["behavioral_features"]["correct"]
        if correct is None:
            continue
        pred = "FLOW" if correct else "CONFUSED"
        y_true.append(gt)
        y_pred.append(pred)
        n_labeled += 1

    if n_labeled == 0:
        note = ("EduAgent310 present but no records carried both a gaze-derived "
                "cognitive label and an accuracy value.")
        print(f"  SKIPPED: {note}")
        result.update({"status": "skipped", "reason": note})
        return result

    metrics = compute_classification_metrics(y_true, y_pred, COGNITIVE_STATES)
    macro_present = [s for s in COGNITIVE_STATES
                     if metrics["per_state"][s]["support"] > 0]
    macro_f1 = (sum(metrics["per_state"][s]["f1"] for s in macro_present)
                / len(macro_present)) if macro_present else 0.0
    print(f"  labeled records: {n_labeled}  (ground truth = real gaze measures)")
    print(f"  label distribution: {dict(Counter(y_true))}")
    print(f"  behavioral-proxy accuracy vs real labels: {metrics['accuracy']:.3f}")
    print(f"  macro-F1 (present classes {macro_present}): {round(macro_f1, 3)}")
    print("  CAVEAT: prediction is a behavioral proxy (no think-aloud text in "
          "EduAgent310); this validates the incorrectness->confusion assumption, "
          "not ARIA's text classifier directly.")
    result.update({
        "status": "ran", "n_labeled": n_labeled,
        "label_distribution": dict(Counter(y_true)),
        "accuracy": metrics["accuracy"],
        "macro_f1": round(macro_f1, 3),
        "present_classes": macro_present,
        "per_state": {s: metrics["per_state"][s] for s in macro_present},
        "confusion_matrix": metrics["confusion_matrix"],
        "caveat": ("prediction is behavioral proxy from correctness; EduAgent310 "
                   "has no text so ARIA's text classifier is not applied. Ground "
                   "truth is real gaze-derived confusion/inattention."),
        "commercial_use_allowed": spec.commercial_use_allowed,
    })
    return result


# ==================================================================
# driver
# ==================================================================

def print_summary(results: list[dict], output_path: str) -> None:
    print("\n" + "=" * 70)
    print("EXTERNAL VALIDATION — SUMMARY")
    print("=" * 70)
    print(f"{'exp':4s}{'dataset':16s}{'tier':16s}{'status':10s} headline")
    print("-" * 70)
    for r in results:
        head = ""
        if r["status"] == "ran":
            if r["experiment"] == "A":
                head = f"chi2 p={r['distribution_check'].get('p_value')}"
            elif r["experiment"] == "B":
                head = (f"degenerate={r['degenerate']} conf drop="
                        f"{r.get('confidence_drop')}")
            elif r["experiment"] == "D":
                head = f"acc={r['accuracy']:.3f} vs real labels (n={r['n_labeled']})"
            elif r["experiment"] == "C":
                head = f"n={r.get('n_records')}"
        else:
            head = r.get("reason", "")[:40]
        print(f"{r['experiment']:4s}{r['dataset']:16s}{r['tier']:16s}"
              f"{r['status']:10s} {head}")
    # non-commercial banner if any non-commercial dataset was actually used
    used_nc = [r for r in results if r.get("status") in ("ran", "limited")
               and r.get("commercial_use_allowed") is False]
    if used_nc:
        names = ", ".join(r["dataset"] for r in used_nc)
        print("\n" + "=" * 64)
        print(" NON-COMMERCIAL DATA USED")
        print(f" Results above include records from non-commercial sources:")
        print(f"   {names}")
        print(" Research use only. Respect each source's licence.")
        print("=" * 64)
        # Eedi additionally carries the strict CC BY-NC-ND banner.
        if any(r["dataset"] == "eedi" for r in used_nc):
            print(NON_COMMERCIAL_BANNER)
    print(f"\nResults saved to {output_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap records per dataset (0 = all).")
    ap.add_argument("--no-llm", action="store_true",
                    help="Classify with heuristics only (Experiments A/B).")
    ap.add_argument("--only", type=str, default="",
                    help="Comma-separated subset of experiments to run (A,B,C,D).")
    ap.add_argument("--output", default=RESULTS_PATH)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    use_llm = not args.no_llm
    want = {x.strip().upper() for x in args.only.split(",") if x.strip()} or {"A", "B", "C", "D"}
    results = []
    if "A" in want:
        results.append(experiment_a(args.limit, use_llm))
    if "B" in want:
        results.append(experiment_b(args.limit, use_llm))
    if "C" in want:
        results.append(experiment_c(args.limit))
    if "D" in want:
        results.append(experiment_d(args.limit))

    payload = {
        "meta": {"classifier_use_llm": use_llm, "limit": args.limit,
                 "tiers": {"DIRECT": "real human/measured labels",
                           "PROXY": "behavior-derived labels",
                           "DISTRIBUTIONAL": "no labels; shape comparison"}},
        "experiments": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print_summary(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
