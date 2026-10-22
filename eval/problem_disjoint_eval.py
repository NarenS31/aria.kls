#!/usr/bin/env python3.11
"""
Leakage-free evaluation of the offline seven-state CognitiveStateAnalyzer.

WHAT IS BEING EVALUATED
-----------------------
`eval/metacognition/analyzer.py :: CognitiveStateAnalyzer`, run offline and
heuristic-only (`use_llm=False`), classifying a think-aloud utterance into one
of PLANNING, FLOW, CONFUSED, RUSHING, FRUSTRATED, STUCK, INSIGHT.

WHAT IS *NOT* BEING EVALUATED (see the generated report for the full statement)
  * the deployed 13-code rule-based reasoning-move detector
    (`agent/reasoning_moves.py`) — no human-labelled 13-code corpus exists;
  * the live hand-written scaffold policy (`agent/nudge.py`,
    `agent/intervention_pipeline.py`);
  * the offline JITAI policy (`eval/metacognition/jitai.py`).
Nothing in this script touches the product's live behaviour.

CONDITIONS
  A  seen-problem / utterance-disjoint  — the shipped test.jsonl (n=350,
     llama3.1:8b). Its 50 problems all appear in train.jsonl too. This is the
     condition that produced the previously reported 0.8457 / 0.8373.
  B  problem-disjoint                   — llama3.1:8b utterances belonging to the
     10 held-out test problems. No test problem's text, paraphrase, persona
     rendering or generator variant occurs in train or val.
  C  generator-stratified               — the same 10 test problems, scored
     separately for llama3.1:8b, mistral:7b, gemma2:9b and phi3:medium.

  Supplementary, clearly labelled as such in the outputs:
    B-train / B-val   the same llama pool restricted to the 30 train / 10 val
                      problems. Holding the utterance pool constant, this is the
                      direct seen-problem vs unseen-problem contrast.
    FLOW hard negatives  the 160-item adversarial FLOW stress set, restricted to
                      the test problems. Excluded from every headline number.

UNCERTAINTY
Confidence intervals bootstrap *problems* as clusters, never individual
utterances: utterances from one problem share its wording, topic and answer and
are not independent draws. Generator-vs-llama differences are computed on the
same resampled problems (paired). The A-vs-B difference cannot be paired — the
two conditions draw on overlapping but different problem sets — and is reported
with independent-bootstrap bounds, flagged as approximate.

    python3.11 eval/problem_disjoint_eval.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from metacognition.analyzer import (  # noqa: E402
    CognitiveStateAnalyzer,
    COGNITIVE_STATES,
    compute_classification_metrics,
)
from build_problem_disjoint_split import (  # noqa: E402
    DATA_DIR,
    DEFAULT_SEED,
    GENERATORS,
    build,
    join_think_aloud,
    problem_hash,
    write_manifest,
)

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 4242

# Files whose content defines the frozen classifier. Hashed before any test
# result is computed, so the report can prove the classifier was not touched.
FROZEN_SOURCES = [
    "eval/metacognition/analyzer.py",
    "eval/metacognition/behavioral.py",
    "eval/metacognition/transfer.py",
]


# ------------------------------------------------------------------
# Provenance
# ------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"<unavailable: {exc}>"


def freeze_record() -> dict:
    """Snapshot the classifier + environment. Called before any scoring runs."""
    pkgs = {}
    for name in ("numpy", "scipy", "sklearn", "matplotlib", "ollama"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "<no __version__>")
        except ImportError:
            pkgs[name] = "<not installed>"

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "source_sha256": {p: _sha256_file(os.path.join(REPO_ROOT, p))
                          for p in FROZEN_SOURCES},
        "classifier": {
            "class": "eval.metacognition.analyzer.CognitiveStateAnalyzer",
            "use_llm": False,
            "route": "heuristic-only; the Ollama fallback is never consulted",
            "HEURISTIC_TRUST": CognitiveStateAnalyzer.HEURISTIC_TRUST,
            "ARIA_META_MODEL_env": os.environ.get("ARIA_META_MODEL", "<unset>"),
            "audio_features": None,
            "behavioral_features": None,
            "aria_previous_prompt": "",
            "labels": COGNITIVE_STATES,
        },
        "environment": {
            "python": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": pkgs,
        },
        "determinism": "The heuristic route is pure string matching with no RNG, "
                       "no network call and no model weights. Repeated runs on the "
                       "same input return byte-identical predictions.",
    }


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def classify_all(records: list[dict], analyzer: CognitiveStateAnalyzer) -> list[dict]:
    """Attach the analyzer's prediction to every record, in place-safe copies."""
    scored = []
    for r in records:
        res = analyzer.analyze(r["text"])
        scored.append({**r,
                       "predicted_state": res["state"],
                       "confidence": res["confidence"],
                       "method": res["method"],
                       "evidence": res.get("evidence", ""),
                       "state_scores": res.get("state_scores", {})})
    return scored


def _macro_f1(per_state: dict) -> float:
    """Macro-F1 over the states that actually have support in this sample."""
    f1s = [v["f1"] for v in per_state.values() if v["support"] > 0]
    return sum(f1s) / len(f1s) if f1s else 0.0


def _accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["cognitive_state"] == r["predicted_state"]) / len(rows)


def cluster_bootstrap(rows: list[dict], reps: int = BOOTSTRAP_REPS,
                      seed: int = BOOTSTRAP_SEED) -> dict:
    """Percentile CIs from resampling PROBLEMS with replacement.

    Utterances about one problem share its wording, topic and answer, so they are
    not independent. Resampling utterances would understate the interval; the
    cluster is the problem.
    """
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_problem[r["problem_id"]].append(r)
    problems = sorted(by_problem)
    if not problems:
        return {}

    rng = random.Random(seed)
    accs: list[float] = []
    f1s: list[float] = []
    incomplete = 0
    for _ in range(reps):
        draw = [rng.choice(problems) for _ in problems]
        sample = [r for pid in draw for r in by_problem[pid]]
        m = compute_classification_metrics(
            [r["cognitive_state"] for r in sample],
            [r["predicted_state"] for r in sample],
            COGNITIVE_STATES,
        )
        if any(v["support"] == 0 for v in m["per_state"].values()):
            incomplete += 1
        accs.append(m["accuracy"])
        f1s.append(_macro_f1(m["per_state"]))

    def pct(vals: list[float], q: float) -> float:
        s = sorted(vals)
        return round(s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))], 4)

    return {
        "method": "percentile bootstrap over problems (clusters), with replacement",
        "reps": reps,
        "seed": seed,
        "n_clusters": len(problems),
        "replicates_missing_a_class": incomplete,
        "accuracy_ci95": [pct(accs, 0.025), pct(accs, 0.975)],
        "macro_f1_ci95": [pct(f1s, 0.025), pct(f1s, 0.975)],
    }


def majority_baseline(eval_rows: list[dict], train_rows: list[dict]) -> dict:
    """Accuracy of always predicting the training split's most common state.

    The class is chosen on TRAIN only. Picking it on the evaluation set would be
    peeking, and would report the best-case constant predictor rather than the
    one a practitioner could actually have built.
    """
    if not train_rows or not eval_rows:
        return {}
    counts = Counter(r["cognitive_state"] for r in train_rows)
    # Ties broken by label order so the baseline is reproducible.
    top = max(sorted(counts), key=lambda s: counts[s])
    hits = sum(1 for r in eval_rows if r["cognitive_state"] == top)
    return {
        "majority_class": top,
        "chosen_on": "problem-disjoint train split (30 problems)",
        "accuracy": round(hits / len(eval_rows), 4),
        "macro_f1": round(_macro_f1(compute_classification_metrics(
            [r["cognitive_state"] for r in eval_rows],
            [top] * len(eval_rows), COGNITIVE_STATES)["per_state"]), 4),
    }


def score(name: str, description: str, rows: list[dict],
          train_rows: list[dict] | None = None,
          reps: int = BOOTSTRAP_REPS) -> dict:
    m = compute_classification_metrics(
        [r["cognitive_state"] for r in rows],
        [r["predicted_state"] for r in rows],
        COGNITIVE_STATES,
    )
    out = {
        "condition": name,
        "description": description,
        "n_examples": len(rows),
        "n_unique_problems": len({r["problem_id"] for r in rows}),
        "problem_ids": sorted({r["problem_id"] for r in rows}),
        "accuracy": m["accuracy"],
        "macro_f1": round(_macro_f1(m["per_state"]), 4),
        "per_state": m["per_state"],
        "confusion_matrix": m["confusion_matrix"],
        "labels": COGNITIVE_STATES,
        "routing": dict(Counter(r["method"] for r in rows)),
        "by_generator": dict(Counter(r["generator"] for r in rows)),
    }
    if train_rows is not None:
        out["majority_baseline"] = majority_baseline(rows, train_rows)
    out["bootstrap"] = cluster_bootstrap(rows, reps=reps)
    return out


def paired_difference(a_rows: list[dict], b_rows: list[dict],
                      reps: int = BOOTSTRAP_REPS,
                      seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired cluster bootstrap of (b - a) when both are scored on the SAME problems.

    Each replicate resamples the shared problem list once and scores both arms on
    that same resample, so the interval reflects the paired comparison rather
    than two independent ones.
    """
    a_by: dict[str, list[dict]] = defaultdict(list)
    b_by: dict[str, list[dict]] = defaultdict(list)
    for r in a_rows:
        a_by[r["problem_id"]].append(r)
    for r in b_rows:
        b_by[r["problem_id"]].append(r)
    problems = sorted(set(a_by) & set(b_by))
    if not problems:
        return {}

    rng = random.Random(seed)
    d_acc: list[float] = []
    d_f1: list[float] = []
    for _ in range(reps):
        draw = [rng.choice(problems) for _ in problems]
        sa = [r for pid in draw for r in a_by[pid]]
        sb = [r for pid in draw for r in b_by[pid]]
        ma = compute_classification_metrics([r["cognitive_state"] for r in sa],
                                            [r["predicted_state"] for r in sa],
                                            COGNITIVE_STATES)
        mb = compute_classification_metrics([r["cognitive_state"] for r in sb],
                                            [r["predicted_state"] for r in sb],
                                            COGNITIVE_STATES)
        d_acc.append(mb["accuracy"] - ma["accuracy"])
        d_f1.append(_macro_f1(mb["per_state"]) - _macro_f1(ma["per_state"]))

    def pct(vals: list[float], q: float) -> float:
        s = sorted(vals)
        return round(s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))], 4)

    return {
        "paired": True,
        "shared_problems": len(problems),
        "delta_accuracy": round(_accuracy(b_rows) - _accuracy(a_rows), 4),
        "delta_accuracy_ci95": [pct(d_acc, 0.025), pct(d_acc, 0.975)],
        "delta_macro_f1_ci95": [pct(d_f1, 0.025), pct(d_f1, 0.975)],
        "excludes_zero_accuracy": not (pct(d_acc, 0.025) <= 0.0 <= pct(d_acc, 0.975)),
    }


def unpaired_difference(a: dict, b: dict) -> dict:
    """Approximate (b - a) bounds when the two arms are not on the same problems.

    Condition A's 50 problems are a superset of condition B's 10, so the two
    samples overlap and are neither paired nor independent. Combining the two
    independent intervals is therefore an approximation; it is labelled as one
    and never used as a significance test.
    """
    return {
        "paired": False,
        "approximate": True,
        "note": "A and B are scored on overlapping problem sets (B's 10 test "
                "problems are among A's 50), so this is neither a paired nor an "
                "independent comparison. Bounds combine the two separate cluster "
                "bootstraps and are indicative only, not a significance test.",
        "delta_accuracy": round(b["accuracy"] - a["accuracy"], 4),
        "delta_macro_f1": round(b["macro_f1"] - a["macro_f1"], 4),
        "a_accuracy_ci95": a.get("bootstrap", {}).get("accuracy_ci95"),
        "b_accuracy_ci95": b.get("bootstrap", {}).get("accuracy_ci95"),
        "a_macro_f1_ci95": a.get("bootstrap", {}).get("macro_f1_ci95"),
        "b_macro_f1_ci95": b.get("bootstrap", {}).get("macro_f1_ci95"),
    }


# ------------------------------------------------------------------
# Failure analysis
# ------------------------------------------------------------------

def _case(row: dict, why: str, kind: str) -> dict:
    scores = row.get("state_scores") or {}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
    return {
        "kind": kind,
        "problem_id": row["problem_id"],
        "problem_text": row["problem_text"],
        "subject": row["subject"],
        "topic": row["topic"],
        "generator": row["generator"],
        "source_file": row["source_file"],
        "source_line": row["source_line"],
        "record_id": row["record_id"],
        "persona": row["persona"],
        "ground_truth_state": row["cognitive_state"],
        "predicted_state": row["predicted_state"],
        "confidence": row["confidence"],
        "confidence_definition": "analyzer heuristic score for the winning state: "
                                 "0.45 + 0.09*evidence + 0.12*(margin over runner-up), "
                                 "capped at 0.97. A marker-weight blend, not a "
                                 "calibrated probability.",
        "top_state_scores": [[s, v] for s, v in ranked],
        "matched_evidence": row["evidence"],
        "student_think_aloud": row["text"],
        "explanation": why,
    }


def _pick(pool: list[dict], used_problems: set[str]) -> dict | None:
    """Most-confident error in `pool`, preferring a problem not already reported.

    Confidence saturates at the 0.97 ceiling for many errors, so the tie-break
    would otherwise hand every case to whichever problem sorts first and the
    three failures would all describe the same problem. Preferring an unused
    problem keeps the selection deterministic while spreading the cases.
    """
    if not pool:
        return None
    key = lambda r: (-r["confidence"], r["problem_id"], r["source_line"])  # noqa: E731
    fresh = [r for r in pool if r["problem_id"] not in used_problems]
    return sorted(fresh or pool, key=key)[0]


def failure_cases(test_rows: list[dict], per_gen: dict[str, list[dict]]) -> list[dict]:
    """Pick failures by fixed, stated rules — never by hand.

    Each selector filters its candidate pool by an explicit predicate and takes
    the most-confident survivor, so the same cases come back on every run.
    """
    cases: list[dict] = []
    used: set[str] = set()
    errors = [r for r in test_rows if r["cognitive_state"] != r["predicted_state"]]

    def add(row: dict, why: str, kind: str) -> None:
        cases.append(_case(row, why, kind))
        used.add(row["problem_id"])

    # 1. FLOW false negative: true FLOW, predicted otherwise.
    flow_fn = [r for r in errors if r["cognitive_state"] == "FLOW"]
    top = _pick(flow_fn, used)
    if top is not None:
        pairs = Counter(r["predicted_state"] for r in flow_fn)
        worst, worst_n = pairs.most_common(1)[0]
        n_flow = sum(1 for r in test_rows if r["cognitive_state"] == "FLOW")
        add(top,
            f"FLOW false negative, predicted {top['predicted_state']} on markers "
            f"[{top['evidence']}]. FLOW is missed on {len(flow_fn)} of {n_flow} "
            f"true-FLOW test utterances ({len(flow_fn) / n_flow:.1%}), most often "
            f"as {worst} ({worst_n} of them) — by a wide margin the weakest class "
            f"in every condition. The cause is structural: FLOW has no lexical "
            f"marker of its own. Sustained productive work is narrated in ordinary "
            f"procedural language, whereas the states it loses to carry "
            f"high-weight distinctive markers (PLANNING's 'first I'll', INSIGHT's "
            f"'oh wait', CONFUSED's 'wait, what'). FLOW's evidence is therefore "
            f"diffuse and low-weight, and a single competing marker anywhere in a "
            f"long utterance is enough to take the argmax away from it. Note the "
            f"utterance here does contain a genuine self-check ('but wait, what "
            f"if'), so part of this gap is label ambiguity rather than pure "
            f"classifier error — a fluent student checking their work is not "
            f"cleanly separable from a confused one on surface lexis alone.",
            "flow_false_negative")

    # 2. phi3-specific: the (true -> predicted) confusion whose rate on
    #    phi3:medium most exceeds its rate on llama3.1:8b, over the same problems.
    phi3 = per_gen.get("phi3:medium", [])
    llama = per_gen.get("llama3.1:8b", [])
    if phi3 and llama:
        def rates(rows):
            n = len(rows)
            c = Counter((r["cognitive_state"], r["predicted_state"]) for r in rows
                        if r["cognitive_state"] != r["predicted_state"])
            return {k: v / n for k, v in c.items()}, c
        p_rate, p_cnt = rates(phi3)
        l_rate, _ = rates(llama)
        excess = {k: p_rate[k] - l_rate.get(k, 0.0) for k in p_rate}
        if excess:
            pair = max(sorted(excess), key=lambda k: excess[k])
            pool = [r for r in phi3 if (r["cognitive_state"], r["predicted_state"]) == pair]
            top = _pick(pool, used)
            if top is not None:
                add(top,
                    f"phi3-specific failure, predicted {pair[1]} on markers "
                    f"[{top['evidence']}]. The {pair[0]}->{pair[1]} confusion hits "
                    f"{p_cnt[pair]}/{len(phi3)} phi3 test utterances "
                    f"({p_rate[pair]:.1%}) against {l_rate.get(pair, 0.0):.1%} on "
                    f"llama3.1:8b over the same 10 problems — an excess of "
                    f"{excess[pair]:+.1%}, and phi3 overall is "
                    f"{abs(_accuracy(phi3) - _accuracy(llama)):.1%} below llama. "
                    f"phi3:medium narrates in long, formal, fully-punctuated prose "
                    f"that walks the entire solution path in one utterance. That "
                    f"style trips markers belonging to several states at once, so "
                    f"the winner is decided by which marker bank carries more "
                    f"weight rather than by the student's actual state. The marker "
                    f"banks were audited for generator-agnosticism on token "
                    f"identity (eval/mine_signals.py), which does not control for "
                    f"the utterance length and register that phi3 changes: more "
                    f"sentences means more chances to fire an off-state marker, "
                    f"and the additive scoring has no length normalisation.",
                    "phi3_generator_specific")

    # 3. Highest-confidence error not already reported.
    seen_rows = {(c["source_file"], c["source_line"]) for c in cases}
    remaining = [r for r in errors if (r["source_file"], r["source_line"]) not in seen_rows]
    top = _pick(remaining, used)
    if top is not None:
        scores = top.get("state_scores") or {}
        gt_score = scores.get(top["cognitive_state"], 0.0)
        pred_score = scores.get(top["predicted_state"], 0.0)
        n_at_ceiling = sum(1 for r in errors if r["confidence"] >= 0.97)
        add(top,
            f"Highest-confidence error on an otherwise unreported problem "
            f"(confidence {top['confidence']:.3f}, the 0.97 ceiling). The true "
            f"state {top['cognitive_state']} scored {gt_score:.2f} against "
            f"{top['predicted_state']}'s {pred_score:.2f} on markers "
            f"[{top['evidence']}], so this is not a near-tie the analyzer had no "
            f"basis to call — the evidence actively points the wrong way. "
            f"{n_at_ceiling} of the {len(errors)} problem-disjoint test errors "
            f"({n_at_ceiling / len(errors):.1%}) sit at that same ceiling. This is "
            f"the structurally important one: because the reported confidence "
            f"rises with the winning margin and contains no term for whether the "
            f"winning evidence is correct, a maximally-confident wrong answer is "
            f"indistinguishable from a maximally-confident right one to any "
            f"downstream consumer. It is also the concrete reason no ECE is "
            f"reported here — the score is a marker-weight margin, not a "
            f"probability that could be compared against observed correctness.",
            "highest_confidence_error")

    # 4. Extra: the most frequent confusion that is not a FLOW false negative, so
    #    the case set is not dominated by the one class already known to be weak.
    non_flow = [r for r in errors if r["cognitive_state"] != "FLOW"]
    if non_flow:
        pairs = Counter((r["cognitive_state"], r["predicted_state"]) for r in non_flow)
        pair, count = max(sorted(pairs.items()), key=lambda kv: kv[1])
        pool = [r for r in non_flow
                if (r["cognitive_state"], r["predicted_state"]) == pair
                and (r["source_file"], r["source_line"])
                not in {(c["source_file"], c["source_line"]) for c in cases}]
        top = _pick(pool, used)
        if top is not None:
            n_true = sum(1 for r in test_rows if r["cognitive_state"] == pair[0])
            add(top,
                f"Most frequent non-FLOW confusion: {pair[0]}->{pair[1]}, "
                f"{count} occurrences, i.e. {count / n_true:.1%} of all true "
                f"{pair[0]} test utterances. Predicted on markers "
                f"[{top['evidence']}]. Included so the failure set is not "
                f"dominated by the single class already known to be weakest. "
                f"{pair[0]} and {pair[1]} are adjacent in the ARIA state model and "
                f"share much of their vocabulary of difficulty, but they call for "
                f"different scaffolds, so this confusion is the one with the "
                f"clearest downstream cost.",
                "most_frequent_non_flow_confusion")
    return cases


# ------------------------------------------------------------------
# Output artefacts
# ------------------------------------------------------------------

def write_per_class_csv(results: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "state", "precision", "recall", "f1", "support"])
        for cond in results["conditions"].values():
            for state in COGNITIVE_STATES:
                m = cond["per_state"][state]
                w.writerow([cond["condition"], state, m["precision"], m["recall"],
                            m["f1"], m["support"]])


def write_confusion_csv(results: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "true_state"] + [f"pred_{s}" for s in COGNITIVE_STATES])
        for cond in results["conditions"].values():
            for i, state in enumerate(COGNITIVE_STATES):
                w.writerow([cond["condition"], state] + cond["confusion_matrix"][i])


def write_comparison_csv(results: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "description", "n_examples", "n_unique_problems",
                    "accuracy", "accuracy_ci95_low", "accuracy_ci95_high",
                    "macro_f1", "macro_f1_ci95_low", "macro_f1_ci95_high",
                    "majority_baseline_accuracy",
                    "delta_accuracy_vs_utterance_level",
                    "delta_macro_f1_vs_utterance_level"])
        base = results["conditions"]["A_utterance_level"]
        for key in ("A_utterance_level", "B_problem_disjoint",
                    "B_train_problems", "B_val_problems"):
            c = results["conditions"].get(key)
            if not c:
                continue
            bs = c.get("bootstrap", {})
            aci = bs.get("accuracy_ci95", [None, None])
            fci = bs.get("macro_f1_ci95", [None, None])
            w.writerow([c["condition"], c["description"], c["n_examples"],
                        c["n_unique_problems"], c["accuracy"], aci[0], aci[1],
                        c["macro_f1"], fci[0], fci[1],
                        c.get("majority_baseline", {}).get("accuracy"),
                        round(c["accuracy"] - base["accuracy"], 4),
                        round(c["macro_f1"] - base["macro_f1"], 4)])


def write_generator_csv(results: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["generator", "n_examples", "n_unique_problems", "accuracy",
                    "accuracy_ci95_low", "accuracy_ci95_high", "macro_f1",
                    "macro_f1_ci95_low", "macro_f1_ci95_high",
                    "majority_baseline_accuracy",
                    "delta_accuracy_vs_llama", "delta_acc_ci95_low",
                    "delta_acc_ci95_high", "delta_macro_f1_vs_llama",
                    "flow_f1", "flow_recall"])
        for gen in GENERATORS:
            c = results["conditions"].get(f"C_generator::{gen}")
            if not c:
                continue
            bs = c.get("bootstrap", {})
            aci = bs.get("accuracy_ci95", [None, None])
            fci = bs.get("macro_f1_ci95", [None, None])
            d = results["generator_vs_llama"].get(gen, {})
            dci = d.get("delta_accuracy_ci95", [None, None])
            w.writerow([gen, c["n_examples"], c["n_unique_problems"], c["accuracy"],
                        aci[0], aci[1], c["macro_f1"], fci[0], fci[1],
                        c.get("majority_baseline", {}).get("accuracy"),
                        d.get("delta_accuracy"), dci[0], dci[1],
                        d.get("delta_macro_f1"),
                        c["per_state"]["FLOW"]["f1"],
                        c["per_state"]["FLOW"]["recall"]])


def write_confusion_figure(results: dict, path: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:  # pragma: no cover - optional dependency
        return None

    panels = [("A_utterance_level", "A. Seen-problem / utterance-disjoint\n"
                                    "(test.jsonl, llama3.1:8b)"),
              ("B_problem_disjoint", "B. Problem-disjoint\n"
                                     "(10 held-out problems, llama3.1:8b)")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
    abbr = [s[:4].title() for s in COGNITIVE_STATES]

    for ax, (key, title) in zip(axes, panels):
        cond = results["conditions"][key]
        cm = np.array(cond["confusion_matrix"], dtype=float)
        rows = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
        im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(abbr)), abbr, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(abbr)), abbr, fontsize=9)
        ax.set_xlabel("predicted state", fontsize=10)
        ax.set_ylabel("true state", fontsize=10)
        ax.set_title(f"{title}\nn={cond['n_examples']}, "
                     f"{cond['n_unique_problems']} problems, "
                     f"acc={cond['accuracy']:.3f}, macro-F1={cond['macro_f1']:.3f}",
                     fontsize=10)
        for i in range(len(abbr)):
            for j in range(len(abbr)):
                if cm[i, j] == 0:
                    continue
                ax.text(j, i, f"{norm[i, j]:.2f}\n{int(cm[i, j])}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.55 else "#22303c")
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("row-normalised rate (recall on the diagonal)", fontsize=9)
    fig.suptitle("ARIA offline seven-state CognitiveStateAnalyzer — "
                 "effect of removing problem-level leakage", fontsize=12.5, y=0.99)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="problem-partition seed")
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    ap.add_argument("--out-dir", default=os.path.join(
        EVAL_ROOT, "data", "eval", "problem_disjoint_20260826"))
    args = ap.parse_args()
    reps = args.bootstrap_reps

    out_dir = args.out_dir
    if os.path.exists(out_dir) and os.listdir(out_dir):
        print(f"NOTE: {out_dir} already exists and is non-empty; files in it will "
              f"be replaced. No file outside this directory is written.")
    os.makedirs(out_dir, exist_ok=True)

    # ---- freeze BEFORE any test result is computed -----------------
    frozen = freeze_record()
    print(f"frozen at commit {frozen['git_commit'][:12]} "
          f"(analyzer.py sha256 "
          f"{frozen['source_sha256']['eval/metacognition/analyzer.py'][:12]})")

    # ---- split ------------------------------------------------------
    records, manifest = build(args.seed)
    write_manifest(manifest, out_dir)
    print(f"split: {manifest['partition_procedure']}, seed {manifest['seed']}, "
          f"{manifest['totals']['n_problems']} problems / "
          f"{manifest['totals']['n_examples']} utterances")

    analyzer = CognitiveStateAnalyzer(use_llm=False)
    scored = classify_all(records, analyzer)

    train_rows = [r for r in scored if r["split"] == "train"]
    val_rows = [r for r in scored if r["split"] == "val"]
    test_rows = [r for r in scored if r["split"] == "test"]

    llama_train = [r for r in train_rows if r["generator"] == "llama3.1:8b"]
    llama_val = [r for r in val_rows if r["generator"] == "llama3.1:8b"]
    llama_test = [r for r in test_rows if r["generator"] == "llama3.1:8b"]

    # ---- condition A: the shipped utterance-level test split --------
    hash_to_id = {p["normalized_problem_sha256"]: p["problem_id"]
                  for p in manifest["problems"]}
    a_records = []
    with open(os.path.join(DATA_DIR, "test.jsonl"), encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            h = problem_hash(rec["problem"]["text"])
            a_records.append({
                "generator": "llama3.1:8b",
                "source_file": "eval/data/synthetic_thinkaloud/test.jsonl",
                "source_line": line_no,
                "record_id": rec.get("id", ""),
                "problem_text": rec["problem"]["text"],
                "problem_hash": h,
                "problem_id": hash_to_id[h],
                "subject": rec["problem"].get("subject", ""),
                "topic": rec["problem"].get("topic", ""),
                "difficulty": rec["problem"].get("difficulty", ""),
                "persona": (rec.get("student_profile") or {}).get("adhd_type", ""),
                "cognitive_state": rec["cognitive_state"],
                "text": join_think_aloud(rec.get("think_aloud")),
            })
    a_scored = classify_all(a_records, analyzer)

    conditions: dict[str, dict] = {}
    conditions["A_utterance_level"] = score(
        "A_utterance_level",
        "Seen-problem / utterance-disjoint. The shipped test.jsonl: new utterances "
        "about problems that also appear in train.jsonl and val.jsonl. This is the "
        "condition behind the previously reported 0.8457 / 0.8373.",
        a_scored, train_rows=llama_train, reps=reps)

    conditions["B_problem_disjoint"] = score(
        "B_problem_disjoint",
        "Problem-disjoint. Every llama3.1:8b utterance whose problem is one of the "
        "10 held-out test problems. No test problem, paraphrase, persona rendering "
        "or generator variant appears in train or val.",
        llama_test, train_rows=llama_train, reps=reps)

    conditions["B_train_problems"] = score(
        "B_train_problems",
        "Supplementary. The same llama3.1:8b pool restricted to the 30 TRAIN "
        "problems. Holding the utterance pool constant, the gap to "
        "B_problem_disjoint isolates the seen-problem effect.",
        llama_train, train_rows=llama_train, reps=reps)

    conditions["B_val_problems"] = score(
        "B_val_problems",
        "Supplementary. The same llama3.1:8b pool restricted to the 10 VALIDATION "
        "problems. Reported for completeness; nothing was tuned on it.",
        llama_val, train_rows=llama_train, reps=reps)

    # ---- condition C: generator-stratified, same 10 test problems ----
    per_gen: dict[str, list[dict]] = {}
    for gen in GENERATORS:
        rows = [r for r in test_rows if r["generator"] == gen]
        per_gen[gen] = rows
        conditions[f"C_generator::{gen}"] = score(
            f"C_generator::{gen}",
            f"Generator-stratified. {gen} utterances on the SAME 10 problem-"
            f"disjoint test problems as condition B.",
            rows, train_rows=[r for r in train_rows if r["generator"] == gen], reps=reps)

    conditions["C_all_generators_pooled"] = score(
        "C_all_generators_pooled",
        "All four generators pooled on the 10 problem-disjoint test problems.",
        test_rows, train_rows=train_rows, reps=reps)

    # ---- supplementary: adversarial FLOW stress set -----------------
    fhn_path = os.path.join(DATA_DIR, "flow_hard_negatives.jsonl")
    fhn_rows = []
    if os.path.exists(fhn_path):
        test_ids = set(manifest["distributions"]["test"]["problem_ids"])
        with open(fhn_path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                h = problem_hash(rec["problem"]["text"])
                pid = hash_to_id.get(h)
                if pid not in test_ids:
                    continue
                fhn_rows.append({
                    "generator": rec.get("generator_model", "unknown"),
                    "source_file": "eval/data/synthetic_thinkaloud/flow_hard_negatives.jsonl",
                    "source_line": line_no,
                    "record_id": rec.get("id", ""),
                    "problem_text": rec["problem"]["text"],
                    "problem_hash": h, "problem_id": pid,
                    "subject": rec["problem"].get("subject", ""),
                    "topic": rec["problem"].get("topic", ""),
                    "difficulty": rec["problem"].get("difficulty", ""),
                    "persona": (rec.get("student_profile") or {}).get("adhd_type", ""),
                    "cognitive_state": rec["cognitive_state"],
                    "text": join_think_aloud(rec.get("think_aloud")),
                })
    if fhn_rows:
        conditions["S_flow_hard_negatives"] = score(
            "S_flow_hard_negatives",
            "SUPPLEMENTARY, NOT A HEADLINE NUMBER. The adversarial FLOW "
            "hard-negative set restricted to the 10 test problems. Adversarially "
            "constructed rather than sampled from the corpus, so it is excluded "
            "from every other condition and must not be pooled with them.",
            classify_all(fhn_rows, analyzer), train_rows=train_rows, reps=reps)

    # ---- differences -------------------------------------------------
    results: dict = {
        "schema": "aria.problem_disjoint_eval/1",
        "generated_for": "NeurIPS IAEval workshop paper — leakage-free evaluation "
                         "of the offline seven-state CognitiveStateAnalyzer",
        "frozen": frozen,
        "split": {
            "seed": manifest["seed"],
            "procedure": manifest["partition_procedure"],
            "sizes": manifest["split_sizes"],
            "test_problem_ids": manifest["distributions"]["test"]["problem_ids"],
            "val_problem_ids": manifest["distributions"]["val"]["problem_ids"],
            "train_problem_ids": manifest["distributions"]["train"]["problem_ids"],
            "leakage_check": manifest["leakage_check"],
            "distributions": manifest["distributions"],
        },
        "conditions": conditions,
    }

    results["utterance_level_vs_problem_disjoint"] = unpaired_difference(
        conditions["A_utterance_level"], conditions["B_problem_disjoint"])
    results["seen_vs_unseen_problems_same_pool"] = {
        "note": "Both arms are the same llama3.1:8b pool and the same classifier; "
                "only the problem group differs. Unpaired (disjoint problem sets) "
                "but non-overlapping, so the two intervals can be read side by side.",
        "train_problems_accuracy": conditions["B_train_problems"]["accuracy"],
        "train_problems_accuracy_ci95":
            conditions["B_train_problems"]["bootstrap"]["accuracy_ci95"],
        "test_problems_accuracy": conditions["B_problem_disjoint"]["accuracy"],
        "test_problems_accuracy_ci95":
            conditions["B_problem_disjoint"]["bootstrap"]["accuracy_ci95"],
        "delta_accuracy": round(conditions["B_problem_disjoint"]["accuracy"]
                                - conditions["B_train_problems"]["accuracy"], 4),
        "delta_macro_f1": round(conditions["B_problem_disjoint"]["macro_f1"]
                                - conditions["B_train_problems"]["macro_f1"], 4),
    }

    results["generator_vs_llama"] = {}
    for gen in GENERATORS:
        if gen == "llama3.1:8b":
            results["generator_vs_llama"][gen] = {
                "baseline": True, "delta_accuracy": 0.0, "delta_macro_f1": 0.0,
                "delta_accuracy_ci95": [0.0, 0.0],
            }
            continue
        d = paired_difference(per_gen["llama3.1:8b"], per_gen[gen],
                              reps=args.bootstrap_reps)
        d["delta_macro_f1"] = round(
            conditions[f"C_generator::{gen}"]["macro_f1"]
            - conditions["C_generator::llama3.1:8b"]["macro_f1"], 4)
        results["generator_vs_llama"][gen] = d

    # ---- calibration -------------------------------------------------
    conf_bins = []
    edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        rows = [r for r in test_rows if lo <= r["confidence"] < hi]
        if rows:
            conf_bins.append({
                "bin": f"[{lo:.2f}, {hi:.2f})", "n": len(rows),
                "accuracy": round(_accuracy(rows), 4),
                "mean_confidence": round(sum(r["confidence"] for r in rows) / len(rows), 4),
            })
    results["calibration"] = {
        "ece_reported": False,
        "reason": "ECE is deliberately omitted. The analyzer's `confidence` is "
                  "0.45 + 0.09*evidence + 0.12*margin, capped at 0.97 — a "
                  "hand-set affine function of keyword-marker weights. It is not "
                  "normalised over the seven states, cannot fall below 0.45 "
                  "whenever any marker fires, and was never fitted against "
                  "observed correctness. Binning it and calling the gap an "
                  "expected calibration error would report a number for a "
                  "quantity that is not a probability. The repository's "
                  "eval/metacognition/calibration.py measures a different thing "
                  "entirely — the STUDENT's metacognitive calibration "
                  "(confidence_before vs correct) — and is not a classifier "
                  "confidence calibration.",
        "descriptive_only_confidence_bins": conf_bins,
        "descriptive_bins_caveat": "Accuracy per confidence bin on the "
                                   "problem-disjoint test set, all generators. "
                                   "Descriptive only; NOT a reliability diagram "
                                   "and NOT an ECE.",
    }

    results["failure_cases"] = failure_cases(test_rows, per_gen)

    # ---- write artefacts ---------------------------------------------
    paths: dict[str, str] = {}
    p = os.path.join(out_dir, "metrics.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    paths["metrics_json"] = p

    for name, fn in (("per_class_metrics.csv", write_per_class_csv),
                     ("confusion_matrices.csv", write_confusion_csv),
                     ("comparison_utterance_vs_problem_disjoint.csv", write_comparison_csv),
                     ("generator_comparison.csv", write_generator_csv)):
        p = os.path.join(out_dir, name)
        fn(results, p)
        paths[name] = p

    p = os.path.join(out_dir, "failure_cases.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"selection_rules":
                   "Fixed rules, applied in order, no manual choice: (1) the "
                   "highest-confidence true-FLOW error; (2) the phi3:medium "
                   "true->predicted confusion whose rate most exceeds "
                   "llama3.1:8b's on the same 10 problems, then that pair's "
                   "highest-confidence instance; (3) the highest-confidence "
                   "remaining error; (4) the most frequent non-FLOW confusion "
                   "pair, highest-confidence instance. Each selector prefers a "
                   "problem not already reported, because confidence saturates at "
                   "the 0.97 ceiling and would otherwise send every case to the "
                   "same problem. Ties break on problem_id then source_line.",
                   "cases": results["failure_cases"]}, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    paths["failure_cases_json"] = p

    fig = write_confusion_figure(results, os.path.join(out_dir, "confusion_matrix.png"))
    if fig:
        paths["confusion_matrix_png"] = fig

    # ---- checksums ---------------------------------------------------
    checks = {}
    for fname in sorted(os.listdir(out_dir)):
        if fname == "CHECKSUMS.sha256":
            continue
        checks[fname] = _sha256_file(os.path.join(out_dir, fname))
    for src in FROZEN_SOURCES + [
            "eval/data/synthetic_thinkaloud/dataset.jsonl",
            "eval/data/synthetic_thinkaloud/test.jsonl",
            "eval/data/synthetic_thinkaloud/holdout_mistral.jsonl",
            "eval/data/synthetic_thinkaloud/holdout_gemma2.jsonl",
            "eval/data/synthetic_thinkaloud/holdout_phi3.jsonl",
            "eval/data/synthetic_thinkaloud/augment_mistral.jsonl",
            "eval/data/synthetic_thinkaloud/augment_gemma2.jsonl",
            "eval/data/synthetic_thinkaloud/augment_phi3.jsonl",
            "eval/build_problem_disjoint_split.py",
            "eval/problem_disjoint_eval.py"]:
        checks[src] = _sha256_file(os.path.join(REPO_ROOT, src))
    p = os.path.join(out_dir, "CHECKSUMS.sha256")
    with open(p, "w", encoding="utf-8") as fh:
        for k in sorted(checks):
            fh.write(f"{checks[k]}  {k}\n")
    paths["checksums"] = p

    # ---- console summary ---------------------------------------------
    print("\n" + "=" * 74)
    for key in ("A_utterance_level", "B_problem_disjoint", "B_train_problems",
                "B_val_problems"):
        c = conditions[key]
        bs = c["bootstrap"]
        print(f"{c['condition']:24s} n={c['n_examples']:5d} "
              f"problems={c['n_unique_problems']:3d} "
              f"acc={c['accuracy']:.4f} {bs['accuracy_ci95']} "
              f"macroF1={c['macro_f1']:.4f} {bs['macro_f1_ci95']}")
    print("-" * 74)
    for gen in GENERATORS:
        c = conditions[f"C_generator::{gen}"]
        d = results["generator_vs_llama"][gen]
        print(f"{gen:24s} n={c['n_examples']:5d} "
              f"problems={c['n_unique_problems']:3d} "
              f"acc={c['accuracy']:.4f} {c['bootstrap']['accuracy_ci95']} "
              f"macroF1={c['macro_f1']:.4f} "
              f"dAcc={d.get('delta_accuracy', 0.0):+.4f} "
              f"{d.get('delta_accuracy_ci95')}")
    print("=" * 74)
    print(f"\nartefacts in {out_dir}")
    for k, v in paths.items():
        print(f"  {k:42s} {os.path.relpath(v, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
