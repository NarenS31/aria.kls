#!/usr/bin/env python3.11
"""
Does enabling the analyzer's LLM fallback improve cross-generator transfer?

WHAT THIS RUNS
--------------
Four configurations over the SAME frozen 1,356-example matched manifest produced
by `matched_generator_eval.py`. The manifest is read from disk and every example
is resolved back to its source line; nothing is re-matched, re-sampled or dropped.

  A  lexical_only    The canonical route: `use_llm=False`. Reproduced here purely
                     as a control — it must return the published numbers exactly.
  B  hybrid_shipped  `use_llm=True` with the few-shot block EXACTLY as shipped.
                     *** CONTAMINATED — see the leakage note below. ***
  B' hybrid_decontam `use_llm=True` with the shipped few-shot block filtered down
                     to the examples whose source problem is in the TRAIN split.
  C  forced_llm      ABLATION, NOT PRODUCT BEHAVIOUR. Every example is sent to the
                     LLM regardless of heuristic confidence, decontaminated block.

THE LEAKAGE NOTE
----------------
`data/eval/mixed_fewshot_examples.json` supplies 56 labelled think-alouds that are
pasted verbatim into every LLM prompt. Resolving each one against the corpus:

    25 come from TRAIN problems
    14 come from VALIDATION problems
    17 come from TEST problems (P14, P39)

So switching the fallback on injects 17 labelled examples drawn from held-out test
problems into the prompt for every classification. Configuration B measures the
system as it actually ships and is reported for that reason, but it CANNOT support
a claim about transfer to unseen problems: 30% of its prompt is test data. B' is
the arm that answers the scientific question. The difference B - B' is a direct
measurement of what that contamination is worth.

B' only ever REMOVES examples from the shipped block. Nothing is added, rewritten
or re-selected, so this is decontamination, not the in-context "retraining" the
analyzer's own comments describe. The cost is a smaller block (25 vs 56 examples),
which is a confound in the B-vs-B' comparison and is reported as one.

The classifier source is never modified. The few-shot block is swapped by rebinding
the module attribute the prompt builder already reads at call time.

    python3.11 eval/llm_fallback_eval.py --configs A,B,Bprime,C
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
import time
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

import metacognition.analyzer as analyzer_mod  # noqa: E402
from metacognition.analyzer import (  # noqa: E402
    CognitiveStateAnalyzer,
    COGNITIVE_STATES,
    compute_classification_metrics,
)
from build_problem_disjoint_split import (  # noqa: E402
    DEFAULT_SEED,
    GENERATORS,
    build,
    problem_hash,
)
from matched_generator_eval import MATCH_SEED, match_cells  # noqa: E402
from problem_disjoint_eval import (  # noqa: E402
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    _accuracy,
    _macro_f1,
    _sha256_file,
    cluster_bootstrap,
)

CANONICAL_DIR = os.path.join(EVAL_ROOT, "data", "eval", "problem_disjoint_20260826")
FEWSHOT_PATH = analyzer_mod.FEWSHOT_PATH
LLAMA = "llama3.1:8b"

CONFIG_ORDER = ["A_lexical_only", "B_hybrid_shipped", "Bprime_hybrid_decontaminated",
                "C_forced_llm_ablation"]
CONFIG_ALIAS = {"A": "A_lexical_only", "B": "B_hybrid_shipped",
                "Bprime": "Bprime_hybrid_decontaminated", "C": "C_forced_llm_ablation"}


# ------------------------------------------------------------------
# Frozen manifest -> examples
# ------------------------------------------------------------------

def load_manifest_examples() -> tuple[list[dict], dict]:
    """Rehydrate the canonical matched manifest without re-matching anything.

    Each manifest row names a source file and line; the utterance is read back
    from exactly that coordinate. A freshly computed match is then compared
    against the manifest and any disagreement is fatal — the experiment is only
    meaningful on the identical example set.
    """
    path = os.path.join(CANONICAL_DIR, "matched_manifest.csv")
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    records, split_manifest = build(DEFAULT_SEED)
    by_coord = {(r["source_file"], str(r["source_line"])): r for r in records}

    examples = []
    for row in rows:
        key = (row["source_file"], row["source_line"])
        if key not in by_coord:
            raise AssertionError(f"manifest row not found in corpus: {key}")
        rec = by_coord[key]
        if rec["cognitive_state"] != row["cognitive_state"]:
            raise AssertionError(f"label drift at {key}: manifest says "
                                 f"{row['cognitive_state']}, corpus says "
                                 f"{rec['cognitive_state']}")
        if rec["problem_id"] != row["problem_id"] or rec["generator"] != row["generator"]:
            raise AssertionError(f"identity drift at {key}")
        examples.append({**rec, "matched_cell": row["matched_cell"]})

    fresh, _ = match_cells([r for r in records if r["split"] == "test"], MATCH_SEED)
    fresh_keys = sorted((r["source_file"], r["source_line"]) for r in fresh)
    manifest_keys = sorted((r["source_file"], int(r["source_line"])) for r in rows)
    if fresh_keys != manifest_keys:
        raise AssertionError("a fresh match disagrees with the stored manifest; "
                             "the frozen example set has drifted")
    return examples, split_manifest


# ------------------------------------------------------------------
# Few-shot provenance and decontamination
# ------------------------------------------------------------------

def fewshot_audit(records: list[dict], split_manifest: dict) -> dict:
    """Resolve every example in the shipped few-shot block to a split.

    Matching is on the exact normalized utterance. Anything the pooled corpus
    cannot resolve is retried against the resampled `mixed_train.jsonl`, which
    is excluded from the pool only to avoid double-counting.
    """
    def norm(s):
        return " ".join(str(s).split())

    hash_to_id = {p["normalized_problem_sha256"]: p["problem_id"]
                  for p in split_manifest["problems"]}
    id_to_split = {p["problem_id"]: p["split"] for p in split_manifest["problems"]}
    by_text = {norm(r["text"]): r for r in records}

    spill = {}
    spill_path = os.path.join(EVAL_ROOT, "data", "synthetic_thinkaloud", "mixed_train.jsonl")
    if os.path.exists(spill_path):
        with open(spill_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                ta = r["think_aloud"]
                text = (" ".join(str(ta[k]) for k in
                                 ("pre_attempt", "during_attempt", "post_attempt") if k in ta)
                        if isinstance(ta, dict) else str(ta))
                spill[norm(text)] = r["problem"]["text"]

    with open(FEWSHOT_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    selected, audit = [], []
    for state in COGNITIVE_STATES:
        per_gen: dict[str, int] = {}
        for item in data.get("examples", {}).get(state, []):
            gen = str(item.get("generator_model", "?"))
            if per_gen.get(gen, 0) >= analyzer_mod.FEWSHOT_PER_GENERATOR:
                continue
            per_gen[gen] = per_gen.get(gen, 0) + 1
            text = norm(item.get("think_aloud", ""))
            if not text:
                continue
            if text in by_text:
                pid = by_text[text]["problem_id"]
            elif text in spill:
                pid = hash_to_id.get(problem_hash(spill[text]))
            else:
                pid = None
            split = id_to_split.get(pid) if pid else None
            entry = {"state": state, "generator": gen, "problem_id": pid,
                     "split": split or "UNRESOLVED", "source_id": item.get("source_id", ""),
                     "text": text}
            audit.append(entry)
            selected.append((item, entry))

    by_split = Counter(e["split"] for e in audit)
    leaking = sorted({e["problem_id"] for e in audit
                      if e["split"] in ("test", "val") and e["problem_id"]})
    return {
        "fewshot_file": os.path.relpath(FEWSHOT_PATH, REPO_ROOT),
        "fewshot_file_sha256": _sha256_file(FEWSHOT_PATH),
        "n_examples_in_prompt": len(audit),
        "by_split": dict(by_split),
        "test_problems_in_prompt": sorted({e["problem_id"] for e in audit
                                           if e["split"] == "test"}),
        "val_problems_in_prompt": sorted({e["problem_id"] for e in audit
                                          if e["split"] == "val"}),
        "held_out_problems_in_prompt": leaking,
        "leakage": by_split.get("test", 0) > 0,
        "entries": audit,
        "_selected": selected,
    }


def build_block(entries, keep_splits) -> str:
    """Re-render a few-shot block from a subset, using the shipped formatting."""
    lines = []
    for item, entry in entries:
        if entry["split"] not in keep_splits:
            continue
        think = " ".join(str(item.get("think_aloud", "")).split())
        if len(think) > analyzer_mod._FEWSHOT_MAX_CHARS:
            think = think[:analyzer_mod._FEWSHOT_MAX_CHARS].rstrip() + "…"
        if think:
            lines.append(f'[{entry["state"]}] "{think}"')
    if not lines:
        return ""
    return ("Here are labelled think-aloud examples spanning multiple LLM "
            "generators and writing styles:\n" + "\n".join(lines) + "\n\n")


# ------------------------------------------------------------------
# LLM call cache
# ------------------------------------------------------------------

class CallCache:
    """Disk-backed cache keyed by (model, few-shot block, utterance).

    Makes a re-run free and makes the raw model output auditable: every call is
    appended verbatim, including failures.
    """

    def __init__(self, path: str):
        self.path = path
        self.mem: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self.mem[rec["key"]] = rec
        self.fh = open(path, "a", encoding="utf-8")
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model: str, block: str, text: str) -> str:
        h = hashlib.sha256()
        h.update(model.encode()); h.update(b"\x00")
        h.update(block.encode()); h.update(b"\x00")
        h.update(text.encode())
        return h.hexdigest()

    def get(self, k):
        rec = self.mem.get(k)
        if rec is not None:
            self.hits += 1
        return rec

    def put(self, k, payload):
        rec = {"key": k, **payload}
        self.mem[k] = rec
        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fh.flush()
        self.misses += 1
        return rec

    def close(self):
        self.fh.close()


# ------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------

def run_config(name: str, examples: list[dict], block: str | None,
               force: bool, cache: CallCache, analyzer: CognitiveStateAnalyzer,
               verbose: bool = True) -> list[dict]:
    """Classify every example under one configuration.

    `block is None` means the lexical-only route (the LLM is never constructed).
    `force` sends every example to the LLM regardless of heuristic confidence.
    """
    lexical = CognitiveStateAnalyzer(use_llm=False)
    trust = CognitiveStateAnalyzer.HEURISTIC_TRUST
    rows: list[dict] = []
    t_start = time.time()
    n_calls = 0

    for i, ex in enumerate(examples):
        base = lexical.analyze(ex["text"])
        h_state, h_conf = base["heuristic_state"], base["heuristic_confidence"]
        triggered = block is not None and (force or h_conf < trust)

        llm_state = llm_conf = llm_evidence = None
        llm_error = None
        latency = None
        cached = False

        if triggered:
            prev = analyzer_mod._FEWSHOT_BLOCK
            analyzer_mod._FEWSHOT_BLOCK = block
            try:
                k = CallCache.key(analyzer.model, block, ex["text"])
                rec = cache.get(k)
                if rec is None:
                    t0 = time.time()
                    out = analyzer._llm_classify(ex["text"], h_state)
                    latency = time.time() - t0
                    rec = cache.put(k, {
                        "model": analyzer.model,
                        "text": ex["text"],
                        "result": out,
                        "latency_s": round(latency, 3),
                    })
                    n_calls += 1
                else:
                    cached = True
                    latency = rec.get("latency_s")
            finally:
                analyzer_mod._FEWSHOT_BLOCK = prev

            out = rec.get("result")
            if out is None:
                # Mirrors the analyzer: an unavailable/failed LLM leaves the
                # heuristic verdict standing.
                llm_error = "llm_returned_none"
            else:
                llm_state = out["state"]
                llm_conf = out["confidence"]
                llm_evidence = out["evidence"]

        if triggered and llm_state is not None:
            final_state, final_conf, method = llm_state, llm_conf, "llm"
        else:
            final_state, final_conf, method = h_state, h_conf, "heuristic"

        rows.append({
            "config": name,
            "problem_id": ex["problem_id"],
            "matched_cell": ex["matched_cell"],
            "generator": ex["generator"],
            "condition": ex["cognitive_state"],
            "true_state": ex["cognitive_state"],
            # Same value under the name the shared metric helpers expect.
            "cognitive_state": ex["cognitive_state"],
            "source_file": ex["source_file"],
            "source_line": ex["source_line"],
            "record_id": ex["record_id"],
            "heuristic_state": h_state,
            "heuristic_confidence": h_conf,
            "fallback_triggered": bool(triggered),
            "llm_state": llm_state,
            "llm_confidence": llm_conf,
            "llm_evidence": llm_evidence,
            "llm_error": llm_error,
            "cached": cached,
            "latency_s": latency,
            "predicted_state": final_state,
            "confidence": final_conf,
            "method": method,
            "correct": final_state == ex["cognitive_state"],
            "text": ex["text"],
        })

        if verbose and triggered and (i + 1) % 25 == 0:
            done = sum(1 for r in rows if r["fallback_triggered"])
            print(f"  [{name}] {i+1}/{len(examples)} examples, {done} fallback calls, "
                  f"{time.time()-t_start:.0f}s elapsed", flush=True)

    return rows, {"wall_seconds": round(time.time() - t_start, 1),
                  "live_llm_calls": n_calls}


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def score_rows(rows: list[dict]) -> dict:
    m = compute_classification_metrics([r["true_state"] for r in rows],
                                       [r["predicted_state"] for r in rows],
                                       COGNITIVE_STATES)
    return {
        "n_examples": len(rows),
        "n_unique_problems": len({r["problem_id"] for r in rows}),
        "accuracy": m["accuracy"],
        "macro_f1": round(_macro_f1(m["per_state"]), 4),
        "per_condition": m["per_state"],
        "confusion_matrix": m["confusion_matrix"],
        "labels": COGNITIVE_STATES,
    }


def paired_delta(base_rows: list[dict], arm_rows: list[dict],
                 reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired cluster bootstrap over the 10 held-out problems.

    Both arms classify the identical example set, so a replicate resamples
    problems once and scores both arms on exactly the same drawn utterances.
    """
    key = lambda r: (r["source_file"], r["source_line"])  # noqa: E731
    base = {key(r): r for r in base_rows}
    arm = {key(r): r for r in arm_rows}
    shared = sorted(set(base) & set(arm))
    if not shared:
        return {}
    by_problem: dict[str, list] = defaultdict(list)
    for k in shared:
        by_problem[base[k]["problem_id"]].append(k)
    problems = sorted(by_problem)

    rng = random.Random(seed)
    d_acc, d_f1 = [], []
    for _ in range(reps):
        draw = [rng.choice(problems) for _ in problems]
        keys = [k for pid in draw for k in by_problem[pid]]
        b = [base[k] for k in keys]
        a = [arm[k] for k in keys]
        mb = compute_classification_metrics([r["true_state"] for r in b],
                                            [r["predicted_state"] for r in b],
                                            COGNITIVE_STATES)
        ma = compute_classification_metrics([r["true_state"] for r in a],
                                            [r["predicted_state"] for r in a],
                                            COGNITIVE_STATES)
        d_acc.append(ma["accuracy"] - mb["accuracy"])
        d_f1.append(_macro_f1(ma["per_state"]) - _macro_f1(mb["per_state"]))

    def pct(vals, q):
        s = sorted(vals)
        return round(s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))], 4)

    lo_a, hi_a = pct(d_acc, 0.025), pct(d_acc, 0.975)
    lo_f, hi_f = pct(d_f1, 0.025), pct(d_f1, 0.975)
    return {
        "paired": True, "reps": reps, "seed": seed, "n_clusters": len(problems),
        "delta_accuracy": round(_accuracy(arm_rows) - _accuracy(base_rows), 4),
        "delta_accuracy_ci95": [lo_a, hi_a],
        "delta_accuracy_excludes_zero": not (lo_a <= 0.0 <= hi_a),
        "delta_macro_f1": round(
            _macro_f1(score_rows(arm_rows)["per_condition"])
            - _macro_f1(score_rows(base_rows)["per_condition"]), 4),
        "delta_macro_f1_ci95": [lo_f, hi_f],
        "delta_macro_f1_excludes_zero": not (lo_f <= 0.0 <= hi_f),
    }


def change_analysis(base_rows, arm_rows):
    """Every prediction the fallback moved, and whether the move helped."""
    key = lambda r: (r["source_file"], r["source_line"])  # noqa: E731
    base = {key(r): r for r in base_rows}
    changes = []
    for r in arm_rows:
        b = base.get(key(r))
        if b is None or b["predicted_state"] == r["predicted_state"]:
            continue
        was, now = b["correct"], r["correct"]
        kind = ("corrected" if (not was and now) else
                "broke" if (was and not now) else "wrong_to_wrong")
        changes.append({
            "problem_id": r["problem_id"], "generator": r["generator"],
            "condition": r["condition"], "true_state": r["true_state"],
            "lexical_state": b["predicted_state"],
            "lexical_confidence": b["confidence"],
            "llm_state": r["llm_state"], "llm_confidence": r["llm_confidence"],
            "final_state": r["predicted_state"], "effect": kind,
            "llm_evidence": r["llm_evidence"],
            "source_file": r["source_file"], "source_line": r["source_line"],
            "record_id": r["record_id"], "text": r["text"],
        })
    counts = Counter(c["effect"] for c in changes)
    return changes, {
        "n_changed": len(changes),
        "corrected": counts.get("corrected", 0),
        "broke": counts.get("broke", 0),
        "wrong_to_wrong": counts.get("wrong_to_wrong", 0),
        "net": counts.get("corrected", 0) - counts.get("broke", 0),
    }


def representative_changes(changes, k=4):
    """Pick examples by a fixed rule, never by hand.

    One corrected and one broken case from the generator with the most changes,
    then one of each from the next most affected generator. Within a bucket the
    case with the highest LLM confidence wins; ties break on problem_id then
    source_line, so the selection is reproducible.
    """
    if not changes:
        return []
    by_gen = Counter(c["generator"] for c in changes)
    picks, seen = [], set()
    for gen, _ in by_gen.most_common():
        for effect in ("corrected", "broke"):
            pool = [c for c in changes
                    if c["generator"] == gen and c["effect"] == effect
                    and (c["source_file"], c["source_line"]) not in seen]
            if not pool:
                continue
            top = sorted(pool, key=lambda c: (-(c["llm_confidence"] or 0.0),
                                              c["problem_id"], c["source_line"]))[0]
            picks.append({**top, "selection_rule":
                          f"highest-confidence '{effect}' change for {gen}, the "
                          f"generator with {'the most' if gen == by_gen.most_common(1)[0][0] else 'the next most'} "
                          f"changed predictions ({by_gen[gen]})"})
            seen.add((top["source_file"], top["source_line"]))
            if len(picks) >= k:
                return picks
    return picks[:k]


# ------------------------------------------------------------------
# Provenance
# ------------------------------------------------------------------

def _git(*a):
    try:
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def ollama_digest(model: str) -> dict:
    info = {"model_tag": model}
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                             check=True, timeout=30).stdout
        for line in out.splitlines()[1:]:
            parts = line.split()
            if parts and parts[0] == model:
                info["model_id"] = parts[1]
                info["size"] = parts[2] + " " + parts[3] if len(parts) > 3 else parts[2]
        ver = subprocess.run(["ollama", "--version"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
        info["ollama_version"] = ver
    except Exception as exc:  # pragma: no cover
        info["error"] = str(exc)
    return info


def freeze(model: str) -> dict:
    pkgs = {}
    for n in ("numpy", "scipy", "sklearn", "matplotlib", "ollama"):
        try:
            pkgs[n] = getattr(__import__(n), "__version__", "<no __version__>")
        except ImportError:
            pkgs[n] = "<not installed>"
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "classifier_sources_sha256": {
            p: _sha256_file(os.path.join(REPO_ROOT, p)) for p in
            ("eval/metacognition/analyzer.py", "eval/metacognition/behavioral.py",
             "eval/metacognition/transfer.py")},
        "classifier_unmodified": _git("diff", "--stat", "HEAD", "--",
                                      "eval/metacognition/analyzer.py") == "",
        "llm": {
            **ollama_digest(model),
            "decoding": {"temperature": 0.0, "num_predict": 150, "num_ctx": 4096,
                         "format": "json", "seed": "not set by the analyzer"},
            "determinism": "Greedy decoding (temperature 0.0). Verified empirically: "
                           "5 repeats of 3 utterances returned byte-identical state, "
                           "confidence and free-text evidence; a 12-example probe at "
                           "4-way concurrency matched the sequential result 12/12.",
            "system_prompt": "You classify a student's cognitive state from their "
                             "think-aloud. Respond with JSON only, no prose.",
            "parser": "strip ``` fences -> json.loads -> regex {.*} fallback; an "
                      "out-of-vocabulary state falls back to substring match, then "
                      "to the heuristic verdict",
            "failure_behaviour": "any exception or missing ollama returns None and "
                                 "the heuristic verdict stands",
            "trigger": f"use_llm AND heuristic_confidence < "
                       f"{CognitiveStateAnalyzer.HEURISTIC_TRUST}",
        },
        "seeds": {"split_seed": DEFAULT_SEED, "match_seed": MATCH_SEED,
                  "bootstrap_seed": BOOTSTRAP_SEED},
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "machine": platform.machine(), "packages": pkgs},
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="A,B,Bprime,C")
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    ap.add_argument("--out-dir", default=os.path.join(
        EVAL_ROOT, "data", "eval", "llm_fallback_20260827"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    wanted = [CONFIG_ALIAS.get(c.strip(), c.strip()) for c in args.configs.split(",")]

    analyzer = CognitiveStateAnalyzer(use_llm=True)
    frozen = freeze(analyzer.model)
    print(f"commit {frozen['git_commit'][:12]} | model {frozen['llm'].get('model_tag')} "
          f"({frozen['llm'].get('model_id','?')})")

    examples, split_manifest = load_manifest_examples()
    print(f"frozen manifest rehydrated: {len(examples)} examples, "
          f"{len({e['problem_id'] for e in examples})} problems, "
          f"{len({e['matched_cell'] for e in examples})} cells")

    records, _ = build(DEFAULT_SEED)
    audit = fewshot_audit(records, split_manifest)
    selected = audit.pop("_selected")
    shipped_block = build_block(selected, {"train", "val", "test", "UNRESOLVED"})
    decon_block = build_block(selected, {"train"})
    audit["shipped_block_chars"] = len(shipped_block)
    audit["decontaminated_block_chars"] = len(decon_block)
    audit["decontaminated_n_examples"] = sum(
        1 for e in audit["entries"] if e["split"] == "train")
    audit["shipped_block_sha256"] = hashlib.sha256(shipped_block.encode()).hexdigest()
    audit["decontaminated_block_sha256"] = hashlib.sha256(decon_block.encode()).hexdigest()
    if shipped_block != analyzer_mod._FEWSHOT_BLOCK:
        raise AssertionError("re-rendered shipped block differs from the imported one")
    print(f"few-shot audit: {audit['by_split']} | test problems in prompt: "
          f"{audit['test_problems_in_prompt']}")

    cache = CallCache(os.path.join(args.out_dir, "llm_calls.jsonl"))
    specs = {
        "A_lexical_only": (None, False),
        "B_hybrid_shipped": (shipped_block, False),
        "Bprime_hybrid_decontaminated": (decon_block, False),
        "C_forced_llm_ablation": (decon_block, True),
    }

    all_rows, timings = {}, {}
    for name in CONFIG_ORDER:
        if name not in wanted:
            continue
        block, force = specs[name]
        print(f"\n== {name} ==", flush=True)
        rows, t = run_config(name, examples, block, force, cache, analyzer)
        all_rows[name] = rows
        timings[name] = t
        n_trig = sum(1 for r in rows if r["fallback_triggered"])
        print(f"  done: acc={_accuracy(rows):.4f} triggered={n_trig} "
              f"live_calls={t['live_llm_calls']} wall={t['wall_seconds']}s", flush=True)
    cache.close()

    # ---------------- metrics ----------------
    results = {
        "schema": "aria.llm_fallback_eval/1",
        "purpose": "Does enabling the analyzer's LLM fallback improve cross-generator "
                   "transfer on the frozen matched manifest?",
        "frozen": frozen,
        "fewshot_audit": audit,
        "manifest": {
            "source": os.path.relpath(
                os.path.join(CANONICAL_DIR, "matched_manifest.csv"), REPO_ROOT),
            "sha256": _sha256_file(os.path.join(CANONICAL_DIR, "matched_manifest.csv")),
            "n_examples": len(examples),
            "n_problems": len({e["problem_id"] for e in examples}),
            "n_cells": len({e["matched_cell"] for e in examples}),
            "per_generator": dict(Counter(e["generator"] for e in examples)),
        },
        "configs": {},
        "vs_lexical": {},
        "changes": {},
        "timings": timings,
    }

    for name, rows in all_rows.items():
        block, force = specs[name]
        overall = score_rows(rows)
        trig = [r for r in rows if r["fallback_triggered"]]
        results["configs"][name] = {
            "config": name,
            "route": ("lexical-only, LLM never constructed" if block is None else
                      ("FORCED: every example sent to the LLM (ablation, not product "
                       "behaviour)" if force else
                       "natural hybrid: LLM consulted iff heuristic_confidence < "
                       f"{CognitiveStateAnalyzer.HEURISTIC_TRUST}")),
            "fewshot_block": ("none" if block is None else
                              ("shipped (CONTAMINATED: contains test-problem examples)"
                               if block == shipped_block else
                               "decontaminated (train-split examples only)")),
            "fewshot_block_sha256": (None if block is None else
                                     hashlib.sha256(block.encode()).hexdigest()),
            **overall,
            "bootstrap": cluster_bootstrap(rows, reps=args.bootstrap_reps),
            "by_generator": {
                g: {**score_rows([r for r in rows if r["generator"] == g]),
                    "fallback_triggered": sum(1 for r in trig if r["generator"] == g)}
                for g in GENERATORS},
            "by_condition_activation": {
                c: {"n": sum(1 for r in rows if r["condition"] == c),
                    "triggered": sum(1 for r in trig if r["condition"] == c)}
                for c in COGNITIVE_STATES},
            "fallback_activation": {
                "n_triggered": len(trig),
                "pct": round(100.0 * len(trig) / len(rows), 2) if rows else 0.0,
                "n_llm_returned_state": sum(1 for r in trig if r["llm_state"]),
                "n_llm_errors": sum(1 for r in trig if r["llm_error"]),
                "by_generator": {g: round(100.0 * sum(
                    1 for r in trig if r["generator"] == g) / max(1, sum(
                        1 for r in rows if r["generator"] == g)), 2) for g in GENERATORS},
            },
        }

    base = all_rows.get("A_lexical_only")
    if base:
        for name, rows in all_rows.items():
            if name == "A_lexical_only":
                continue
            results["vs_lexical"][name] = paired_delta(base, rows, args.bootstrap_reps)
            ch, summary = change_analysis(base, rows)
            results["changes"][name] = {
                **summary, "representative": representative_changes(ch)}
            with open(os.path.join(args.out_dir, f"changed_predictions_{name}.json"),
                      "w", encoding="utf-8") as fh:
                json.dump({"config": name, "summary": summary,
                           "selection_rule": representative_changes.__doc__.strip(),
                           "changes": ch}, fh, indent=2, ensure_ascii=False)
                fh.write("\n")

    with open(os.path.join(args.out_dir, "fallback_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    with open(os.path.join(args.out_dir, "per_example_predictions.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "problem_id", "matched_cell", "generator", "condition",
                    "true_state", "heuristic_state", "heuristic_confidence",
                    "fallback_triggered", "llm_state", "llm_confidence", "llm_error",
                    "predicted_state", "method", "correct", "source_file", "source_line"])
        for name in CONFIG_ORDER:
            for r in all_rows.get(name, []):
                w.writerow([r["config"], r["problem_id"], r["matched_cell"],
                            r["generator"], r["condition"], r["true_state"],
                            r["heuristic_state"], r["heuristic_confidence"],
                            r["fallback_triggered"], r["llm_state"], r["llm_confidence"],
                            r["llm_error"], r["predicted_state"], r["method"],
                            r["correct"], r["source_file"], r["source_line"]])

    print("\n" + "=" * 78)
    for name in CONFIG_ORDER:
        c = results["configs"].get(name)
        if not c:
            continue
        d = results["vs_lexical"].get(name, {})
        print(f"{name:30s} acc={c['accuracy']:.4f} f1={c['macro_f1']:.4f} "
              f"trig={c['fallback_activation']['n_triggered']:4d} "
              f"dAcc={d.get('delta_accuracy', 0.0):+.4f} "
              f"{d.get('delta_accuracy_ci95', '')}")
    print("=" * 78)
    print(f"artefacts in {os.path.relpath(args.out_dir, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
