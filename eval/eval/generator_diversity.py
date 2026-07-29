#!/usr/bin/env python3.11
"""
Generator linguistic-diversity report (Part 6).

Quantifies how differently the four generators (llama3.1:8b, mistral:7b,
gemma2:9b, phi3:medium) render each cognitive state. Low overlap between
generators' characteristic vocabulary is the diversity that justifies training
on a mixed corpus instead of a single generator.

For each (cognitive_state x generator) cell it reports:
  - top-20 discriminative tokens (log-odds of the token in this state vs all
    other states, within that generator's samples)
  - mean sentence length (words / sentence)
  - filler-word rate (um, uh, wait, like, "okay so", ...)
  - question rate (questions / sentence)
  - negation rate (don't, can't, not, never, ...)

Then, per state, the Jaccard overlap of the top-20 token sets across every
generator pair (low overlap = diverse = good), plus the mean over pairs.

Reads the mixed corpus (data/synthetic_thinkaloud/dataset_mixed.jsonl) when
present, else falls back to concatenating dataset.jsonl + augment_*.jsonl +
flow_hard_negatives.jsonl. Writes data/eval/generator_diversity.json.

    python3.11 eval/generator_diversity.py
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO_ROOT, "data")
ST_DIR = os.path.join(DATA_DIR, "synthetic_thinkaloud")
MIXED_PATH = os.path.join(ST_DIR, "dataset_mixed.jsonl")
OUT_PATH = os.path.join(DATA_DIR, "eval", "generator_diversity.json")

COGNITIVE_STATES = ["PLANNING", "FLOW", "CONFUSED", "RUSHING",
                    "FRUSTRATED", "STUCK", "INSIGHT"]

_WORD_RE = re.compile(r"[a-z']+")
FILLER_UNIGRAMS = {"um", "umm", "uh", "uhh", "hmm", "hmmm", "wait", "like",
                   "well", "er", "erm", "okay", "ok"}
FILLER_BIGRAMS = {("okay", "so"), ("ok", "so"), ("i", "mean")}
NEGATIONS = {"not", "no", "never", "none", "cannot", "cant", "dont", "doesnt",
             "didnt", "wont", "wouldnt", "couldnt", "shouldnt", "isnt", "arent",
             "wasnt", "werent", "nothing", "nowhere"}
# A short stoplist so discriminative tokens are content-bearing, not "the/a/of".
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "as", "is", "it", "this", "that", "i", "im", "you", "he",
    "she", "we", "they", "my", "so", "do", "does", "did", "be", "been", "am",
    "are", "was", "were", "will", "would", "can", "could", "should", "have",
    "has", "had", "s", "t", "m", "re", "ll", "ve", "d", "just", "like", "okay",
    "ok", "its", "me", "up", "out", "get", "got", "what", "how", "why", "then",
    "now", "here", "there", "some", "one", "no", "not", "well",
}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower())]


def _think_aloud_text(rec: dict) -> str:
    ta = rec.get("think_aloud", {})
    if isinstance(ta, str):
        return ta
    return " ".join(str(ta.get(k, "")) for k in
                    ("pre_attempt", "during_attempt", "post_attempt"))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"[.!?]+", text) if s.strip()]


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_corpus() -> list[dict]:
    if os.path.exists(MIXED_PATH):
        return _load_jsonl(MIXED_PATH)
    records: list[dict] = []
    records.extend(_load_jsonl(os.path.join(ST_DIR, "dataset.jsonl")))
    for path in sorted(glob.glob(os.path.join(ST_DIR, "augment_*.jsonl"))):
        records.extend(_load_jsonl(path))
    records.extend(_load_jsonl(os.path.join(ST_DIR, "flow_hard_negatives.jsonl")))
    return records


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def discriminative_tokens(state_counts: Counter, other_counts: Counter,
                          top_n: int = 20, alpha: float = 1.0) -> list[tuple[str, float]]:
    """Log-odds-with-smoothing of each token in `state_counts` vs `other_counts`.

    Returns the top_n content tokens (stopwords removed) by log-odds ratio."""
    vocab = set(state_counts) | set(other_counts)
    total_s = sum(state_counts.values()) + alpha * len(vocab)
    total_o = sum(other_counts.values()) + alpha * len(vocab)
    scored: list[tuple[str, float]] = []
    for tok in vocab:
        if tok in STOPWORDS or len(tok) < 3:
            continue
        # require a minimum presence in-state so rare noise doesn't dominate
        if state_counts.get(tok, 0) < 2:
            continue
        p_s = (state_counts.get(tok, 0) + alpha) / total_s
        p_o = (other_counts.get(tok, 0) + alpha) / total_o
        scored.append((tok, math.log(p_s / p_o)))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:top_n]


def cell_style_metrics(records: list[dict]) -> dict:
    """Mean sentence length, filler / question / negation rates for a set of records."""
    n_tokens = 0
    n_sentences = 0
    n_questions = 0
    n_filler = 0
    n_neg = 0
    sent_lengths: list[int] = []
    for rec in records:
        text = _think_aloud_text(rec)
        toks = _tokens(text)
        n_tokens += len(toks)
        sents = _sentences(text)
        n_sentences += len(sents)
        n_questions += text.count("?")
        for s in sents:
            sent_lengths.append(len(s.split()))
        n_filler += sum(1 for t in toks if t in FILLER_UNIGRAMS)
        n_filler += sum(1 for a, b in zip(toks, toks[1:]) if (a, b) in FILLER_BIGRAMS)
        n_neg += sum(1 for t in toks if t in NEGATIONS)
        n_neg += len(re.findall(r"n't\b", text.lower()))
    denom_tok = max(1, n_tokens)
    denom_sent = max(1, n_sentences)
    return {
        "n_samples": len(records),
        "mean_sentence_length": round(sum(sent_lengths) / max(1, len(sent_lengths)), 2),
        "filler_rate": round(n_filler / denom_tok, 4),
        "question_rate": round(n_questions / denom_sent, 4),
        "negation_rate": round(n_neg / denom_tok, 4),
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

def run(records: list[dict]) -> dict:
    # group records by (generator, state)
    by_gen_state: dict[tuple, list] = defaultdict(list)
    generators: set[str] = set()
    for rec in records:
        gen = rec.get("generator_model", "unknown")
        st = rec.get("cognitive_state", "")
        if st not in COGNITIVE_STATES:
            continue
        generators.add(gen)
        by_gen_state[(gen, st)].append(rec)
    generators = sorted(generators)

    # token counts per (generator, state)
    tok_counts: dict[tuple, Counter] = {}
    for (gen, st), recs in by_gen_state.items():
        c: Counter = Counter()
        for rec in recs:
            c.update(_tokens(_think_aloud_text(rec)))
        tok_counts[(gen, st)] = c

    cells: dict[str, dict] = {}
    top_tokens: dict[str, dict[str, list[str]]] = {s: {} for s in COGNITIVE_STATES}
    for gen in generators:
        for st in COGNITIVE_STATES:
            recs = by_gen_state.get((gen, st), [])
            if not recs:
                continue
            state_c = tok_counts.get((gen, st), Counter())
            other_c: Counter = Counter()
            for other_st in COGNITIVE_STATES:
                if other_st != st:
                    other_c.update(tok_counts.get((gen, other_st), Counter()))
            disc = discriminative_tokens(state_c, other_c)
            style = cell_style_metrics(recs)
            cells[f"{gen}|{st}"] = {
                "generator": gen, "state": st,
                "top_discriminative_tokens": [
                    {"token": t, "log_odds": round(v, 3)} for t, v in disc],
                **style,
            }
            top_tokens[st][gen] = [t for t, _ in disc]

    # Jaccard overlap of top-20 token sets across generator pairs, per state.
    per_state_overlap: dict[str, dict] = {}
    all_pair_means: list[float] = []
    for st in COGNITIVE_STATES:
        gens_here = [g for g in generators if top_tokens[st].get(g)]
        pair_scores = {}
        for g1, g2 in itertools.combinations(gens_here, 2):
            j = jaccard(set(top_tokens[st][g1]), set(top_tokens[st][g2]))
            pair_scores[f"{g1} vs {g2}"] = round(j, 3)
        mean_overlap = (round(sum(pair_scores.values()) / len(pair_scores), 3)
                        if pair_scores else None)
        if mean_overlap is not None:
            all_pair_means.append(mean_overlap)
        per_state_overlap[st] = {
            "generators_present": gens_here,
            "pairwise_jaccard": pair_scores,
            "mean_jaccard_overlap": mean_overlap,
        }

    overall_mean = (round(sum(all_pair_means) / len(all_pair_means), 3)
                    if all_pair_means else None)

    return {
        "n_records": len(records),
        "generators": generators,
        "cells": cells,
        "per_state_top20_jaccard_overlap": per_state_overlap,
        "overall_mean_jaccard_overlap": overall_mean,
        "interpretation": (
            "Lower Jaccard overlap of top-20 discriminative tokens across "
            "generators = more linguistic diversity in the mixed corpus, which is "
            "what justifies multi-generator training."),
    }


def print_summary(results: dict) -> None:
    print("\n" + "=" * 70)
    print("GENERATOR DIVERSITY REPORT")
    print("=" * 70)
    print(f"records: {results['n_records']}   generators: {results['generators']}")
    print(f"\n{'state':12s}{'mean top-20 Jaccard overlap (lower = more diverse)':>50s}")
    print("-" * 62)
    for st in COGNITIVE_STATES:
        ov = results["per_state_top20_jaccard_overlap"].get(st, {})
        mo = ov.get("mean_jaccard_overlap")
        n_gen = len(ov.get("generators_present", []))
        val = f"{mo:.3f}" if mo is not None else "n/a"
        print(f"{st:12s}{val:>40s}   ({n_gen} generators)")
    print("-" * 62)
    om = results["overall_mean_jaccard_overlap"]
    print(f"{'OVERALL':12s}{(f'{om:.3f}' if om is not None else 'n/a'):>40s}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generator linguistic-diversity report.")
    ap.add_argument("--output", default=OUT_PATH)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    records = load_corpus()
    if not records:
        print("ERROR: no corpus found. Run generation + --merge first.", file=sys.stderr)
        return 1

    results = run(records)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print_summary(results)
    print(f"\nResults saved to {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
