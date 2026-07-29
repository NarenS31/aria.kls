#!/usr/bin/env python3.11
"""
Linguistic profiling of the four think-aloud generators, to explain WHY the
cross-generator accuracy gap varies so much between them.

Observed heuristic gaps vs the llama3.1 baseline (the numbers this script
correlates against):

    llama = 0.0   gemma2 = 4.3   mistral = 17.4   phi3 = 33.4

For each generator it measures eight surface features (rate features are per
100 words so corpora of different sizes are comparable), then Pearson-correlates
each feature across the four generators with the gap. A strong positive
correlation means "generators that do more of this are harder for the
llama-tuned classifier" — i.e. a candidate explanation for the gap.

    python3.11 eval/generator_linguistics.py      # < 5 min, no model calls

Writes:
    data/eval/generator_linguistics.json      per-generator feature values
    data/eval/gap_correlation_analysis.json    per-feature correlation with gap
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from metacognition.analyzer import _MARKERS, _join_think_aloud  # noqa: E402

D = os.path.join(REPO_ROOT, "data", "synthetic_thinkaloud")
RESULTS_DIR = os.path.join(REPO_ROOT, "data", "eval")

# generator tag -> source corpus (order also fixes the table column order)
SOURCES = {
    "llama":   os.path.join(D, "train.jsonl"),          # llama3.1:8b train split
    "mistral": os.path.join(D, "augment_mistral.jsonl"),
    "gemma2":  os.path.join(D, "augment_gemma2.jsonl"),
    "phi3":    os.path.join(D, "augment_phi3.jsonl"),
}
GENS = list(SOURCES)

# Cross-generator gaps supplied for the correlation (points below the llama
# baseline). llama is the baseline, so its gap is 0.
GAP = {"llama": 0.0, "mistral": 17.4, "gemma2": 4.3, "phi3": 33.4}

# Every marker regex across all states, flattened, for the signal-density metric.
_ALL_MARKERS = [pat for markers in _MARKERS.values() for pat, _ in markers]

# --- feature lexicons (word-boundaried; apostrophes optional) ---------------
_WORD = re.compile(r"[a-z']+")
_SENT_SPLIT = re.compile(r"[.!?]+")
_FILLER = re.compile(r"\b(um+|uh+|like|okay\s+so|wait)\b", re.I)
_FIRST_PERSON = re.compile(r"\b(i|me|my)\b", re.I)
_NEGATION = re.compile(r"\b(don'?t|can'?t|not|never)\b", re.I)
_HEDGE = re.compile(r"\b(maybe|perhaps|i\s+think|i'?m\s+not\s+sure)\b", re.I)


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _clean(ta) -> str:
    """Join think_aloud parts and strip the stringified-list wrapper chars some
    generators emit (e.g. "['...']")."""
    t = _join_think_aloud(ta)
    for a, b in (("['", " "), ("']", " "), ('["', " "), ('"]', " ")):
        t = t.replace(a, b)
    return t.replace("\\'", "'").replace('\\"', '"')


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w != "'"]


def _unique_per_100(all_tokens: list[str]) -> float:
    """Mean distinct word types in a 100-token window (vocabulary richness,
    length-normalised so big and small corpora are comparable)."""
    if len(all_tokens) < 100:
        return round(100.0 * len(set(all_tokens)) / max(1, len(all_tokens)), 2)
    counts = []
    for i in range(0, len(all_tokens) - 99, 100):     # non-overlapping windows
        counts.append(len(set(all_tokens[i:i + 100])))
    return round(sum(counts) / len(counts), 2)


def profile(texts: list[str]) -> dict:
    total_words = 0
    total_sents = 0
    filler = first_person = questions = negation = hedge = signal = 0
    all_tokens: list[str] = []

    for t in texts:
        toks = _tokens(t)
        n = len(toks)
        if n == 0:
            continue
        total_words += n
        all_tokens.extend(toks)
        # sentences: number of terminal-punctuation groups (>=1 per non-empty)
        sents = [s for s in _SENT_SPLIT.split(t) if s.strip()]
        total_sents += max(1, len(sents))
        questions += t.count("?")
        filler += len(_FILLER.findall(t))
        first_person += len(_FIRST_PERSON.findall(t))
        negation += len(_NEGATION.findall(t))
        hedge += len(_HEDGE.findall(t))
        signal += sum(len(pat.findall(t)) for pat in _ALL_MARKERS)

    per100 = lambda c: round(100.0 * c / total_words, 2) if total_words else 0.0
    return {
        "n_records": len(texts),
        "total_words": total_words,
        "unique_words_per_100w": _unique_per_100(all_tokens),
        "mean_sentence_length_words": round(total_words / total_sents, 2) if total_sents else 0.0,
        "filler_rate_per_100w": per100(filler),
        "first_person_rate_per_100w": per100(first_person),
        "question_rate_per_100w": per100(questions),
        "negation_rate_per_100w": per100(negation),
        "hedging_rate_per_100w": per100(hedge),
        "signal_word_density_per_100w": per100(signal),
    }


# Features that go in the correlation table (label -> profile key).
FEATURES = [
    ("unique_words (per 100w)", "unique_words_per_100w"),
    ("mean_sentence_len (words)", "mean_sentence_length_words"),
    ("filler_rate (per 100w)", "filler_rate_per_100w"),
    ("first_person_rate (per 100w)", "first_person_rate_per_100w"),
    ("question_rate (per 100w)", "question_rate_per_100w"),
    ("negation_rate (per 100w)", "negation_rate_per_100w"),
    ("hedging_rate (per 100w)", "hedging_rate_per_100w"),
    ("signal_word_density (per 100w)", "signal_word_density_per_100w"),
]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None  # no variance -> correlation undefined
    return cov / math.sqrt(vx * vy)


def main() -> int:
    profiles: dict[str, dict] = {}
    for gen, path in SOURCES.items():
        if not os.path.exists(path):
            print(f"[warn] missing {path}; skipping {gen}", file=sys.stderr)
            continue
        texts = [_clean(r.get("think_aloud", "")) for r in _load(path)]
        profiles[gen] = profile(texts)

    present = [g for g in GENS if g in profiles]
    gaps = [GAP[g] for g in present]

    # correlate each feature (across generators) with the gap
    correlations: dict[str, dict] = {}
    for label, key in FEATURES:
        vals = [profiles[g][key] for g in present]
        r = pearson(vals, gaps)
        correlations[key] = {
            "label": label,
            "values": {g: profiles[g][key] for g in present},
            "correlation_with_gap": round(r, 3) if r is not None else None,
        }

    # ---- print table ----
    print("=" * 92)
    print("GENERATOR LINGUISTIC PROFILE  (rate features are per 100 words)")
    print("=" * 92)
    print(f"gaps vs llama baseline: " +
          "  ".join(f"{g}={GAP[g]}" for g in present))
    print()
    hdr = f"{'Feature':32s}" + "".join(f"{g:>9s}" for g in present) + f"{'corr_gap':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for label, key in FEATURES:
        row = f"{label:32s}"
        for g in present:
            row += f"{profiles[g][key]:>9.2f}"
        r = correlations[key]["correlation_with_gap"]
        row += f"{(r if r is not None else float('nan')):>10.3f}"
        print(row)
    print("-" * len(hdr))

    # ranked drivers
    ranked = sorted(
        (c for c in correlations.values() if c["correlation_with_gap"] is not None),
        key=lambda c: abs(c["correlation_with_gap"]), reverse=True)
    print("\nFeatures ranked by |correlation with gap| (strongest gap drivers first):")
    for c in ranked:
        r = c["correlation_with_gap"]
        direction = "more -> larger gap" if r > 0 else "more -> smaller gap"
        print(f"  {r:+.3f}  {c['label']:32s} ({direction})")

    # ---- save ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ling_path = os.path.join(RESULTS_DIR, "generator_linguistics.json")
    with open(ling_path, "w", encoding="utf-8") as fh:
        json.dump({
            "gaps": {g: GAP[g] for g in present},
            "per_generator": profiles,
            "note": ("Rate features are counts per 100 words. "
                     "unique_words_per_100w = mean distinct types in a 100-token "
                     "window (MATTR-style). mean_sentence_length is in words. "
                     "signal_word_density = total _MARKERS regex matches per 100 words."),
        }, fh, indent=2, ensure_ascii=False)

    corr_path = os.path.join(RESULTS_DIR, "gap_correlation_analysis.json")
    with open(corr_path, "w", encoding="utf-8") as fh:
        json.dump({
            "gap_values": {g: GAP[g] for g in present},
            "generator_order": present,
            "method": "Pearson correlation across 4 generators (n=4, exploratory).",
            "per_feature": correlations,
            "ranked_by_abs_correlation": [
                {"feature": next(k for _, k in FEATURES if correlations[k] is c),
                 "label": c["label"],
                 "correlation_with_gap": c["correlation_with_gap"]}
                for c in ranked
            ],
        }, fh, indent=2, ensure_ascii=False)

    print(f"\nSaved: {ling_path}")
    print(f"Saved: {corr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
