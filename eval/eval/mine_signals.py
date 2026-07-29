#!/usr/bin/env python3.11
"""
Marker-audit tool for the cross-generator ARIA cognitive-state heuristic.

Why this exists
---------------
The keyword banks in ``metacognition/analyzer._MARKERS`` were originally tuned
on llama3.1-generated think-alouds. When the classifier is evaluated on other
generators (mistral/gemma2/phi3) the accuracy drops — a chunk of the markers
turn out to be llama's stylistic fingerprint rather than a real cognitive
signal. This script quantifies that so the marker banks can be repaired, and
re-run whenever a new generator is added.

For each cognitive state it reports:

  1. TOP CROSS-GENERATOR SIGNALS — the discriminative n-grams (Monroe et al.
     2008 weighted log-odds, state-vs-rest) computed per generator, ranked by
     how many of the four generators share them ("coverage"). High-coverage
     tokens are the ones a generator-agnostic marker should key on; tokens that
     only light up for llama are flagged LLAMA-ONLY.

  2. EXISTING MARKER AUDIT — for every regex currently in ``_MARKERS``, the
     fraction of each generator's own-state samples it fires on. Markers that
     fire almost only on llama are flagged LLAMA-SPECIFIC; markers that fire
     nowhere are flagged DEAD.

Sources: the llama signal is mined from the llama training split; the other
three from the augment_<gen>.jsonl corpora (kept disjoint from the holdout sets
the classifier is actually scored on, so this is not evaluating on test data).

    python3.11 eval/mine_signals.py
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

from metacognition.analyzer import (  # noqa: E402
    _MARKERS, COGNITIVE_STATES, _join_think_aloud,
)

D = os.path.join(REPO_ROOT, "data", "synthetic_thinkaloud")
SOURCES = {
    "llama":   os.path.join(D, "train.jsonl"),          # llama3.1:8b train split
    "mistral": os.path.join(D, "augment_mistral.jsonl"),
    "gemma2":  os.path.join(D, "augment_gemma2.jsonl"),
    "phi3":    os.path.join(D, "augment_phi3.jsonl"),
}
GENS = list(SOURCES)
Z_STRONG = 1.5   # a token counts as a "signal" for a generator if its z >= this
MIN_DF = 4       # ...and it occurs at least this many times in that state
TOP_N = 30


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


_TOK = re.compile(r"[a-z']+")


def _ngrams(text: str) -> list[str]:
    ws = [w for w in _TOK.findall(text.lower()) if len(w) >= 2 and w != "'"]
    return ws + [f"{a} {b}" for a, b in zip(ws, ws[1:])]


def _logodds(ci: Counter, crest: Counter, alpha: float = 0.25) -> dict[str, float]:
    """Monroe/Colaresi/Quinn (2008) weighted log-odds z-score per token."""
    vocab = set(ci) | set(crest)
    ni, nr = sum(ci.values()), sum(crest.values())
    a0 = alpha * len(vocab)
    z: dict[str, float] = {}
    for w in vocab:
        yi, yr = ci.get(w, 0), crest.get(w, 0)
        numi, deni = yi + alpha, ni + a0 - yi - alpha
        numr, denr = yr + alpha, nr + a0 - yr - alpha
        if deni <= 0 or denr <= 0:
            continue
        delta = math.log(numi / deni) - math.log(numr / denr)
        z[w] = delta / math.sqrt(1.0 / numi + 1.0 / numr)
    return z


def main() -> int:
    by_gs: dict[str, dict[str, list[str]]] = {
        g: {s: [] for s in COGNITIVE_STATES} for g in GENS}
    for g, path in SOURCES.items():
        if not os.path.exists(path):
            print(f"[warn] missing {path}; skipping {g}")
            continue
        for r in _load(path):
            s = r.get("cognitive_state", "")
            if s in COGNITIVE_STATES:
                by_gs[g][s].append(_clean(r.get("think_aloud", "")))

    print("=== corpus sizes (records per generator per state) ===")
    print("state".ljust(12) + "".join(g.rjust(9) for g in GENS))
    for s in COGNITIVE_STATES:
        print(s.ljust(12) + "".join(str(len(by_gs[g][s])).rjust(9) for g in GENS))

    # per generator: state -> {token: z}, and state -> counts
    gen_state_z: dict[str, dict[str, dict[str, float]]] = {g: {} for g in GENS}
    gen_state_counts: dict[str, dict[str, Counter]] = {g: {} for g in GENS}
    for g in GENS:
        allc = {s: Counter(t for txt in by_gs[g][s] for t in _ngrams(txt))
                for s in COGNITIVE_STATES}
        total = Counter()
        for c in allc.values():
            total.update(c)
        for s in COGNITIVE_STATES:
            ci = allc[s]
            crest = total.copy()
            crest.subtract(ci)
            crest = Counter({w: c for w, c in crest.items() if c > 0})
            gen_state_z[g][s] = _logodds(ci, crest)
            gen_state_counts[g][s] = ci

    print("\n\n=== TOP CROSS-GENERATOR DISCRIMINATIVE SIGNALS PER STATE ===")
    print("(coverage = # of generators where token is a signal; ranked by "
          "coverage then mean z)")
    for s in COGNITIVE_STATES:
        cand = {w for g in GENS for w, z in gen_state_z[g][s].items()
                if z >= Z_STRONG and gen_state_counts[g][s].get(w, 0) >= MIN_DF}
        rows = []
        for w in cand:
            zs = {g: gen_state_z[g][s].get(w, float("-inf")) for g in GENS}
            present = [g for g in GENS if zs[g] >= Z_STRONG
                       and gen_state_counts[g][s].get(w, 0) >= MIN_DF]
            rows.append((w, len(present), sum(zs[g] for g in present) / len(present),
                         present, zs))
        rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
        print(f"\n---- {s}  (top {TOP_N}) ----")
        for w, cov, mz, present, zs in rows[:TOP_N]:
            tag = "  <== LLAMA-ONLY" if present == ["llama"] else (
                "" if cov >= 3 else "  (weak coverage)")
            zstr = " ".join(f"{g[:3]}:{zs[g]:+.1f}" for g in GENS)
            print(f"  cov{cov} mz{mz:+.1f}  {w:<22} [{zstr}]{tag}")

    print("\n\n=== EXISTING MARKER AUDIT (own-state firing rate per generator) ===")
    for s in COGNITIVE_STATES:
        print(f"\n---- {s} ----")
        own_texts = {g: by_gs[g][s] for g in GENS}
        for pat, weight in _MARKERS[s]:
            own = {}
            for g in GENS:
                ot = own_texts[g]
                own[g] = (sum(1 for t in ot if pat.search(t)) / len(ot)) if ot else 0.0
            ownstr = " ".join(f"{g[:3]}:{own[g] * 100:4.1f}" for g in GENS)
            non_llama = [own[g] for g in GENS if g != "llama"]
            mean_nl = sum(non_llama) / len(non_llama) if non_llama else 0.0
            flag = ""
            if own["llama"] >= 0.05 and mean_nl < own["llama"] * 0.34:
                flag = "  <== LLAMA-SPECIFIC"
            elif max(own.values()) < 0.02:
                flag = "  <== DEAD"
            print(f"  w{weight:>3} own[{ownstr}]{flag}   rx: {pat.pattern[:46]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
