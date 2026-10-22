#!/usr/bin/env python
"""Cohen's kappa for the blind 150-item annotation study (two raters, seven cognitive-state categories).

    python scripts/annotation_kappa.py --rater1 R1.csv --rater2 R2.csv --key PRIVATE_KEY.json --out kappa.json

Inputs are the two returned blind sheets (columns item_id, problem_text, think_aloud, predicted_category, confidence;
format of eval/data/annotation/rater*_blind.csv, written by eval/build_annotation_study.py) and the PRIVATE key
(eval/data/annotation/annotation_private_key.json, never committed: .gitignore:57). This script lives outside eval/ and
does not edit anything there.

Reports: unweighted Cohen's kappa between the two raters (95% bootstrap interval over items), raw agreement, per-category
agreement, each rater against the key, and the same agreement with and without the items whose text contains their own
category word (the lexical confound the build script measures). It refuses incomplete sheets, unknown categories, item-id
mismatches, and two sheets that are identical (a copied sheet would give kappa = 1).

It does not use `eval/score_calibration.py`'s output: that 14-item recognition check is deliberately not a kappa.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

N_BOOT = 2000
BOOT_SEED = 20261022


def read_sheet(path: Path, categories: list, n_expected: int = None) -> dict:
    rows = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            item = (row.get("item_id") or "").strip()
            cat = (row.get("predicted_category") or "").strip().upper()
            if not item:
                raise ValueError(f"{path}: a row has no item_id")
            if item in rows:
                raise ValueError(f"{path}: duplicate item {item}")
            if not cat:
                raise ValueError(f"{path}: {item} has no predicted_category (incomplete sheet)")
            if cat not in categories:
                raise ValueError(f"{path}: {item} has unknown category {cat!r}; allowed {categories}")
            conf = (row.get("confidence") or "").strip()
            if conf and conf not in {"1", "2", "3", "4", "5"}:
                raise ValueError(f"{path}: {item} has confidence {conf!r} (allowed 1-5 or blank)")
            rows[item] = {"cat": cat, "conf": int(conf) if conf else None}
    if n_expected is not None and len(rows) != n_expected:
        raise ValueError(f"{path}: {len(rows)} items, expected {n_expected}")
    return rows


def cohen_kappa(a: list, b: list, labels: list = None) -> float:
    """Unweighted multi-class Cohen's kappa; nan if chance agreement is 1."""
    n = len(a)
    if n == 0:
        return float("nan")
    labels = labels or sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def boot_ci(a: list, b: list, labels: list, seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> list:
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx], labels)
        if k == k:
            vals.append(k)
    vals.sort()
    if not vals:
        return [float("nan"), float("nan")]
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]


def summarize(a: list, b: list, labels: list) -> dict:
    n = len(a)
    return {"n": n, "raw_agreement": sum(x == y for x, y in zip(a, b)) / n if n else float("nan"),
            "kappa": cohen_kappa(a, b, labels), "kappa_ci95": boot_ci(a, b, labels) if n else [float("nan")] * 2}


def analyse(r1: dict, r2: dict, key: dict, label: str) -> dict:
    cats = list(key["categories"])
    items = sorted(r1)
    if set(r1) != set(r2):
        raise ValueError(f"the two sheets do not contain the same items: {sorted(set(r1) ^ set(r2))[:5]}")
    if set(items) != set(key["items"]):
        raise ValueError("sheet item ids do not match the private key")
    a = [r1[i]["cat"] for i in items]
    b = [r2[i]["cat"] for i in items]
    if a == b and len(a) > 20 and all(r1[i]["conf"] == r2[i]["conf"] for i in items):
        raise ValueError("the two sheets are identical; a copied sheet is not an independent second rater")
    truth = [key["items"][i]["true_category"] for i in items]
    own = [bool(key["items"][i].get("contains_own_label_word")) for i in items]
    out = {"label": label, "n_items": len(items), "categories": cats,
           "inter_rater": summarize(a, b, cats),
           "rater1_vs_key": summarize(a, truth, cats), "rater2_vs_key": summarize(b, truth, cats)}
    for name, flag in (("items_with_own_label_word", True), ("items_without_own_label_word", False)):
        sel = [k for k, f in enumerate(own) if f is flag]
        out[name] = summarize([a[k] for k in sel], [b[k] for k in sel], cats) if sel else {"n": 0}
    per = {}
    for c in cats:
        both = sum(1 for x, y in zip(a, b) if x == c and y == c)
        either = sum(1 for x, y in zip(a, b) if x == c or y == c)
        per[c] = {"rater1_n": a.count(c), "rater2_n": b.count(c), "both_chose": both,
                  "positive_specific_agreement": (2 * both / (a.count(c) + b.count(c))) if (a.count(c) + b.count(c)) else None,
                  "jaccard": both / either if either else None}
    out["per_category"] = per
    confusion = {}
    for x, y in zip(a, b):
        confusion[f"{x}|{y}"] = confusion.get(f"{x}|{y}", 0) + 1
    out["rater_confusion_rater1|rater2"] = confusion
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rater1", required=True, type=Path)
    ap.add_argument("--rater2", required=True, type=Path)
    ap.add_argument("--key", required=True, type=Path, help="PRIVATE key json (never committed)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", default="", help="free text stored in the output, e.g. 'CODE TEST ONLY (synthetic annotators)'")
    a = ap.parse_args(argv)
    key = json.loads(a.key.read_text(encoding="utf-8"))
    cats = list(key["categories"])
    n = int(key.get("n_items", len(key["items"])))
    res = analyse(read_sheet(a.rater1, cats, n), read_sheet(a.rater2, cats, n), key, a.label)
    a.out.write_text(json.dumps(res, indent=1))
    ir = res["inter_rater"]
    print(f"[{a.label or 'real run'}] n={ir['n']}  kappa={ir['kappa']:.3f} (95% {ir['kappa_ci95'][0]:.3f}..{ir['kappa_ci95'][1]:.3f})  raw agreement={ir['raw_agreement']:.3f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        print("REFUSED:", e, file=sys.stderr)
        raise SystemExit(2)
