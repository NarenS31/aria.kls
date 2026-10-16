#!/usr/bin/env python
"""Analyse data/demo_logs/teacher_taps.csv exactly as docs/teacher_timing_analysis.md says (descriptive; n >= 15 or nothing is reported).

    /usr/bin/python3 scripts/analyze_teacher_taps.py data/demo_logs/teacher_taps.csv --moments data/teacher_moments.json [--json out.json]

Pure standard library. Seeds: subsample 20261024, bootstrap 20261025 (1,000 draws).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

MIN_N = 15
SMALL = 5
SUBSAMPLE_SEED = 20261024
BOOT_SEED = 20261025
BOOT_DRAWS = 1000
SPLIT_BAND = (0.40, 0.60)


def fleiss_kappa(counts: list):
    """counts: per-moment [n_step_in, n_stay_quiet], the SAME number of raters n for every moment. None if undefined."""
    N, n = len(counts), sum(counts[0])
    if N == 0 or n < 2 or any(sum(c) != n for c in counts):
        return None
    p_bar = sum((sum(x * x for x in c) - n) / (n * (n - 1)) for c in counts) / N
    pj = [sum(c[j] for c in counts) / (N * n) for j in range(2)]
    pe = sum(p * p for p in pj)
    return None if pe >= 1 else (p_bar - pe) / (1 - pe)


def kappa_for(moments: list, ratings: dict, rng: random.Random):
    """ratings: moment_id -> list of decisions (one per session). Subsample every moment to the smallest count (seeded)."""
    lists = [ratings.get(m, []) for m in moments]
    n = min((len(l) for l in lists), default=0)
    if n < 2:
        return None, n
    counts = []
    for l in lists:
        pick = rng.sample(l, n) if len(l) > n else list(l)
        counts.append([sum(d == "step_in" for d in pick), sum(d == "stay_quiet" for d in pick)])
    return fleiss_kappa(counts), n


def boot_ci(moments: list, rows_by_session: dict, draws=BOOT_DRAWS, seed=BOOT_SEED):
    rng = random.Random(seed)
    sessions = list(rows_by_session)
    vals = []
    for _ in range(draws):
        pick = [rng.choice(sessions) for _ in sessions]
        ratings = defaultdict(list)
        for s in pick:
            for mid, dec in rows_by_session[s].items():
                if mid in moments:
                    ratings[mid].append(dec)
        k, _ = kappa_for(moments, ratings, rng)
        if k is not None:
            vals.append(k)
    if len(vals) < draws // 2:
        return None
    vals.sort()
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]


def show(n: int) -> str:
    return f"<{SMALL}" if n < SMALL else str(n)


def analyse(rows: list, moments: dict) -> dict:
    by_mom = defaultdict(list)
    by_sess = defaultdict(dict)
    for r in rows:
        if r["moment_id"] in moments:
            by_mom[r["moment_id"]].append(r)
            by_sess[r["session_id"]][r["moment_id"]] = r["teacher_decision"]
    n_per = {m: len({r["session_id"] for r in by_mom[m]}) for m in moments}
    out = {"n_sessions_total": len({r["session_id"] for r in rows}), "n_per_moment": n_per, "sets": {}, "min_n": MIN_N}
    sets = {"all": list(moments), "math": [m for m, v in moments.items() if not v["words_only"]],
            "words_only": [m for m, v in moments.items() if v["words_only"]],
            "verified_wrong": [m for m, v in moments.items() if v["verified_wrong"]]}
    for name, ids in sets.items():
        res = {"moments": ids}
        low = {m: n_per[m] for m in ids if n_per[m] < MIN_N}
        if not ids:
            res["status"] = "no moments in this set"
        elif low:
            res["status"] = f"not reported (n < {MIN_N} for: " + ", ".join(f"{m}={show(n)}" for m, n in sorted(low.items())) + ")"
        else:
            res["status"] = "reported"
            ratings = {m: [r["teacher_decision"] for r in by_mom[m]] for m in ids}
            share = {m: sum(d == "step_in" for d in ratings[m]) / len(ratings[m]) for m in ids}
            majority = {m: ("split" if SPLIT_BAND[0] <= share[m] <= SPLIT_BAND[1] else "step_in" if share[m] > SPLIT_BAND[1] else "stay_quiet") for m in ids}
            res["step_in_share"] = share
            res["majority"] = majority
            res["aria"] = {m: moments[m]["aria_decision"] for m in ids}
            if name != "verified_wrong":
                k, nused = kappa_for(ids, ratings, random.Random(SUBSAMPLE_SEED))
                res["fleiss_kappa"] = k
                res["raters_per_moment_used"] = nused
                res["fleiss_kappa_ci95"] = boot_ci(ids, {s: d for s, d in by_sess.items()}) if k is not None else None
                # ARIA-vs-teacher comparisons leave out verified-wrong moments (ARIA's decision there is fixed by design; plan section 5d)
                cmp_ids = [m for m in ids if not moments[m]["verified_wrong"]]
                res["moments_in_aria_comparison"] = cmp_ids
                conf = Counter((majority[m], res["aria"][m]) for m in cmp_ids if majority[m] != "split")
                res["confusion_majority_vs_aria"] = {f"majority={a}|aria={b}": conf[(a, b)] for a in ("step_in", "stay_quiet") for b in ("step_in", "stay_quiet")}
                res["split_moments"] = [m for m in cmp_ids if majority[m] == "split"]
                agree = [r["teacher_decision"] == moments[r["moment_id"]]["aria_decision"] for m in cmp_ids for r in by_mom[m]]
                res["response_level_agreement_with_aria"] = (sum(agree) / len(agree)) if agree else None
                res["response_level_ci95"] = _boot_rate(cmp_ids, by_mom, moments) if agree else None
                both = [(r["teacher_type"], r["aria_type"]) for m in cmp_ids for r in by_mom[m] if r["teacher_decision"] == "step_in" and r["aria_decision"] == "step_in"]
                res["type_match_when_both_step_in"] = {"n": len(both), "share": (sum(a == b for a, b in both) / len(both)) if both else None}
            else:
                res["teacher_stay_quiet_rate_on_verified_wrong"] = {m: 1 - share[m] for m in ids}
                res["note"] = "ARIA never stays quiet on a verified wrong step (by design), so these moments are not in the ARIA-vs-majority table."
        out["sets"][name] = res
    after = [(r["teacher_decision"] == r["aria_decision"], r["agree"] == "agree") for r in rows if r["moment_id"] in moments]
    if all(n_per[m] >= MIN_N for m in moments) and after:
        out["after_reveal"] = {
            "agree_share": sum(a for _, a in after) / len(after),
            "agree_share_when_decision_matched_aria": _share([a for m, a in after if m]),
            "agree_share_when_decision_differed": _share([a for m, a in after if not m])}
    else:
        out["after_reveal"] = f"not reported (n < {MIN_N} for at least one moment)"
    out["subgroups"] = _subgroups(rows, moments)
    return out


def _share(xs):
    return (sum(xs) / len(xs)) if xs else None


def _boot_rate(ids, by_mom, moments, draws=BOOT_DRAWS, seed=BOOT_SEED):
    rng = random.Random(seed)
    per = defaultdict(list)
    for m in ids:
        for r in by_mom[m]:
            per[r["session_id"]].append(r["teacher_decision"] == moments[m]["aria_decision"])
    sessions = list(per)
    vals = []
    for _ in range(draws):
        pick = [rng.choice(sessions) for _ in sessions]
        flat = [x for s in pick for x in per[s]]
        vals.append(sum(flat) / len(flat))
    vals.sort()
    return [vals[int(0.025 * (draws - 1))], vals[int(0.975 * (draws - 1))]]


def _subgroups(rows, moments):
    """step_in share per subgroup, suppressed below MIN_N sessions; counts below SMALL shown as '<5'. Descriptive only."""
    out = {}
    for col in ("role", "years_band", "grade_band"):
        sess = defaultdict(set)
        dec = defaultdict(list)
        for r in rows:
            if r["moment_id"] in moments:
                sess[r[col]].add(r["session_id"])
                dec[r[col]].append(r["teacher_decision"] == "step_in")
        out[col] = {g: ({"sessions": show(len(sess[g])), "step_in_share": None, "status": f"suppressed (n < {MIN_N})"} if len(sess[g]) < MIN_N
                        else {"sessions": len(sess[g]), "step_in_share": sum(dec[g]) / len(dec[g])}) for g in sorted(sess)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--moments", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    moments = {m["id"]: m for m in json.loads(Path(a.moments).read_text(encoding="utf-8"))}
    with open(a.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    res = analyse(rows, moments)
    print(f"sessions: {res['n_sessions_total']} | per moment: {res['n_per_moment']} | minimum n to report: {MIN_N}")
    for name, s in res["sets"].items():
        print(f"\n[{name}] {s['status']}")
        for k, v in s.items():
            if k not in ("status", "moments"):
                print(f"   {k}: {json.dumps(v)}")
    print("\nafter reveal:", json.dumps(res["after_reveal"]))
    print("subgroups:", json.dumps(res["subgroups"]))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
