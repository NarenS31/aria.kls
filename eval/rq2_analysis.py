#!/usr/bin/env python3.11
"""Unblind and analyse the RQ2 rating form responses.

    python3.11 eval/rq2_analysis.py --responses path/to/form_responses.csv

UNTESTED AGAINST A REAL EXPORT. No responses have been collected yet, so this
has only been run against a simulated Google Forms export (see
tests/test_rq2_analysis.py, which plants a known effect and a known A/B
arrangement and checks that both come back out). The logic is complete and
exercised; what has never been seen is a CSV that Google actually produced. The
first thing to check on real data is the column mapping — the script fails loudly
rather than guessing if the headers are not what it expects.

WHY THE COLUMNS ARE MATCHED BY ORDER
The five per-item questions are worded identically on all ten items, by design:
changing the wording per item would change what is being asked. So a Forms
export has ten columns titled "Rate Response A — ...", and they are told apart
by position, which is the order the items appear in the form. That order is
recorded as `form_item_number` in data/rq2_answer_key.json. If the number of
matching columns is not exactly ten, the mapping is not trustworthy and this
script stops instead of producing plausible numbers from a bad join.

WHAT IS REPORTED
  - mean helpfulness for scaffolded vs answer-giving, unblinded per item
  - preference rates on Q5 (most helpful) and Q8 (what I'd want when stuck)
  - the Q5/Q8 divergence per rater: the same person naming one style as more
    useful for learning and the other as what they'd rather receive is the
    finding, not noise, so it is counted and broken out by direction
  - per-item breakdown, since the effect need not be uniform across items
  - Wilcoxon signed-rank (primary) and paired t-test (secondary) on the
    helpfulness scores

Wilcoxon is primary because a 1-5 helpfulness rating is ordinal: the distance
from 1 to 2 is not known to equal the distance from 4 to 5, which is what a
t-test on the raw scale assumes. The t-test is reported alongside it because
reviewers expect it and because agreement between the two is worth seeing.
Raters, not items, are the unit: each rater contributes one scaffolded mean and
one answer-giving mean, paired. The item-level test is reported second.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import unicodedata
from collections import Counter
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_KEY = os.path.join(REPO_ROOT, "data", "rq2_answer_key.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "eval", "rq2_results.json")


def display_path(path: str) -> str:
    """Repo-relative when the file is in the repo, absolute when it is not.

    A downloaded response export usually sits in ~/Downloads, and rendering that
    as a chain of `../..` makes the provenance line in the results file harder
    to read than the full path.
    """
    absolute = os.path.abspath(path)
    relative = os.path.relpath(absolute, REPO_ROOT)
    return absolute if relative.startswith(os.pardir) else relative


SCAFFOLDED = "scaffolded"
ANSWER_GIVING = "answer_giving"
SAME = "about_the_same"

CONDITIONS = (SCAFFOLDED, ANSWER_GIVING)


# ------------------------------------------------------------------
# Column mapping
# ------------------------------------------------------------------

def normalize(header: str) -> str:
    """Compare question text without tripping over transport differences.

    A CSV that has been through a spreadsheet can come back with the em dash or
    the curly apostrophe rewritten, and pandas appends `.1`, `.2` to duplicate
    headers. None of that changes which question it is.
    """
    text = unicodedata.normalize("NFKC", header or "").strip()
    text = re.sub(r"\.\d+$", "", text)          # pandas duplicate suffix
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def find_columns(headers: list[str], question: str, expected: int) -> list[int]:
    """Indices of the columns for `question`, in form order."""
    target = normalize(question)
    found = [i for i, h in enumerate(headers) if normalize(h) == target]
    if len(found) != expected:
        raise SystemExit(
            f"Column mapping failed.\n"
            f"  Expected {expected} column(s) titled:\n    {question}\n"
            f"  Found {len(found)}.\n"
            f"  Headers in the file:\n"
            + "\n".join(f"    [{i}] {h}" for i, h in enumerate(headers))
            + "\n\nThe per-item questions are matched by position, so a partial "
              "match cannot be repaired by guessing. Check that the CSV is the "
              "response export for this form and not a different one."
        )
    return found


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

def parse_rating(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    match = re.match(r"^([1-5])\b", value)
    if not match:
        return None
    return int(match.group(1))


def parse_choice(value: str, item_key: dict) -> str | None:
    """Map a rater's "A"/"B"/"About the same" onto the condition it names."""
    value = (value or "").strip()
    if not value:
        return None
    letter = value[0].upper()
    if letter == "A" and not value.lower().startswith("about"):
        return item_key["A"]
    if letter == "B":
        return item_key["B"]
    if value.lower().startswith("about"):
        return SAME
    return None


def load_key(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        key = json.load(fh)
    by_number = {v["form_item_number"]: (pid, v) for pid, v in key["key"].items()}
    missing = [n for n in range(1, key["n_items"] + 1) if n not in by_number]
    if missing:
        raise SystemExit(f"{path}: answer key has no entry for item(s) {missing}")
    key["_by_number"] = by_number
    return key


def read_responses(path: str, key: dict) -> tuple[list[dict], list[str]]:
    """One record per (rater, item), plus each rater's background answers."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f"{path}: file is empty")
    headers, body = rows[0], rows[1:]
    if not body:
        raise SystemExit(f"{path}: no responses in the export (header row only)")

    q = key["questions"]
    n_items = key["n_items"]
    which_cols = find_columns(headers, q["which_helps"], n_items)
    rate_a_cols = find_columns(headers, q["rate_a"], n_items)
    rate_b_cols = find_columns(headers, q["rate_b"], n_items)
    stuck_cols = find_columns(headers, q["when_stuck"], n_items)
    free_cols = find_columns(headers, q["free_text"], n_items)
    overall_col = find_columns(headers, q["overall"], 1)[0]
    grade_col = find_columns(headers, q["grade"], 1)[0]
    math_col = find_columns(headers, q["math_relationship"], 1)[0]

    def cell(row: list[str], index: int) -> str:
        return row[index] if index < len(row) else ""

    records, raters = [], []
    for r, row in enumerate(body, 1):
        rater = f"R{r:02d}"
        raters.append({
            "rater": rater,
            "overall_preference": cell(row, overall_col).strip(),
            "grade": cell(row, grade_col).strip(),
            "math_relationship": cell(row, math_col).strip(),
        })
        for number in range(1, n_items + 1):
            problem_id, item_key = key["_by_number"][number]
            i = number - 1
            rating_a = parse_rating(cell(row, rate_a_cols[i]))
            rating_b = parse_rating(cell(row, rate_b_cols[i]))
            ratings = {}
            if rating_a is not None:
                ratings[item_key["A"]] = rating_a
            if rating_b is not None:
                ratings[item_key["B"]] = rating_b
            records.append({
                "rater": rater,
                "form_item_number": number,
                "problem_id": problem_id,
                "shown_as_A": item_key["A"],
                "rating": ratings,
                "most_helpful": parse_choice(cell(row, which_cols[i]), item_key),
                "when_stuck": parse_choice(cell(row, stuck_cols[i]), item_key),
                "comment": cell(row, free_cols[i]).strip(),
            })
    return records, raters


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def paired_tests(pairs: list[tuple[float, float]]) -> dict:
    """Wilcoxon signed-rank (primary) and paired t-test on the same pairs."""
    out = {
        "n_pairs": len(pairs),
        "mean_scaffolded": None,
        "mean_answer_giving": None,
        "mean_difference": None,
        "wilcoxon": {"statistic": None, "p_value": None,
                     "rank_biserial_r": None, "note": None},
        "paired_t": {"statistic": None, "p_value": None,
                     "cohens_dz": None, "note": None},
    }
    if len(pairs) < 2:
        note = f"only {len(pairs)} complete pair(s); no test run"
        out["wilcoxon"]["note"] = out["paired_t"]["note"] = note
        if pairs:
            out["mean_scaffolded"] = pairs[0][0]
            out["mean_answer_giving"] = pairs[0][1]
            out["mean_difference"] = pairs[0][0] - pairs[0][1]
        return out

    scaffolded = [p[0] for p in pairs]
    answering = [p[1] for p in pairs]
    diffs = [s - a for s, a in pairs]
    out["mean_scaffolded"] = round(statistics.mean(scaffolded), 3)
    out["mean_answer_giving"] = round(statistics.mean(answering), 3)
    out["mean_difference"] = round(statistics.mean(diffs), 3)

    from scipy import stats  # imported here so --help works without scipy

    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        note = "every rater scored the two conditions identically; no test run"
        out["wilcoxon"]["note"] = out["paired_t"]["note"] = note
        return out

    try:
        w_stat, w_p = stats.wilcoxon(scaffolded, answering)
        out["wilcoxon"]["statistic"] = float(w_stat)
        out["wilcoxon"]["p_value"] = float(w_p)
        out["wilcoxon"]["rank_biserial_r"] = round(rank_biserial(diffs), 3)
    except ValueError as exc:
        out["wilcoxon"]["note"] = f"scipy declined to run the test: {exc}"

    # A t-test needs variance in the differences. If every pair differs by the
    # same amount the statistic is division by zero, and scipy returns infinity
    # with a precision-loss warning rather than refusing — so refuse here. The
    # Wilcoxon result above still stands; it only needs the signs.
    sd = statistics.stdev(diffs)
    if not sd:
        out["paired_t"]["note"] = (
            f"every pair differs by exactly {out['mean_difference']}; with no "
            "variance in the differences t is undefined (Wilcoxon still applies)"
        )
        return out

    try:
        t_stat, t_p = stats.ttest_rel(scaffolded, answering)
        out["paired_t"]["statistic"] = float(t_stat)
        out["paired_t"]["p_value"] = float(t_p)
        out["paired_t"]["cohens_dz"] = round(statistics.mean(diffs) / sd, 3)
    except ValueError as exc:
        out["paired_t"]["note"] = f"scipy declined to run the test: {exc}"
    return out


def rank_biserial(diffs: list[float]) -> float:
    """Matched-pairs rank-biserial correlation: (T+ - T-) / (T+ + T-).

    Computed here rather than taken from scipy so the sign convention is fixed
    and readable: positive favours scaffolded, since the differences are
    scaffolded minus answer-giving.
    """
    nonzero = [d for d in diffs if d != 0]
    ordered = sorted(nonzero, key=abs)
    ranks, i = [0.0] * len(ordered), 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and abs(ordered[j + 1]) == abs(ordered[i]):
            j += 1
        average = (i + j + 2) / 2  # ranks are 1-based; ties share the average
        for k in range(i, j + 1):
            ranks[k] = average
        i = j + 1
    positive = sum(r for d, r in zip(ordered, ranks) if d > 0)
    negative = sum(r for d, r in zip(ordered, ranks) if d < 0)
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def summarise_ratings(records: list[dict]) -> dict:
    values = {c: [r["rating"][c] for r in records if c in r["rating"]]
              for c in CONDITIONS}
    summary = {}
    for condition, scores in values.items():
        summary[condition] = {
            "n_ratings": len(scores),
            "mean": round(statistics.mean(scores), 3) if scores else None,
            "median": statistics.median(scores) if scores else None,
            "sd": round(statistics.stdev(scores), 3) if len(scores) > 1 else None,
            "distribution": {str(v): c for v, c in
                             sorted(Counter(scores).items())},
        }
    both = [r for r in records if len(r["rating"]) == 2]
    summary["complete_pairs"] = len(both)
    summary["mean_difference_scaffolded_minus_answer_giving"] = (
        round(statistics.mean([r["rating"][SCAFFOLDED] - r["rating"][ANSWER_GIVING]
                               for r in both]), 3) if both else None
    )
    return summary


def preference_rates(records: list[dict], field: str) -> dict:
    answered = [r[field] for r in records if r[field]]
    counts = Counter(answered)
    total = len(answered)
    return {
        "n_answered": total,
        "counts": {k: counts.get(k, 0) for k in (SCAFFOLDED, ANSWER_GIVING, SAME)},
        "rates": {k: round(counts.get(k, 0) / total, 3) if total else None
                  for k in (SCAFFOLDED, ANSWER_GIVING, SAME)},
    }


def divergence(records: list[dict], raters: list[dict]) -> dict:
    """Where "most helpful" and "what I'd want when stuck" disagree.

    Counted per rater, and broken out by direction: a rater who names the
    scaffolded response as more useful for learning but wants the answer-giving
    one when stressed is a different observation from the reverse, and the
    asymmetry between those two cells is the interesting number.
    """
    per_rater, flips = {}, Counter()
    for rater in raters:
        rid = rater["rater"]
        mine = [r for r in records
                if r["rater"] == rid and r["most_helpful"] and r["when_stuck"]]
        differing = [r for r in mine if r["most_helpful"] != r["when_stuck"]]
        for r in differing:
            flips[f"{r['most_helpful']} -> {r['when_stuck']}"] += 1
        per_rater[rid] = {
            "items_answering_both": len(mine),
            "items_diverging": len(differing),
            "divergence_rate": round(len(differing) / len(mine), 3) if mine else None,
            "diverging_items": [r["problem_id"] for r in differing],
        }
    comparable = sum(v["items_answering_both"] for v in per_rater.values())
    diverging = sum(v["items_diverging"] for v in per_rater.values())
    return {
        "item_level": {
            "n_comparable": comparable,
            "n_diverging": diverging,
            "rate": round(diverging / comparable, 3) if comparable else None,
        },
        "directions": dict(flips.most_common()),
        "learning_says_scaffolded_but_stressed_wants_answer": flips.get(
            f"{SCAFFOLDED} -> {ANSWER_GIVING}", 0),
        "learning_says_answer_but_stressed_wants_scaffolded": flips.get(
            f"{ANSWER_GIVING} -> {SCAFFOLDED}", 0),
        "raters_with_any_divergence": sum(
            1 for v in per_rater.values() if v["items_diverging"]),
        "per_rater": per_rater,
    }


def per_item(records: list[dict], key: dict) -> list[dict]:
    rows = []
    for number in range(1, key["n_items"] + 1):
        problem_id, item_key = key["_by_number"][number]
        mine = [r for r in records if r["form_item_number"] == number]
        scaffolded = [r["rating"][SCAFFOLDED] for r in mine if SCAFFOLDED in r["rating"]]
        answering = [r["rating"][ANSWER_GIVING] for r in mine
                     if ANSWER_GIVING in r["rating"]]
        helpful = preference_rates(mine, "most_helpful")
        stuck = preference_rates(mine, "when_stuck")
        rows.append({
            "form_item_number": number,
            "problem_id": problem_id,
            "subject": item_key.get("subject"),
            "topic": item_key.get("topic"),
            "scaffolded_letter": item_key["scaffolded_letter"],
            "n_raters": len(mine),
            "mean_scaffolded": round(statistics.mean(scaffolded), 3) if scaffolded else None,
            "mean_answer_giving": round(statistics.mean(answering), 3) if answering else None,
            "difference": (round(statistics.mean(scaffolded) - statistics.mean(answering), 3)
                           if scaffolded and answering else None),
            "most_helpful": helpful["counts"],
            "when_stuck": stuck["counts"],
            "comments": [r["comment"] for r in mine if r["comment"]],
        })
    return rows


def rater_level_pairs(records: list[dict], raters: list[dict]) -> list[tuple[float, float]]:
    pairs = []
    for rater in raters:
        mine = [r for r in records
                if r["rater"] == rater["rater"] and len(r["rating"]) == 2]
        if not mine:
            continue
        pairs.append((
            statistics.mean([r["rating"][SCAFFOLDED] for r in mine]),
            statistics.mean([r["rating"][ANSWER_GIVING] for r in mine]),
        ))
    return pairs


def item_level_pairs(records: list[dict], key: dict) -> list[tuple[float, float]]:
    pairs = []
    for number in range(1, key["n_items"] + 1):
        mine = [r for r in records
                if r["form_item_number"] == number and len(r["rating"]) == 2]
        if not mine:
            continue
        pairs.append((
            statistics.mean([r["rating"][SCAFFOLDED] for r in mine]),
            statistics.mean([r["rating"][ANSWER_GIVING] for r in mine]),
        ))
    return pairs


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

def fmt(value, width: int = 6, places: int = 2) -> str:
    if value is None:
        return "—".rjust(width)
    # `round(3.0, 3)` is an int, so a mean that lands exactly on a whole number
    # arrives here as one. Format every number the same way or the columns break
    # on precisely the rows most likely to be read closely.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):{width}.{places}f}"
    return str(value).rjust(width)


def p_value_text(p) -> str:
    if p is None:
        return "not run"
    return "p < 0.001" if p < 0.001 else f"p = {p:.4f}"


def print_report(results: dict) -> None:
    rule = "=" * 78
    print(rule)
    print("RQ2 — scaffolded vs answer-giving tutoring responses")
    print(rule)
    meta = results["meta"]
    print(f"responses     {meta['responses_file']}")
    print(f"answer key    {meta['answer_key_file']} (frozen {meta['frozen_on']})")
    print(f"raters        {meta['n_raters']}     items {meta['n_items']}")
    print(f"length ratio  {meta['measured_length_ratio']}× "
          "(answer-giving / scaffolded, measured on the frozen set)")
    print()

    print("-" * 78)
    print("HELPFULNESS (1-5, higher is more helpful)")
    print("-" * 78)
    ratings = results["ratings"]
    print(f"{'condition':<16}{'n':>6}{'mean':>8}{'median':>8}{'sd':>8}   distribution")
    for condition in CONDITIONS:
        row = ratings[condition]
        dist = " ".join(f"{k}:{v}" for k, v in row["distribution"].items())
        print(f"{condition:<16}{row['n_ratings']:>6}{fmt(row['mean'], 8)}"
              f"{fmt(row['median'], 8)}{fmt(row['sd'], 8)}   {dist}")
    print(f"\nmean difference (scaffolded - answer-giving): "
          f"{ratings['mean_difference_scaffolded_minus_answer_giving']}"
          f"   over {ratings['complete_pairs']} complete rating pairs")
    print()

    print("-" * 78)
    print("SIGNIFICANCE — rater-level pairing (primary)")
    print("-" * 78)
    for label, block in (("rater-level (primary)", results["tests"]["rater_level"]),
                         ("item-level (secondary)", results["tests"]["item_level"])):
        print(f"{label}: n = {block['n_pairs']} pairs, "
              f"scaffolded {block['mean_scaffolded']} vs "
              f"answer-giving {block['mean_answer_giving']} "
              f"(difference {block['mean_difference']})")
        w, t = block["wilcoxon"], block["paired_t"]
        if w["note"]:
            print(f"    Wilcoxon signed-rank: {w['note']}")
        else:
            print(f"    Wilcoxon signed-rank: W = {w['statistic']}, "
                  f"{p_value_text(w['p_value'])}, rank-biserial r = "
                  f"{w['rank_biserial_r']}")
        if t["note"]:
            print(f"    paired t-test:        {t['note']}")
        else:
            print(f"    paired t-test:        t = {t['statistic']:.3f}, "
                  f"{p_value_text(t['p_value'])}, Cohen's dz = {t['cohens_dz']}")
        print()

    print("-" * 78)
    print("PREFERENCE")
    print("-" * 78)
    for label, block in (("Q5  most helpful for understanding", results["most_helpful"]),
                         ("Q8  what I'd want when stuck", results["when_stuck"])):
        print(f"{label}  (n = {block['n_answered']})")
        for condition in (SCAFFOLDED, ANSWER_GIVING, SAME):
            rate = block["rates"][condition]
            count = block["counts"][condition]
            bar = "█" * int(round((rate or 0) * 40))
            print(f"    {condition:<16}{count:>4}  "
                  f"{'' if rate is None else f'{rate:5.1%}'}  {bar}")
        print()

    print("-" * 78)
    print("Q5 / Q8 DIVERGENCE — the same rater wanting different things")
    print("-" * 78)
    div = results["divergence"]
    item_level = div["item_level"]
    rate = item_level["rate"]
    rate_text = "" if rate is None else f" ({rate:.1%})"
    print(f"items where a rater's two answers disagree: "
          f"{item_level['n_diverging']}/{item_level['n_comparable']}{rate_text}")
    print(f"raters showing any divergence: {div['raters_with_any_divergence']}"
          f"/{meta['n_raters']}")
    print(f"    named scaffolded most helpful, wanted answer-giving when stuck: "
          f"{div['learning_says_scaffolded_but_stressed_wants_answer']}")
    print(f"    named answer-giving most helpful, wanted scaffolded when stuck: "
          f"{div['learning_says_answer_but_stressed_wants_scaffolded']}")
    if div["directions"]:
        print("    all directions:")
        for direction, count in div["directions"].items():
            print(f"        {direction:<42}{count}")
    print()

    print("-" * 78)
    print("PER ITEM")
    print("-" * 78)
    print(f"{'#':>2} {'problem_id':<20}{'n':>4}{'scaf':>7}{'ans':>7}{'diff':>7}"
          f"   {'Q5 s/a/=':<12}{'Q8 s/a/=':<12}")
    for row in results["per_item"]:
        helpful = row["most_helpful"]
        stuck = row["when_stuck"]
        print(f"{row['form_item_number']:>2} {row['problem_id']:<20}"
              f"{row['n_raters']:>4}{fmt(row['mean_scaffolded'], 7)}"
              f"{fmt(row['mean_answer_giving'], 7)}{fmt(row['difference'], 7)}"
              f"   {f'{helpful[SCAFFOLDED]}/{helpful[ANSWER_GIVING]}/{helpful[SAME]}':<12}"
              f"{f'{stuck[SCAFFOLDED]}/{stuck[ANSWER_GIVING]}/{stuck[SAME]}':<12}")
    print()

    print("-" * 78)
    print("OVERALL PREFERENCE AND BACKGROUND")
    print("-" * 78)
    for label, counts in (("overall style preference", results["overall_preference"]),
                          ("grade or year", results["background"]["grade"]),
                          ("relationship with math",
                           results["background"]["math_relationship"])):
        print(f"{label}:")
        for value, count in counts.items():
            print(f"    {value or '(blank)':<52}{count}")
        print()

    print(rule)
    print("Reminder for the write-up: the conditions are not length-matched "
          f"({meta['measured_length_ratio']}×).")
    print("The overall-preference question asks about letters (A/B), which were "
          "shuffled per item, so it measures a rater's impression, not a "
          "condition. Read it alongside Q5/Q8, not instead of them.")
    print(rule)


# ------------------------------------------------------------------

def analyse(responses_path: str, key_path: str) -> dict:
    key = load_key(key_path)
    records, raters = read_responses(responses_path, key)

    results = {
        "meta": {
            "generated_on": date.today().isoformat(),
            "responses_file": display_path(responses_path),
            "answer_key_file": display_path(key_path),
            "frozen_on": key["frozen_on"],
            "measured_length_ratio": key[
                "measured_length_ratio_answer_giving_over_scaffolded"],
            "n_raters": len(raters),
            "n_items": key["n_items"],
            "n_records": len(records),
            "unit_of_analysis": "rater (primary), item (secondary)",
        },
        "ratings": summarise_ratings(records),
        "tests": {
            "rater_level": paired_tests(rater_level_pairs(records, raters)),
            "item_level": paired_tests(item_level_pairs(records, key)),
        },
        "most_helpful": preference_rates(records, "most_helpful"),
        "when_stuck": preference_rates(records, "when_stuck"),
        "divergence": divergence(records, raters),
        "per_item": per_item(records, key),
        "overall_preference": dict(
            Counter(r["overall_preference"] for r in raters).most_common()),
        "background": {
            "grade": dict(Counter(r["grade"] for r in raters).most_common()),
            "math_relationship": dict(
                Counter(r["math_relationship"] for r in raters).most_common()),
        },
    }
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--responses", required=True,
                    help="CSV export of the form's responses.")
    ap.add_argument("--key", default=DEFAULT_KEY,
                    help="Unblinding key written by eval/generate_rq2_form.py.")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="Where to write the results JSON.")
    ap.add_argument("--quiet", action="store_true",
                    help="Write the results file without printing the report.")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    results = analyse(args.responses, args.key)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    if not args.quiet:
        print_report(results)
    print(f"\nWrote {display_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
