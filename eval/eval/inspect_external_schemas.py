#!/usr/bin/env python3.11
"""Inspect the raw schema of every external dataset before any mapping.

    python3.11 eval/eval/inspect_external_schemas.py
    python3.11 eval/eval/inspect_external_schemas.py --quiet   # summary table only

Nothing here maps anything. It reads each file's header and first rows, prints
the columns with their inferred types and real sample values, and proposes which
column (if any) could carry each of the seven fields the behavioral proxy needs.
Those proposals are evidence for writing the adapters, not the mapping itself —
the adapters name their columns explicitly.

The datasets do NOT share a schema. ASSISTments 2004 and 2005 are per-problem
tutor logs; 2015 is a four-column correctness export; Aleven Fractions is
DataShop step-level tutor transactions; OULAD is a distance-learning VLE with
assessment scores and click counts and no per-problem interaction at all; EdNet
and Eedi are single-answer question platforms with no retries and no hints. What
each one can and cannot support is the point of this report.

Row counts are exact and come from a buffered byte scan, not from loading the
file. EdNet is reported from its 500k sample plus the population size recorded
when the sample was drawn; its 5.8 GB source is never opened here.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EVAL_ROOT)
EXTERNAL = os.path.join(EVAL_ROOT, "data", "external")

csv.field_size_limit(10_000_000)

# The seven fields the proxy rules read, and the words that hint at each. Order
# matters: the first pattern that matches a column name wins for that field.
FIELD_HINTS = {
    "student_id": (r"anon.*student", r"^student.?id$", r"^user.?id$",
                   r"id_student", r"^student$", r"student", r"user"),
    "problem_id": (r"^item.?id$", r"^question.?id$", r"^problem.?name$",
                   r"^problem.?id$", r"content_id", r"id_assessment",
                   r"step.?name", r"sequence_id", r"id_site", r"problem",
                   r"question", r"item"),
    "correct": (r"correctonfirstattempt", r"^is.?correct$", r"answered_correctly",
                r"^outcome$", r"^correct$", r"^score$", r"^response$",
                r"correct", r"score", r"grade"),
    "attempt_count": (r"attempt.?at.?step", r"^attempt.?count$", r"^attempts?$",
                      r"num_of_prev_attempts", r"attempt"),
    "hint_count": (r"total.?num.?hints", r"^hint.?count$", r"^hints?$",
                   r"help.?level", r"hint", r"help"),
    "response_time": (r"^duration", r"total_seconds", r"^seconds$",
                      r"elapsed_time", r"^timestamp$", r"date.?answered",
                      r"^time$", r"date_submitted", r"time"),
    "skill_topic": (r"^skill.?name$", r"^skill$", r"curriculum_id",
                    r"subject", r"activity_type", r"level\(", r"topic",
                    r"knowledge"),
}


class FileSpec:
    def __init__(self, path: str, role: str = "", note: str = "",
                 count_rows: bool = True):
        self.path = path
        self.role = role
        self.note = note
        self.count_rows = count_rows


class DatasetSpec:
    def __init__(self, key: str, label: str, platform: str, year: str,
                 files: list[FileSpec], primary: str, note: str = ""):
        self.key = key
        self.label = label
        self.platform = platform
        self.year = year
        self.files = files
        self.primary = primary          # basename of the transaction table
        self.note = note


def _p(*parts: str) -> str:
    return os.path.join(EXTERNAL, *parts)


DATASETS = [
    DatasetSpec(
        "assistments_2004", "ASSISTments 2004-05", "ASSISTments", "2004",
        [FileSpec(_p("assistments_2004", "raw", "dataframe3_aggr.txt.output.txt"),
                  "per-problem tutor log")],
        "dataframe3_aggr.txt.output.txt"),
    DatasetSpec(
        "assistments_2005", "ASSISTments 2005-06", "ASSISTments", "2005",
        [FileSpec(_p("assistments_2005", "raw", "dataframe3_aggr_06.txt.output.txt"),
                  "per-problem tutor log")],
        "dataframe3_aggr_06.txt.output.txt"),
    DatasetSpec(
        "assistments_2015", "ASSISTments 2015 skill builders", "ASSISTments", "2015",
        [FileSpec(_p("assistments_2015", "raw",
                     "2015_100_skill_builders_main_problems.csv"),
                  "correctness export")],
        "2015_100_skill_builders_main_problems.csv"),
    DatasetSpec(
        "aleven_fractions", "Aleven Fractions Lab (DataShop)", "Fractions Lab",
        "2012-13",
        [FileSpec(_p("aleven_fractions", "raw",
                     "All_Data_1826_2013_0408_075715_forReimport.txt"),
                  "DataShop step-level transactions")],
        "All_Data_1826_2013_0408_075715_forReimport.txt"),
    DatasetSpec(
        "oulad", "Open University Learning Analytics", "OULAD", "2013-14",
        [FileSpec(_p("oulad", "raw", "studentInfo.csv"), "one row per enrolment"),
         FileSpec(_p("oulad", "raw", "studentAssessment.csv"),
                  "assessment submissions (scores)"),
         FileSpec(_p("oulad", "raw", "assessments.csv"), "assessment due dates"),
         FileSpec(_p("oulad", "raw", "studentVle.csv"),
                  "daily click counts per VLE resource"),
         FileSpec(_p("oulad", "raw", "studentRegistration.csv"), "registration dates"),
         FileSpec(_p("oulad", "raw", "vle.csv"), "VLE resource types"),
         FileSpec(_p("oulad", "raw", "courses.csv"), "module lengths")],
        "studentAssessment.csv",
        "Distance-learning VLE. No per-problem interaction exists anywhere in "
        "these tables."),
    DatasetSpec(
        "ednet_riiid", "EdNet / Riiid AIEd Challenge", "EdNet (Santa)", "2020",
        [FileSpec(_p("ednet_riiid", "sample_500k.csv"),
                  "500k uniform random sample of train.csv")],
        "sample_500k.csv",
        "Full train.csv (5.85 GB, 101,230,332 rows) is never opened after "
        "sampling."),
    DatasetSpec(
        "eedi_neurips2020", "Eedi NeurIPS 2020 Education Challenge", "Eedi", "2020",
        [FileSpec(_p("eedi_neurips2020", "raw", "train_data", "train_task_1_2.csv"),
                  "answer log (tasks 1 & 2)"),
         FileSpec(_p("eedi_neurips2020", "raw", "train_data", "train_task_3_4.csv"),
                  "answer log (tasks 3 & 4)"),
         FileSpec(_p("eedi_neurips2020", "raw", "metadata",
                     "answer_metadata_task_1_2.csv"),
                  "per-answer timestamp / confidence", count_rows=True),
         FileSpec(_p("eedi_neurips2020", "raw", "metadata",
                     "question_metadata_task_1_2.csv"), "question subjects"),
         FileSpec(_p("eedi_neurips2020", "raw", "metadata",
                     "student_metadata_task_1_2.csv"), "student demographics"),
         FileSpec(_p("eedi_neurips2020", "raw", "metadata", "subject_metadata.csv"),
                  "subject taxonomy")],
        "train_task_1_2.csv",
        "CC BY-NC-ND 4.0 — non-commercial use only."),
]


# ------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------

def sniff_delimiter(sample: str) -> str:
    first = sample.splitlines()[0] if sample.splitlines() else ""
    counts = {d: first.count(d) for d in ("\t", ",", ";", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def count_rows(path: str) -> int:
    """Exact data-row count via a buffered byte scan. Never loads the file."""
    total = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            total += chunk.count(b"\n")
    # Header row, and a trailing newline that does not start a row.
    with open(path, "rb") as fh:
        fh.seek(max(0, os.path.getsize(path) - 1))
        ends_with_newline = fh.read(1) == b"\n"
    return max(0, total - (1 if ends_with_newline else 0))


def read_head(path: str, n: int = 20) -> tuple[list[str], list[list[str]], str]:
    with open(path, "rb") as fh:
        raw = fh.read(1024 * 512)
    for encoding in ("utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
        encoding = "utf-8 (with replacements)"
    delimiter = sniff_delimiter(text)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = []
    for i, row in enumerate(reader):
        if i > n:
            break
        rows.append(row)
    header = rows[0] if rows else []
    body = rows[1:1 + n]
    # The final row of a truncated read is usually cut mid-line.
    if body and len(body[-1]) != len(header):
        body = body[:-1]
    return header, body, f"{encoding}, delimiter={delimiter!r}"


def infer_type(values: list[str]) -> str:
    seen = [v for v in values if v not in ("", "?", "NA", "null", "None")]
    if not seen:
        return "empty"
    def all_match(fn) -> bool:
        return all(fn(v) for v in seen)
    def is_int(v: str) -> bool:
        try:
            int(v)
            return True
        except ValueError:
            return False
    def is_float(v: str) -> bool:
        try:
            float(v)
            return True
        except ValueError:
            return False
    if all_match(is_int):
        lo, hi = min(int(v) for v in seen), max(int(v) for v in seen)
        if {lo, hi} <= {0, 1}:
            return "int(0/1)"
        return f"int[{lo}..{hi}]"
    if all_match(is_float):
        return "float"
    if all_match(lambda v: bool(re.match(
            r"^\d{4}-\d{2}-\d{2}|^\d{2}[-/]\w{3}[-/]\d{2}|^\d{1,2}/\d{1,2}/\d{2}", v))):
        return "datetime"
    if all_match(lambda v: v.upper() in {"TRUE", "FALSE", "Y", "N"}):
        return "bool-ish"
    return "str"


def propose_mapping(header: list[str]) -> dict[str, str | None]:
    """Best-guess column per canonical field. Advisory only."""
    proposal: dict[str, str | None] = {}
    used: set[str] = set()
    for field, patterns in FIELD_HINTS.items():
        match = None
        for pattern in patterns:
            for column in header:
                if not column:
                    continue
                if column in used:
                    continue
                if re.search(pattern, column.strip(), re.IGNORECASE):
                    match = column
                    break
            if match:
                break
        if match:
            used.add(match)
        proposal[field] = match
    return proposal


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

def inspect_file(spec: FileSpec, verbose: bool) -> dict:
    if not os.path.exists(spec.path):
        if verbose:
            print(f"  MISSING: {spec.path}")
        return {"path": os.path.relpath(spec.path, REPO_ROOT), "present": False}

    header, body, fmt = read_head(spec.path)
    n_rows = count_rows(spec.path) if spec.count_rows else None
    size = os.path.getsize(spec.path)
    proposal = propose_mapping(header)

    if verbose:
        print(f"\n  FILE  {os.path.relpath(spec.path, REPO_ROOT)}")
        print(f"        {spec.role} · {size:,} bytes · "
              f"{'{:,}'.format(n_rows) if n_rows is not None else '?'} data rows "
              f"· {fmt}")
        print(f"        {len(header)} columns")
        width = max((len(c) for c in header if c), default=10)
        width = min(max(width, 12), 32)
        for i, column in enumerate(header):
            values = [row[i] for row in body if i < len(row)]
            shown = ", ".join(repr(v)[:26] for v in values[:3]) or "(no rows read)"
            name = column if column else "(unnamed)"
            print(f"          [{i:>2}] {name[:32]:<{width}}  "
                  f"{infer_type(values):<14} {shown}")

    return {
        "path": os.path.relpath(spec.path, REPO_ROOT),
        "present": True,
        "role": spec.role,
        "size_bytes": size,
        "n_data_rows": n_rows,
        "format": fmt,
        "n_columns": len(header),
        "columns": header,
        "dtypes": {(c or f"col_{i}"): infer_type(
            [r[i] for r in body if i < len(r)])
            for i, c in enumerate(header)},
        "proposed_mapping": proposal,
    }


def inspect_dataset(ds: DatasetSpec, verbose: bool) -> dict:
    if verbose:
        print("\n" + "=" * 78)
        print(f"{ds.key}  —  {ds.label}  ({ds.platform}, {ds.year})")
        print("=" * 78)
        if ds.note:
            print(f"  NOTE: {ds.note}")

    files = [inspect_file(f, verbose) for f in ds.files]
    present = [f for f in files if f.get("present")]
    primary = next((f for f in present
                    if os.path.basename(f["path"]) == ds.primary), None)

    entry = {
        "key": ds.key, "label": ds.label, "platform": ds.platform,
        "year": ds.year, "note": ds.note,
        "files": files,
        "primary_file": primary["path"] if primary else None,
        "n_rows_primary": primary["n_data_rows"] if primary else None,
    }

    if ds.key == "ednet_riiid":
        meta_path = _p("ednet_riiid", "sample_500k_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                entry["sampling"] = json.load(fh)

    if primary and verbose:
        print(f"\n  PROPOSED FIELD MAPPING (from {os.path.basename(primary['path'])}):")
        for field, column in primary["proposed_mapping"].items():
            print(f"        {field:<16} {column if column else '— none found —'}")
    return entry


AVAILABILITY_OVERRIDES: dict[str, dict[str, str]] = {
    # Filled in after reading the proposals: a keyword match is not proof the
    # column means what the proxy needs. See each adapter for the final word.
    "assistments_2015": {
        "attempt_count": "absent", "hint_count": "absent",
        "response_time": "absent",
        "notes": "Only user_id/log_id/sequence_id/correct. No behavioral "
                 "columns of any kind.",
    },
    "oulad": {
        "attempt_count": "absent", "hint_count": "absent",
        "response_time": "absent",
        "notes": "Assessment-level scores only. `date_submitted` is a day "
                 "index vs a deadline, not a response time; "
                 "`num_of_prev_attempts` counts module retakes, not problem "
                 "attempts.",
    },
    "ednet_riiid": {
        "attempt_count": "platform=1", "hint_count": "absent",
        "response_time": "derived",
        "notes": "One row per answered question, no retries in the platform, "
                 "so attempts=1 is a fact about Santa rather than a guess. "
                 "`prior_question_elapsed_time` describes the PREVIOUS "
                 "question and must be shifted forward one row per user. "
                 "`prior_question_had_explanation` is not a hint request.",
    },
    "eedi_neurips2020": {
        "attempt_count": "platform=1", "hint_count": "absent",
        "response_time": "derived",
        "notes": "One row per answer, no retries and no hints. Timing only "
                 "from consecutive DateAnswered differences in the 1 GB "
                 "answer metadata; check granularity before trusting it.",
    },
}


def summary_row(entry: dict) -> dict:
    primary = next((f for f in entry["files"]
                    if f.get("present") and f["path"] == entry["primary_file"]), None)
    proposal = primary["proposed_mapping"] if primary else {}
    override = AVAILABILITY_OVERRIDES.get(entry["key"], {})

    def status(field: str) -> str:
        if field in override:
            return override[field]
        return "yes" if proposal.get(field) else "absent"

    return {
        "dataset": entry["key"],
        "n_rows": entry["n_rows_primary"],
        "correct": status("correct"),
        "hints": status("hint_count"),
        "timing": status("response_time"),
        "attempts": status("attempt_count"),
        "notes": override.get("notes", entry.get("note", "")),
    }


def print_summary(entries: list[dict]) -> None:
    rows = [summary_row(e) for e in entries]
    print("\n" + "=" * 78)
    print("SCHEMA SUMMARY — what each dataset actually carries")
    print("=" * 78)
    print(f"{'Dataset':<20}{'N rows':>13}  {'Correct':<9}{'Hints':<11}"
          f"{'Timing':<11}{'Attempts':<11}")
    print("-" * 78)
    for row in rows:
        n = f"{row['n_rows']:,}" if row["n_rows"] is not None else "?"
        print(f"{row['dataset']:<20}{n:>13}  {row['correct']:<9}"
              f"{row['hints']:<11}{row['timing']:<11}{row['attempts']:<11}")
    print("-" * 78)
    print("absent = the field does not exist in the source at all")
    print("platform=1 = one attempt per row is a property of the platform's log,")
    print("             not a value inferred from anything")
    print("derived = must be computed (shift or difference), see notes")
    print()
    for row in rows:
        if row["notes"]:
            print(f"  {row['dataset']}: {row['notes']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true",
                    help="Print the summary table only.")
    ap.add_argument("--out", default=os.path.join(
        EVAL_ROOT, "data", "eval", "external_schema_inspection.json"))
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    verbose = not args.quiet
    if verbose:
        print("SCHEMA INSPECTION — run before any mapping is attempted")
        print(f"external data root: {os.path.relpath(EXTERNAL, REPO_ROOT)}")

    entries = [inspect_dataset(ds, verbose) for ds in DATASETS]
    print_summary(entries)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report = {
        "generated_by": "eval/eval/inspect_external_schemas.py",
        "n_datasets": len(entries),
        "datasets": entries,
        "summary": [summary_row(e) for e in entries],
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"\nWrote {os.path.relpath(args.out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
