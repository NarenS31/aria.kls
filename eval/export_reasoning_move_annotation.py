#!/usr/bin/env python3.11
"""Export a private, prediction-blind packet from real Eedi dialogue.

The output contains licensed real dialogue and must remain in the ignored
research/packets/private directory unless a data-sharing review permits more.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.reasoning_moves import MOVE_DEFINITIONS  # noqa: E402
DEFAULT_SOURCE = (
    ROOT / "eval" / "data" / "external" / "eedi_dialogues" / "raw" / "test.csv"
)
PRIVATE_DIR = ROOT / "research" / "packets" / "private"


def deidentify(text: str) -> str:
    text = re.sub(
        r"\b[A-Z][a-z]{2,}\b(?=,|!|\s+(?:how|are|can|could|would)\b)",
        "[NAME]",
        text,
    )
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"\b(?:\+?\d[\d ()-]{7,}\d)\b", "[PHONE]", text)
    return text


def load_windows(path: Path) -> list[dict]:
    sessions = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sessions[row["InterventionId"]].append(row)
    windows = []
    for intervention_id, rows in sessions.items():
        rows.sort(key=lambda row: int(row["MessageSequence"]))
        for index, row in enumerate(rows):
            if row["IsTutor"].strip().lower() not in {"0", "false"}:
                continue
            context_rows = rows[max(0, index - 2):index]
            context = "\n".join(
                f"{'TUTOR' if prior['IsTutor'].strip().lower() in {'1', 'true'} else 'STUDENT'}: "
                f"{deidentify(prior['MessageString'])}"
                for prior in context_rows
            )
            windows.append({
                "intervention_id": intervention_id,
                "question_id": row["QuestionId_DQ"],
                "context": context,
                "student_text": deidentify(row["MessageString"]),
            })
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=PRIVATE_DIR / "moves_rater.csv")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    windows = load_windows(args.source)
    rng = random.Random(args.seed)
    rng.shuffle(windows)
    selected = windows[:min(args.n, len(windows))]
    rows = []
    for index, row in enumerate(selected, 1):
        packet_row = {
            "item_id": f"MOVE-{index:04d}",
            **row,
            "unclear_yes_no": "",
            "annotator_notes": "",
        }
        for code in MOVE_DEFINITIONS:
            packet_row[f"{code}_yes_no"] = ""
            packet_row[f"{code}_evidence"] = ""
        rows.append(packet_row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"Wrote {len(rows)} prediction-blind items to {args.output}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
