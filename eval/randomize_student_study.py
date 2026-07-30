#!/usr/bin/env python3.11
"""Create concealed, reproducible blocked assignment for an approved study.

The allocation file is identifiable research infrastructure and must be stored
outside the public repository with institution-approved access controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--block-columns",
        nargs="+",
        default=["site_id", "classroom_id", "pretest_band"],
    )
    args = parser.parse_args()
    with args.enrollment.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"study_id", *args.block_columns}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    if len({row["study_id"] for row in rows}) != len(rows):
        raise SystemExit("study_id values must be unique")

    blocks = defaultdict(list)
    for row in rows:
        blocks[tuple(row[column] for column in args.block_columns)].append(row)
    rng = random.Random(args.seed)
    assigned = []
    for block, members in sorted(blocks.items()):
        rng.shuffle(members)
        start = rng.randrange(2)
        for index, row in enumerate(members):
            row = dict(row)
            row["condition"] = (
                "aria" if (index + start) % 2 == 0 else "active_control"
            )
            row["block_id"] = hashlib.sha256(
                "\x1f".join(block).encode()
            ).hexdigest()[:12]
            assigned.append(row)
    rng.shuffle(assigned)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assigned[0]))
        writer.writeheader()
        writer.writerows(assigned)
    print(f"Wrote concealed assignment for {len(assigned)} participants.")
    print(f"SHA-256: {hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
