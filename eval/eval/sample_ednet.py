#!/usr/bin/env python3.11
"""Draw a fixed random sample from the EdNet/Riiid train.csv without loading it.

    python3.11 eval/eval/sample_ednet.py

train.csv is ~5.8 GB and ~101M rows. Nothing downstream may load it, so this
runs once, streams it a single time, and writes a sample that every later step
uses instead.

Reservoir sampling (Vitter's Algorithm L) gives a uniform random sample of a
stream whose length is not known in advance, in one pass and constant memory.
Algorithm L rather than the naive reservoir because the naive form draws a
random number for every one of the 101M rows; Algorithm L draws one per
*accepted* row — about k·ln(n/k) ≈ 2.6M draws instead of 101M — by sampling the
gap to the next acceptance directly.

Memory is bounded by the sample, not the file: 500k lines of ~50 bytes is about
25 MB. Peak RSS is measured and reported so the claim is checked, not asserted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EVAL_ROOT)

DEFAULT_SOURCE = os.path.join(
    EVAL_ROOT, "data", "external", "ednet_riiid", "raw", "train.csv")
DEFAULT_OUT = os.path.join(
    EVAL_ROOT, "data", "external", "ednet_riiid", "sample_500k.csv")
DEFAULT_K = 500_000
DEFAULT_SEED = 20260823


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB (ru_maxrss is bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def reservoir_sample(path: str, k: int, seed: int) -> tuple[bytes, list[bytes], int]:
    """Return (header, sampled lines, total data rows). One streaming pass.

    Lines are kept as raw bytes: they are written straight back out, so decoding
    and re-encoding 500k rows would be work done for nothing.
    """
    rng = random.Random(seed)
    reservoir: list[bytes] = []
    total = 0
    started = time.time()

    with open(path, "rb") as fh:
        header = fh.readline()

        # Fill the reservoir with the first k rows.
        for line in fh:
            reservoir.append(line)
            total += 1
            if total >= k:
                break

        if total < k:
            return header, reservoir, total

        # Algorithm L. `w` shrinks geometrically; the gap to the next accepted
        # row is drawn from it, so most rows are only counted, never sampled.
        w = math.exp(math.log(rng.random()) / k)
        next_hit = total + math.floor(math.log(rng.random()) / math.log1p(-w)) + 1

        for line in fh:
            total += 1
            if total == next_hit:
                reservoir[rng.randrange(k)] = line
                w *= math.exp(math.log(rng.random()) / k)
                next_hit = total + math.floor(
                    math.log(rng.random()) / math.log1p(-w)) + 1
            if total % 20_000_000 == 0:
                print(f"    ...{total:,} rows scanned "
                      f"({time.time() - started:.0f}s, peak RSS "
                      f"{peak_rss_mb()} MB)", flush=True)

    return header, reservoir, total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if not os.path.exists(args.source):
        print(f"ERROR: {args.source} not found.", file=sys.stderr)
        return 1

    size_bytes = os.path.getsize(args.source)
    print(f"source      {args.source}")
    print(f"size        {size_bytes:,} bytes ({size_bytes / 1e9:.2f} GB)")
    print(f"sampling    k={args.k:,}  seed={args.seed}  "
          f"(reservoir, single streaming pass)")
    started = time.time()

    header, sample, total = reservoir_sample(args.source, args.k, args.seed)
    elapsed = time.time() - started

    if total < args.k:
        print(f"WARNING: file has {total:,} data rows, fewer than k={args.k:,}. "
              "Sample is the whole file.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        fh.write(header)
        fh.writelines(sample)

    # Provenance sidecar, so downstream steps can report the population this
    # sample stands for without re-reading (or hardcoding) the 5.8 GB source.
    sidecar = os.path.splitext(args.out)[0] + "_meta.json"
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump({
            "source_file": os.path.relpath(args.source, REPO_ROOT),
            "source_size_bytes": size_bytes,
            "population_rows": total,
            "sampled_rows": len(sample),
            "sampling_fraction": round(len(sample) / total, 8),
            "method": "reservoir sampling (Vitter Algorithm L), single "
                      "streaming pass, uniform without replacement",
            "seed": args.seed,
            "peak_rss_mb": peak_rss_mb(),
            "elapsed_seconds": round(elapsed, 1),
        }, fh, indent=2)
        fh.write("\n")

    out_size = os.path.getsize(args.out)
    print()
    print(f"total data rows streamed   {total:,}")
    print(f"rows sampled               {len(sample):,}")
    print(f"sampling fraction          {len(sample) / total:.6f}")
    print(f"elapsed                    {elapsed:.1f}s "
          f"({total / elapsed / 1e6:.1f}M rows/s)")
    print(f"peak RSS                   {peak_rss_mb()} MB  "
          f"(full file is {size_bytes / 1e9:.2f} GB — never held in memory)")
    print(f"wrote                      {args.out} ({out_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
