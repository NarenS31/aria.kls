"""Amendment A6: run the parametric closed-loop sensitivity grid. ASSUMPTION-DEPENDENT, NOT EVIDENCE OF EFFECT.  python -m aria_pace.run_closed_loop"""
import json
import time

from . import closed_loop as CL
from .constants import RESULTS


def main():
    t0 = time.time()
    out = CL.run_grid(log=lambda s: print(f"[{time.time() - t0:4.0f}s] {s}", flush=True))
    out["env"] = {"seconds": round(time.time() - t0)}
    with open(RESULTS / "closed_loop_sensitivity.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote results/closed_loop_sensitivity.json ({len(out['points'])} points)")


if __name__ == "__main__":
    main()
