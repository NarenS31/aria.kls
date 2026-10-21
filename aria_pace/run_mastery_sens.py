"""Amendment A4: mastery-definition sensitivity on 2009 TRAIN / VALIDATION only (exploratory; see SPEC.md A4).

    python -m aria_pace.run_mastery_sens          (from the repo root)

Step 0 is the pre-registered control check (A4.3): under D3 the pipeline must reproduce frozen_val.json (platt.bkt and configs.main.frozen.taus.bkt)
to 1e-9; otherwise the run stops before any other definition is computed. Writes results/mastery_sensitivity_val.json only. The 2009 test split is
dropped right after the split (as freeze_val.py:22-24) and the 2012 file is never loaded.
"""
import json
import sys
import time

from . import baselines as B
from . import mastery_sens as MS
from .constants import RAW_CSV, RESULTS
from .loader import build_sequences, load_raw
from .split import make_split

STATUS = "exploratory sensitivity (validation only; not pre-registered). Control check A4.3 is pre-registered."


def main():
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731
    df, _ = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique())
    test_ids = set(parts["test"])
    seqs = [q for q in build_sequences(df) if q.user_id not in test_ids]
    train_ids = set(parts["train"])
    train = [q for q in seqs if q.user_id in train_ids]
    val = [q for q in seqs if q.user_id not in train_ids]
    assert not {q.user_id for q in seqs} & test_ids
    sel = json.load(open(RESULTS / "baselines_val.json"))["selection"]["chosen"]
    bkt = B.BKT(sel["bkt_min_seq"]).fit(train)
    log(f"train sequences {len(train)}, validation {len(val)}; BKT fitted")

    frozen = json.load(open(RESULTS / "frozen_val.json"))
    ftau, fplatt = frozen["configs"]["main"]["frozen"]["taus"]["bkt"], frozen["platt"]["bkt"]
    out = {"status": STATUS, "spec": "aria_pace/SPEC.md Amendment 4", "definitions": {}}

    control = MS.run_definition(val, bkt, MS.DEFS[0])
    ok_platt = abs(control["platt"]["intercept"] - fplatt["intercept"]) < 1e-9 and abs(control["platt"]["slope"] - fplatt["slope"]) < 1e-9
    ok_tau = all(control["taus"][p] == ftau[p] for p in ftau)
    out["control_check_A4_3"] = {"platt_equal_1e-9": bool(ok_platt), "taus_equal": bool(ok_tau),
                                 "control_platt": control["platt"], "frozen_platt": fplatt, "control_taus": control["taus"], "frozen_taus": ftau}
    log(f"CONTROL D3: platt equal={ok_platt} taus equal={ok_tau}  control taus {control['taus']} frozen {ftau}")
    if not (ok_platt and ok_tau):
        json.dump(out, open("/tmp/mastery_sens_control_FAILED.json", "w"), indent=2)
        sys.exit("CONTROL CHECK FAILED (A4.3): D3 does not reproduce frozen_val.json. Stopping before any other definition.")
    out["definitions"]["D3"] = control
    for d in MS.DEFS[1:]:
        out["definitions"][d.name] = MS.run_definition(val, bkt, d)
        r = out["definitions"][d.name]
        log(f"{d.name}: label_rate={r['scope']['label_rate']:.3f} n={r['scope']['in_scope_labelled_sequences']} taus={r['taus']}")
    out["env"] = {"seconds": round(time.time() - t0)}
    with open(RESULTS / "mastery_sensitivity_val.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    log("wrote results/mastery_sensitivity_val.json")


if __name__ == "__main__":
    main()
