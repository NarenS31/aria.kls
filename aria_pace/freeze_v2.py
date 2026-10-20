"""C5 step 1 (SPEC A3.6): freeze everything the single 2012 evaluation will use. Reads NO 2012 data. Writes results/frozen_v2.json (+ .sha256)."""
import hashlib
import json

from .constants import RESULTS


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    aff = json.load(open(RESULTS / "affect_val.json"))
    frozen = {
        "note": "Frozen before any 2012 test row is read (SPEC A3.6). Do not edit.",
        "cross_dataset_a": {
            "policy_and_calibration_source": "results/frozen_val.json", "frozen_val_sha256": sha(RESULTS / "frozen_val.json"),
            "models_fitted_on": "2009 train split only (as in run_test.py); PFA/BKT/head hyperparameters from frozen_val.json",
            "primary_risk_score": "bkt", "dkt_sakt_promotion_decision": "keep BKT (neither promoted; results/dkt_val.json, results/sakt_val.json)",
            "dkt": {"weights": "results/dkt_2009.pt", "sha256": sha(RESULTS / "dkt_2009.pt"), "meta": "results/dkt_2009_meta.json", "meta_sha256": sha(RESULTS / "dkt_2009_meta.json")},
            "sakt": {"weights": "results/sakt_2009.pt", "sha256": sha(RESULTS / "sakt_2009.pt"), "meta": "results/sakt_2009_meta.json", "meta_sha256": sha(RESULTS / "sakt_2009_meta.json")},
            "kc_mapping": "2012 skill_id as integer string; missing -> UNK; KCs unseen in 2009 use pooled parameters (BKT/PFA) or the UNK id (DKT/SAKT)",
            "bootstrap_seed": 20261025,
        },
        "within_dataset_b": {
            "split": "results/split_2012.json", "split_sha256": sha(RESULTS / "split_2012.json"),
            "validation_selection": "results/affect_val.json", "affect_val_sha256": sha(RESULTS / "affect_val.json"),
            "head_C": {n: aff["selection"][n]["C"] for n in ("without", "with_primary", "with_lagged")},
            "rank_transform": "per-column mid-rank ECDF fitted on 2012-train skill-builder rows",
            "primary_endpoint": "mean over k in {3,5,8} of AUC(with) - AUC(without), variant A, rows <= k; lagged (rows <= k-1) as sensitivity",
            "rule": "affect adds information only if CI lower bound > 0 AND point estimate >= 0.005; primary/lagged disagreement = not robust",
            "bootstrap_seed": 20261026, "bootstrap_draws": 1000,
        },
        "pre_registered_endpoints": ["a-P1 AUC of BKT at k=3,5,8", "a-P2 quiet-profile intervention reduction vs always-help and missed-help share",
                                     "a-P3 J difference to always-help in three profiles", "b-P1 as above"],
    }
    text = json.dumps(frozen, indent=2, sort_keys=True)
    (RESULTS / "frozen_v2.json").write_text(text)
    (RESULTS / "frozen_v2.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
    print(hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
