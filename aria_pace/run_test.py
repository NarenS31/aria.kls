"""Stage 5b: ONE-TIME test evaluation with the frozen validation values (SPEC A2.4).

python -m aria_pace.run_test            evaluates test once; refuses if results/TEST_LOCK exists.
python -m aria_pace.run_test --verify   recomputes and compares with results/test_results.json, writes nothing.
Nothing is chosen on test: taus, n, calibrators and model hyperparameters come from results/frozen_val.json.
"""
import datetime
import hashlib
import json
import sys

from . import stage5 as S5
from .constants import RAW_CSV, RESULTS
from .loader import build_sequences, load_raw
from .split import make_split


def main(argv):
    verify = "--verify" in argv
    lock = RESULTS / "TEST_LOCK"
    if lock.exists() and not verify:
        sys.exit("TEST_LOCK exists: the test split was already evaluated. Use --verify to recompute without writing.")
    text = (RESULTS / "frozen_val.json").read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != (RESULTS / "frozen_val.sha256").read_text().strip():
        sys.exit("frozen_val.json does not match frozen_val.sha256")
    frozen = json.loads(text)
    if not verify:
        lock.write_text(json.dumps({"started_utc": datetime.datetime.utcnow().isoformat(), "frozen_sha256": sha,
                                    "status": "started"}, indent=2))
    df, _ = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique())
    seqs = build_sequences(df)
    train_ids, test_ids = set(parts["train"]), set(parts["test"])
    train = [q for q in seqs if q.user_id in train_ids]
    test = [q for q in seqs if q.user_id in test_ids]
    assert not {q.user_id for q in train} & {q.user_id for q in test}
    m = frozen["models"]
    models = S5.fit_models(train, m["pfa_C"], m["bkt_min_seq"], m["head_C"])
    res = S5.evaluate_split(test, frozen, models)
    res["_frozen_sha256"] = sha
    res["_note"] = "Single evaluation with frozen validation values. Test-sweep optimal taus are diagnostic only."
    path = RESULTS / "test_results.json"
    if verify:
        old = json.loads(path.read_text())
        same = json.dumps(old, sort_keys=True) == json.dumps(json.loads(json.dumps(res)), sort_keys=True)
        print("VERIFY identical:", same)
        return
    with open(path, "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    lock.write_text(json.dumps({"started_utc": json.loads(lock.read_text())["started_utc"],
                                "completed_utc": datetime.datetime.utcnow().isoformat(),
                                "frozen_sha256": sha, "status": "completed"}, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1:])
