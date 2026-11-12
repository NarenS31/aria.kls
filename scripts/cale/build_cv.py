"""Build (or reuse) the cached pooled CV held-out predictions."""
import common as C
import cv_heldout as CVH

if __name__ == "__main__":
    log = C.log_fn()
    _, parts, train, val, _ = C.load_pool(log)
    d = CVH.get(train, val, log)
    C.assert_no_test(d["H"].sid, parts)
    log(f"pooled held-out: {len(d['H'].y)} sequences, {len(set(d['H'].sid.tolist()))} students")
