"""Risk head: L2 logistic regression on windowed features of the first i responses (SPEC A1.8)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["i", "n_correct", "frac_correct", "c_i", "c_i-1", "c_i-2", "wrong_last3", "wrong_last5",
            "correct_run", "wrong_run", "longest_wrong_run"]


def features(correct, i):
    x = np.asarray(correct[:i], dtype=float)
    w = 1.0 - x
    cr = 0
    for v in x[::-1]:
        if v == 1: cr += 1
        else: break
    cw = 0
    for v in x[::-1]:
        if v == 0: cw += 1
        else: break
    lw = run = 0
    for v in w:
        run = run + 1 if v == 1 else 0
        lw = max(lw, run)
    return [i, x.sum(), x.mean(), x[-1], x[-2], x[-3], w[-3:].sum(), w[-5:].sum(), cr, cw, lw]


class RiskHead:
    name = "risk_head"

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(self.scaler.transform(X), y)
        return self

    def predict_rows(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]

    def score(self, seq, i):
        return float(self.predict_rows(np.array([features(seq.correct, i)]))[0])

    def coefficients(self):
        return {f: float(c) for f, c in zip(FEATURES, self.clf.coef_[0])} | {"intercept": float(self.clf.intercept_[0])}
