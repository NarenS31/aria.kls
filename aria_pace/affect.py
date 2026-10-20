"""Rank-transformed detector-affect features (SPEC Amendment 3, A3.4)."""
import numpy as np


class RankTransform:
    """Per-column mid-rank empirical CDF fitted on TRAIN rows only; applied unchanged to validation/test. Output in [0, 1]."""

    def fit(self, values):                                         # values: (N, 4)
        self.cols = [np.sort(values[:, j].astype(np.float64)) for j in range(values.shape[1])]
        return self

    def transform(self, values):
        out = np.empty(values.shape, dtype=np.float32)
        for j, col in enumerate(self.cols):
            v = values[:, j].astype(np.float64)
            lo, hi = np.searchsorted(col, v, side="left"), np.searchsorted(col, v, side="right")
            out[:, j] = ((lo + hi) / 2.0) / len(col)
        return out


def cumulative(a):
    """Per-row running mean and running max of an (n, 4) array: row r summarises rows 0..r."""
    n = len(a)
    return np.cumsum(a, axis=0) / np.arange(1, n + 1)[:, None], np.maximum.accumulate(a, axis=0)


def affect_features(a, cm, cx, i, lag=0):
    """12 features at prediction step i from rows <= i-lag (1-based): per column last value, mean, max. lag=0: rows <= i; lag=1: rows <= i-1."""
    r = i - lag - 1                                                # 0-based index of the last row used
    return np.concatenate([a[r], cm[r], cx[r]])


AFFECT_FEATURE_NAMES = [f"{col}_{stat}" for stat in ("last", "mean", "max") for col in ("FRUSTRATED", "CONFUSED", "CONCENTRATING", "BORED")]
