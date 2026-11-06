"""Router features (CURRENT8).

All eight features are functions of the retrieval score matrix alone.  Nothing
here reads the gold concept, a cross-encoder score, or the mention context, so
every feature is available before any reranking call is issued.

    s1                                the top-1 retrieval score
    margin_s1_s2                      s1 - s2
    margin_s1_s5                      s1 - s5
    std_top5                          standard deviation of the top-5 scores
    gap_s2_s3                         s2 - s3
    max_adj_gap_top5                  largest gap between adjacent top-5 scores
    near_top_count_top5               top-5 candidates within TIE_EPS of s1
    effective_competitor_count_top5   exp(H), H = Shannon entropy of softmax(top-5)

``near_top_count_top5`` counts the top-1 itself, so it is >= 1 by construction and
equals the size of the tie group at the top of the retrieval ranking.
"""
from __future__ import annotations

import numpy as np

CURRENT8 = ['s1', 'margin_s1_s2', 'margin_s1_s5', 'std_top5', 'gap_s2_s3',
            'max_adj_gap_top5', 'near_top_count_top5', 'effective_competitor_count_top5']
TIE_EPS = 1e-6


def cheap_features(R, tie_eps: float = TIE_EPS) -> dict[str, np.ndarray]:
    """CURRENT8 from a retrieval score matrix (n, >= 5).  Empty slots must be -inf."""
    s = np.where(np.isfinite(R), R, np.nan)
    top5 = s[:, :5]
    p = np.exp(top5 - np.nanmax(top5, axis=1, keepdims=True))
    p = p / np.nansum(p, axis=1, keepdims=True)
    H = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    adj = top5[:, :-1] - top5[:, 1:]
    return {
        's1': s[:, 0],
        'margin_s1_s2': s[:, 0] - s[:, 1],
        'margin_s1_s5': s[:, 0] - s[:, 4],
        'std_top5': np.nanstd(top5, axis=1),
        'gap_s2_s3': s[:, 1] - s[:, 2],
        'max_adj_gap_top5': np.nanmax(adj, axis=1),
        'near_top_count_top5': (np.abs(s[:, :1] - top5) <= tie_eps).sum(axis=1).astype(float),
        'effective_competitor_count_top5': np.exp(H),
    }


def feature_matrix(R, tie_eps: float = TIE_EPS):
    """(X, feature dict) with NaNs mapped to 0, columns in CURRENT8 order."""
    F = cheap_features(R, tie_eps)
    X = np.column_stack([np.nan_to_num(F[c], nan=0.0) for c in CURRENT8])
    return X, F


def fit_standardiser(X):
    """Training-split mean/sd; zero-variance columns get sd 1."""
    mu, sd = X.mean(0), X.std(0)
    return mu, np.where(sd > 0, sd, 1.0)


def apply_standardiser(X, mu, sd):
    return (X - mu) / sd
