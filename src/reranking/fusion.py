"""Score standardisation, retriever/cross-encoder fusion and candidate depth.

The reranking score of a candidate is

    (1 - alpha) * z(retrieval score) + alpha * z(cross-encoder score)

with the z statistics (mean/sd) taken over the valid candidates of the first
``k_ref`` positions.  Both ``alpha`` and the z statistics are estimated on the
train-internal validation split only and then frozen; the dev split is used to
select the candidate depth K_high, and the test split is used for neither.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def z_statistics(ret, ce, valid, k_ref: int = 50) -> dict[str, Any]:
    m = valid[:, :k_ref]
    r = np.asarray(ret)[:, :k_ref][m]
    c = np.asarray(ce)[:, :k_ref][m]
    return dict(ret_mu=float(r.mean()), ret_sd=float(r.std()),
                ce_mu=float(c.mean()), ce_sd=float(c.std()), K_ref=k_ref,
                estimated_on='train-internal-validation', frozen_across_K=True)


def fuse(ret, ce, valid, cal: dict[str, Any], alpha: float) -> np.ndarray:
    zr = (np.where(valid, ret, 0.0) - cal['ret_mu']) / cal['ret_sd']
    zc = (np.where(valid, ce, 0.0) - cal['ce_mu']) / cal['ce_sd']
    return np.where(valid, (1 - alpha) * zr + alpha * zc, -np.inf)


def correct_prefix(scores, gold_idx, ) -> np.ndarray:
    """(n, k) bool: at each prefix depth K = 1..k, is the argmax the gold concept?

    The running argmax keeps the earliest maximum, so ties are resolved by the
    original candidate (retrieval) order.
    """
    s = np.asarray(scores)
    n, p = s.shape
    best = np.full(n, -np.inf)
    bidx = np.zeros(n, dtype=np.int64)
    out = np.empty((n, p), dtype=bool)
    for k in range(p):
        upd = s[:, k] > best
        best = np.where(upd, s[:, k], best)
        bidx = np.where(upd, k, bidx)
        out[:, k] = (bidx == gold_idx) & (gold_idx >= 0)
    return out


def select_alpha(iv_ret, iv_ce, iv_valid, gold_idx, alpha_grid, k_ref: int = 50,
                 k_select: int = 64):
    """Pick alpha on train-internal validation Acc@1 at K = ``k_select``.

    Ties go to the first value in the declared grid.
    """
    cal = z_statistics(iv_ret, iv_ce, iv_valid, k_ref)
    rows = []
    for a in alpha_grid:
        C = correct_prefix(fuse(iv_ret, iv_ce, iv_valid, cal, a), gold_idx)
        rows.append(dict(alpha=float(a),
                         iv_acc1_at_K64=round(float(C[:, k_select - 1].mean()), 6),
                         iv_acc1_at_K20=round(float(C[:, 19].mean()), 6),
                         iv_acc1_at_K5=round(float(C[:, 4].mean()), 6),
                         n_internal_val=int(len(C))))
    best = max(r['iv_acc1_at_K64'] for r in rows)
    alpha = next(r['alpha'] for r in rows if r['iv_acc1_at_K64'] == best)
    return alpha, cal, rows


def select_candidate_depth(acc_by_k: dict[int, float], tolerance: float = 0.005,
                           rule: str = '') -> dict[str, Any]:
    """Smallest K whose Acc@1 is within ``tolerance`` of the best fixed K."""
    best_k = max(acc_by_k, key=lambda k: acc_by_k[k])
    thr = acc_by_k[best_k] - tolerance
    adm = sorted(k for k in acc_by_k if acc_by_k[k] >= thr)
    return dict(selected_K_high=int(adm[0]), best_fixed_K=int(best_k),
                best_fixed_acc=round(float(acc_by_k[best_k]), 6),
                threshold_acc=round(float(thr), 6),
                acc_at_selected=round(float(acc_by_k[adm[0]]), 6),
                delta_selected_vs_best_pp=round(
                    100 * (acc_by_k[adm[0]] - acc_by_k[best_k]), 4),
                admissible_K=[int(k) for k in adm], tolerance=tolerance,
                rule=rule or 'smallest K within tolerance of the best fixed K',
                K_HIGH_SELECTED_FROM_ROUTING=False)


def ce_pairs(k) -> np.ndarray:
    """Cost convention: K <= 1 issues no cross-encoder call, K > 1 costs K pairs."""
    k = np.asarray(k)
    return np.where(k <= 1, 0, k).astype(np.int64)


def predict_concepts(pool, ret, ce, valid, cal, alpha, K: int, selected) -> np.ndarray:
    """Final linked concept per mention.

    Mentions that are not selected keep the retriever top-1.  Selected mentions
    take the argmax of the fused score over the first K candidates; ties keep the
    original pool order (lowest index).  Candidates beyond K are never touched.
    """
    pred = pool[:, 0].copy()
    idx = np.flatnonzero(selected)
    if len(idx):
        f = fuse(ret[idx, :K], ce[idx, :K], valid[idx, :K], cal, alpha)
        w = np.argmax(np.where(np.isfinite(f), f, -np.inf), axis=1)
        pred[idx] = pool[idx, :][np.arange(len(idx)), w]
    return pred
