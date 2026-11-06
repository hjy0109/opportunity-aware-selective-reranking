"""Budgeted selection and the selective-reranking outcome.

``q = B / N`` is the normalised share of mentions that may be reranked.  At a
given q every policy reranks exactly ``round(q * N)`` mentions, so the accuracy
comparison between policies is at matched reranking compute.

Selection is a descending stable argsort of the routing score, so ties are broken
by the original candidate-frame row order, identically for every policy.  A
routing score is always oriented so that larger means "rerank sooner".
"""
from __future__ import annotations

import numpy as np


def select_top_q(score, q: float, n: int) -> np.ndarray:
    """Boolean mask over ``n`` mentions selecting the ``round(q * n)`` highest scores."""
    kq = int(round(float(q) * n))
    sel = np.zeros(n, bool)
    if kq > 0:
        s = np.asarray(score, float)
        s = np.where(np.isfinite(s), s, -np.inf)
        sel[np.argsort(-s, kind='stable')[:kq]] = True
    return sel


def apply_selection(selected, correct_at_1, correct_at_k_high) -> np.ndarray:
    """Per-mention correctness under selective reranking.

    Selected mentions take the reranked prediction; unselected mentions keep the
    retriever top-1 prediction.
    """
    return np.where(np.asarray(selected, bool),
                    np.asarray(correct_at_k_high, bool),
                    np.asarray(correct_at_1, bool))


def selection_summary(selected, correct_at_1, correct_at_k_high, k_high: int) -> dict:
    from ..reranking.fusion import ce_pairs
    sel = np.asarray(selected, bool)
    c1 = np.asarray(correct_at_1, bool)
    ckh = np.asarray(correct_at_k_high, bool)
    ok = apply_selection(sel, c1, ckh)
    pairs = ce_pairs(np.where(sel, k_high, 1))
    full_pairs = float(ce_pairs(np.full(len(c1), k_high)).mean())
    return dict(
        n=int(len(c1)), n_reranked=int(sel.sum()), n_kept=int((~sel).sum()),
        rerank_rate=float(sel.mean()), acc_at_1=float(ok.mean()),
        ce_pairs_per_mention=float(pairs.mean()), ce_pairs_total=int(pairs.sum()),
        normalised_compute_fraction=(float(pairs.mean()) / full_pairs if full_pairs else None),
        acc_full_reranking=float(ckh.mean()), acc_retriever_only=float(c1.mean()),
        delta_vs_full_pp=round(100 * (float(ok.mean()) - float(ckh.mean())), 6),
        delta_vs_retriever_pp=round(100 * (float(ok.mean()) - float(c1.mean())), 6),
        corrections_gained=int((sel & (~c1) & ckh).sum()),
        rerank_induced_losses=int((sel & c1 & (~ckh)).sum()))


def selection_composition(selected, hard, opp, ben, correct_at_1) -> dict:
    """How the reranking budget was spent, and how much of each class it captured."""
    sel = np.asarray(selected, bool)
    hard, opp, ben = (np.asarray(x, bool) for x in (hard, opp, ben))
    c1 = np.asarray(correct_at_1, bool)
    ns = int(sel.sum())

    def frac(mask):
        return float(np.asarray(mask)[sel].mean()) if ns else float('nan')

    def capture(mask):
        m = np.asarray(mask)
        return float((m & sel).sum()) / max(int(m.sum()), 1)

    return dict(
        n_selected=ns,
        pct_BEN=frac(ben),
        pct_accessible_not_corrected=frac(opp & ~ben),
        pct_unreachable_error=frac(hard & ~opp),
        pct_retriever_already_correct=frac(c1),
        BEN_capture=capture(ben), OPP_capture=capture(opp), HARD_capture=capture(hard),
        selected_HARD_frac=frac(hard), selected_OPP_frac=frac(opp))
