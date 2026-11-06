"""ARR -- adaptive re-ranking, adapted to biomedical entity linking.

This is *our adaptation* of the published method, not the original
implementation, and it is reported as such (marked with a dagger in the paper).
No code from the original repository is included here.

What is preserved from the original formulation
    utility(route) = eff(route) - lambda_cost * (latency(route) / max latency)
    eff            = 0.5 * nDCG@10 + 0.5 * MRR@10  (binary relevance, one gold)
    label          = argmax over routes, ties going to the first declared route
    router         = a BERT-base sequence classifier over the mention with context,
                     AdamW, lr 3e-5, 20 epochs, minority-class down-sampling,
                     75/25 train/validation, native decision = argmax
    lambda_cost    = the value reported by the original work (0.05); it is not
                     tuned on our data, because the original provides no
                     selection procedure for it

What is adapted to this task
    The action space is our frozen binary one:
        route 0 = stop, ranked list = the retriever ranking as-is
        route 1 = rerank@K_high, i.e. the first K_high candidates are re-ordered
                  by the frozen fused score and the tail keeps the retriever order
    Route latency is our measured mean per route (retrieval, and retrieval plus
    reranking at K_high).
    For budget-matched comparison the continuous class-1 probability is used as
    the routing score; the native operating point is the original argmax.
"""
from __future__ import annotations

import numpy as np

EVAL_AT_K = 10


def route_ranks(fused, retrieval_gold_rank, k_high: int, gold_idx) -> dict[str, np.ndarray]:
    """Gold rank under each route.

    Stop keeps the retriever rank.  Rerank@K_high re-orders only the first K_high
    candidates by the fused score (ties broken by the original pool order) and
    leaves deeper candidates where they are, so a gold concept beyond K_high keeps
    its retriever rank.  A gold concept outside the pool has rank 0 on both routes.
    """
    g = np.asarray(gold_idx, dtype=np.int64)
    rank_stop = np.where(g >= 0, g + 1, 0).astype(np.int64)
    rank_ce = rank_stop.copy()
    inK = (g >= 0) & (g < int(k_high))
    if inK.any():
        sub = np.asarray(fused)[inK][:, :int(k_high)]
        gi = g[inK]
        order = np.argsort(-sub, axis=1, kind='stable')
        pos = np.argsort(order, axis=1, kind='stable')
        rank_ce[inK] = pos[np.arange(len(gi)), gi] + 1
    return dict(rank_stop=rank_stop, rank_ce=rank_ce)


def graded_utility(rank, k: int = EVAL_AT_K):
    """(combined, nDCG@k, MRR@k) for a single binary-relevant gold document.

    nDCG@k = 1 / log2(rank + 1) because the ideal DCG is 1; MRR@k = 1 / rank; both
    are 0 beyond rank k.  The combined score is the mean of the two.
    """
    r = np.asarray(rank, dtype=float)
    ok = (r >= 1) & (r <= k)
    ndcg = np.where(ok, 1.0 / np.log2(np.where(ok, r, 2) + 1), 0.0)
    mrr = np.where(ok, 1.0 / np.where(ok, r, 1), 0.0)
    return 0.5 * ndcg + 0.5 * mrr, ndcg, mrr


def utility_labels(eff_stop, eff_rerank, latency_stop: float, latency_rerank: float,
                   lambda_cost: float) -> np.ndarray:
    """Route label per mention: 1 = rerank.  Ties resolve to route 0 (stop),
    matching the original argmax over routes in declaration order."""
    mx = max(latency_stop, latency_rerank, 1e-12)
    u0 = np.asarray(eff_stop) - lambda_cost * (latency_stop / mx)
    u1 = np.asarray(eff_rerank) - lambda_cost * (latency_rerank / mx)
    return (u1 > u0).astype(np.int8)


def downsample_balanced(y, seed: int):
    """Minority-class down-sampling, as in the original recipe (no class weights)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    m = min(len(i1), len(i0))
    sel = np.concatenate([rng.choice(i1, m, replace=False),
                          rng.choice(i0, m, replace=False)])
    rng.shuffle(sel)
    return sel
