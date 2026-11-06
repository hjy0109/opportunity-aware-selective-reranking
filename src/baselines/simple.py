"""Margin and Random.

Both are routing policies over the same frozen host: same retriever, same
candidate pool, same reranker, same candidate depth, same reranking rate.  Only
the ordering of mentions differs.

Margin
    Routing score = ``-margin_s1_s2``, i.e. the negated gap between the top-1 and
    top-2 retrieval scores.  A *smaller* retrieval margin gets a *higher* routing
    priority, so the mentions where the retriever is least separated are reranked
    first.  The negation exists only to keep every policy on the same "larger
    score = rerank sooner" convention.

Random
    A uniform random subset of exactly ``round(q * N)`` mentions.  Reported values
    are the mean over 20 replicates with the declared seed, and the spread across
    replicates is reported next to it.
"""
from __future__ import annotations

import numpy as np


def margin_score(table) -> np.ndarray:
    """``-margin_s1_s2`` from a routing table."""
    return -table.margin_s1_s2.to_numpy(float)


def random_accuracy(correct_at_1, correct_at_k_high, q: float, seed: int,
                    replicates: int = 20):
    """(mean Acc@1, sd over replicates) for uniform selection at rate q.

    Each replicate draws exactly ``round(q * N)`` mentions without replacement, so
    the invocation count matches every other policy at the same q.
    """
    from ..routing.budget import apply_selection
    rng = np.random.default_rng(seed)
    n = len(correct_at_1)
    m = int(round(float(q) * n))
    acc = []
    for _ in range(replicates):
        sel = np.zeros(n, bool)
        if m:
            sel[rng.choice(n, size=m, replace=False)] = True
        acc.append(float(apply_selection(sel, correct_at_1, correct_at_k_high).mean()))
    return float(np.mean(acc)), float(np.std(acc))
