"""Supervision targets.

These are the definitions the experiments used, and they are what separates
Opportunity from the Difficulty control.  All three are training-time labels: they
require the gold concept and, for BEN, the outcome of a reranking call.  None of
them is available at inference time, and none is a router input.

    HARD (Difficulty)  1[retriever top-1 != gold]
    OPP  (Opportunity) 1[retriever top-1 != gold AND 1 < gold_rank <= K_high]
    BEN  (diagnostic)  1[retriever top-1 != gold AND rerank@K_high == gold]

By construction BEN is a subset of OPP is a subset of HARD; ``check_containment``
asserts it rather than assuming it.

Opportunity is Difficulty intersected with reachability at the deployed candidate
depth: a mention whose gold concept is not in the top K_high cannot be fixed by
reranking to depth K_high, no matter how hard it is.
"""
from __future__ import annotations

import numpy as np


def hardness(correct_at_1) -> np.ndarray:
    return ~np.asarray(correct_at_1, bool)


def opportunity(correct_at_1, gold_rank, k_high: int) -> np.ndarray:
    """``gold_rank`` is the 1-based retrieval rank of the gold concept, or <= 0 when
    the gold concept is not in the candidate pool."""
    gr = np.asarray(gold_rank, dtype=np.int64)
    return (~np.asarray(correct_at_1, bool)) & (gr > 1) & (gr <= int(k_high))


def benefit(correct_at_1, correct_at_k_high) -> np.ndarray:
    return (~np.asarray(correct_at_1, bool)) & np.asarray(correct_at_k_high, bool)


def check_containment(hard, opp, ben) -> None:
    hard, opp, ben = (np.asarray(x, bool) for x in (hard, opp, ben))
    n_ben = int((ben & ~opp).sum())
    n_opp = int((opp & ~hard).sum())
    if n_ben:
        raise SystemExit(f'BEN is not a subset of OPP ({n_ben} mentions)')
    if n_opp:
        raise SystemExit(f'OPP is not a subset of HARD ({n_opp} mentions)')


TARGETS = {
    'OPP': dict(label='Opportunity', tier='primary', fn=opportunity,
                definition='1[retriever top-1 != gold AND 1 < gold_rank <= K_high]'),
    'HARD': dict(label='Difficulty', tier='control', fn=hardness,
                 definition='1[retriever top-1 != gold]'),
    'BEN': dict(label='Benefit', tier='diagnostic', fn=benefit,
                definition='1[retriever top-1 != gold AND rerank@K_high == gold]'),
}
# Targets a router can be fitted on.  BEN is the appendix diagnostic: it needs a
# reranking outcome, so its training labels come from the out-of-fold rerankers
# built by src.routing.table, never from the reranker it is later applied with.
ROUTED_TARGETS = ('OPP', 'HARD', 'BEN')
TARGET_LABEL = {k: v['label'] for k, v in TARGETS.items()}
