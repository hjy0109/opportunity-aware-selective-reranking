"""Routing policies available to the evaluation.

Every policy is a per-mention score with the same orientation (larger = rerank
sooner) over the same frozen host, so at a given reranking rate they all rerank
exactly the same number of mentions with the same reranker at the same depth.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

POLICIES = ('retriever', 'full', 'random', 'margin', 'arr', 'arter',
            'difficulty', 'opportunity', 'benefit')
POLICY_LABEL = {
    'retriever': 'Retriever only', 'full': 'Full reranking', 'random': 'Random',
    'margin': 'Margin', 'arr': 'ARR', 'arter': 'ARTER',
    'difficulty': 'Difficulty', 'opportunity': 'Opportunity',
    'benefit': 'Benefit',
}
ROUTED = ('random', 'margin', 'arr', 'arter', 'difficulty', 'opportunity', 'benefit')
# Learned routers, and the supervision target each one is fitted on.  Benefit is
# the appendix diagnostic: same features, same classifier, same hyperparameters,
# different label.
SUPERVISION = {'opportunity': 'OPP', 'difficulty': 'HARD', 'benefit': 'BEN'}


def load_arr_router(cfg, device: str) -> dict:
    """The trained ARR router, for timing its per-mention forward pass."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    d = cfg.model_path(f'arr_router_{cfg.tag}')
    if not os.path.isdir(d):
        raise SystemExit(f'ARR router checkpoint not found: {d}')
    return dict(tok=AutoTokenizer.from_pretrained(d),
                model=AutoModelForSequenceClassification.from_pretrained(d).to(
                    device).eval(),
                max_len=int(cfg.method['baselines']['arr']['max_len']))


def load_arter_encoder(cfg, device: str) -> dict:
    """The ARTER feature encoder, for timing its three per-mention forward passes."""
    from transformers import AutoModel, AutoTokenizer
    ckpt = cfg.method['baselines']['arter']['encoder']
    return dict(tok=AutoTokenizer.from_pretrained(ckpt),
                model=AutoModel.from_pretrained(ckpt).to(device).eval())


def _aligned(path: str, table, column: str = 'score'):
    d = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    s = pd.Series(d[column].astype(float).to_numpy(),
                  index=pd.Index(d.mention_id.astype(str)))
    v = s.reindex(table.mention_id.astype(str)).to_numpy()
    if pd.isna(v).any():
        raise SystemExit(f'{path}: {int(pd.isna(v).sum())} mentions have no score')
    return v


def available(cfg, table, split: str) -> dict[str, np.ndarray]:
    """Routing scores that exist on disk for this dataset and split.

    ``retriever`` and ``full`` are the two endpoints and carry no score.
    ``random`` is handled separately because it is averaged over replicates.
    """
    from ..baselines.simple import margin_score
    from ..routing.infer import router_scores

    out: dict[str, np.ndarray] = {'margin': margin_score(table)}
    for pol, target in SUPERVISION.items():
        if os.path.exists(cfg.router_scores_path(target)):
            out[pol] = router_scores(cfg, target, split, np)[1]
    for pol, sub in (('arr', 'arr'), ('arter', 'arter')):
        p = cfg.out_path('baselines', sub, f'scores_{split}.parquet')
        if os.path.exists(p):
            out[pol] = _aligned(p, table)
    return out
