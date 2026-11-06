"""Split helpers.

Two grouped splits are used, both at document level so that no document's
mentions ever straddle a boundary:

    * a 3-fold assignment over the training split, used for the leakage-safe
      out-of-fold reranking that the BEN diagnostic label needs;
    * a 10% train-internal validation holdout, the only signal used to select the
      cross-encoder checkpoint, the fusion weight alpha and the z statistics.

The dev split is never used for fitting, and the test split is never read by any
selection step.
"""
from __future__ import annotations

import numpy as np

from ..utils.io import sha256_ids

SPLIT_RULE = 'DOCUMENT_GROUPED_3FOLD'
IV_RULE = 'document-level 10% holdout of the training split'


def document_folds(doc_ids, n_folds: int = 3, seed: int = 1):
    """Greedy balanced document-grouped folds.

    Documents are shuffled with the given seed, ordered by descending mention
    count, then assigned to the currently smallest fold.  This keeps the fold
    mention counts close while keeping each document whole.
    """
    doc_ids = np.asarray(doc_ids).astype(str)
    docs, inv, cnt = np.unique(doc_ids, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(docs))
    order = perm[np.argsort(-cnt[perm], kind='stable')]
    load = np.zeros(n_folds, dtype=np.int64)
    doc_fold = np.empty(len(docs), dtype=np.int8)
    for d in order:
        f = int(np.argmin(load))
        doc_fold[d] = f
        load[f] += cnt[d]
    return doc_fold[inv], docs, doc_fold, cnt


def split_internal_val(df, seed: int = 1, rule: str = 'document_holdout_10pct'):
    """Train-internal validation holdout.

    ``rule='stored_flag'`` reuses a boolean ``is_internal_val`` column when the
    dataset ships one (MedMentions reuses its stored 3,000-mention holdout);
    otherwise 10% of documents are drawn with the given seed.
    """
    if rule == 'stored_flag' and 'is_internal_val' in df.columns:
        iv = df.is_internal_val.fillna(False).astype(bool).to_numpy()
        return (df[~iv].reset_index(drop=True), df[iv].reset_index(drop=True),
                'stored is_internal_val flag',
                dict(rule='stored is_internal_val', seed=None,
                     n_holdout_mentions=int(iv.sum())))
    docs = df.document_id.astype(str).to_numpy()
    uniq = np.unique(docs)
    rng = np.random.default_rng(seed)
    hold = set(rng.choice(uniq, size=max(1, int(round(0.10 * len(uniq)))), replace=False))
    iv = np.array([d in hold for d in docs])
    return (df[~iv].reset_index(drop=True), df[iv].reset_index(drop=True), IV_RULE,
            dict(rule=IV_RULE, seed=seed, n_documents=int(len(uniq)),
                 n_holdout_documents=int(len(hold)),
                 holdout_document_ids_sha256=sha256_ids(sorted(hold))))
