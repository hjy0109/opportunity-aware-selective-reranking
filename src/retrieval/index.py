"""Candidate generation over the knowledge base.

Two aggregations are used in the reported experiments and both are kept, because
they are not interchangeable:

``concept_max_streaming``
    The whole KB is streamed in concept-aligned blocks; each concept's score is
    the maximum over its aliases, and a global top-k of concepts is maintained.
    Every concept in the KB competes.  Retrieval scores are raw inner products
    of CLS embeddings (``normalize: false``).

``alias_topn_dedup``
    A global top-N over aliases is taken first (N = ``name_topn``), then folded
    to concepts: a concept's first appearance in the alias ranking gives both its
    position and its score (its best-matching alias).  A concept whose aliases
    all fall outside the alias top-N cannot enter the pool.  Queries and aliases
    are L2-normalised, so scores are cosines (``normalize: true``).

Both write the same candidate frame, so every downstream stage is shared.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .encoder import encode_dense


def concept_rank_streaming(model, tok, names, name2concept, n_concepts, queries, device,
                           k_max: int = 512, max_len: int = 32, encode_batch: int = 512,
                           name_block: int = 400_000, q_chunk: int = 2048,
                           verbose: bool = True):
    """Concept-max aggregation with a streamed KB; returns a global top-k of concepts.

    ``names`` must be ordered by concept so that a block never splits a concept.
    Returns (top concept indices [nq, k], top scores [nq, k], encode s, search s).
    """
    nq = queries.shape[0]
    k = min(k_max, n_concepts)
    best_s = torch.full((nq, k), float('-inf'), device=device, dtype=torch.float32)
    best_i = torch.zeros((nq, k), device=device, dtype=torch.int32)
    enc_sec = search_sec = 0.0
    n = len(names)
    start = 0
    while start < n:
        end = min(start + name_block, n)
        while end < n and name2concept[end] == name2concept[end - 1]:   # to concept boundary
            end += 1
        blk_c = name2concept[start:end]
        c_lo, c_hi = int(blk_c[0]), int(blk_c[-1])
        n_blk_c = c_hi - c_lo + 1
        local = torch.as_tensor(blk_c - c_lo, device=device, dtype=torch.int64)
        nd, dt = encode_dense(model, tok, names[start:end], device, max_len=max_len,
                              batch=encode_batch)
        enc_sec += dt
        t0 = time.time()
        for a in range(0, nq, q_chunk):
            b = min(a + q_chunk, nq)
            cmax = torch.full((b - a, n_blk_c), float('-inf'), device=device,
                              dtype=torch.float32)
            sc = queries[a:b] @ nd.T
            cmax.index_reduce_(1, local, sc, 'amax', include_self=True)
            del sc
            kk = min(k, n_blk_c)
            ts, ti = torch.topk(cmax, kk, dim=1)
            del cmax
            cs = torch.cat([best_s[a:b], ts], dim=1)
            ci = torch.cat([best_i[a:b], (ti + c_lo).to(torch.int32)], dim=1)
            ms, mi = torch.topk(cs, k, dim=1)
            best_s[a:b] = ms
            best_i[a:b] = torch.gather(ci, 1, mi)
            del cs, ci, ms, mi, ts, ti
        search_sec += time.time() - t0
        del nd, local
        torch.cuda.empty_cache()
        if verbose:
            print(f'  block {start:,}-{end:,} / {n:,}', flush=True)
        start = end
    return best_i.cpu().numpy(), best_s.cpu().numpy(), enc_sec, search_sec


def alias_topn(model, tok, names, queries, device, topn: int = 4096, max_len: int = 32,
               encode_batch: int = 512, name_block: int = 400_000, q_chunk: int = 1024,
               verbose: bool = True):
    """Global top-N over aliases, cosine similarity (queries and aliases normalised)."""
    nq = queries.shape[0]
    q = F.normalize(queries, dim=-1)
    n = len(names)
    k = min(topn, n)
    best_s = torch.full((nq, k), float('-inf'), device=device, dtype=torch.float32)
    best_i = torch.zeros((nq, k), device=device, dtype=torch.int32)
    enc_sec = search_sec = 0.0
    start = 0
    while start < n:
        end = min(start + name_block, n)
        nd, dt = encode_dense(model, tok, names[start:end], device, max_len=max_len,
                              batch=encode_batch)
        nd = F.normalize(nd, dim=-1)
        enc_sec += dt
        t0 = time.time()
        for a in range(0, nq, q_chunk):
            b = min(a + q_chunk, nq)
            sc = q[a:b] @ nd.T
            kk = min(k, sc.shape[1])
            ts, ti = torch.topk(sc, kk, dim=1)
            del sc
            cs = torch.cat([best_s[a:b], ts], dim=1)
            ci = torch.cat([best_i[a:b], (ti + start).to(torch.int32)], dim=1)
            ms, mi = torch.topk(cs, k, dim=1)
            best_s[a:b] = ms
            best_i[a:b] = torch.gather(ci, 1, mi)
            del cs, ci, ms, mi, ts, ti
        search_sec += time.time() - t0
        del nd
        torch.cuda.empty_cache()
        if verbose:
            print(f'  block {start:,}-{end:,} / {n:,}', flush=True)
        start = end
    return best_i.cpu().numpy(), best_s.cpu().numpy(), enc_sec, search_sec


def dedup_to_concepts(top_name_idx, top_score, name2concept, k_pool: int = 64):
    """Fold an alias ranking into a concept ranking.

    A concept's first appearance is its best-matching alias, and that alias's
    score becomes the concept's retrieval score.
    """
    nq = top_name_idx.shape[0]
    cc = np.full((nq, k_pool), -1, dtype=np.int64)
    cn = np.full((nq, k_pool), -1, dtype=np.int64)
    cs = np.full((nq, k_pool), np.nan, dtype=np.float32)
    nf = np.zeros(nq, dtype=np.int32)
    for i in range(nq):
        seen: set[int] = set()
        j = 0
        for pos in range(top_name_idx.shape[1]):
            ni = int(top_name_idx[i, pos])
            ci = int(name2concept[ni])
            if ci in seen:
                continue
            seen.add(ci)
            cc[i, j] = ci
            cn[i, j] = ni
            cs[i, j] = top_score[i, pos]
            j += 1
            if j >= k_pool:
                break
        nf[i] = j
    return cc, cn, cs, nf


def gold_rank(top_concept_idx, gold_concept_index) -> np.ndarray:
    """1-based rank of the gold concept in the retrieved ranking, or -1 if absent."""
    r = np.full(len(gold_concept_index), -1, dtype=np.int32)
    for i, g in enumerate(gold_concept_index):
        if g < 0:
            continue
        w = np.where(top_concept_idx[i] == g)[0]
        if len(w):
            r[i] = int(w[0]) + 1
    return r


def retrieval_score_matrix(cand, k: int = 64) -> np.ndarray:
    """(n, k) retrieval scores from a candidate frame; empty slots are -inf."""
    R = np.full((len(cand), k), -np.inf, dtype=np.float64)
    for i, row in enumerate(cand.cand_ret_score):
        kk = min(k, len(row))
        R[i, :kk] = np.asarray(row[:kk], dtype=np.float64)
    return R


def pool_concept_ids(cand, k: int = 64) -> np.ndarray:
    """(n, k) candidate concept ids as strings; empty slots are ''."""
    out = np.full((len(cand), k), '', dtype=object)
    for i, row in enumerate(cand.cand_concept_id):
        a = np.asarray([('' if c is None else c) for c in row]).astype(str)
        kk = min(k, len(a))
        out[i, :kk] = a[:kk]
    return out
