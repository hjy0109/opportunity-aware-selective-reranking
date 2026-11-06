"""BiomedBERT pointwise cross-encoder.

One (query, candidate) pair per forward pass, two-way classification head, and
the candidate score is ``logit[1] - logit[0]``.  The negative sampling recipe,
the pair construction and the checkpoint-selection metric are the ones used for
the reported models.
"""
from __future__ import annotations

import numpy as np
import torch


def setup_precision() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def amp(enabled: bool = True):
    return torch.autocast('cuda', torch.bfloat16, enabled=enabled)


def build_model(backbone: str, device: str = 'cuda'):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=2)
    return model.to(device), tok


def load_checkpoint(path: str, device: str = 'cuda'):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    return model, tok


@torch.no_grad()
def score_pairs(model, tok, first, second, device, batch: int, max_len: int,
                truncation: str = 'longest_first', use_amp: bool = True):
    out = np.empty(len(first), dtype=np.float32)
    for i in range(0, len(first), batch):
        enc = tok(first[i:i + batch], second[i:i + batch], padding=True,
                  truncation=truncation, max_length=max_len,
                  return_tensors='pt').to(device)
        with amp(use_amp and device.startswith('cuda')):
            lg = model(**enc).logits
        out[i:i + lg.shape[0]] = (lg[:, 1] - lg[:, 0]).float().cpu().numpy()
    return out


def pick_negatives(row, rng, self_hard, n_a: int = 3, n_b: int = 3, n_c: int = 2):
    """Negative sampling: 3 from ranks 2-10, 3 from ranks 11-64 (at least one deeper
    than rank 20), and up to 2 self-hard negatives mined from the current model."""
    gr, n = int(row.gold_rank_in_pool), int(row.n_candidates)
    A = [j for j in range(1, min(10, n)) if j + 1 != gr]          # 0-based 1..9 = ranks 2..10
    B = [j for j in range(10, n) if j + 1 != gr]                  # ranks 11..64
    Bdeep = [j for j in B if j + 1 > 20]
    out: list[int] = []
    if A:
        out += [A[i] for i in rng.choice(len(A), min(n_a, len(A)), replace=False)]
    if B:
        take = min(n_b, len(B))
        forced = ([Bdeep[int(rng.integers(len(Bdeep)))]] if Bdeep and take else [])
        rest = [j for j in B if j not in forced]
        need = take - len(forced)
        if need > 0 and rest:
            forced += [rest[i] for i in rng.choice(len(rest), min(need, len(rest)),
                                                   replace=False)]
        out += forced
    if n_c and self_hard is not None:
        sh = [j for j in self_hard.get(row.Index, ()) if j + 1 != gr and j not in out][:n_c]
        out += sh
    return list(dict.fromkeys(out))


def build_pairs(cand, queries: dict[str, str], repr_map: dict[str, str], rng, self_hard,
                n_look=None, n_keep=None):
    """Training pairs for one epoch.

    The positive is the gold concept representation.  When the gold concept is
    outside the retrieved pool its representation is injected anyway, so every
    mention with a known gold contributes one positive.
    """
    first, second, y = [], [], []
    n_inj = n_dup = 0
    for r in cand.itertuples():
        q = queries[str(r.mention_id)]
        gr = int(r.gold_rank_in_pool)
        gold_txt = repr_map[str(r.gold_concept_id)]
        if gr > 0:
            pos = repr_map[str(r.cand_concept_id[gr - 1])]
        else:
            pos = gold_txt
            n_inj += 1
        first.append(q); second.append(pos); y.append(1)
        for j in pick_negatives(r, rng, self_hard):
            neg = repr_map[str(r.cand_concept_id[j])]
            n_dup += int(neg == pos)
            first.append(q); second.append(neg); y.append(0)
    return first, second, np.array(y, dtype=np.int64), n_inj, n_dup


@torch.no_grad()
def mine_self_hard(model, tok, cand, queries, repr_map, device, batch, max_len, rng,
                   n_look: int, n_keep: int, truncation: str = 'longest_first'):
    """Highest-scoring non-gold candidates under the current model."""
    model.eval()
    first, second, own = [], [], []
    for r in cand.itertuples():
        gr, n = int(r.gold_rank_in_pool), int(r.n_candidates)
        pool = [j for j in range(1, n) if j + 1 != gr]
        if not pool:
            continue
        q = queries[str(r.mention_id)]
        for j in [pool[i] for i in rng.choice(len(pool), min(n_look, len(pool)),
                                              replace=False)]:
            first.append(q); second.append(repr_map[str(r.cand_concept_id[j])])
            own.append((r.Index, j))
    s = score_pairs(model, tok, first, second, device, batch, max_len, truncation)
    agg: dict[int, list[tuple[float, int]]] = {}
    for (mi, j), v in zip(own, s):
        agg.setdefault(mi, []).append((v, j))
    model.train()
    return {mi: [j for _, j in sorted(v, reverse=True)[:n_keep]]
            for mi, v in agg.items()}, len(first)


@torch.no_grad()
def internal_val_metrics(model, tok, iv, queries, repr_map, device, batch, max_len,
                         k_pool: int = 64, truncation: str = 'longest_first'):
    """Reranker-only Acc@1 over the full candidate pool of the internal validation split.

    This is the only checkpoint-selection signal; dev and test are not read.
    """
    model.eval()
    first, second, own = [], [], []
    for qi, r in enumerate(iv.itertuples()):
        q = queries[str(r.mention_id)]
        for j in range(min(k_pool, int(r.n_candidates))):
            first.append(q); second.append(repr_map[str(r.cand_concept_id[j])])
            own.append((qi, j))
    s = score_pairs(model, tok, first, second, device, batch, max_len, truncation)
    best: dict[int, tuple[int, float]] = {}
    for (qi, j), v in zip(own, s):
        if qi not in best or v > best[qi][1]:
            best[qi] = (j, v)
    ok = np.zeros(len(iv), dtype=bool)
    for qi, r in enumerate(iv.itertuples()):
        j = best.get(qi, (-1, 0.0))[0]
        ok[qi] = j >= 0 and r.cand_concept_id[j] == r.gold_concept_id
    inpool = iv['gold_rank_in_pool'].to_numpy() > 0
    model.train()
    return (float(ok.mean()), float(ok[inpool].mean()),
            float(ok[~inpool].mean()) if (~inpool).any() else 0.0, len(first))


@torch.no_grad()
def score_candidate_pool(model, tok, cand, queries, repr_map, device, k_pool: int,
                         max_len: int, batch: int = 256,
                         truncation: str = 'longest_first', flush_at: int = 40_000):
    """(n, k_pool) cross-encoder scores plus a validity mask; empty slots are -inf."""
    import time
    n = len(cand)
    CE = np.full((n, k_pool), -np.inf, dtype=np.float64)
    V = np.zeros((n, k_pool), dtype=bool)
    first, second, own = [], [], []
    t0 = time.time()
    npair = 0

    def flush():
        nonlocal first, second, own, npair
        if not first:
            return
        s = score_pairs(model, tok, first, second, device, batch, max_len, truncation)
        for (qi, j), v in zip(own, s):
            CE[qi, j] = float(v)
            V[qi, j] = True
        npair += len(first)
        first, second, own = [], [], []

    for qi, r in enumerate(cand.itertuples()):
        q = queries[str(r.mention_id)]
        for j in range(min(k_pool, int(r.n_candidates))):
            first.append(q); second.append(repr_map[str(r.cand_concept_id[j])])
            own.append((qi, j))
        if len(first) >= flush_at:
            flush()
    flush()
    return CE, V, npair, time.time() - t0
