"""ARTER -- easy/hard routing, adapted to biomedical entity linking.

This is *our adaptation* of the published method, not the original
implementation, and it is reported as such (marked with a dagger in the paper).
No code from the original repository is included here.

Preserved from the original
    routing target       easy/hard, i.e. whether the frozen base linker's top-1
                         prediction differs from the gold concept.  This is
                         formally the same as our HARD (Difficulty) label.
    router               Random Forest.  The original reports no hyperparameters,
                         so library defaults with a fixed seed are used and that
                         fact is recorded.
    operating point      threshold maximising Youden's J on our dev split, the
                         native ARTER rule (our own delta rule is not imposed).
    margin               penalised top-1 minus penalised top-2.
    sent_len             token length of the mention's context.

Adapted, and why
    entity representation   the original uses a Wikipedia title plus a Wikidata
                            description; our knowledge bases carry no definitions,
                            so the concept representation (alias pair) is encoded.
    theta1, theta2, theta3  recomputed with a biomedical sentence encoder.  The
                            retriever cannot be reused for theta1: it encodes the
                            mention surface form only, at max_len 32, so it never
                            sees the context.
    theta2                  computed with the same encoder as theta1 and theta3 so
                            that the three terms share a scale.  The frozen
                            retrieval score is on a per-dataset scale (unnormalised
                            inner products on some datasets), which would dominate
                            the sum; it is reported alongside as
                            ``retrieval_top1`` rather than substituted in.
    penalised score         ``(theta1 + theta2 - theta3) / 2``.  The original divides
                            a three-term sum by 3; the LLM-confidence term is not
                            reproduced here, so the denominator matches the two
                            terms that remain.
    entropy                 the original does not state its normalisation, so ours is
                            fixed and stated: base-2 Shannon entropy over
                            ``p_i = (s_i - min) / sum(s - min)`` on the top-10
                            retrieval scores.
    n_cands                 degenerate in our setting (constant 64, zero variance),
                            so it is excluded from the router input and the
                            exclusion is stated rather than silent.
    LLM confidence (phi)    omitted.
    downstream action       our frozen cross-encoder reranking at K_high, in place
                            of the original's targeted LLM reasoning.
"""
from __future__ import annotations

import numpy as np
import torch

FEATURES = ['penalized_top1', 'penalized_top2', 'penalized_margin',
            'avg_theta1', 'avg_theta2', 'avg_theta3', 'entropy_top10', 'sent_len']
OMITTED = {
    'n_cands': 'degenerate here: constant 64 with zero variance, so it carries no '
               'information for the router; excluded and stated',
    'phi (LLM confidence)': 'omitted; not reproduced in this adaptation',
}


@torch.no_grad()
def encode_normalised(model, tok, texts, max_len: int, device: str, batch: int = 256):
    out = []
    for i in range(0, len(texts), batch):
        e = tok(texts[i:i + batch], truncation=True, max_length=max_len,
                padding=True, return_tensors='pt').to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=device.startswith('cuda')):
            h = model(**e).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h.float(), dim=1).cpu())
    return torch.cat(out) if out else torch.zeros(0, model.config.hidden_size)


def entropy_top10(scores, top_m: int = 10) -> float:
    s = np.asarray(scores, dtype=np.float64)[:top_m]
    v = s - s.min()
    tot = v.sum()
    if tot <= 0:
        return 0.0
    p = v / tot
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def theta_block(entity_vecs, context_vecs, mention_vecs):
    """theta1/theta2/theta3 and the penalised scores for one split.

    ``entity_vecs`` is (n, top_m, d) with the candidate concept embeddings in
    retrieval order; theta3 is the largest cosine to another candidate in the
    same top-m set.
    """
    th1 = torch.einsum('nkd,nd->nk', entity_vecs, context_vecs).numpy()
    th2 = torch.einsum('nkd,nd->nk', entity_vecs, mention_vecs).numpy()
    G = torch.einsum('nkd,nld->nkl', entity_vecs, entity_vecs).numpy()
    np.einsum('nkk->nk', G)[...] = -np.inf          # exclude self-similarity
    th3 = G.max(axis=2)
    pen = (th1 + th2 - th3) / 2.0
    return th1, th2, th3, pen


def youden_threshold(y, score):
    """Threshold maximising Youden's J -- the native ARTER operating rule."""
    y = np.asarray(y, bool)
    s = np.asarray(score, float)
    o = np.argsort(-s, kind='stable')
    ys = y[o]
    P, N = int(ys.sum()), int((~ys).sum())
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    j = tp / max(P, 1) - fp / max(N, 1)
    i = int(np.argmax(j))
    return float(s[o][i]), float(j[i])
