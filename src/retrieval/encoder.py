"""SapBERT dense encoding.

The retriever is the off-the-shelf SapBERT checkpoint, CLS pooling, fp32, and the
query is the mention surface form only -- no context, no abbreviation expansion.
"""
from __future__ import annotations

import time

import torch


def load_encoder(model_name: str, device: str = 'cuda'):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, tok


@torch.no_grad()
def encode_dense(model, tok, texts, device, max_len: int = 32, batch: int = 512,
                 desc: str = '', out=None):
    """CLS embeddings, fp32.

    Filled with ``.copy_`` into a preallocated tensor: the CLS slice is a view, so
    collecting the slices in a list would keep every parent activation alive.
    """
    t0 = time.time()
    v = out if out is not None else torch.empty(
        len(texts), model.config.hidden_size, device=device, dtype=torch.float32)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, max_length=max_len,
                  truncation=True, return_tensors='pt').to(device)
        h = model(**enc).last_hidden_state[:, 0, :]
        v[i:i + h.shape[0]].copy_(h)
        del h, enc
    dt = time.time() - t0
    if desc:
        print(f'  dense[{desc}] {len(texts):,} in {dt:.1f}s '
              f'({len(texts) / max(dt, 1e-9):,.0f}/s)', flush=True)
    return v, dt
