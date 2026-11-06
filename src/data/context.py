"""Reranker query construction.

The cross-encoder sees two sequences:

    sequence 1  "Target mention: {mention}\\nContext: {window}"
    sequence 2  the candidate concept representation (see data/ontology.py)

The window is the mention sentence plus one sentence on either side, taken from
the full document, then cropped to ``max_context_tokens`` around the mention.
The mention span itself is never dropped by the crop.
"""
from __future__ import annotations

import re

_SENT = re.compile(r'(?<=[.!?])\s+')


def query_of(mention_text: str, context: str) -> str:
    return f'Target mention: {mention_text}\nContext: {context}'


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for part in _SENT.split(text):
        spans.append((pos, pos + len(part)))
        pos += len(part) + 1
    return spans


def sentence_window(text: str, s: int, e: int, pad: int = 1,
                    spans: list[tuple[int, int]] | None = None):
    """Mention sentence +/- ``pad`` sentences, plus the mention offsets inside it."""
    spans = sentence_spans(text) if spans is None else spans
    for i, (a, b) in enumerate(spans):
        if a <= s < b or (s < a and e > a):
            lo = max(0, i - pad)
            hi = min(len(spans), i + pad + 1)
            w0, w1 = spans[lo][0], spans[hi - 1][1]
            return text[w0:w1], s - w0, e - w0
    w0 = max(0, s - 150)
    return text[w0:e + 150], s - w0, e - w0


def _covers(om, lo, hi, ms, me) -> bool:
    return any(b > ms and a < me for a, b in om[lo:hi + 1])


def _slice(window: str, om, lo: int, hi: int) -> str:
    return window[om[lo][0]:om[hi][1]].strip()


def crop_tokens(tok, window: str, ms: int, me: int, budget: int):
    """Target-centred, balanced crop to ``budget`` tokens.

    Returns ``(cropped_text, n_context_tokens, mention_span_kept)``.  When the
    mention alone exceeds the budget the mention wins and the budget is exceeded.
    """
    if not window:
        return '', 0, False
    enc = tok(window, add_special_tokens=False, return_offsets_mapping=True)
    om = enc['offset_mapping']
    if not om:
        return '', 0, False
    if len(om) <= budget:
        return window, len(om), _covers(om, 0, len(om) - 1, ms, me)
    idx = [i for i, (a, b) in enumerate(om) if b > ms and a < me]
    if not idx:                                   # mention not locatable: crop from the front
        return _slice(window, om, 0, budget - 1), budget, False
    m0, m1 = idx[0], idx[-1]
    need = m1 - m0 + 1
    if need >= budget:
        return _slice(window, om, m0, m1), need, True
    extra = budget - need
    lo, hi = m0 - extra // 2, m1 + (extra - extra // 2)
    if lo < 0:
        hi += -lo
        lo = 0
    if hi > len(om) - 1:
        lo = max(0, lo - (hi - (len(om) - 1)))
        hi = len(om) - 1
    return _slice(window, om, lo, hi), hi - lo + 1, True


def build_queries(tok, mentions, documents: dict[str, str], budget: int, pad: int = 1):
    """Per-mention query text for one split.

    ``documents`` maps document_id to the full document string.  When a mention
    cannot be located in its document, the stored ``mention_context`` column is
    used as the window (the fallback the original pipeline applied to a handful
    of MedMentions mentions).
    """
    import numpy as np
    import pandas as pd

    span_cache: dict[str, list[tuple[int, int]]] = {}
    ctx, ntok, kept, present, located = [], [], [], [], []
    for r in mentions.itertuples(index=False):
        did = str(r.document_id)
        mt = str(r.mention_text)
        full = documents.get(did, '')
        s, e = int(r.char_start), int(r.char_end)
        if did not in span_cache:
            span_cache[did] = sentence_spans(full)
        w, ms, me = sentence_window(full, s, e, pad=pad, spans=span_cache[did])
        ok = bool(w) and 0 <= ms < len(w)
        if not ok:
            w = str(getattr(r, 'mention_context', '') or '')
            k = w.lower().find(mt.lower())
            ms, me = (k, k + len(mt)) if k >= 0 else (-1, -1)
        located.append(ms >= 0)
        c, n, k = crop_tokens(tok, w, ms, me, budget)
        ctx.append(c)
        ntok.append(n)
        kept.append(k)
        present.append(mt.lower() in c.lower())
    q = [query_of(str(mt), c) for mt, c in zip(mentions.mention_text, ctx)]
    q_tok = np.array([len(x) for x in tok(q, add_special_tokens=False)['input_ids']])
    return pd.DataFrame(dict(
        mention_id=mentions.mention_id.astype(str).to_numpy(),
        mention_text=mentions.mention_text.astype(str).to_numpy(),
        query_text=q, context_text=ctx,
        ctx_tokens=np.asarray(ntok), query_tokens=q_tok,
        mention_span_kept=np.asarray(kept), mention_text_present=np.asarray(present),
        mention_span_located=np.asarray(located)))
