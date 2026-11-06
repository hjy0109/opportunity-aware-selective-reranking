"""Reranker query construction entry point.

    python -m src.data.queries --dataset ncbi_disease

Writes ``queries/queries_{split}.parquet`` plus a structural audit (mention-span
retention, pair-length overflow, candidate-text truncation).  The context and
pair token budgets are read from the config and are not re-selected here.
"""
from __future__ import annotations

import argparse
import gzip
import json

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, write_efficiency
from ..utils.io import jdump, write_table
from ..utils.threads import limit_threads


def _pct(a, ps=(50, 90, 95, 99, 100)):
    import numpy as np
    a = np.asarray(a)
    return {f'p{p}': (int(np.percentile(a, p)) if len(a) else 0) for p in ps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    from ..data.context import build_queries
    from ..data.ontology import load_concept_representation

    cfg = load_config(args.dataset, args, args.method_config)
    ctx_cfg = cfg.method['context']
    clock = Clock()
    tok = AutoTokenizer.from_pretrained(cfg.method['reranker']['backbone'])
    with gzip.open(cfg.documents_path(), 'rt', encoding='utf8') as f:
        documents = json.load(f)
    repr_map = load_concept_representation(cfg.concept_repr_path())

    audit = {}
    for split in args.splits:
        cand = pd.read_parquet(cfg.candidates_path(split))
        q = build_queries(tok, cand, documents[split], int(ctx_cfg['max_context_tokens']),
                          int(ctx_cfg['sentence_pad']))
        write_table(q, cfg.queries_path(split))

        uniq = sorted({repr_map[str(c)] for row in cand.cand_concept_id
                       for c in row if c is not None})
        clen = dict(zip(uniq, [len(x) for x in tok(uniq, add_special_tokens=False)['input_ids']]))
        max_pair = int(ctx_cfg['max_pair_tokens'])
        tot, ctrunc = [], 0
        qtok = q.query_tokens.to_numpy()
        for i, row in enumerate(cand.cand_concept_id):
            qq = int(qtok[i])
            for cc in row:
                if cc is None:
                    continue
                t = qq + clen[repr_map[str(cc)]] + 3      # [CLS] q [SEP] cand [SEP]
                tot.append(t)
                ctrunc += int(t > max_pair)
        tot = np.asarray(tot)
        audit[split] = dict(
            n=int(len(cand)), n_documents=int(cand.document_id.nunique()),
            target_mention_span_retained=round(float(q.mention_span_kept.mean()), 6),
            mention_text_in_context=round(float(q.mention_text_present.mean()), 6),
            mention_span_located=round(float(q.mention_span_located.mean()), 6),
            context_tokens=_pct(q.ctx_tokens), query_tokens=_pct(q.query_tokens),
            pair_tokens=_pct(tot), n_pairs_measured=int(len(tot)),
            pair_overflow_rate=round(float((tot > max_pair).mean()), 6),
            candidate_text_truncated_pairs=int(ctrunc))
        print(f'  [{cfg.key}/{split}] n {audit[split]["n"]:,} · span retained '
              f'{audit[split]["target_mention_span_retained"]:.5f} · context p99 '
              f'{audit[split]["context_tokens"]["p99"]} · pair overflow '
              f'{audit[split]["pair_overflow_rate"]:.5f}', flush=True)

    jdump(dict(dataset=cfg.key, recipe=ctx_cfg, budget_reselected_here=False,
               splits=audit), cfg.out_path('audit', 'context.json'))
    write_efficiency([eff_row('build_queries', clock, dataset=cfg.key, device='CPU',
                              gpu_allocations=0, neural_inference_calls=0)],
                     cfg.out_path('efficiency', 'queries.csv'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
