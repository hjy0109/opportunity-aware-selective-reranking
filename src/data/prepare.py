"""Dataset preparation entry point.

    python -m src.data.prepare --dataset ncbi_disease

Writes, under ``OUTPUT_DIR/<dataset>/``:

    kb/kb.parquet                        concept_id, name  (one row per alias)
    kb/concept_representation.parquet     concept_id, a1, a2, n_distinct, representation
    mentions/mentions_{split}.parquet     one row per mention
    mentions/documents.json.gz            {split: {document_id: full text}}
    mentions/folds.parquet                document-grouped 3-fold assignment (train)
    audit/data_preparation.json           counts, exclusions and coverage
    efficiency/prepare.csv                wall time and peak memory

CPU only; no model is loaded.
"""
from __future__ import annotations

import argparse
import gzip
import json

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, write_efficiency
from ..utils.io import ensure_parent, jdump, sha256_ids, write_table
from ..utils.threads import limit_threads


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'],
                    help='splits to prepare (default: all three official splits)')
    ap.add_argument('--n-folds', type=int, default=3,
                    help='document-grouped folds over the training split')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd

    from . import corpora, ontology
    from .splits import SPLIT_RULE, document_folds

    cfg = load_config(args.dataset, args, args.method_config)
    clock = Clock()

    kb, kb_meta = ontology.build_kb(cfg)
    write_table(kb, cfg.kb_path())
    rep = ontology.build_concept_representation(
        kb, cfg.method['concept_representation']['separator'])
    write_table(rep, cfg.concept_repr_path())
    universe = set(kb.concept_id.astype(str))
    n_single = int(rep.n_distinct.eq(1).sum())

    loaded = corpora.load_corpus(cfg, tuple(args.splits))
    stats, docs_all = {}, {}
    for sp in args.splits:
        m, docs, st = loaded[sp]
        m = m.copy()
        m['mention_context'] = corpora.mention_context_column(
            m, docs, cfg.method['context']['sentence_pad'])
        st.update(corpora.gold_coverage(m, universe))
        write_table(m, cfg.mentions_path(sp))
        stats[sp] = st
        docs_all[sp] = docs
        print(f'  [{cfg.key}/{sp}] mentions {st["n_mentions"]:,} · '
              f'documents {st["n_documents"]:,} · '
              f'gold in universe {st["gold_in_universe_pct"]}% · '
              f'composite excluded {st["n_composite"]}', flush=True)

    ensure_parent(cfg.documents_path())
    with gzip.open(cfg.documents_path(), 'wt', encoding='utf8') as f:
        json.dump(docs_all, f)

    fold_meta = {}
    if 'train' in args.splits:
        tr = pd.read_parquet(cfg.mentions_path('train'))
        fid, _, doc_fold, _ = document_folds(
            tr.document_id.astype(str).to_numpy(), args.n_folds,
            cfg.method['seeds']['fold'])
        write_table(pd.DataFrame(dict(mention_id=tr.mention_id, example_id=tr.example_id,
                                      document_id=tr.document_id, fold=fid.astype(int))),
                    cfg.folds_path())
        per_doc = (pd.DataFrame(dict(d=tr.document_id.astype(str), f=fid))
                   .groupby('d').f.nunique())
        leak = []
        for i in range(args.n_folds):
            rest = set(tr.document_id.astype(str)[fid != i])
            leak.append(round(100 * float(np.isin(tr.document_id.astype(str)[fid == i],
                                                  list(rest)).mean()), 6))
        fold_meta = dict(rule=SPLIT_RULE, group_key=cfg.dataset['grouping']['key'],
                         n_folds=args.n_folds, seed=cfg.method['seeds']['fold'],
                         sizes=[int((fid == i).sum()) for i in range(args.n_folds)],
                         n_documents=[int((doc_fold == i).sum()) for i in range(args.n_folds)],
                         n_documents_split_across_folds=int((per_doc > 1).sum()),
                         held_out_document_leakage_pct=leak,
                         group_leakage_zero=bool(int((per_doc > 1).sum()) == 0
                                                 and max(leak) == 0.0),
                         stratified_on_outcome=False)

    audit = dict(
        dataset=cfg.key, label=cfg.label, splits_prepared=list(args.splits),
        knowledge_base=dict(name=cfg.dataset['knowledge_base']['name'], **kb_meta),
        concept_representation=dict(
            n_concepts=int(len(rep)), n_single_alias_concepts=n_single,
            pct_single_alias=round(100 * n_single / max(len(rep), 1), 3),
            degenerates_to_single_alias_note=(
                'a concept with one distinct alias has no second alias to append'),
            **{k: v for k, v in cfg.method['concept_representation'].items()}),
        grouping=cfg.dataset['grouping'], split_statistics=stats, folds=fold_meta,
        mention_ids_sha256={sp: sha256_ids(
            pd.read_parquet(cfg.mentions_path(sp)).mention_id) for sp in args.splits})
    jdump(audit, cfg.out_path('audit', 'data_preparation.json'))
    write_efficiency([eff_row('prepare_data', clock, dataset=cfg.key, device='CPU',
                              gpu_allocations=0, neural_inference_calls=0,
                              n_concepts=int(rep.shape[0]), n_aliases=int(len(kb)),
                              **{f'n_mentions_{sp}': stats[sp]['n_mentions']
                                 for sp in args.splits})],
                     cfg.out_path('efficiency', 'prepare.csv'))
    print(f'prepared {cfg.key} -> {cfg.out_path()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
