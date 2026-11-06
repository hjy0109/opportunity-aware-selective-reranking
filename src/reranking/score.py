"""Cross-encoder scoring entry point.

    CUDA_VISIBLE_DEVICES=0 python -m src.reranking.score --dataset ncbi_disease --fold -1 --splits internal_val dev test
    CUDA_VISIBLE_DEVICES=0 python -m src.reranking.score --dataset ncbi_disease --fold 0  --splits internal_val out_of_fold

``internal_val`` is the train-internal validation holdout, the split on which the
fusion weight and z statistics are estimated.  ``out_of_fold`` is the held-out
fold of an out-of-fold reranker.  Writes ``ce_scores/{split}_{fold}.npz`` with
``ce`` (n, k_pool), ``valid``, ``mention_id`` and ``gold_rank_in_pool``.

Scoring the dev or test split here does not constitute using it for selection:
nothing downstream selects alpha, K_high, a router or a threshold on those scores.
"""
from __future__ import annotations

import argparse
import json

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, model_size, write_efficiency
from ..utils.io import ensure_parent
from ..utils.threads import limit_threads

SPLIT_CHOICES = ('internal_val', 'out_of_fold', 'train', 'dev', 'test')


def _frames(cfg, fold, which, np, pd, split_internal_val):
    """(frame, query map) for a requested scoring target."""
    cand_tr = pd.read_parquet(cfg.candidates_path('train'))
    q_tr = pd.read_parquet(cfg.queries_path('train'))
    qmap_tr = dict(zip(q_tr.mention_id.astype(str), q_tr.query_text.astype(str)))
    if which in ('internal_val', 'out_of_fold', 'train'):
        folds = pd.read_parquet(cfg.folds_path())
        fid = folds.fold.to_numpy()
        part = cand_tr if fold < 0 else cand_tr[fid != fold].reset_index(drop=True)
        if which == 'train':
            return cand_tr, qmap_tr
        if which == 'out_of_fold':
            if fold < 0:
                raise SystemExit('--splits out_of_fold requires --fold 0/1/2')
            return cand_tr[fid == fold].reset_index(drop=True), qmap_tr
        _, iv, _, _ = split_internal_val(
            part, int(cfg.method['seeds']['internal_val']),
            cfg.method['reranker']['internal_val'])
        return iv, qmap_tr
    cand = pd.read_parquet(cfg.candidates_path(which))
    q = pd.read_parquet(cfg.queries_path(which))
    return cand, dict(zip(q.mention_id.astype(str), q.query_text.astype(str)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--fold', type=int, default=-1,
                    help='-1 = the full-train reranker; 0/1/2 = an out-of-fold reranker')
    ap.add_argument('--splits', nargs='+', default=['internal_val', 'dev', 'test'],
                    choices=SPLIT_CHOICES)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=None, help='override reranker.eval_batch')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd
    import torch

    from ..data.ontology import load_concept_representation
    from ..data.splits import split_internal_val
    from . import cross_encoder as CE

    cfg = load_config(args.dataset, args, args.method_config)
    R = cfg.method['reranker']
    k_pool = int(cfg.method['retriever']['k_pool'])
    max_len = int(cfg.method['context']['max_pair_tokens'])
    batch = int(args.batch or R['eval_batch'])
    device = args.device
    fold = args.fold
    fkey = 'full' if fold < 0 else f'f{fold}'
    clock = Clock()

    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
    model, tok = CE.load_checkpoint(cfg.ce_checkpoint(fold), device)
    size = model_size(model)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    repr_map = load_concept_representation(cfg.concept_repr_path())

    eff_rows = []
    for which in args.splits:
        c1 = Clock()
        frame, qmap = _frames(cfg, fold, which, np, pd, split_internal_val)
        ce, valid, npair, sec = CE.score_candidate_pool(
            model, tok, frame, qmap, repr_map, device, k_pool, max_len, batch,
            R['truncation'])
        peak = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                if device.startswith('cuda') else None)
        path = cfg.ce_scores_path(which, fold)
        ensure_parent(path)
        np.savez_compressed(
            path, ce=ce.astype(np.float32), valid=valid,
            mention_id=frame.mention_id.astype(str).to_numpy(),
            example_id=frame.example_id.astype(str).to_numpy(),
            gold_rank_in_pool=frame.gold_rank_in_pool.to_numpy().astype(np.int32),
            meta=json.dumps(dict(dataset=cfg.key, fold=int(fold), fold_key=fkey,
                                 which=which, checkpoint=cfg.ce_checkpoint(fold),
                                 n=int(len(frame)), n_pairs=int(npair), k_pool=k_pool,
                                 truncation=R['truncation'], max_len=max_len,
                                 precision=R['precision'],
                                 USED_FOR_SELECTION=which in ('internal_val',)),
                            ensure_ascii=False))
        eff_rows.append(eff_row(
            'score_candidates', c1, dataset=cfg.key, fold_key=fkey, split=which,
            device=('CUDA' if device.startswith('cuda') else 'CPU'),
            gpu=(torch.cuda.get_device_name() if device.startswith('cuda') else None),
            model=R['backbone'], model_parameters_M=size['n_parameters_M'],
            n_mentions=int(len(frame)), n_ce_pairs=int(npair),
            inference_sec=round(sec, 1),
            pairs_per_sec=round(npair / max(sec, 1e-9), 1),
            ms_per_mention=round(1000 * sec / max(len(frame), 1), 4),
            gpu_peak_GiB=peak, eval_batch=batch))
        print(f'  [{cfg.key}/{fkey}/{which}] n {len(frame):,} · pairs {npair:,} · '
              f'{sec:.0f}s · {npair / max(sec, 1e-9):.0f} pair/s', flush=True)

    eff_rows.append(eff_row('score_candidates_total', clock, dataset=cfg.key,
                            fold_key=fkey))
    write_efficiency(eff_rows, cfg.out_path('efficiency', f'score_{fkey}.csv'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
