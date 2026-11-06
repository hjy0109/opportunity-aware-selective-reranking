"""Candidate retrieval entry point.

    CUDA_VISIBLE_DEVICES=0 python -m src.retrieval.run --dataset ncbi_disease

For every requested split this caches the top-``k_pool`` candidate concepts with
their retrieval scores and the gold rank, and reports retrieval headroom
(Acc@1, Recall@K, the share of mentions whose gold falls outside the pool).

Writes:
    candidates/candidates_{split}.parquet
    audit/retrieval_headroom.csv
    efficiency/retrieval_{split}.csv     encode/search seconds, throughput, VRAM peak
"""
from __future__ import annotations

import argparse
import time

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, model_size, write_efficiency
from ..utils.io import write_table
from ..utils.threads import limit_threads


def headroom_row(cfg, split, cand, k_grid):
    import numpy as np
    gr = cand.gold_concept_rank.to_numpy()
    c1 = (gr == 1)
    row = dict(dataset=cfg.key, label=cfg.label, split=split,
               n_mentions=int(len(cand)), n_documents=int(cand.document_id.nunique()),
               retriever=cfg.method['retriever']['model'],
               aggregation=cfg.method['retriever']['aggregation'],
               normalize=cfg.method['retriever']['normalize'],
               gold_in_universe_pct=round(100 * float(cand.gold_in_universe.mean()), 4),
               acc_at_1=round(float(c1.mean()), 6),
               HARD_prevalence=round(float((~c1).mean()), 6))
    for K in (1, 3, 5, 10, 20, 50, 64):
        row[f'recall_at_{K}'] = round(float(((gr > 0) & (gr <= K)).mean()), 6)
    for K in k_grid:
        row[f'opportunity_prevalence_K{K}'] = round(
            float(((~c1) & (gr > 1) & (gr <= K)).mean()), 6)
    row['recall_at_64_minus_acc_at_1_pp'] = round(
        100 * (row['recall_at_64'] - row['acc_at_1']), 4)
    row['frac_gold_rank_gt_64'] = round(float(((gr < 0) | (gr > 64)).mean()), 6)
    inb = gr[gr > 0]
    row['median_gold_rank'] = float(np.median(inb)) if len(inb) else None
    for p in (90, 95, 99):
        row[f'p{p}_gold_rank'] = float(np.percentile(inb, p)) if len(inb) else None
    row['gold_found_within_rank_depth'] = round(float((gr > 0).mean()), 6)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    ap.add_argument('--device', default='cuda', help='cuda or cpu')
    ap.add_argument('--k-pool', type=int, default=None,
                    help='override retriever.k_pool (candidates cached per mention)')
    ap.add_argument('--rank-depth', type=int, default=None,
                    help='override retriever.rank_depth (depth at which gold rank is measured)')
    ap.add_argument('--encode-batch', type=int, default=None)
    ap.add_argument('--name-block', type=int, default=None)
    ap.add_argument('--query-chunk', type=int, default=None)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd
    import torch

    from ..data import ontology
    from . import index
    from .encoder import encode_dense, load_encoder

    cfg = load_config(args.dataset, args, args.method_config)
    R = dict(cfg.method['retriever'])
    for flag, key in (('k_pool', 'k_pool'), ('rank_depth', 'rank_depth'),
                      ('encode_batch', 'encode_batch'), ('name_block', 'name_block'),
                      ('query_chunk', 'query_chunk')):
        v = getattr(args, flag)
        if v is not None:
            R[key] = v
    k_pool = int(R['k_pool'])
    device = args.device
    clock = Clock()

    names, name2concept, concept_ids = ontology.kb_arrays(cfg.kb_path())
    model, tok = load_encoder(R['model'], device)
    size = model_size(model)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    print(f'[{cfg.key}] KB {len(concept_ids):,} concepts / {len(names):,} aliases · '
          f'retriever {R["model"]} ({size["n_parameters_M"]}M) · aggregation '
          f'{R["aggregation"]} · device {device}', flush=True)

    eff_rows, head_rows = [], []
    for split in args.splits:
        c1 = Clock()
        m = pd.read_parquet(cfg.mentions_path(split))
        cid_index = {c: i for i, c in enumerate(concept_ids)}
        gold_idx = np.array([cid_index.get(str(g), -1) for g in m.gold_concept_id],
                            dtype=np.int64)
        t0 = time.time()
        q, q_sec = encode_dense(model, tok, m.mention_text.astype(str).tolist(), device,
                                max_len=int(R['max_len']), batch=int(R['encode_batch']),
                                desc=f'{cfg.key}/{split}:queries')
        if R['aggregation'] == 'concept_max_streaming':
            top_i, top_s, kb_sec, search_sec = index.concept_rank_streaming(
                model, tok, names, name2concept, len(concept_ids), q, device,
                k_max=int(R['rank_depth']), max_len=int(R['max_len']),
                encode_batch=int(R['encode_batch']), name_block=int(R['name_block']),
                q_chunk=int(R['query_chunk']))
            alias_idx = None
        elif R['aggregation'] == 'alias_topn_dedup':
            ni, ns, kb_sec, search_sec = index.alias_topn(
                model, tok, names, q, device, topn=int(R['name_topn']),
                max_len=int(R['max_len']), encode_batch=int(R['encode_batch']),
                name_block=int(R['name_block']), q_chunk=int(R['query_chunk']))
            top_i, alias_idx, top_s, n_found = index.dedup_to_concepts(
                ni, ns, name2concept, k_pool)
        else:
            raise ValueError(f'unknown retriever.aggregation: {R["aggregation"]}')
        total_sec = time.time() - t0
        del q
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        rank = index.gold_rank(top_i, gold_idx)
        kp = min(k_pool, top_i.shape[1])
        cand = m.copy()
        cand['gold_concept_index'] = gold_idx
        cand['gold_in_universe'] = gold_idx >= 0
        cand['gold_concept_rank'] = rank
        cand['gold_rank_in_pool'] = np.where((rank >= 1) & (rank <= kp), rank, -1).astype(np.int32)
        if R['aggregation'] == 'alias_topn_dedup':
            cand['n_candidates'] = n_found.astype(np.int32)
            cand['cand_concept_id'] = [
                [(concept_ids[j] if j >= 0 else None) for j in row] for row in top_i[:, :kp]]
            cand['cand_alias'] = [
                [(names[j] if j >= 0 else None) for j in row] for row in alias_idx[:, :kp]]
            cand['cand_ret_score'] = [row.astype(np.float32).tolist() for row in top_s[:, :kp]]
        else:
            cand['n_candidates'] = np.int32(kp)
            cand['cand_concept_id'] = [list(concept_ids[top_i[i, :kp]]) for i in range(len(m))]
            cand['cand_ret_score'] = [list(top_s[i, :kp].astype(np.float32))
                                      for i in range(len(m))]
        cand['gold_injected'] = cand.gold_rank_in_pool.to_numpy() < 0
        write_table(cand, cfg.candidates_path(split))
        head_rows.append(headroom_row(cfg, split, cand,
                                      cfg.method['candidate_depth']['K_grid']))

        peak = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                if device.startswith('cuda') else None)
        eff_rows.append(eff_row(
            'retrieval', c1, dataset=cfg.key, split=split,
            device=('CUDA' if device.startswith('cuda') else 'CPU'),
            gpu=(torch.cuda.get_device_name() if device.startswith('cuda') else None),
            model=R['model'], model_dtype=R['dtype'],
            model_parameters_M=size['n_parameters_M'],
            n_concepts=len(concept_ids), n_aliases=len(names), n_mentions=int(len(m)),
            encode_batch=int(R['encode_batch']), max_len=int(R['max_len']),
            kb_encode_sec=round(kb_sec, 2),
            aliases_per_sec=round(len(names) / max(kb_sec, 1e-9), 1),
            query_encode_sec=round(q_sec, 2),
            queries_per_sec=round(len(m) / max(q_sec, 1e-9), 1),
            search_sec=round(search_sec, 2), retrieval_total_sec=round(total_sec, 1),
            ms_per_mention=round(1000 * (q_sec + search_sec) / max(len(m), 1), 4),
            gpu_peak_GiB=peak))
        h = head_rows[-1]
        print(f'  [{cfg.key}/{split}] n {len(m):,} · Acc@1 {h["acc_at_1"]:.4f} · '
              f'R@64 {h["recall_at_64"]:.4f} · '
              f'R@64-Acc@1 {h["recall_at_64_minus_acc_at_1_pp"]:.2f} pp · '
              f'{total_sec:.0f}s', flush=True)

    write_table(pd.DataFrame(head_rows), cfg.out_path('audit', 'retrieval_headroom.csv'))
    eff_rows.append(eff_row('retrieval_total', clock, dataset=cfg.key,
                            device=('CUDA' if device.startswith('cuda') else 'CPU')))
    write_efficiency(eff_rows, cfg.out_path('efficiency', 'retrieval.csv'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
