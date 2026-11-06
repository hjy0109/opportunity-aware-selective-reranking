"""Primary evaluation.

    python -m src.evaluation.evaluate --dataset ncbi_disease --split test

Produces the numbers the primary results rest on, for every policy that has a
score on disk and every reranking rate in the grid:

    Acc@1, the number of mentions reranked, the cross-encoder pairs per mention,
    the composition of the selected set, and -- for each matched-rate contrast --
    a paired clustered bootstrap interval.

At a given rate every routed policy reranks exactly ``round(rate * N)`` mentions,
with the same reranker at the same candidate depth; unselected mentions keep the
retriever top-1.  ``Retriever only`` and ``Full reranking`` are the two endpoints.
``Random`` is the mean over the declared number of replicates and its spread
across replicates is reported alongside.

Writes ``evaluation/results.csv``, ``evaluation/results.json``,
``evaluation/contrasts.csv`` and ``evaluation/composition.csv``.  No manuscript
formatting, no LaTeX, no figures.
"""
from __future__ import annotations

import argparse
import os

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, write_efficiency
from ..utils.io import jdump, write_table
from ..utils.threads import limit_threads


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--split', default='test', choices=['train', 'dev', 'test'])
    ap.add_argument('--rates', nargs='+', type=float, default=None,
                    help='reranking rates (default: the configured q grid)')
    ap.add_argument('--policies', nargs='+', default=None,
                    help='restrict to these policies (default: all available)')
    ap.add_argument('--no-bootstrap', action='store_true',
                    help='skip the confidence intervals (much faster)')
    ap.add_argument('--latency', default=None,
                    help='latency CSV to join in (default: the dataset latency output '
                         'if present)')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd

    from ..baselines.simple import random_accuracy
    from ..routing.budget import (apply_selection, selection_composition,
                                  selection_summary, select_top_q)
    from . import policies as PO
    from .bootstrap import clustered_paired_ci, n_clusters

    cfg = load_config(args.dataset, args, args.method_config)
    Bud = cfg.method['budget']
    St = cfg.method['statistics']
    rates = [float(q) for q in (args.rates if args.rates is not None else Bud['q_grid'])]
    clock = Clock()

    t = pd.read_parquet(cfg.routing_table_path(args.split))
    n = len(t)
    K = int(t.K_high.iloc[0])
    groups = t.group.astype(str).to_numpy()
    c1 = t.correct_K1.to_numpy().astype(bool)
    ckh = t.correct_Khigh.to_numpy().astype(bool)
    hard, opp, ben = (t[c].to_numpy().astype(bool) for c in ('HARD', 'OPP', 'BEN'))

    scores = PO.available(cfg, t, args.split)
    wanted = args.policies or list(PO.POLICIES)
    pol_list = [p for p in wanted
                if p in ('retriever', 'full', 'random') or p in scores]
    missing = [p for p in wanted if p not in pol_list]

    lat = None
    lat_path = args.latency or cfg.out_path('latency', 'e2e_latency.csv')
    if os.path.exists(lat_path):
        lat = pd.read_csv(lat_path)

    def latency_for(policy, q):
        if lat is None:
            return None, None
        if policy == 'retriever':
            cell = 'retriever_only'
        elif policy == 'full':
            cell = 'full_reranking'
        else:
            cell = f'{policy}@q{int(round(100 * q))}'
        r = lat[lat.cell == cell]
        if not len(r):
            return None, None
        return (float(r.median_ms_per_mention.iloc[0]),
                (float(r.speedup_vs_full.iloc[0]) if 'speedup_vs_full' in r else None))

    rows, comp_rows, correctness = [], [], {}
    for pol in pol_list:
        qs = ((0.0,) if pol == 'retriever' else (1.0,) if pol == 'full' else tuple(rates))
        for q in qs:
            if pol == 'retriever':
                sel, correct, sd = np.zeros(n, bool), c1, None
            elif pol == 'full':
                sel, correct, sd = np.ones(n, bool), ckh, None
            elif pol == 'random':
                sel = None
                acc, sd = random_accuracy(c1, ckh, q, int(cfg.method['seeds']['bootstrap']),
                                          int(cfg.method['baselines']['random']['replicates']))
                correct = None
            else:
                sel = select_top_q(scores[pol], q, n)
                correct = apply_selection(sel, c1, ckh)
                sd = None
            if pol == 'random':
                nr = int(round(q * n))
                summ = dict(n=n, n_reranked=nr, rerank_rate=nr / n, acc_at_1=acc,
                            ce_pairs_per_mention=(nr / n) * (K if K > 1 else 0),
                            acc_full_reranking=float(ckh.mean()),
                            acc_retriever_only=float(c1.mean()))
            else:
                summ = selection_summary(sel, c1, ckh, K)
                correctness[(pol, round(float(q), 6))] = (correct, sel)
                comp_rows.append(dict(
                    dataset=cfg.label, dataset_key=cfg.key, split=args.split,
                    method=PO.POLICY_LABEL[pol], policy=pol, reranking_rate=round(q, 6),
                    **selection_composition(sel, hard, opp, ben, c1)))
            ms, sp = latency_for(pol, q)
            rows.append(dict(
                dataset=cfg.label, dataset_key=cfg.key, split=args.split,
                method=PO.POLICY_LABEL[pol], policy=pol,
                supervision_target=PO.SUPERVISION.get(pol, ''),
                reranking_rate=round(float(q), 6),
                reranking_rate_pct=round(100 * float(q), 4),
                n_mentions=n, n_clusters=n_clusters(groups), candidate_depth_K=K,
                number_reranked=int(summ['n_reranked']),
                actual_rate=round(float(summ['rerank_rate']), 6),
                exact_count_match=bool(int(summ['n_reranked']) == int(round(q * n))),
                acc_at_1=round(float(summ['acc_at_1']), 6),
                acc_sd_over_replicates=(None if sd is None else round(sd, 6)),
                ce_pairs_per_mention=round(float(summ['ce_pairs_per_mention']), 4),
                acc_retriever_only=round(float(summ['acc_retriever_only']), 6),
                acc_full_reranking=round(float(summ['acc_full_reranking']), 6),
                latency_ms_per_mention=ms, latency_speedup_vs_full=sp,
                is_endpoint=bool(pol in ('retriever', 'full'))))

    contrasts = []
    if not args.no_bootstrap:
        pairs = []
        for q in rates:
            key_o = ('opportunity', round(float(q), 6))
            if key_o not in correctness:
                continue
            for other in ('difficulty', 'margin', 'arr', 'arter'):
                key_b = (other, round(float(q), 6))
                if key_b in correctness:
                    pairs.append((key_o, key_b,
                                  'PRIMARY' if other == 'difficulty' else 'SECONDARY'))
            for ref, refq in (('retriever', 0.0), ('full', 1.0)):
                if (ref, refq) in correctness:
                    pairs.append((key_o, (ref, refq), 'REFERENCE'))
        for (pa, qa), (pb, qb), kind in pairs:
            ca, sa = correctness[(pa, qa)]
            cb, sb = correctness[(pb, qb)]
            pt, lo, hi, sig = clustered_paired_ci(ca, cb, groups, int(St['n_boot']),
                                                  int(St['seed']))
            contrasts.append(dict(
                dataset=cfg.label, dataset_key=cfg.key, split=args.split,
                contrast_kind=kind, method_a=PO.POLICY_LABEL[pa],
                method_b=PO.POLICY_LABEL[pb], reranking_rate=qa,
                reranking_rate_pct=round(100 * qa, 2),
                acc_a=round(float(ca.mean()), 6), acc_b=round(float(cb.mean()), 6),
                delta_acc_pp=pt, ci_low=lo, ci_high=hi, ci_excludes_zero=sig,
                number_reranked_a=int(sa.sum()), number_reranked_b=int(sb.sum()),
                matched_invocation=bool(int(sa.sum()) == int(sb.sum())),
                n_mentions=n, n_clusters=n_clusters(groups),
                n_boot=int(St['n_boot']), seed=int(St['seed']),
                bootstrap_unit='document/source cluster'))

    R = pd.DataFrame(rows)
    write_table(R, cfg.out_path('evaluation', 'results.csv'))
    write_table(pd.DataFrame(comp_rows), cfg.out_path('evaluation', 'composition.csv'))
    if contrasts:
        write_table(pd.DataFrame(contrasts), cfg.out_path('evaluation', 'contrasts.csv'))
    jdump(dict(dataset=cfg.key, label=cfg.label, split=args.split,
               candidate_depth_K=K, n_mentions=n, n_clusters=n_clusters(groups),
               reranking_rates=rates, policies_evaluated=pol_list,
               policies_unavailable=missing,
               budget_definition=Bud['definition'], selection_rule=Bud['selection'],
               unselected_rule=Bud['unselected'],
               latency_source=(lat_path if lat is not None else None),
               statistics=St, results=R.to_dict(orient='records'),
               contrasts=contrasts),
          cfg.out_path('evaluation', 'results.json'))

    show = R[R.reranking_rate.isin([0.0, 0.10, 0.30, 0.50, 1.0])]
    print(show.pivot_table(index='method', columns='reranking_rate_pct',
                           values='acc_at_1').round(6).to_string())
    if missing:
        print(f'\nnot evaluated (no score on disk): {", ".join(missing)}')
    if contrasts:
        C = pd.DataFrame(contrasts)
        print('\nOpportunity vs Difficulty (matched rate):')
        print(C[C.contrast_kind == 'PRIMARY'][
            ['reranking_rate_pct', 'acc_a', 'acc_b', 'delta_acc_pp', 'ci_low', 'ci_high',
             'ci_excludes_zero', 'matched_invocation']].to_string(index=False))
    write_efficiency([eff_row('evaluate', clock, dataset=cfg.key, split=args.split,
                              device='CPU', gpu_allocations=0, neural_inference_calls=0,
                              n_rows=len(R), n_contrasts=len(contrasts))],
                     cfg.out_path('efficiency', 'evaluate.csv'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
