"""Per-mention routing table.

    python -m src.routing.table --dataset ncbi_disease --splits train dev test

One row per mention holding everything the router, the baselines and the
evaluation need:

    mention_id, example_id, group          identity and the bootstrap cluster
    gold_rank                              1-based retrieval rank of the gold concept
                                            (0 when it is outside the pool)
    correct_K1                             retriever top-1 is the gold concept
    correct_Khigh                          reranking to K_high yields the gold concept
    HARD, OPP, BEN                         supervision targets (training-time labels)
    <CURRENT8>                             the eight router features
    K_high                                 the deployed candidate depth

For the training split, ``correct_Khigh`` comes from the leakage-safe out-of-fold
rerankers, each fused with its own fold's alpha; for dev and test it comes from the
full-train reranker with the frozen alpha.  This asymmetry is deliberate: an
in-sample reranking outcome would make the BEN diagnostic look optimistic.

``correct_K1``, ``gold_rank`` and the eight features are retrieval-only, so the
Opportunity and Difficulty labels do not depend on any cross-encoder call.
"""
from __future__ import annotations

import argparse

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, write_efficiency
from ..utils.io import jload, write_table
from ..utils.threads import limit_threads


def _fold_alpha(cfg, fold, FU):
    from ..reranking.select_depth import load_scored
    F = cfg.method['fusion']
    _, ret, ce, valid, gold = load_scored(cfg, 'internal_val', fold)
    return FU.select_alpha(ret, ce, valid, gold, F['alpha_grid'],
                           int(F['k_ref']), int(F['k_select']))[:2]


def correct_at_depth(cfg, split, k_high, FU, np, pd):
    """(mention ids, correct_Khigh) for one split under the frozen reranking action."""
    from ..reranking.select_depth import load_scored

    if split != 'train':
        fz = jload(cfg.fusion_path())
        cand, ret, ce, valid, gold = load_scored(cfg, split, -1)
        fused = FU.fuse(ret, ce, valid, fz['calibration'], float(fz['selected_alpha']))
        corr = FU.correct_prefix(fused, gold)
        return (cand.mention_id.astype(str).to_numpy(),
                corr[:, int(k_high) - 1], f'full-train reranker, frozen alpha '
                f'{fz["selected_alpha"]}')

    folds = pd.read_parquet(cfg.folds_path())
    n_folds = int(folds.fold.max()) + 1
    ids, ok = [], []
    for f in range(n_folds):
        alpha, cal = _fold_alpha(cfg, f, FU)
        cand, ret, ce, valid, gold = load_scored(cfg, 'out_of_fold', f)
        fused = FU.fuse(ret, ce, valid, cal, alpha)
        corr = FU.correct_prefix(fused, gold)
        ids.append(cand.mention_id.astype(str).to_numpy())
        ok.append(corr[:, int(k_high) - 1])
    return (np.concatenate(ids), np.concatenate(ok),
            f'out-of-fold rerankers ({n_folds} folds), per-fold alpha')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    ap.add_argument('--k-high', type=int, default=None,
                    help='override the candidate depth (defaults to the selected value)')
    ap.add_argument('--no-benefit', action='store_true',
                    help='skip BEN (leaves correct_Khigh unset); use when the '
                         'out-of-fold rerankers have not been trained')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd

    from ..reranking import fusion as FU
    from ..retrieval.index import retrieval_score_matrix
    from . import targets as TG
    from .features import CURRENT8, TIE_EPS, feature_matrix

    cfg = load_config(args.dataset, args, args.method_config)
    clock = Clock()
    k_high = args.k_high
    if k_high is None:
        try:
            k_high = int(jload(cfg.depth_path())['selected_K_high'])
        except OSError:
            k_high = cfg.method['candidate_depth']['K_high']
    if k_high is None:
        raise SystemExit('no candidate depth: run src.reranking.select_depth or pass --k-high')
    k_high = int(k_high)
    group_key = cfg.dataset['grouping']['key']

    summary = []
    for split in args.splits:
        cand = pd.read_parquet(cfg.candidates_path(split))
        R = retrieval_score_matrix(cand, int(cfg.method['retriever']['k_pool']))
        _, F = feature_matrix(R, TIE_EPS)
        gr = cand.gold_rank_in_pool.to_numpy(dtype=np.int64)
        gold_rank = np.where(gr >= 1, gr, 0)
        c1 = gold_rank == 1
        hard = TG.hardness(c1)
        opp = TG.opportunity(c1, gold_rank, k_high)

        if args.no_benefit:
            ckh = np.zeros(len(cand), bool)
            ben = np.zeros(len(cand), bool)
            prov = 'correct_Khigh NOT COMPUTED (--no-benefit)'
        else:
            ids, ok, prov = correct_at_depth(cfg, split, k_high, FU, np, pd)
            s = pd.Series(ok, index=pd.Index(ids.astype(str)))
            ckh = s.reindex(cand.mention_id.astype(str)).to_numpy()
            if pd.isna(ckh).any():
                raise SystemExit(f'{cfg.key}/{split}: missing reranking outcome for '
                                 f'{int(pd.isna(ckh).sum())} mentions')
            ckh = ckh.astype(bool)
            ben = TG.benefit(c1, ckh)
            TG.check_containment(hard, opp, ben)

        t = pd.DataFrame(dict(
            dataset=cfg.key, split=split, K_high=k_high,
            mention_id=cand.mention_id.astype(str), example_id=cand.example_id.astype(str),
            group=cand[group_key].astype(str), gold_rank=gold_rank,
            correct_K1=c1, correct_Khigh=ckh, HARD=hard, OPP=opp, BEN=ben,
            **{c: np.asarray(F[c], dtype=np.float64) for c in CURRENT8}))
        write_table(t, cfg.routing_table_path(split))
        summary.append(dict(
            dataset=cfg.key, split=split, n=len(t), n_clusters=int(t.group.nunique()),
            K_high=k_high, acc_retriever_only=round(float(c1.mean()), 6),
            acc_full_reranking=round(float(ckh.mean()), 6),
            HARD_pct=round(100 * float(hard.mean()), 4),
            OPP_pct=round(100 * float(opp.mean()), 4),
            BEN_pct=round(100 * float(ben.mean()), 4),
            P_OPP_given_HARD=round(float(opp.sum() / max(hard.sum(), 1)), 6),
            P_BEN_given_OPP=round(float(ben.sum() / max(opp.sum(), 1)), 6),
            reranking_outcome_provenance=prov))
        s = summary[-1]
        print(f'  [{cfg.key}/{split}] n {s["n"]:,} · clusters {s["n_clusters"]:,} · '
              f'Acc@1 {s["acc_retriever_only"]:.4f} · full {s["acc_full_reranking"]:.4f} · '
              f'HARD {s["HARD_pct"]:.2f}% · OPP {s["OPP_pct"]:.2f}% · '
              f'BEN {s["BEN_pct"]:.2f}%', flush=True)

    write_table(pd.DataFrame(summary), cfg.out_path('audit', 'routing_table_summary.csv'))
    write_efficiency([eff_row('routing_table', clock, dataset=cfg.key, device='CPU',
                              gpu_allocations=0, neural_inference_calls=0, K_high=k_high)],
                     cfg.out_path('efficiency', 'routing_table.csv'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
