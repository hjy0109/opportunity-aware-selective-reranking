"""Router inference and selective reranking.

    python -m src.routing.infer --dataset ncbi_disease --target OPP --split test --rate 0.10

At inference time the router sees only the eight retrieval-score features.  The
gold concept, the candidate depth label, and any cross-encoder output are not
inputs.  Mentions are ranked by router score, the top ``round(rate * N)`` are
reranked, and the rest keep the retriever top-1 prediction.

Writes ``routing/decisions_{target}_{split}_q{rate}.parquet`` with, per mention,
the router score, the routing decision and the resulting prediction, plus a JSON
summary of accuracy, invocation count and cost.
"""
from __future__ import annotations

import argparse

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, write_efficiency
from ..utils.io import jdump, write_table
from ..utils.threads import limit_threads
from .targets import TARGET_LABEL


def router_scores(cfg, target, split, np):
    """Cached router scores for a split, aligned to the routing table."""
    import pandas as pd
    z = np.load(cfg.router_scores_path(target), allow_pickle=True)
    t = pd.read_parquet(cfg.routing_table_path(split))
    ids = z[f'mention_id_{split}'].astype(str)
    s = pd.Series(z[split].astype(float), index=pd.Index(ids))
    v = s.reindex(t.mention_id.astype(str)).to_numpy()
    if pd.isna(v).any():
        raise SystemExit(f'{cfg.key}/{split}/{target}: {int(pd.isna(v).sum())} '
                         f'mentions have no router score')
    return t, v


def score_from_features(cfg, target, X, np):
    """Score a fresh feature matrix with the saved router (used by the latency
    benchmark and by anyone applying the router to new mentions)."""
    from . import classifiers as CL
    from .features import apply_standardiser
    z = np.load(cfg.router_scores_path(target), allow_pickle=True)
    model = CL.load(cfg.method['router']['family'], cfg.router_model_path(target))
    return model.predict_proba(apply_standardiser(X, z['mu'], z['sd']))[:, 1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--target', default='OPP', choices=['OPP', 'HARD', 'BEN'])
    ap.add_argument('--split', default='test', choices=['train', 'dev', 'test'])
    ap.add_argument('--rate', type=float, default=0.10,
                    help='reranking rate q = B / N (0.10 = rerank 10%% of mentions)')
    ap.add_argument('--recompute-features', action='store_true',
                    help='recompute the features from the candidate cache and score them '
                         'with the saved router, instead of reading the cached scores')
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd

    from .budget import apply_selection, selection_composition, selection_summary, select_top_q

    cfg = load_config(args.dataset, args, args.method_config)
    clock = Clock()
    t, s = router_scores(cfg, args.target, args.split, np)
    if args.recompute_features:
        from ..retrieval.index import retrieval_score_matrix
        from .features import TIE_EPS, feature_matrix
        cand = pd.read_parquet(cfg.candidates_path(args.split))
        X, _ = feature_matrix(retrieval_score_matrix(
            cand, int(cfg.method['retriever']['k_pool'])), TIE_EPS)
        s2 = score_from_features(cfg, args.target, X, np)
        print(f'  cached vs recomputed router score: max|delta| '
              f'{float(np.abs(s - s2).max()):.3e}', flush=True)
        s = s2

    n = len(t)
    K = int(t.K_high.iloc[0])
    sel = select_top_q(s, args.rate, n)
    correct = apply_selection(sel, t.correct_K1.to_numpy(), t.correct_Khigh.to_numpy())
    summ = selection_summary(sel, t.correct_K1.to_numpy(), t.correct_Khigh.to_numpy(), K)
    comp = selection_composition(sel, t.HARD.to_numpy(), t.OPP.to_numpy(),
                                 t.BEN.to_numpy(), t.correct_K1.to_numpy())

    qtag = f'q{int(round(100 * args.rate))}'
    write_table(pd.DataFrame(dict(
        mention_id=t.mention_id, group=t.group, router_score=s, reranked=sel,
        correct=correct, correct_K1=t.correct_K1, correct_Khigh=t.correct_Khigh,
        HARD=t.HARD, OPP=t.OPP, BEN=t.BEN)),
        cfg.out_path('routing', f'decisions_{args.target}_{args.split}_{qtag}.parquet'))
    out = dict(dataset=cfg.key, target=args.target,
               target_label=TARGET_LABEL[args.target],
               split=args.split, requested_rate=args.rate, K_high=K,
               n_selected_expected=int(round(args.rate * n)),
               exact_count_match=bool(int(sel.sum()) == int(round(args.rate * n))),
               selection_rule=cfg.method['budget']['selection'],
               unselected_rule=cfg.method['budget']['unselected'],
               **summ, **comp)
    jdump(out, cfg.out_path('routing', f'summary_{args.target}_{args.split}_{qtag}.json'))
    print(f'[{cfg.key}/{args.split}] {out["target_label"]} @ q={100 * args.rate:.0f}% · '
          f'reranked {summ["n_reranked"]:,}/{n:,} · Acc@1 {summ["acc_at_1"]:.6f} '
          f'(retriever {summ["acc_retriever_only"]:.6f} · '
          f'full {summ["acc_full_reranking"]:.6f}) · '
          f'CE pairs/mention {summ["ce_pairs_per_mention"]:.2f}', flush=True)
    write_efficiency([eff_row('router_inference', clock, dataset=cfg.key,
                              target=args.target, split=args.split, device='CPU',
                              gpu_allocations=0, n_mentions=n,
                              n_reranked=summ['n_reranked'])],
                     cfg.out_path('efficiency', f'infer_{args.target}_{args.split}.csv'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
