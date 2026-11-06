"""Fusion weight and candidate depth selection.

    python -m src.reranking.select_depth --dataset ncbi_disease

Two selections, in the order the experiments made them:

1. ``alpha`` and the z statistics, on the **train-internal validation** split only,
   by Acc@1 at K = 64.  Ties go to the first value in the declared grid.
2. ``K_high``, on the **dev** split: the smallest K in the declared grid whose
   fused Acc@1 is within 0.5 pp of the best fixed K.  Routing results are not
   consulted, and the test split is not read.

Writes ``fusion/fusion_selection.json`` and ``fusion/candidate_depth.json``.
Both files record the frozen reference value from the config next to the value
selected here, so a mismatch is visible rather than silent.
"""
from __future__ import annotations

import argparse

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, write_efficiency
from ..utils.io import jdump, write_table
from ..utils.threads import limit_threads


def load_scored(cfg, which, fold=-1):
    """(candidate frame aligned to the npz, retrieval matrix, ce, valid, gold_idx)."""
    import numpy as np
    import pandas as pd

    from ..retrieval.index import retrieval_score_matrix

    z = np.load(cfg.ce_scores_path(which, fold), allow_pickle=True)
    ids = z['mention_id'].astype(str)
    base = 'train' if which in ('internal_val', 'out_of_fold', 'train') else which
    cand = pd.read_parquet(cfg.candidates_path(base))
    cand = cand.set_index(cand.mention_id.astype(str)).loc[ids].reset_index(drop=True)
    ret = retrieval_score_matrix(cand, int(cfg.method['retriever']['k_pool']))
    gr = cand.gold_rank_in_pool.to_numpy(dtype=np.int64)
    return cand, ret, z['ce'].astype(np.float64), z['valid'], np.where(gr >= 1, gr - 1, -1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import pandas as pd

    from . import fusion as FU

    cfg = load_config(args.dataset, args, args.method_config)
    F = cfg.method['fusion']
    D = cfg.method['candidate_depth']
    clock = Clock()

    _, iv_ret, iv_ce, iv_valid, iv_gold = load_scored(cfg, 'internal_val', -1)
    alpha, cal, grid = FU.select_alpha(iv_ret, iv_ce, iv_valid, iv_gold,
                                       F['alpha_grid'], int(F['k_ref']), int(F['k_select']))
    frozen_alpha = F.get('alpha')
    jdump(dict(dataset=cfg.key, formula=F['formula'], alpha_grid=F['alpha_grid'],
               selected_alpha=alpha, frozen_alpha_in_config=frozen_alpha,
               matches_frozen=(None if frozen_alpha is None
                               else bool(abs(alpha - float(frozen_alpha)) < 1e-12)),
               selection=F['alpha_selection'], selection_split='train-internal-validation',
               k_ref=int(F['k_ref']), k_select=int(F['k_select']), calibration=cal,
               ALPHA_SELECTION_USES_DEV=False, ALPHA_SELECTION_USES_TEST=False),
          cfg.fusion_path())
    write_table(pd.DataFrame(grid), cfg.out_path('fusion', 'alpha_grid.csv'))
    print(f'[{cfg.key}] alpha {alpha} (config: {frozen_alpha}) · '
          f'z {["%.4f" % cal[k] for k in ("ret_mu", "ret_sd", "ce_mu", "ce_sd")]}',
          flush=True)

    _, dv_ret, dv_ce, dv_valid, dv_gold = load_scored(cfg, 'dev', -1)
    fused = FU.fuse(dv_ret, dv_ce, dv_valid, cal, alpha)
    corr = FU.correct_prefix(fused, dv_gold)
    acc_by_k = {int(k): float(corr[:, int(k) - 1].mean()) for k in D['K_grid']}
    depth = FU.select_candidate_depth(acc_by_k, float(D['tolerance']), D['rule'])
    frozen_k = D.get('K_high')
    depth.update(dataset=cfg.key, alpha=alpha, dev_fixed_K_acc=acc_by_k,
                 frozen_K_high_in_config=frozen_k,
                 matches_frozen=(None if frozen_k is None
                                 else bool(depth['selected_K_high'] == int(frozen_k))),
                 selection_split='dev (routing results not consulted)',
                 TEST_READ=False, cost_convention=D['cost_convention'])
    jdump(depth, cfg.depth_path())
    print(f'[{cfg.key}] K_high {depth["selected_K_high"]} (config: {frozen_k}) · '
          f'best fixed K {depth["best_fixed_K"]} · '
          f'delta {depth["delta_selected_vs_best_pp"]} pp', flush=True)

    write_efficiency([eff_row('select_depth', clock, dataset=cfg.key, device='CPU',
                              gpu_allocations=0, neural_inference_calls=0,
                              selected_alpha=alpha,
                              selected_K_high=depth['selected_K_high'])],
                     cfg.out_path('efficiency', 'select_depth.csv'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
