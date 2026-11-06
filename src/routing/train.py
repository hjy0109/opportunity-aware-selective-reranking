"""Router training.

    python -m src.routing.train --dataset ncbi_disease --target OPP
    python -m src.routing.train --dataset ncbi_disease --target HARD

Opportunity and Difficulty are trained by the *same* call with a different
``--target``: identical features, classifier family, hyperparameters, random
state, class balancing, standardiser and training split.  The supervision label
is the only difference, which is what makes the comparison controlled.

Hyperparameter selection (``--select``) runs the declared grid and scores each
cell by dev head-mean Acc@1 over q in {5, 10, 20, 30}%, i.e. by the downstream
budgeted accuracy the router is meant to deliver rather than by a ranking metric;
ties go to dev OPP AUPRC.  Selection uses the OPP target only and the HARD router
reuses the winning hyperparameters, so capacity is held fixed across the two.
Without ``--select`` the frozen per-dataset hyperparameters from the config are
used and no search is run.  The test split is never read here.

Writes ``router/router_{target}.json``, ``router/scores_{target}.npz`` (train, dev
and test scores) and ``router/{target}_training.json``.
"""
from __future__ import annotations

import argparse
import os
import time

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, write_efficiency
from ..utils.io import ensure_parent, jdump, write_table
from ..utils.threads import limit_threads
from .targets import TARGET_LABEL


def load_block(cfg, split, pd, np):
    from .features import CURRENT8
    t = pd.read_parquet(cfg.routing_table_path(split))
    return dict(X=t[CURRENT8].to_numpy(float),
                OPP=t.OPP.to_numpy().astype(int), HARD=t.HARD.to_numpy().astype(int),
                BEN=t.BEN.to_numpy().astype(int),
                correct_K1=t.correct_K1.to_numpy().astype(bool),
                correct_Khigh=t.correct_Khigh.to_numpy().astype(bool),
                groups=t.group.astype(str).to_numpy(),
                mention_id=t.mention_id.astype(str).to_numpy(),
                K_high=int(t.K_high.iloc[0]), n=len(t))


def head_mean_accuracy(score, block, q_head, np):
    """Dev head-mean Acc@1: the selection metric for the router hyperparameters."""
    from .budget import apply_selection, select_top_q
    accs = {}
    for q in q_head:
        sel = select_top_q(score, q, block['n'])
        accs[float(q)] = float(apply_selection(
            sel, block['correct_K1'], block['correct_Khigh']).mean())
    return float(np.mean([accs[float(q)] for q in q_head])), accs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--target', default='OPP', choices=['OPP', 'HARD', 'BEN'],
                    help='OPP = Opportunity (proposed), HARD = Difficulty (control), '
                         'BEN = Benefit (appendix diagnostic).  All three use the same '
                         'features, classifier and hyperparameters; only the label '
                         'differs, and the hyperparameters are always the ones selected '
                         'on OPP.')
    ap.add_argument('--family', default=None,
                    help='classifier family (default: the frozen router.family)')
    ap.add_argument('--select', action='store_true',
                    help='run the declared hyperparameter grid on dev instead of using '
                         'the frozen per-dataset values')
    ap.add_argument('--threads', type=int, default=1,
                    help='BLAS threads; the reported router timings used 1')
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd

    from . import classifiers as CL
    from .features import CURRENT8, apply_standardiser, fit_standardiser

    cfg = load_config(args.dataset, args, args.method_config)
    RT = cfg.method['router']
    family = args.family or RT['family']
    seed = int(cfg.method['seeds']['router_random_state'])
    q_head = cfg.method['budget']['q_head']
    clock = Clock()

    blocks = {sp: load_block(cfg, sp, pd, np) for sp in ('train', 'dev', 'test')}
    mu, sd = fit_standardiser(blocks['train']['X'])
    Z = {sp: apply_standardiser(blocks[sp]['X'], mu, sd) for sp in blocks}
    y_tr = blocks['train'][args.target]

    search_rows, hp = [], RT.get('selected_hyperparameters')
    if args.select:
        from sklearn.metrics import average_precision_score, roc_auc_score
        y_dev_opp = blocks['dev']['OPP']
        for cell in CL.expand_grid(RT['grid']):
            m = CL.make(family, cell, RT['fixed'], seed, blocks['train']['OPP'])
            t0 = time.time()
            m.fit(Z['train'], blocks['train']['OPP'])
            fit_s = time.time() - t0
            s = m.predict_proba(Z['dev'])[:, 1]
            hm, _ = head_mean_accuracy(s, blocks['dev'], q_head, np)
            search_rows.append(dict(
                dataset=cfg.key, family=family, target='OPP', **cell,
                dev_head_mean_acc=round(hm, 6),
                dev_OPP_auroc=round(float(roc_auc_score(y_dev_opp, s)), 6),
                dev_OPP_auprc=round(float(average_precision_score(y_dev_opp, s)), 6),
                fit_s=round(fit_s, 3), n_dev=blocks['dev']['n']))
            print(f'  {cell} head-mean {hm:.6f} ({fit_s:.1f}s)', flush=True)
        best = max(search_rows, key=lambda r: (r['dev_head_mean_acc'], r['dev_OPP_auprc']))
        hp = {k: best[k] for k in RT['grid']}
        write_table(pd.DataFrame(search_rows),
                    cfg.out_path('router', f'hp_search_{family}.csv'))
        print(f'  selected {hp} (dev head-mean {best["dev_head_mean_acc"]:.6f})', flush=True)
    if hp is None:
        raise SystemExit('no hyperparameters: pass --select or set '
                         'router.selected_hyperparameters in the dataset config')

    model = CL.make(family, hp, RT['fixed'], seed, y_tr)
    t0 = time.time()
    model.fit(Z['train'], y_tr)
    fit_s = time.time() - t0
    scores = {sp: model.predict_proba(Z[sp])[:, 1] for sp in blocks}
    path = cfg.router_model_path(args.target)
    CL.save(family, model, path)
    ensure_parent(cfg.router_scores_path(args.target))
    np.savez_compressed(cfg.router_scores_path(args.target),
                        **{sp: scores[sp] for sp in blocks},
                        **{f'mention_id_{sp}': blocks[sp]['mention_id'] for sp in blocks},
                        mu=mu, sd=sd)

    dev_hm, dev_accs = head_mean_accuracy(scores['dev'], blocks['dev'], q_head, np)
    jdump(dict(
        dataset=cfg.key, target=args.target,
        target_label=TARGET_LABEL[args.target],
        target_definition=cfg.method['targets'][args.target],
        family=family, family_label=CL.FAMILY_LABEL[family],
        hyperparameters=hp,
        hyperparameters_source=('grid search on dev (this run)' if args.select
                                else 'frozen per-dataset value from the config'),
        selection_metric=RT['selection_metric'], selection_target=RT['selection_target'],
        fixed=RT['fixed'], random_state=seed, class_balance=RT['class_balance'],
        features=CURRENT8, n_features=len(CURRENT8),
        standardisation=cfg.method['features']['standardisation'],
        standardiser=dict(mu=[float(v) for v in mu], sd=[float(v) for v in sd]),
        K_high=blocks['train']['K_high'],
        n_train=blocks['train']['n'], n_dev=blocks['dev']['n'], n_test=blocks['test']['n'],
        positive_rate_train=round(float(np.mean(y_tr)), 6),
        positive_rate_dev=round(float(np.mean(blocks['dev'][args.target])), 6),
        fit_seconds=round(fit_s, 3),
        model_bytes=(os.path.getsize(path) if os.path.exists(path) else None),
        dev_head_mean_acc=round(dev_hm, 6),
        dev_acc_by_q={str(k): round(v, 6) for k, v in dev_accs.items()},
        TEST_USED_FOR_SELECTION=False, GOLD_IN_INFERENCE_FEATURES=False),
        cfg.out_path('router', f'{args.target}_training.json'))
    write_efficiency([eff_row(
        'train_router', clock, dataset=cfg.key, target=args.target, family=family,
        device='CPU', gpu_allocations=0, n_features=len(CURRENT8),
        n_train=blocks['train']['n'], fit_seconds=round(fit_s, 3),
        model_bytes=(os.path.getsize(path) if os.path.exists(path) else None),
        blas_threads=args.threads, dev_head_mean_acc=round(dev_hm, 6))],
        cfg.out_path('efficiency', f'train_router_{args.target}.csv'))
    print(f'[{cfg.key}] {args.target} router ({family}) fitted in {fit_s:.2f}s · '
          f'dev head-mean Acc@1 {dev_hm:.6f} · -> {path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
