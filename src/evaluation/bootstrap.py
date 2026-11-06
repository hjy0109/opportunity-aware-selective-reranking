"""Paired clustered bootstrap.

    python -m src.evaluation.bootstrap --dataset ncbi_disease \
        --pred-a outputs/ncbi_disease/routing/decisions_OPP_test_q10.parquet \
        --pred-b outputs/ncbi_disease/routing/decisions_HARD_test_q10.parquet

The primary inference in this work is a **document-clustered paired bootstrap**:
clusters (documents, or the source example where a corpus has no documents) are
resampled with replacement, and the statistic is the difference in Acc@1 between
two systems evaluated on the *same* resampled mentions.  Mention-level i.i.d.
intervals are not a substitute -- most of these corpora put many mentions in one
document, so mention-level resampling understates the uncertainty.

The estimator resamples clusters, sums the paired per-mention difference within
each cluster, and divides by the resampled mention count, so clusters contribute
in proportion to their size:

    delta_b = sum_{c in sample} sum_{i in c} (a_i - b_i) / sum_{c in sample} |c|

The interval is the 2.5/97.5 percentile of the ``n_boot`` replicates.  Both
``n_boot`` and the seed are fixed in configs/method.yaml, so the interval is
reproducible.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 20260823


def clustered_paired_ci(a, b, groups, n_boot: int = N_BOOT, seed: int = SEED):
    """(delta pp, lo pp, hi pp, excludes_zero) for Acc@1(a) - Acc@1(b)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    g = pd.factorize(np.asarray(groups))[0]
    ng = g.max() + 1
    sums = np.bincount(g, weights=d, minlength=ng)
    cnts = np.bincount(g, minlength=ng).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ng, size=(n_boot, ng))
    bs = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pt = float(d.mean())
    return (round(100 * pt, 4), round(100 * float(lo), 4), round(100 * float(hi), 4),
            bool(lo > 0 or hi < 0))


def clustered_ratio_ci(num_a, den_a, num_b, den_b, groups, n_boot: int = N_BOOT,
                       seed: int = SEED):
    """Same resampling for a ratio contrast whose denominator is not the mention count
    (for example BEN capture, or the composition of the selected set)."""
    g = pd.factorize(np.asarray(groups))[0]
    ng = g.max() + 1

    def S(v):
        return np.bincount(g, weights=np.asarray(v, float), minlength=ng)

    na, da, nb, db = S(num_a), S(den_a), S(num_b), S(den_b)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, ng, size=(n_boot, ng))
    ra = na[pick].sum(1) / np.maximum(da[pick].sum(1), 1e-12)
    rb = nb[pick].sum(1) / np.maximum(db[pick].sum(1), 1e-12)
    d = 100.0 * (ra - rb)
    lo, hi = np.percentile(d, [2.5, 97.5])
    point = 100.0 * (np.asarray(num_a).sum() / max(np.asarray(den_a).sum(), 1e-12)
                     - np.asarray(num_b).sum() / max(np.asarray(den_b).sum(), 1e-12))
    return (round(float(point), 4), round(float(lo), 4), round(float(hi), 4),
            bool(lo > 0 or hi < 0))


def n_clusters(groups) -> int:
    return int(pd.factorize(np.asarray(groups))[0].max() + 1)


def _load(path: str):
    d = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    for c in ('mention_id', 'correct'):
        if c not in d.columns:
            raise SystemExit(f'{path}: missing required column "{c}"')
    return d


def main(argv=None) -> int:
    from ..config import add_common_arguments, load_config
    from ..utils.io import jdump

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap, dataset_required=False)
    ap.add_argument('--pred-a', required=True,
                    help='parquet/csv with mention_id, correct (and optionally group)')
    ap.add_argument('--pred-b', required=True, help='the system compared against')
    ap.add_argument('--label-a', default='A')
    ap.add_argument('--label-b', default='B')
    ap.add_argument('--split', default='test')
    ap.add_argument('--n-boot', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out', default=None, help='write the result as JSON to this path')
    args = ap.parse_args(argv)

    A = _load(args.pred_a)
    B = _load(args.pred_b)
    n_boot, seed = args.n_boot, args.seed
    groups = None
    if args.dataset:
        cfg = load_config(args.dataset, args, args.method_config)
        st = cfg.method['statistics']
        n_boot = n_boot or int(st['n_boot'])
        seed = seed or int(st['seed'])
        t = pd.read_parquet(cfg.routing_table_path(args.split))
        gmap = dict(zip(t.mention_id.astype(str), t.group.astype(str)))
        groups = np.asarray([gmap[str(i)] for i in A.mention_id.astype(str)])
    n_boot = n_boot or N_BOOT
    seed = seed or SEED
    if groups is None:
        if 'group' not in A.columns:
            raise SystemExit('no cluster column: pass --dataset, or include "group" in --pred-a')
        groups = A.group.astype(str).to_numpy()

    B = B.set_index(B.mention_id.astype(str)).reindex(A.mention_id.astype(str))
    if B.correct.isna().any():
        raise SystemExit(f'--pred-b is missing {int(B.correct.isna().sum())} mentions '
                         f'present in --pred-a; the bootstrap must be paired')
    a = A.correct.to_numpy().astype(float)
    b = B.correct.to_numpy().astype(float)
    pt, lo, hi, sig = clustered_paired_ci(a, b, groups, n_boot, seed)
    out = dict(dataset=args.dataset, split=args.split,
               system_a=args.label_a, system_b=args.label_b,
               pred_a=args.pred_a, pred_b=args.pred_b,
               n_mentions=int(len(a)), n_clusters=n_clusters(groups),
               acc_a=round(float(a.mean()), 6), acc_b=round(float(b.mean()), 6),
               delta_acc_pp=pt, ci_low=lo, ci_high=hi, ci_excludes_zero=sig,
               n_boot=n_boot, seed=seed, bootstrap_unit='document/source cluster',
               interval='95% percentile, paired')
    print(f'{args.label_a} - {args.label_b}: {pt:+.4f} pp '
          f'[{lo:+.4f}, {hi:+.4f}] · clusters {out["n_clusters"]:,} · '
          f'n_boot {n_boot} · seed {seed} · '
          f'{"excludes 0" if sig else "includes 0"}')
    if args.out:
        jdump(out, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
