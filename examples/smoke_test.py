#!/usr/bin/env python3
"""Offline smoke test of the scientific core.

    python examples/smoke_test.py

Builds a tiny synthetic retrieval cache under a temporary output directory and
exercises the parts of the pipeline that need neither the licensed corpora nor a
downloaded model:

    * CURRENT8 feature computation
    * Opportunity / Difficulty / Benefit label construction and containment
    * budget selection at matched reranking rates
    * the selective-reranking outcome (unselected mentions keep the retriever top-1)
    * a router fitted on the synthetic features, for both targets, through the
      shared training path
    * the evaluation table and the paired clustered bootstrap

It checks invariants, not accuracy: the synthetic data carries no signal worth
reporting.  A non-zero exit status means something in the released code is broken.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np                                                        # noqa: E402
import pandas as pd                                                       # noqa: E402

N_MENTIONS = 400
N_DOCS = 40
K_POOL = 64
K_HIGH = 20
SEED = 20260101

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f' -- {detail}' if detail else ''))


def synthetic_retrieval(rng):
    """A retrieval score matrix with a plausible shape: descending scores per row."""
    base = rng.normal(0.8, 0.1, size=(N_MENTIONS, 1))
    decay = np.sort(rng.exponential(0.02, size=(N_MENTIONS, K_POOL)), axis=1)[:, ::-1]
    R = base - np.cumsum(decay, axis=1)
    gold_rank = rng.choice(np.arange(0, K_POOL + 1), size=N_MENTIONS,
                           p=_rank_prior(K_POOL))
    return R, gold_rank


def _rank_prior(k):
    p = np.concatenate([[0.06], 0.55 * np.ones(1), np.linspace(0.05, 0.005, k - 1)])
    return p / p.sum()


def main() -> int:
    from src.routing.budget import apply_selection, select_top_q
    from src.routing.features import CURRENT8, feature_matrix
    from src.routing import targets as TG

    rng = np.random.default_rng(SEED)
    R, gold_rank = synthetic_retrieval(rng)

    X, F = feature_matrix(R)
    check('feature matrix shape', X.shape == (N_MENTIONS, 8), f'{X.shape}')
    check('all features finite', bool(np.isfinite(X).all()))
    check('near_top_count >= 1', bool((F['near_top_count_top5'] >= 1).all()))
    check('margin_s1_s2 >= 0 for descending scores',
          bool((F['margin_s1_s2'] >= -1e-12).all()))

    c1 = gold_rank == 1
    hard = TG.hardness(c1)
    opp = TG.opportunity(c1, gold_rank, K_HIGH)
    # Synthetic reranking outcome: a fixed share of the reachable errors is fixed.
    fixable = opp & (rng.random(N_MENTIONS) < 0.55)
    broken = c1 & (rng.random(N_MENTIONS) < 0.03)
    ckh = (c1 & ~broken) | fixable
    ben = TG.benefit(c1, ckh)
    TG.check_containment(hard, opp, ben)
    check('OPP subset of HARD', bool(not (opp & ~hard).any()))
    check('BEN subset of OPP', bool(not (ben & ~opp).any()))
    check('OPP excludes rank 1 and ranks beyond K_high',
          bool(not opp[(gold_rank <= 1) | (gold_rank > K_HIGH)].any()))
    check('HARD equals "top-1 wrong"', bool((hard == ~c1).all()))

    n = N_MENTIONS
    score = rng.random(n)
    for q in (0.0, 0.10, 0.30, 0.50, 1.0):
        sel = select_top_q(score, q, n)
        check(f'selection count at rate {q}', int(sel.sum()) == int(round(q * n)),
              f'{int(sel.sum())} vs {int(round(q * n))}')
    sel = select_top_q(score, 0.30, n)
    correct = apply_selection(sel, c1, ckh)
    check('unselected mentions keep the retriever top-1',
          bool((correct[~sel] == c1[~sel]).all()))
    check('selected mentions take the reranked outcome',
          bool((correct[sel] == ckh[sel]).all()))
    s2 = np.zeros(n)
    check('ties broken by row order (stable sort)',
          bool(np.array_equal(np.flatnonzero(select_top_q(s2, 0.25, n)),
                              np.arange(int(round(0.25 * n))))))

    docs = np.array([f'doc{i % N_DOCS}' for i in range(n)])
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'outputs')
        cfg_dir = os.path.join(tmp, 'configs')
        os.makedirs(os.path.join(cfg_dir, 'datasets'))
        import shutil
        shutil.copy(os.path.join(REPO, 'configs', 'method.yaml'),
                    os.path.join(cfg_dir, 'method.yaml'))
        with open(os.path.join(cfg_dir, 'datasets', 'smoke.yaml'), 'w') as f:
            f.write('key: smoke\nlabel: Smoke\ntag: smoke\n'
                    'corpus: {kind: bioc, files: {}, mention_type: null}\n'
                    'knowledge_base: {kind: ontology_json, name: none, ontology: none, '
                    'type_map: null, universe_type: null}\n'
                    'grouping: {key: document_id, description: synthetic}\n'
                    'method_overrides:\n'
                    f'  candidate_depth: {{K_high: {K_HIGH}}}\n'
                    '  fusion: {alpha: 0.5}\n'
                    '  router:\n'
                    '    selected_hyperparameters:\n'
                    '      n_estimators: 60\n      max_depth: 3\n'
                    '      learning_rate: 0.1\n      subsample: 1.0\n')
        table = pd.DataFrame(dict(
            dataset='smoke', split='x', K_high=K_HIGH,
            mention_id=[f'm{i}' for i in range(n)],
            example_id=[f'm{i}#g' for i in range(n)], group=docs,
            gold_rank=gold_rank, correct_K1=c1, correct_Khigh=ckh,
            HARD=hard, OPP=opp, BEN=ben,
            **{c: np.nan_to_num(F[c], nan=0.0) for c in CURRENT8}))
        for sp in ('train', 'dev', 'test'):
            d = os.path.join(out, 'smoke', 'routing')
            os.makedirs(d, exist_ok=True)
            table.assign(split=sp).to_parquet(
                os.path.join(d, f'routing_table_{sp}.parquet'), index=False)

        pp = os.pathsep.join(filter(None, [REPO, os.environ.get('PYTHONPATH', '')]))
        env = dict(os.environ, OUTPUT_DIR=out, PYTHONPATH=pp)
        mcfg = os.path.join(cfg_dir, 'method.yaml')
        dcfg = os.path.join(cfg_dir, 'datasets', 'smoke.yaml')

        def run(mod, *a):
            r = subprocess.run([sys.executable, '-m', mod, '--dataset', dcfg,
                                '--method-config', mcfg, *a],
                               cwd=REPO, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-3000:]); print(r.stderr[-3000:], file=sys.stderr)
            return r

        ok_router = True
        for target in ('OPP', 'HARD'):
            r = run('src.routing.train', '--target', target)
            ok_router &= r.returncode == 0
        check('router training runs for both targets through the shared path', ok_router)

        if ok_router:
            zo = np.load(os.path.join(out, 'smoke', 'router', 'scores_OPP.npz'))
            zh = np.load(os.path.join(out, 'smoke', 'router', 'scores_HARD.npz'))
            check('the two routers share the standardiser',
                  bool(np.allclose(zo['mu'], zh['mu']) and np.allclose(zo['sd'], zh['sd'])))
            check('the two routers give different scores (labels differ)',
                  bool(not np.allclose(zo['test'], zh['test'])))

            for target in ('OPP', 'HARD'):
                r = run('src.routing.infer', '--target', target, '--split', 'test',
                        '--rate', '0.30')
                check(f'router inference runs ({target})', r.returncode == 0)
            a = os.path.join(out, 'smoke', 'routing', 'decisions_OPP_test_q30.parquet')
            b = os.path.join(out, 'smoke', 'routing', 'decisions_HARD_test_q30.parquet')
            if os.path.exists(a) and os.path.exists(b):
                da, db = pd.read_parquet(a), pd.read_parquet(b)
                check('matched policies rerank the same number of mentions',
                      int(da.reranked.sum()) == int(db.reranked.sum()),
                      f'{int(da.reranked.sum())} vs {int(db.reranked.sum())}')

            r = run('src.evaluation.evaluate', '--split', 'test',
                    '--rates', '0.0', '0.10', '0.30', '0.50', '1.0')
            check('evaluation runs', r.returncode == 0)
            res = os.path.join(out, 'smoke', 'evaluation', 'results.csv')
            if os.path.exists(res):
                E = pd.read_csv(res)
                check('evaluation covers every rate in the grid',
                      set(E[E.policy == 'opportunity'].reranking_rate)
                      == {0.0, 0.1, 0.3, 0.5, 1.0})
                check('every routed cell reranked exactly round(rate * N)',
                      bool(E[~E.is_endpoint].exact_count_match.all()))
                ep = E[E.policy == 'retriever'].acc_at_1.iloc[0]
                check('rate 0 equals retriever-only accuracy',
                      bool(abs(E[(E.policy == 'opportunity')
                                 & (E.reranking_rate == 0.0)].acc_at_1.iloc[0] - ep) < 1e-9))
                fu = E[E.policy == 'full'].acc_at_1.iloc[0]
                check('rate 1 equals full-reranking accuracy',
                      bool(abs(E[(E.policy == 'opportunity')
                                 & (E.reranking_rate == 1.0)].acc_at_1.iloc[0] - fu) < 1e-9))

            r = run('src.evaluation.bootstrap', '--pred-a', a, '--pred-b', b,
                    '--label-a', 'Opportunity', '--label-b', 'Difficulty',
                    '--split', 'test')
            check('bootstrap runs on a paired pair of predictions', r.returncode == 0,
                  r.stdout.strip().splitlines()[-1] if r.returncode == 0 else '')

    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print(f'\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed')
    return 1 if n_fail else 0


if __name__ == '__main__':
    raise SystemExit(main())
