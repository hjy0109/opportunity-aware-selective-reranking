#!/usr/bin/env python3
"""Regression tests for the latency benchmark's routing-feature path.

    python tests/test_latency_wiring.py
    python -m unittest discover -s tests

The defect these tests exist for: an earlier version of the benchmark read a
precomputed CURRENT8 row inside the ``routing_features`` timing block, so that
stage measured a dataframe slice instead of the feature construction a served
mention actually pays for.  The reported end-to-end latency was then lower than
the definition the paper uses.

The tests are static (AST) plus numeric, so they need neither a GPU nor the
licensed corpora.  A static test is the point here: the wiring is what broke,
and it broke in a way that still ran and still produced plausible numbers.
"""
from __future__ import annotations

import ast
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np                                                        # noqa: E402
import yaml                                                               # noqa: E402

from src.routing.features import CURRENT8, TIE_EPS, cheap_features, feature_matrix  # noqa: E402

LATENCY_SRC = os.path.join(REPO, 'src', 'evaluation', 'latency.py')
METHOD_YAML = os.path.join(REPO, 'configs', 'method.yaml')
LIVE_TAG = 'current8_live'
CACHED_TAG = 'current8_cached'


def latency_source() -> str:
    with open(LATENCY_SRC) as f:
        return f.read()


def latency_tree():
    return ast.parse(latency_source(), filename=LATENCY_SRC)


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name}() not found in {LATENCY_SRC}')


def stage_block(fn, stage):
    """The body of the ``with sw('<stage>')`` block inside a function."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            c = item.context_expr
            if (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    and c.func.id == 'sw' and c.args
                    and isinstance(c.args[0], ast.Constant) and c.args[0].value == stage):
                return node
    raise AssertionError(f"no `with sw({stage!r})` block in {fn.name}()")


def called_names(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def guard_constants(node):
    """String constants any enclosing/inner ``if`` in this block compares against."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Compare):
            for c in [n.left, *n.comparators]:
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    out.add(c.value)
    return out


def subscripted_names(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            out.add(n.value.id)
    return out


def reference_current8(R, tie_eps=TIE_EPS):
    """An independent transcription of the frozen CURRENT8 definition."""
    R = np.asarray(R, dtype=np.float64)
    n = R.shape[0]
    out = np.zeros((n, 8), dtype=np.float64)
    for i in range(n):
        row = [v for v in R[i, :5]]
        s = [v if np.isfinite(v) else np.nan for v in row]
        fin = [v for v in s if not np.isnan(v)]
        mx = max(fin)
        ex = [np.exp(v - mx) for v in fin]
        tot = sum(ex)
        p = [v / tot for v in ex]
        H = -sum(v * np.log(v) for v in p if v > 0)
        mean = sum(fin) / len(fin)
        var = sum((v - mean) ** 2 for v in fin) / len(fin)
        adj = [s[j] - s[j + 1] for j in range(4)]
        out[i] = [s[0], s[0] - s[1], s[0] - s[4], np.sqrt(var), s[1] - s[2],
                  max(a for a in adj if not np.isnan(a)),
                  float(sum(1 for v in s if abs(s[0] - v) <= tie_eps)),
                  np.exp(H)]
    return np.nan_to_num(out, nan=0.0)


class LatencyWiring(unittest.TestCase):
    """1-3: the timed stage really builds the features, under the public name."""

    def setUp(self):
        self.tree = latency_tree()
        self.one_pass = find_function(self.tree, 'one_pass')
        self.block = stage_block(self.one_pass, 'routing_features')

    def test_routing_features_block_calls_feature_matrix(self):
        self.assertIn('feature_matrix', called_names(self.block),
                      'the routing_features timing block does not call feature_matrix; '
                      'the stage is not measuring feature construction')

    def test_feature_matrix_is_the_public_name(self):
        src = latency_source()
        self.assertIn('from ..routing.features import', src)
        self.assertIn('feature_matrix', src)
        self.assertNotIn('featmat', src,
                         'featmat is an internal name; the released module exports '
                         'feature_matrix')
        self.assertTrue(callable(feature_matrix))

    def test_live_construction_reads_the_retrieval_scores_not_a_cached_row(self):
        names = subscripted_names(self.block)
        cached = {n for n in names if 'X8' in n or 'cached' in n.lower()}
        self.assertTrue(cached <= {'X8_cached'},
                        f'unexpected cached feature source in the timed block: {cached}')
        if cached:
            self.assertIn(CACHED_TAG, guard_constants(self.block),
                          'a cached feature row is read without being guarded by the '
                          f'{CACHED_TAG!r} diagnostic mode')
        # the live branch must consume the tensor produced by the retrieval stage
        live_branch = [n for n in ast.walk(self.block)
                       if isinstance(n, ast.If)
                       and LIVE_TAG in guard_constants(ast.Module(body=[n.test],
                                                                  type_ignores=[]))]
        self.assertTrue(live_branch, f'no {LIVE_TAG!r} branch in the timed block')
        srcseg = ast.unparse(live_branch[0])
        self.assertIn('tv[:5]', srcseg,
                      'the live branch does not read the top-5 retrieval scores that '
                      'the retrieval stage just produced')
        self.assertIn('feature_matrix', srcseg)

    def test_retrieval_stage_keeps_the_topk_values(self):
        blk = stage_block(self.one_pass, 'retrieval')
        self.assertIn('tv', {n.id for n in ast.walk(blk) if isinstance(n, ast.Name)},
                      'the retrieval stage discards the top-k values, so the live '
                      'feature branch would have nothing to build from')

    def test_live_is_the_default_mode(self):
        src = latency_source()
        i = src.index("dest='routing_feature_mode'")
        seg = src[i - 200:i + 400]
        self.assertIn("default='live'", seg,
                      'the manuscript path must be the default; cached is diagnostic')


class LatencyProtocol(unittest.TestCase):
    """Gate F: the timing protocol the reported numbers were produced under."""

    def setUp(self):
        self.tree = latency_tree()
        with open(METHOD_YAML) as f:
            self.method = yaml.safe_load(f)

    def test_round_robin_sweeps(self):
        src = latency_source()
        self.assertIn('for sweep in range(warmup + reps):', src,
                      'the timed loop is not sweep-major; per-cell loops let thermal '
                      'drift accumulate against whichever cell is timed last')
        i = src.index('for sweep in range(warmup + reps):')
        seg = src[i:i + 400]
        self.assertIn('for c in cells:', seg,
                      'a sweep must run every cell once')

    def test_warmup_and_timed_reps_are_the_frozen_values(self):
        self.assertEqual(int(self.method['latency']['warmup']), 3)
        self.assertEqual(int(self.method['latency']['timed_reps']), 9)

    def test_statistic_is_the_median_over_timed_sweeps(self):
        src = latency_source()
        self.assertIn('passes[cell[\'key\']][warmup:]', src,
                      'the warm-up sweeps are not dropped before aggregating')
        self.assertIn('np.median(tot)', src)
        self.assertEqual(self.method['latency']['statistic'],
                         'median over timed repetitions')

    def test_batch_one_online_regime(self):
        self.assertEqual(self.method['latency']['regime'], 'batch=1 online')

    def test_every_stage_is_timed(self):
        from src.evaluation.latency import STAGES
        one_pass = find_function(self.tree, 'one_pass')
        timed = set()
        for node in ast.walk(one_pass):
            if isinstance(node, ast.With):
                for item in node.items:
                    c = item.context_expr
                    if (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                            and c.func.id == 'sw' and c.args):
                        timed.add(c.args[0].value)
        self.assertEqual(timed, set(STAGES))


class Current8Definition(unittest.TestCase):
    """4-5: the eight features, and their values, are the frozen definition."""

    def test_eight_features(self):
        self.assertEqual(len(CURRENT8), 8)
        X, _ = feature_matrix(np.linspace(1, 0, 64)[None, :])
        self.assertEqual(X.shape, (1, 8))

    def test_feature_list_matches_the_frozen_config(self):
        with open(METHOD_YAML) as f:
            m = yaml.safe_load(f)
        self.assertEqual(list(m['features']['list']), list(CURRENT8))
        self.assertEqual(float(m['features']['tie_eps']), float(TIE_EPS))

    def test_values_match_an_independent_implementation(self):
        rng = np.random.default_rng(20260910)
        rows = []
        for _ in range(256):
            r = np.sort(rng.normal(0, 1, 64))[::-1].copy()
            rows.append(r)
        for _ in range(128):                       # exact ties at the top
            r = np.sort(rng.normal(0, 1, 64))[::-1].copy()
            r[1] = r[0]
            rows.append(r)
        for _ in range(128):                       # near-ties inside tie_eps
            r = np.sort(rng.normal(0, 1, 64))[::-1].copy()
            r[1] = r[0] - TIE_EPS / 2
            r[2] = r[0] - TIE_EPS / 3
            rows.append(r)
        R = np.asarray(rows, dtype=np.float64)
        X, _ = feature_matrix(R)
        self.assertTrue(np.array_equal(X, reference_current8(R)),
                        'CURRENT8 values differ from the independent transcription of '
                        f'the frozen definition (max|delta| '
                        f'{np.abs(X - reference_current8(R)).max():.3e})')

    def test_features_are_a_function_of_the_retrieval_scores_alone(self):
        import inspect
        src = inspect.getsource(cheap_features)
        for forbidden in ('gold', 'ce_', 'cross_encoder', 'context'):
            self.assertNotIn(forbidden, src)

    def test_batch_invariance(self):
        rng = np.random.default_rng(7)
        R = np.sort(rng.normal(0, 1, (64, 64)), axis=1)[:, ::-1].copy()
        Xb, _ = feature_matrix(R)
        Xs = np.vstack([feature_matrix(R[i:i + 1])[0] for i in range(len(R))])
        self.assertTrue(np.array_equal(Xb, Xs),
                        'per-mention (batch=1, the latency path) and batched (the '
                        'accuracy path) feature construction disagree')


if __name__ == '__main__':
    unittest.main(verbosity=2)
