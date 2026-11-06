#!/usr/bin/env python3
"""The ARR cost term reads latencies that the pipeline itself produced.

    python tests/test_arr_route_latency.py

The utility label flips for every mention whose ranking gain is below
``lambda_cost * (1 - stop / rerank)``, so where those two numbers come from is
part of the baseline's definition.  These tests pin the resolution order and
check that the measured path reads the batch-amortised ``ms_per_mention`` the
retrieval and scoring stages write next to their outputs.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import pandas as pd                                                       # noqa: E402

from src.baselines.arr_train import route_latencies                       # noqa: E402

CONFIG_SPEC = dict(retrieval=0.07, reranking=40.602,
                   reranking_basis='full 64-candidate pool (not rescaled to K_high)')


class Cfg:
    """The two attributes route_latencies() uses."""

    def __init__(self, root, spec=CONFIG_SPEC):
        self.root = root
        self.dataset = dict(arr_route_latency_ms=spec) if spec else {}

    def out_path(self, *parts):
        p = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p


def args(route_latency='reported', retrieval_ms=None, rerank_ms=None):
    return argparse.Namespace(route_latency=route_latency,
                              retrieval_ms=retrieval_ms, rerank_ms=rerank_ms)


def write_efficiency(cfg, retrieval_ms, rerank_ms):
    pd.DataFrame([dict(stage='retrieval', split='test', ms_per_mention=retrieval_ms),
                  dict(stage='retrieval_total', split=None, ms_per_mention=None)]
                 ).to_csv(cfg.out_path('efficiency', 'retrieval.csv'), index=False)
    pd.DataFrame([dict(stage='score_candidates', split='test', ms_per_mention=rerank_ms),
                  dict(stage='score_candidates_total', split=None, ms_per_mention=None)]
                 ).to_csv(cfg.out_path('efficiency', 'score_full.csv'), index=False)


class RouteLatencyResolution(unittest.TestCase):

    def test_command_line_wins(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Cfg(d)
            write_efficiency(cfg, 1.0, 2.0)
            stop, rerank, src = route_latencies(cfg, args('measured', 3.0, 4.0))
            self.assertEqual((stop, rerank), (3.0, 7.0))
            self.assertIn('command line', src)

    def test_reported_is_the_default_and_reads_the_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Cfg(d)
            write_efficiency(cfg, 1.0, 2.0)
            stop, rerank, src = route_latencies(cfg, args())
            self.assertAlmostEqual(stop, 0.07)
            self.assertAlmostEqual(rerank, 0.07 + 40.602)
            self.assertIn('dataset config', src)

    def test_measured_reads_the_efficiency_artefacts(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Cfg(d)
            write_efficiency(cfg, 0.9, 30.0)
            stop, rerank, src = route_latencies(cfg, args('measured'))
            self.assertAlmostEqual(stop, 0.9)
            self.assertAlmostEqual(rerank, 30.9)
            self.assertIn('measured in this run', src)

    def test_measured_without_artefacts_stops_with_a_message(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                route_latencies(Cfg(d), args('measured'))

    def test_no_config_falls_back_to_the_measurement(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Cfg(d, spec=None)
            write_efficiency(cfg, 0.5, 10.0)
            stop, rerank, _ = route_latencies(cfg, args())
            self.assertAlmostEqual(stop, 0.5)
            self.assertAlmostEqual(rerank, 10.5)

    def test_stop_route_is_contained_in_the_rerank_route(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Cfg(d)
            write_efficiency(cfg, 0.9, 30.0)
            for a in (args(), args('measured'), args('measured', 2.0, 5.0)):
                stop, rerank, _ = route_latencies(cfg, a)
                self.assertLess(stop, rerank)


if __name__ == '__main__':
    unittest.main(verbosity=2)
