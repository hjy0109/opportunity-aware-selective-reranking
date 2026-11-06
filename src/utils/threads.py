"""Thread limits.

The reported CPU numbers (router fitting time, per-mention router overhead) were
measured with the BLAS/OpenMP thread count pinned, so the helper is applied by
every entry point before numpy/sklearn/xgboost are imported.
"""
from __future__ import annotations

import os

_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
         'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def limit_threads(n: int = 4) -> None:
    for v in _VARS:
        os.environ[v] = str(n)
