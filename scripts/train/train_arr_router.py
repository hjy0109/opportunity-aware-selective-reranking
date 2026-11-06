#!/usr/bin/env python3
"""Thin wrapper: equivalent to `python -m src.baselines.arr_train`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.baselines.arr_train import main   # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
