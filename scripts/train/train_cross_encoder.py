#!/usr/bin/env python3
"""Thin wrapper: equivalent to `python -m src.reranking.train`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.reranking.train import main   # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
