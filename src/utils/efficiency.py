"""Resource accounting.

Efficiency numbers are primary evidence in this work, so every stage writes the
same row shape next to its accuracy output: wall time, device, peak memory and
per-item throughput.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

from .io import write_table


class Clock:
    def __init__(self) -> None:
        self.t = time.time()

    def s(self) -> float:
        return round(time.time() - self.t, 3)


def peak_rss_gib() -> float | None:
    """Process high-water mark, read from /proc (Linux)."""
    try:
        for ln in open('/proc/self/status'):
            if ln.startswith('VmHWM'):
                return round(float(ln.split()[1]) / 2 ** 20, 3)
    except OSError:
        return None
    return None


def eff_row(stage: str, clock: Clock, **kw: Any) -> dict[str, Any]:
    row = dict(stage=stage, ts_utc=pd.Timestamp.utcnow().isoformat(),
               wall_s=clock.s(), peak_rss_GiB=peak_rss_gib())
    row.update(kw)
    return row


def write_efficiency(rows, path: str) -> str:
    return write_table(pd.DataFrame(rows), path)


def model_size(model) -> dict[str, Any]:
    n = sum(p.numel() for p in model.parameters())
    dtypes = {str(p.dtype) for p in model.parameters()}
    return dict(n_parameters=int(n), n_parameters_M=round(n / 1e6, 2),
                dtypes=sorted(dtypes))
