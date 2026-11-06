"""Small IO helpers shared by every stage."""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

_WS = re.compile(r'\s+')


def ensure_parent(path: str) -> str:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    return path


def jdump(obj: Any, path: str) -> str:
    ensure_parent(path)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
    return path


def jload(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def write_table(df, path: str, index: bool = False) -> str:
    """CSV for `.csv`, parquet (zstd) otherwise -- the format each stage used."""
    ensure_parent(path)
    if path.endswith('.csv'):
        df.to_csv(path, index=index)
    else:
        df.to_parquet(path, index=index, compression='zstd')
    return path


def norm_alias(alias: str) -> str:
    """Alias normalisation used for concept-representation de-duplication.

    strip -> collapse repeated whitespace -> casefold.  Punctuation, abbreviations,
    stemming and lemmatisation are deliberately untouched.
    """
    return _WS.sub(' ', str(alias).strip()).casefold()


def sha256_file(path: str, blk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(blk), b''):
            h.update(b)
    return h.hexdigest()


def sha256_ids(ids) -> str:
    return hashlib.sha256('|'.join(map(str, ids)).encode()).hexdigest()
