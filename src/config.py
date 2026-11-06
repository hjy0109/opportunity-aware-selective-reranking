"""Configuration loading.

Everything that was a machine-specific constant in the experiment code is either
a config field or a command-line argument here.  Three roots locate all inputs
and outputs and may be given as environment variables or as CLI flags:

    DATA_ROOT   corpora, ontologies and licensed resources supplied by the user
    MODEL_ROOT  trained checkpoints (cross-encoder, ARR router)
    OUTPUT_DIR  everything this code writes

The frozen experimental settings live in ``configs/method.yaml`` (shared) and
``configs/datasets/<dataset>.yaml`` (per dataset).  A dataset config may only
override the fields listed in ``METHOD_OVERRIDABLE``; anything else raises, so a
config file cannot silently redefine the method.
"""
from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, 'configs')

# Per-dataset host settings.  These differ between datasets in the reported
# experiments and are the only method fields a dataset config may set.
METHOD_OVERRIDABLE = {
    'retriever': {'normalize', 'aggregation', 'name_topn', 'similarity'},
    'reranker': {'max_epochs', 'patience', 'mine_every', 'n_look', 'n_keep',
                 'truncation', 'internal_val'},
    'fusion': {'alpha'},
    'candidate_depth': {'K_high'},
    'router': {'selected_hyperparameters'},
}


@dataclass
class Roots:
    data: str
    model: str
    output: str

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'Roots':
        def pick(flag: str, env: str, default: str) -> str:
            v = getattr(args, flag, None) or os.environ.get(env) or default
            return os.path.abspath(os.path.expanduser(v))
        return Roots(data=pick('data_root', 'DATA_ROOT', 'data'),
                     model=pick('model_root', 'MODEL_ROOT', 'models'),
                     output=pick('output_dir', 'OUTPUT_DIR', 'outputs'))


@dataclass
class Config:
    dataset: dict[str, Any]
    method: dict[str, Any]
    roots: Roots
    extra: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- accessors
    @property
    def key(self) -> str:
        return self.dataset['key']

    @property
    def tag(self) -> str:
        """Filesystem-safe dataset key (``bc5cdr-disease`` -> ``bc5cdr_disease``)."""
        return self.dataset['key'].replace('-', '_')

    @property
    def label(self) -> str:
        return self.dataset['label']

    def section(self, name: str) -> dict[str, Any]:
        return self.method[name]

    def data_path(self, *parts: str) -> str:
        return os.path.join(self.roots.data, *parts)

    def model_path(self, *parts: str) -> str:
        return os.path.join(self.roots.model, *parts)

    def out_path(self, *parts: str) -> str:
        return os.path.join(self.roots.output, self.tag, *parts)

    # ---------------------------------------------------------------- artefacts
    def kb_path(self) -> str:
        return self.out_path('kb', 'kb.parquet')

    def concept_repr_path(self) -> str:
        return self.out_path('kb', 'concept_representation.parquet')

    def mentions_path(self, split: str) -> str:
        return self.out_path('mentions', f'mentions_{split}.parquet')

    def documents_path(self) -> str:
        return self.out_path('mentions', 'documents.json.gz')

    def folds_path(self) -> str:
        return self.out_path('mentions', 'folds.parquet')

    def candidates_path(self, split: str) -> str:
        return self.out_path('candidates', f'candidates_{split}.parquet')

    def queries_path(self, split: str) -> str:
        return self.out_path('queries', f'queries_{split}.parquet')

    def ce_checkpoint(self, fold: int) -> str:
        f = 'full' if fold < 0 else f'f{fold}'
        return self.model_path(f'cross_encoder_{self.tag}_{f}')

    def ce_scores_path(self, split: str, fold: int = -1) -> str:
        f = 'full' if fold < 0 else f'f{fold}'
        return self.out_path('ce_scores', f'{split}_{f}.npz')

    def fusion_path(self) -> str:
        return self.out_path('fusion', 'fusion_selection.json')

    def depth_path(self) -> str:
        return self.out_path('fusion', 'candidate_depth.json')

    def router_dir(self) -> str:
        return self.out_path('router')

    def router_model_path(self, target: str) -> str:
        return os.path.join(self.router_dir(), f'router_{target}.json')

    def router_scores_path(self, target: str) -> str:
        return os.path.join(self.router_dir(), f'scores_{target}.npz')

    def routing_table_path(self, split: str) -> str:
        return self.out_path('routing', f'routing_table_{split}.parquet')


def _merge(method: dict[str, Any], overrides: dict[str, Any], src: str) -> dict[str, Any]:
    out = copy.deepcopy(method)
    for section, values in (overrides or {}).items():
        allowed = METHOD_OVERRIDABLE.get(section)
        if allowed is None:
            raise ValueError(f'{src}: section "{section}" may not be overridden per dataset')
        for k, v in values.items():
            if k not in allowed:
                raise ValueError(
                    f'{src}: "{section}.{k}" is a frozen method setting and may not be '
                    f'overridden per dataset (allowed here: {sorted(allowed)})')
            out[section][k] = v
    return out


def load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def dataset_config_path(name: str) -> str:
    if os.path.exists(name):
        return name
    return os.path.join(CONFIG_DIR, 'datasets', f'{name.replace("-", "_")}.yaml')


def available_datasets() -> list[str]:
    d = os.path.join(CONFIG_DIR, 'datasets')
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith('.yaml'))


def load_config(dataset: str, args: argparse.Namespace | None = None,
                method_path: str | None = None) -> Config:
    args = args or argparse.Namespace()
    dpath = dataset_config_path(dataset)
    raw = load_yaml(dpath)
    method = load_yaml(method_path or os.path.join(CONFIG_DIR, 'method.yaml'))
    overrides = raw.pop('method_overrides', None)
    merged = _merge(method, overrides, os.path.basename(dpath))
    return Config(dataset=raw, method=merged, roots=Roots.from_args(args))


def add_common_arguments(ap: argparse.ArgumentParser, dataset_required: bool = True) -> None:
    """Flags every entry point accepts."""
    ap.add_argument('--dataset', required=dataset_required,
                    help='dataset key or path to a dataset YAML '
                         f'(available: {", ".join(available_datasets())})')
    ap.add_argument('--data-root', dest='data_root', default=None,
                    help='root of user-supplied corpora / ontologies [env DATA_ROOT]')
    ap.add_argument('--model-root', dest='model_root', default=None,
                    help='root for trained checkpoints [env MODEL_ROOT]')
    ap.add_argument('--output-dir', dest='output_dir', default=None,
                    help='root for generated artefacts [env OUTPUT_DIR]')
    ap.add_argument('--method-config', dest='method_config', default=None,
                    help='override configs/method.yaml')
