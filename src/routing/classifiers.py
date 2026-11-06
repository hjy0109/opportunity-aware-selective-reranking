"""Router classifier families.

XGBoost is the frozen choice; the other three families are kept because the
selection among them is part of the reported method development and a reviewer
may want to re-run it.  Every family is instantiated identically for the
Opportunity and the Difficulty target -- same class, same hyperparameters, same
random state, same class balancing -- so the controlled comparison differs only
in the supervision label.
"""
from __future__ import annotations

from typing import Any

FAMILIES = ('XGB', 'LR', 'RF', 'GBDT')
FAMILY_LABEL = {'XGB': 'XGBoost', 'LR': 'Logistic Regression',
                'RF': 'Random Forest', 'GBDT': 'HistGradientBoosting (sklearn)'}

LR_SPEC = dict(solver='lbfgs', C=1.0, penalty='l2', max_iter=2000, tol=1e-4,
               class_weight='balanced')


def expand_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Full factorial over the declared grid, keys in sorted order.

    No random or Bayesian search: the grid is declared in configs/method.yaml and
    is enumerated exhaustively.
    """
    keys = sorted(grid)
    out: list[dict[str, Any]] = [dict()]
    for k in keys:
        out = [dict(o, **{k: v}) for o in out for v in grid[k]]
    return out


def scale_pos_weight(y) -> float:
    """XGBoost's counterpart to sklearn's ``class_weight='balanced'``."""
    import numpy as np
    y = np.asarray(y)
    return float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)


def make(family: str, hyperparams: dict[str, Any], fixed: dict[str, Any],
         random_state: int, y=None):
    if family == 'XGB':
        import xgboost as xgb
        kw = dict(fixed)
        kw['random_state'] = random_state
        if y is not None:
            kw['scale_pos_weight'] = scale_pos_weight(y)
        return xgb.XGBClassifier(**hyperparams, **kw)
    if family == 'LR':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=random_state, **LR_SPEC)
    if family == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=random_state, n_jobs=1,
                                      class_weight='balanced', **hyperparams)
    if family == 'GBDT':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            random_state=random_state, class_weight='balanced', early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, **hyperparams)
    raise ValueError(f'unknown classifier family: {family}')


def save(family: str, model, path: str) -> str:
    from ..utils.io import ensure_parent
    ensure_parent(path)
    if family == 'XGB':
        model.save_model(path)
    else:
        import joblib
        joblib.dump(model, path, compress=0)
    return path


def load(family: str, path: str):
    if family == 'XGB':
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(path)
        m.set_params(n_jobs=1, device='cpu')
        return m
    import joblib
    return joblib.load(path)
