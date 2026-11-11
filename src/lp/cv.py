"""Validação agrupada por prompt: o mesmo prompt aparece em várias batalhas."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from lp.data import labels

Folds = list[tuple[np.ndarray, np.ndarray]]


def grouped_folds(df: pd.DataFrame, n_splits: int = 5) -> Folds:
    return list(GroupKFold(n_splits=n_splits).split(df, groups=df["prompt"]))


def oof_predict(factory: Callable[[], object], df: pd.DataFrame, folds: Folds) -> np.ndarray:
    P = np.zeros((len(df), 3))
    y = labels(df)
    for tr, va in folds:
        m = factory()
        m.fit(df.iloc[tr], y[tr])
        P[va] = m.predict_proba(df.iloc[va])
    return P
