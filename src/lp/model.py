"""LightGBM sobre as features de forma, com aumento por troca A↔B: treina com a cópia
trocada e, na previsão, faz a média com a previsão trocada de volta. Isso força a simetria
(P(A vence | A,B) = P(B vence | B,A)) e reduz o viés de posição do modelo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from lp.features import features


def _swap_features(f: pd.DataFrame) -> pd.DataFrame:
    g = f.copy()
    d = [c for c in f.columns if c.startswith("d_")]
    g[d] = -f[d]
    return g


_SWAP_Y = np.array([1, 0, 2])


class FeatureGBM:
    def __init__(self, **kw):
        params = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        params.update(kw)
        self.clf = LGBMClassifier(objective="multiclass", **params)

    def fit(self, df: pd.DataFrame, y: np.ndarray):
        f = features(df)
        X = pd.concat([f, _swap_features(f)], ignore_index=True)
        self.clf.fit(X, np.concatenate([y, _SWAP_Y[y]]))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        f = features(df)
        p = self.clf.predict_proba(f)
        q = self.clf.predict_proba(_swap_features(f))[:, [1, 0, 2]]
        return (p + q) / 2
