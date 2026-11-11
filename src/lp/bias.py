"""Vieses de preferência na Chatbot Arena.

`bt_style` ajusta um Bradley-Terry (regressão logística) nas batalhas decisivas:
logit P(A vence) = posição + força(model_a) − força(model_b) + Σ β·estilo(A − B).
Com `control_models=False` o estilo leva o crédito da qualidade do modelo (modelos melhores
também escrevem mais); com o controle, β mede o efeito do estilo com o modelo fixo.
É a ideia do "style control" da LMSYS (2024), implementada aqui do zero."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def position_bias(y: np.ndarray) -> dict:
    y = np.asarray(y)
    dec = y[y < 2]
    p = float((dec == 0).mean())
    return {"n_decisive": int(len(dec)), "p_a_given_decisive": p,
            "se": float(np.sqrt(p * (1 - p) / len(dec))), "tie_rate": float((y == 2).mean())}


def bt_style(df: pd.DataFrame, style_cols: Sequence[str], *, control_models: bool = True,
             standardize: bool = True) -> pd.DataFrame:
    """df: model_a, model_b, win_a (0/1, só batalhas decisivas) e as colunas de estilo
    (diferenças A − B). Devolve coef, se e odds ratio por coluna, mais a posição."""
    S = df[list(style_cols)].to_numpy(float)
    if standardize:
        S = S / np.where(S.std(0) > 0, S.std(0), 1)
    cols = [np.ones(len(df))[:, None], S]
    names = ["position (A)", *style_cols]
    if control_models:
        models = sorted(set(df["model_a"]) | set(df["model_b"]))[1:]  # 1º = referência
        idx = {m: i for i, m in enumerate(models)}
        M = np.zeros((len(df), len(models)))
        for r, (a, b) in enumerate(zip(df["model_a"], df["model_b"], strict=True)):
            if a in idx:
                M[r, idx[a]] += 1
            if b in idx:
                M[r, idx[b]] -= 1
        cols.append(M)
        names += [f"model:{m}" for m in models]
    X = np.hstack(cols)
    y = df["win_a"].to_numpy()
    lr = LogisticRegression(C=1e4, fit_intercept=False, max_iter=5000).fit(X, y)
    beta = lr.coef_[0]
    p = 1 / (1 + np.exp(-X @ beta))
    H = X.T @ (X * (p * (1 - p))[:, None]) + np.eye(X.shape[1]) / 1e4
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    out = pd.DataFrame({"coef": beta, "se": se}, index=names)
    out["odds_ratio"] = np.exp(out["coef"])
    return out.loc[["position (A)", *style_cols]]
