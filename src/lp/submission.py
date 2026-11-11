"""Arquivo de submissão: id + probabilidades das 3 classes, normalizadas."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from lp.data import TARGETS


def make_submission(ids: Sequence, probs: np.ndarray) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (len(ids), 3):
        raise ValueError("formato errado")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("probabilidade inválida")
    if len(set(ids)) != len(ids):
        raise ValueError("id repetido")
    probs = probs / probs.sum(1, keepdims=True)
    return pd.DataFrame({"id": list(ids), **dict(zip(TARGETS, probs.T, strict=True))})


def write_submission(s: pd.DataFrame, out_dir: str | Path = ".") -> Path:
    path = Path(out_dir) / "submission.csv"
    s.to_csv(path, index=False)
    return path
