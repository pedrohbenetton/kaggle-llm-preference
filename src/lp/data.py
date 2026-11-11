"""Leitura e preparação dos dados da Chatbot Arena (LLM Classification Finetuning)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["winner_model_a", "winner_model_b", "winner_tie"]
_LOCAL = (
    str(Path(globals()["__file__"]).resolve().parents[3] / "llm-classification-finetuning"
        / "llm-classification-finetuning")
    if "__file__" in globals()
    else ""
)
_CANDIDATES = (
    os.environ.get("LP_DATA", ""),
    "/kaggle/input/llm-classification-finetuning",
    "/kaggle/input/competitions/llm-classification-finetuning",
    _LOCAL,
)


def data_dir() -> Path:
    for c in _CANDIDATES:
        if c and (Path(c) / "train.csv").exists():
            return Path(c)
    raise FileNotFoundError("train.csv não encontrado; defina LP_DATA")


def join_turns(s: str) -> str:
    """Prompt e respostas vêm como lista JSON (uma entrada por turno, às vezes null)."""
    try:
        turns = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return str(s)
    if not isinstance(turns, list):
        return str(turns)
    return "\n\n".join("" if t is None else str(t) for t in turns)


def labels(df: pd.DataFrame) -> np.ndarray:
    return df[TARGETS].to_numpy().argmax(1)


def swap(df: pd.DataFrame) -> pd.DataFrame:
    """Troca A e B (respostas, modelos e rótulos). Serve para aumentar dados e medir viés."""
    out = df.copy()
    for a, b in (("response_a", "response_b"), ("model_a", "model_b"),
                 ("winner_model_a", "winner_model_b")):
        if a in out and b in out:
            out[a], out[b] = df[b].to_numpy(), df[a].to_numpy()
    return out


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(data_dir() / f"{name}.csv")
