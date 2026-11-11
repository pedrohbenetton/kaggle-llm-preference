"""Features de forma das respostas. `d_*` = A − B (troca de sinal ao trocar A e B),
`s_*` = simétricas. Capturam os vieses conhecidos da Arena: verbosidade, formatação
(markdown, listas, código) e recusas."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from lp.data import join_turns

REFUSAL = re.compile(r"\b(I'm sorry|I am sorry|I cannot|I can't|I apologize|As an AI|"
                     r"I'm not able|I am not able|unable to)\b", re.IGNORECASE)
PATTERNS = {
    "bold": r"\*\*", "header": r"(?m)^#{1,6} ", "bullet": r"(?m)^\s*[-*•] ",
    "numbered": r"(?m)^\s*\d+[.)] ", "code": r"```", "newline": r"\n", "question": r"\?",
    "exclaim": r"!", "url": r"https?://",
}


def _one(texts: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=texts.index)
    f["chars"] = texts.str.len()
    f["words"] = texts.str.split().str.len().fillna(0)
    f["log_chars"] = np.log1p(f["chars"])
    for k, p in PATTERNS.items():
        f[k] = texts.str.count(p)
    f["refusal"] = texts.str.count(REFUSAL)
    f["empty"] = (f["chars"] == 0).astype(int)
    return f


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(1, len(sa | sb))


def features(df: pd.DataFrame) -> pd.DataFrame:
    a = df["response_a"].map(join_turns)
    b = df["response_b"].map(join_turns)
    p = df["prompt"].map(join_turns)
    fa, fb = _one(a), _one(b)
    out = pd.DataFrame(index=df.index)
    for c in fa.columns:
        out[f"d_{c}"] = fa[c] - fb[c]
        out[f"s_{c}"] = fa[c] + fb[c]
    out["d_len_ratio"] = np.log1p(fa["chars"]) - np.log1p(fb["chars"])
    out["s_prompt_chars"] = p.str.len()
    out["s_turns"] = df["prompt"].str.count('","') + 1
    out["s_jaccard"] = [_jaccard(x, y) for x, y in zip(a, b, strict=True)]
    out["s_identical"] = (a == b).astype(int)
    return out.astype(float)
