"""Passo 1: vieses humanos na Chatbot Arena (posição, tamanho, formatação, recusas),
com e sem controle pela força de cada modelo. Gera results/bias.json e results/figures/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lp.bias import bt_style, position_bias
from lp.data import labels, load
from lp.features import features

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
OUT = Path("results")
(OUT / "figures").mkdir(parents=True, exist_ok=True)

df = load("train")
y = labels(df)
f = features(df)


def nd(name):  # diferença normalizada (A − B) / (A + B + 1), entre -1 e 1
    return f[f"d_{name}"] / (f[f"s_{name}"] + 1)


style = pd.DataFrame({
    "tamanho": f["d_chars"] / (f["s_chars"] + 1),
    "listas": (f["d_bullet"] + f["d_numbered"]) / (f["s_bullet"] + f["s_numbered"] + 1),
    "títulos": nd("header"),
    "negrito": nd("bold"),
    "blocos de código": nd("code"),
    "recusa": np.sign(f["d_refusal"]),
})
dec = y < 2
bt_df = pd.concat([df.loc[dec, ["model_a", "model_b"]].reset_index(drop=True),
                   style[dec].reset_index(drop=True)], axis=1)
bt_df["win_a"] = (y[dec] == 0).astype(int)
cols = list(style.columns)
naive = bt_style(bt_df, cols, control_models=False)
ctrl = bt_style(bt_df, cols, control_models=True)
pos = position_bias(y)

# taxa de vitória da resposta mais longa por faixa de razão de tamanho
len_a = (f["s_chars"] + f["d_chars"]) / 2
len_b = (f["s_chars"] - f["d_chars"]) / 2
ratio = np.log2(len_a + 1) - np.log2(len_b + 1)
edges = [-np.inf, -3, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 3, np.inf]
b = pd.cut(ratio[dec], edges)
wins = pd.Series(y[dec] == 0).groupby(b.to_numpy(), observed=True).agg(["mean", "size"])
wins["se"] = np.sqrt(wins["mean"] * (1 - wins["mean"]) / wins["size"])

# figura 1
fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
x = np.arange(len(wins))
ax.axhline(0.5, color=MUTED, lw=1, ls="--")
ax.errorbar(x, wins["mean"], yerr=1.96 * wins["se"], color=BLUE, lw=2, marker="o", ms=7,
            capsize=0, elinewidth=1.5)
lab = ["≤ 1/8", "1/8–1/4", "1/4–1/2", "1/2–0,7", "0,7–0,93", "≈ 1", "1,07–1,4", "1,4–2",
       "2–4", "4–8", "≥ 8"]
ax.set_xticks(x, lab, fontsize=8, color=MUTED)
ax.set_xlabel("tamanho de A ÷ tamanho de B (caracteres)", color=MUTED)
ax.set_ylabel("P(A vence | sem empate)", color=MUTED)
ax.set_title("Na Chatbot Arena, a resposta mais longa tende a vencer", color=INK, loc="left")
ax.grid(axis="y", color=GRID)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.annotate(f"{wins['mean'].iloc[-1]:.0%}", (x[-1], wins["mean"].iloc[-1]),
            textcoords="offset points", xytext=(-28, 6), color=INK, fontsize=9)
ax.annotate(f"{wins['mean'].iloc[0]:.0%}", (x[0], wins["mean"].iloc[0]),
            textcoords="offset points", xytext=(-6, -18), color=INK, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "figures" / "length_winrate.png", dpi=150)

# figura 2
fig, ax = plt.subplots(figsize=(8, 4), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
yy = np.arange(len(cols))[::-1]
for tab, color, name, dy in ((naive, ORANGE, "sem controle", 0.15),
                                             (ctrl, BLUE, "controlando pelo modelo", -0.15)):
    c, s = tab.loc[cols, "coef"], tab.loc[cols, "se"]
    ax.errorbar(np.exp(c), yy + dy, xerr=[np.exp(c) - np.exp(c - 1.96 * s),
                np.exp(c + 1.96 * s) - np.exp(c)], fmt="o", color=color, ms=7,
                elinewidth=1.5, capsize=0, label=name)
ax.axvline(1, color=MUTED, lw=1, ls="--")
ax.set_yticks(yy, cols, color=INK)
ax.set_xlabel("odds de vencer por +1 desvio-padrão de diferença (A − B)", color=MUTED)
ax.set_title("Efeito do estilo na preferência, com e sem controle pela força do modelo",
             color=INK, loc="left", fontsize=11)
ax.grid(axis="x", color=GRID)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "figures" / "style_effects.png", dpi=150)

res = {
    "posicao": {k: round(v, 4) if isinstance(v, float) else v for k, v in pos.items()},
    "odds_ratio_sem_controle": naive["odds_ratio"].round(3).to_dict(),
    "odds_ratio_com_controle": ctrl["odds_ratio"].round(3).to_dict(),
    "se_com_controle": ctrl["se"].round(4).to_dict(),
    "vitoria_A_por_faixa_de_tamanho": dict(zip(lab, wins["mean"].round(3), strict=True)),
    "n_por_faixa": dict(zip(lab, wins["size"].astype(int), strict=True)),
}
(OUT / "bias.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
