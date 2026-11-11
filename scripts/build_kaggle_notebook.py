"""Gera o notebook de submissão (autocontido, sem internet) a partir de src/lp."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["data", "features", "model", "submission"]
MAIN = '''
# Treina no train.csv inteiro (com a cópia A<->B) e prevê o teste, com média da previsão trocada.
train, test = load("train"), load("test")
model = FeatureGBM().fit(train, labels(train))
P = model.predict_proba(test)
path = write_submission(make_submission(test["id"].tolist(), P))
print(path, len(test), "linhas")
'''


def module_source(name: str) -> str:
    src = (ROOT / "src" / "lp" / f"{name}.py").read_text(encoding="utf-8")
    src = re.sub(r"^from lp\.\S+ import .*\n", "", src, flags=re.MULTILINE)
    src = re.sub(r"^from __future__ import annotations\n", "", src, flags=re.MULTILINE)
    return f"# ---- lp/{name}.py ----\n" + src


def build() -> dict:
    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.strip("\n").splitlines(True)}

    intro = ("# Chatbot Arena: features de forma + LightGBM com troca A/B\n\n"
             "Gerado por `scripts/build_kaggle_notebook.py` a partir de "
             "[kaggle-llm-preference](https://github.com/pedrohbenetton/kaggle-llm-preference). "
             "CV agrupada por prompt: log loss 1,029.")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(True)}]
    cells += [code(module_source(m)) for m in MODULES] + [code(MAIN)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notebooks/kaggle_submission.ipynb")
    a = ap.parse_args()
    Path(a.out).write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(a.out)
