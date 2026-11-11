import json
import os
import subprocess
import sys

import pandas as pd


def test_notebook_roda_offline_e_gera_submissao(tmp_path, toy):
    data = tmp_path / "data"
    data.mkdir()
    toy.to_csv(data / "train.csv", index=False)
    toy.drop(columns=["model_a", "model_b", "winner_model_a", "winner_model_b",
                      "winner_tie"]).head(7).to_csv(data / "test.csv", index=False)
    nb = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "scripts/build_kaggle_notebook.py", "--out", str(nb)],
                   check=True)
    code = "\n\n".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"]
                       if c["cell_type"] == "code")
    assert "from lp" not in code
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                   env={**os.environ, "LP_DATA": str(data)}, check=True)
    sub = pd.read_csv(tmp_path / "submission.csv")
    assert len(sub) == 7 and abs(sub.iloc[:, 1:].sum(1) - 1).max() < 1e-6
