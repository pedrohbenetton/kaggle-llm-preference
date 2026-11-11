"""CV agrupada por prompt: prior constante contra LightGBM nas features de forma."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

from lp.cv import grouped_folds, oof_predict
from lp.data import labels, load
from lp.model import FeatureGBM

df = load("train")
y = labels(df)
folds = grouped_folds(df, 5)
res = {}
prior = np.bincount(y, minlength=3) / len(y)
res["prior constante"] = log_loss(y, np.tile(prior, (len(y), 1)))
res["uniforme"] = log_loss(y, np.full((len(y), 3), 1 / 3))
t = time.time()
P = oof_predict(FeatureGBM, df, folds)
res["features de forma + LightGBM (troca A/B)"] = log_loss(y, P)
print(f"tempo {time.time() - t:.0f}s")
for k, v in res.items():
    print(f"{k:45} {v:.4f}")
Path("results/raw").mkdir(parents=True, exist_ok=True)
np.save("results/raw/oof_feature_gbm.npy", P)
Path("results/cv_baseline.json").write_text(
    json.dumps({k: round(v, 4) for k, v in res.items()}, indent=2, ensure_ascii=False))
