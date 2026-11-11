import numpy as np
import pandas as pd

from lp.bias import bt_style, position_bias


def _synthetic(n=6000, beta=0.8, conf=1.5, seed=0):
    """Modelos fortes escrevem mais (confundidor); o estilo tem efeito próprio `beta`."""
    rng = np.random.default_rng(seed)
    strength = np.array([-1.0, 0.0, 1.0, 2.0])
    a, b = rng.integers(0, 4, n), rng.integers(0, 4, n)
    keep = a != b
    a, b = a[keep], b[keep]
    style = conf * (strength[a] - strength[b]) / 3 + rng.normal(size=len(a))
    logit = 0.1 + strength[a] - strength[b] + beta * style
    win_a = rng.random(len(a)) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"model_a": [f"m{i}" for i in a], "model_b": [f"m{i}" for i in b],
                         "win_a": win_a.astype(int), "style": style})


def test_bt_style_separa_estilo_da_forca_do_modelo():
    df = _synthetic()
    ctrl = bt_style(df, ["style"], control_models=True, standardize=False)
    naive = bt_style(df, ["style"], control_models=False, standardize=False)
    assert abs(ctrl.loc["style", "coef"] - 0.8) < 0.15
    assert naive.loc["style", "coef"] > ctrl.loc["style", "coef"] + 0.3  # confundido
    assert (ctrl["se"] > 0).all()
    assert abs(ctrl.loc["position (A)", "coef"] - 0.1) < 0.12


def test_position_bias_conta_so_decisivas():
    y = np.array([0, 0, 0, 1, 2, 2])
    r = position_bias(y)
    assert r["n_decisive"] == 4 and abs(r["p_a_given_decisive"] - 0.75) < 1e-9
