import numpy as np

from lp.data import swap
from lp.features import features


def test_features_antissimetricas_na_troca(toy):
    f, g = features(toy), features(swap(toy))
    diff = [c for c in f.columns if c.startswith("d_")]
    assert diff and np.allclose(f[diff].to_numpy(), -g[diff].to_numpy())
    same = [c for c in f.columns if c.startswith("s_")]
    assert np.allclose(f[same].to_numpy(), g[same].to_numpy())
