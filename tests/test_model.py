import numpy as np
from sklearn.metrics import log_loss

from lp.cv import grouped_folds, oof_predict
from lp.data import labels
from lp.model import FeatureGBM


def test_modelo_aprende_verbosidade_e_e_simetrico(toy):
    y = labels(toy)
    P = oof_predict(lambda: FeatureGBM(n_estimators=60), toy, grouped_folds(toy, 4))
    assert np.allclose(P.sum(1), 1)
    assert log_loss(y, P, labels=[0, 1, 2]) < log_loss(y, np.full_like(P, 1 / 3))
