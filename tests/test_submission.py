import numpy as np
import pytest

from lp.submission import make_submission


def test_submissao_normaliza_e_valida():
    s = make_submission([1, 2], np.array([[2.0, 1.0, 1.0], [1.0, 1.0, 2.0]]))
    assert list(s.columns) == ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert np.allclose(s.iloc[:, 1:].sum(1), 1)
    with pytest.raises(ValueError):
        make_submission([1, 1], np.ones((2, 3)))
    with pytest.raises(ValueError):
        make_submission([1], np.array([[np.nan, 1, 1]]))
