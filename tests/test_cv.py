from lp.cv import grouped_folds


def test_folds_nunca_dividem_um_prompt(toy):
    for tr, va in grouped_folds(toy, 4):
        assert not set(toy.prompt.iloc[tr]) & set(toy.prompt.iloc[va])
