import numpy as np

from lp.data import TARGETS, join_turns, labels, swap


def test_join_turns_lida_com_json_null_e_texto_quebrado():
    assert join_turns('["oi", "tudo bem"]') == "oi\n\ntudo bem"
    assert join_turns('["oi", null]') == "oi\n\n"
    assert join_turns("não é json") == "não é json"


def test_labels_inteiros(toy):
    y = labels(toy)
    assert set(np.unique(y)) <= {0, 1, 2}
    assert (y == toy[TARGETS].to_numpy().argmax(1)).all()


def test_swap_troca_respostas_e_rotulos(toy):
    s = swap(toy)
    assert (s["response_a"] == toy["response_b"]).all()
    assert (s["winner_model_a"] == toy["winner_model_b"]).all()
    assert (s["winner_tie"] == toy["winner_tie"]).all()
    assert (swap(s)[toy.columns] == toy).all().all()
