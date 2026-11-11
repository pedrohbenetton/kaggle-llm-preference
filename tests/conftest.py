import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy():
    """120 batalhas: a resposta mais longa vence (viés de verbosidade), prompts repetidos."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        la, lb = rng.integers(5, 60, size=2)
        a, b = " ".join(["word"] * la), " ".join(["text"] * lb)
        win = 0 if la > lb + 5 else 1 if lb > la + 5 else 2
        rows.append({
            "id": i, "model_a": "m1", "model_b": "m2",
            "prompt": json.dumps([f"question {i % 40}"]),
            "response_a": json.dumps([a]), "response_b": json.dumps([b]),
            "winner_model_a": int(win == 0), "winner_model_b": int(win == 1),
            "winner_tie": int(win == 2),
        })
    return pd.DataFrame(rows)
