"""P7 model tests: masked softmax sums to 1, overfit-a-tiny-batch,
checkpoint reload gives identical outputs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gde.features.state_builder import FEATURE_COLUMNS
from gde.models.gbm import GbmScorer
from gde.models.mlp_setencoder import SetEncoderScorer, load_mlp, save_mlp


def _tiny_labels_df(n_states: int = 24, n_cand: int = 5, seed: int = 3):
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_states):
        base = rng.normal(size=len(FEATURE_COLUMNS))
        js = rng.normal(0.1, 0.02, size=n_cand)
        for i in range(n_cand):
            rows.append({
                "state_id": f"s{s}", "split": "train" if s < 18 else "val",
                "action_id": f"RUN:j{i}" if i < n_cand - 1 else "WAIT",
                "job_id": f"j{i}" if i < n_cand - 1 else None,
                "regret": float(js[i] - js.min()),
                "soft_target": float(np.exp(-(js[i]) / 0.05) / np.sum(np.exp(-js / 0.05))),
                "hard_label": "x", "tau": 0.05, "edf_pick": None, "sjf_pick": None,
                "fifo_pick": None,
                **{c: float(base[k] + i * 0.01) for k, c in enumerate(FEATURE_COLUMNS)},
            })
    return pd.DataFrame(rows)


def test_mlp_masked_softmax_sums_to_one():
    model = SetEncoderScorer()
    x = torch.randn(4, 7, len(FEATURE_COLUMNS))
    mask = torch.zeros(4, 7, dtype=torch.bool)
    mask[:, :3] = True
    mask[0, 4] = True
    scores = model.scores(x, mask)
    sums = scores.sum(-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)
    # masked-out entries get zero probability
    assert torch.all(scores[~mask] == 0)


def test_mlp_overfits_tiny_batch():
    torch.manual_seed(0)
    model = SetEncoderScorer(seed=0)
    x = torch.randn(2, 4, len(FEATURE_COLUMNS))
    mask = torch.ones(2, 4, dtype=torch.bool)
    target = torch.zeros(2, 4)
    target[:, 0] = 1.0  # always pick candidate 0
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        loss = model.loss(x, mask, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    probs = model.scores(x, mask)
    assert torch.all(probs.argmax(-1) == torch.zeros(2, dtype=torch.long))


def test_gbm_train_scores_and_reload(tmp_path):
    df = _tiny_labels_df()
    val = df[df["split"] == "val"]
    scorer = GbmScorer.train(df, val)
    rows = [{c: float(v) for c, v in zip(FEATURE_COLUMNS, r)}
            for r in np.random.default_rng(5).normal(size=(4, len(FEATURE_COLUMNS)))]
    scores = scorer.scores(rows)
    assert len(scores) == 4 and pytest.approx(sum(scores), abs=1e-6) == 1.0
    path = tmp_path / "gbm.txt"
    scorer.save(path)
    loaded = GbmScorer.load(path)
    assert loaded.predict_regret(rows) == pytest.approx(scorer.predict_regret(rows))


def test_mlp_checkpoint_reload_identical(tmp_path):
    model = SetEncoderScorer(seed=1)
    model.temperature = 0.7
    save_mlp(model, tmp_path / "mlp.pt")
    loaded = load_mlp(tmp_path / "mlp.pt")
    x = torch.randn(3, 5, len(FEATURE_COLUMNS))
    mask = torch.ones(3, 5, dtype=torch.bool)
    assert torch.allclose(loaded.scores(x, mask), model.scores(x, mask), atol=1e-7)
    assert loaded.temperature == 0.7
