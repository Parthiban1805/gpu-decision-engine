"""V1 training entry: GBM regret regression + set-encoder MLP on the P6 labels.

Offline selection on VAL (regret of the model's argmin vs EDF/SJF picks);
checkpoints to checkpoints/v1/; both heads share the identical data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gde.features.state_builder import FEATURE_COLUMNS
from gde.models.gbm import GbmScorer
from gde.models.mlp_setencoder import mlp_state_rows, train_mlp
from gde.paths import CHECKPOINTS


def offline_metrics(state_rows: list[dict], pick_fn) -> dict:
    """pick_fn(features ndarray, action_ids) -> index of the chosen candidate."""
    regrets, picks = [], []
    for r in state_rows:
        i = pick_fn(r["features"], r["action_ids"])
        picks.append(r["action_ids"][i])
        regrets.append(float(r["regret"][i]))
    return {"mean_regret": float(np.mean(regrets)),
            "median_regret": float(np.median(regrets)),
            "p90_regret": float(np.percentile(regrets, 90)),
            "picks": pd.Series(picks).value_counts(normalize=True).to_dict()}


def _gbm_pick_factory(scorer: GbmScorer):
    def pick(features, action_ids):
        rows = [{c: float(v) for c, v in zip(FEATURE_COLUMNS, row)}
                for row in features]
        return int(np.argmax(scorer.scores(rows)))
    return pick


def _mlp_pick_factory(model):
    import torch

    def pick(features, action_ids):
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
        mask = torch.ones(1, x.size(1), dtype=torch.bool)
        s = model.scores(x, mask)[0].numpy()
        return int(np.argmax(s))
    return pick


def heuristic_picks(state_rows: list[dict]) -> dict:
    """Regret of policy picks, recomputed from candidate features: EDF/SJF
    approximations are taken from the stored picks in the labels frame."""
    return {}  # filled by caller from df (edf_pick/sjf_pick)


def pick_regret_from_df(df: pd.DataFrame, pick_col: str) -> dict:
    """Mean regret of a stored policy pick (edf_pick/sjf_pick) per state."""
    regrets = []
    for _, g in df.groupby("state_id", sort=False):
        pick = g[pick_col].iloc[0]
        sel = g[g["action_id"] == f"RUN:{pick}"] if pick else g[g["is_wait"] == 1]
        regrets.append(float(sel["regret"].iloc[0]) if len(sel) else float(g["regret"].max()))
    return {"mean_regret": float(np.mean(regrets)),
            "median_regret": float(np.median(regrets))}


def train_v1(parquet_path: Path, out_dir: Path = CHECKPOINTS / "v1") -> dict:
    df = pd.read_parquet(parquet_path)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    gbm = GbmScorer.train(train, val)
    gbm.save(out_dir / "gbm.txt")

    train_rows = mlp_state_rows(train)
    val_rows = mlp_state_rows(val)
    test_rows = mlp_state_rows(test)
    mlp, mlp_meta = train_mlp(train_rows, val_rows)
    from gde.models.mlp_setencoder import save_mlp
    save_mlp(mlp, out_dir / "mlp.pt")

    val_states = mlp_state_rows(val)
    metrics = {
        "n_states": {"train": len(train_rows), "val": len(val_rows),
                     "test": len(test_rows)},
        "gbm": {
            "val": offline_metrics(val_states, _gbm_pick_factory(gbm)),
            "test": offline_metrics(test_rows, _gbm_pick_factory(gbm)),
        },
        "mlp": {
            "val": offline_metrics(val_states, _mlp_pick_factory(mlp)),
            "test": offline_metrics(test_rows, _mlp_pick_factory(mlp)),
            **mlp_meta,
        },
        "baselines": {
            "edf": pick_regret_from_df(val, "edf_pick"),
            "sjf": pick_regret_from_df(val, "sjf_pick"),
            "fifo": pick_regret_from_df(val, "fifo_pick"),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str),
                                          encoding="utf-8")
    return {"gbm": gbm, "mlp": mlp, "metrics": metrics}
