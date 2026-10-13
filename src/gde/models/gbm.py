"""V1(a): LightGBM per-candidate REGRET regression (PLAN.md §6).

Score(candidate) = -predicted_regret; softmax over candidates per state.
Trained on TRAIN split only; val used for early stopping + selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from gde.features.state_builder import FEATURE_COLUMNS


@dataclass
class GbmScorer:
    model: lgb.Booster
    feature_columns: list[str]

    @classmethod
    def train(cls, df: pd.DataFrame, val_df: pd.DataFrame,
              params: dict | None = None) -> "GbmScorer":
        params = params or {
            "objective": "regression", "metric": "l2", "learning_rate": 0.05,
            "num_leaves": 31, "min_data_in_leaf": 40, "feature_fraction": 0.9,
            "bagging_fraction": 0.9, "bagging_freq": 1, "seed": 7, "verbose": -1,
        }
        x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = df["regret"].to_numpy(dtype=np.float32)
        xv = val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        yv = val_df["regret"].to_numpy(dtype=np.float32)
        train_set = lgb.Dataset(x, label=y)
        val_set = lgb.Dataset(xv, label=yv, reference=train_set)
        model = lgb.train(params, train_set, num_boost_round=600,
                          valid_sets=[val_set],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        return cls(model=model, feature_columns=list(FEATURE_COLUMNS))

    def predict_regret(self, rows: list[dict]) -> np.ndarray:
        x = np.array([[float(r.get(c, 0.0)) for c in self.feature_columns]
                      for r in rows], dtype=np.float32)
        return self.model.predict(x)

    def scores(self, rows: list[dict]) -> list[float]:
        """Softmax(-regret) over one state's candidates."""
        neg = -self.predict_regret(rows)
        neg = neg - neg.max()
        e = np.exp(neg)
        return (e / e.sum()).tolist()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "GbmScorer":
        return cls(model=lgb.Booster(model_file=str(path)),
                   feature_columns=list(FEATURE_COLUMNS))
