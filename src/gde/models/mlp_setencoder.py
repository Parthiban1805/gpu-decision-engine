"""V1(b): small set-encoder MLP over candidates (DeepSets-style, PLAN.md §6).

Per candidate: encode (candidate||state features) with an MLP; pool over the
candidate set (mean) for a context vector; score each candidate with a second
MLP on (embedding||context). Masked softmax over the state's candidates.
Trained with soft cross-entropy against the rollout soft targets; temperature
scaled on the validation split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gde.features.state_builder import FEATURE_COLUMNS
from gde.runtime import resolve_device


class SetEncoderScorer(nn.Module):
    def __init__(self, n_features: int = len(FEATURE_COLUMNS), hidden: int = 64,
                 seed: int = 7) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.temperature = 1.0

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, N, F); mask: (B, N) bool. Returns logits (B, N)."""
        h = self.enc(x)                                  # (B, N, H)
        ctx = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        logits = self.head(torch.cat([h, ctx.unsqueeze(1).expand(-1, h.size(1), -1)],
                                     dim=-1)).squeeze(-1)
        return logits.masked_fill(~mask, -1e9)

    def loss(self, x, mask, soft_target):
        logits = self.forward(x, mask) / max(self.temperature, 1e-6)
        logp = F.log_softmax(logits, dim=-1)
        return -(soft_target * logp).sum(-1).mean()

    @torch.no_grad()
    def scores(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, mask) / max(self.temperature, 1e-6)
        return F.softmax(logits, dim=-1)


def train_mlp(train_rows, val_rows, epochs: int = 60, batch_states: int = 256,
              lr: float = 1e-3, seed: int = 7) -> tuple[SetEncoderScorer, dict]:
    """train_rows/val_rows: list of dicts {features: (N,F) array, soft: (N,)}."""
    device = resolve_device()
    model = SetEncoderScorer(seed=seed).to(device)

    def to_batch(rows):
        n_max = max(len(r["soft"]) for r in rows)
        x = np.zeros((len(rows), n_max, len(FEATURE_COLUMNS)), dtype=np.float32)
        mask = np.zeros((len(rows), n_max), dtype=bool)
        t = np.zeros((len(rows), n_max), dtype=np.float32)
        for i, r in enumerate(rows):
            n = len(r["soft"])
            x[i, :n] = r["features"]
            mask[i, :n] = True
            t[i, :n] = r["soft"]
        return (torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device),
                torch.from_numpy(t).to(device))

    xt, mt, tt = to_batch(train_rows)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for _ in range(max(1, len(train_rows) // batch_states)):
            idx = rng.choice(len(train_rows), size=min(batch_states, len(train_rows)),
                             replace=False)
            loss = model.loss(xt[idx], mt[idx], tt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()

    # temperature scaling on val (NLL over soft targets)
    xv, mv, tv = to_batch(val_rows)
    with torch.no_grad():
        logits = model.forward(xv, mv)
    best_t, best_nll = 1.0, float("inf")
    for t in np.geomspace(0.05, 5.0, 40):
        logp = F.log_softmax(logits / float(t), dim=-1)
        nll = float(-(tv * logp).sum(-1).mean())
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    model.temperature = best_t
    return model, {"val_nll": best_nll, "temperature": best_t}


def mlp_state_rows(df) -> list[dict]:
    """Group a labels frame into per-state training rows."""
    out = []
    for _, g in df.groupby("state_id", sort=False):
        out.append({"features": g[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
                    "soft": g["soft_target"].to_numpy(dtype=np.float32),
                    "action_ids": g["action_id"].tolist(),
                    "regret": g["regret"].to_numpy(dtype=np.float32)})
    return out


def save_mlp(model: SetEncoderScorer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "temperature": model.temperature}, path)


def load_mlp(path: Path) -> SetEncoderScorer:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = SetEncoderScorer()
    model.load_state_dict(payload["state_dict"])
    model.temperature = payload["temperature"]
    return model
