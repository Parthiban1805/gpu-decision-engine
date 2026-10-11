"""GPU smoke tests; skipped when CUDA is unavailable."""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_matmul_on_cuda_matches_cpu() -> None:
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(256, 128, generator=gen)
    b = torch.randn(128, 64, generator=gen)
    out = (a.cuda() @ b.cuda()).cpu()
    torch.testing.assert_close(out, a @ b, rtol=1e-3, atol=1e-3)


def test_tiny_mlp_forward_backward_on_cuda() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    x = torch.randn(64, 16, device=device)
    y = torch.randn(64, 1, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    loss_before = nn.functional.mse_loss(model(x), y)
    loss_before.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and g.device.type == "cuda" and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    opt.step()

    with torch.no_grad():
        loss_after = nn.functional.mse_loss(model(x), y)
    assert loss_after < loss_before
