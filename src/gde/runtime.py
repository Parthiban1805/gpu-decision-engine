"""Device selection for training and inference, driven by configs/runtime.yaml."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from gde import paths

if TYPE_CHECKING:
    import torch

DEVICE_CHOICES = ("auto", "cuda", "cpu")
ENV_OVERRIDE = "GDE_DEVICE"


def load_runtime_config(path: Path | None = None) -> dict[str, Any]:
    path = path or paths.CONFIGS / "runtime.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_device(requested: str | None = None, config_path: Path | None = None) -> torch.device:
    """Return the torch device to use.

    Precedence: explicit argument > GDE_DEVICE env var > configs/runtime.yaml > "auto".
    Requesting cuda without a usable GPU falls back to cpu with a warning.
    """
    import torch

    choice = requested or os.environ.get(ENV_OVERRIDE) or load_runtime_config(config_path).get("device", "auto")
    choice = str(choice).lower()
    if choice not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}, got {choice!r}")
    if choice == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "cuda":
        warnings.warn("device=cuda requested but CUDA is unavailable; falling back to cpu", RuntimeWarning, stacklevel=2)
    return torch.device("cpu")
