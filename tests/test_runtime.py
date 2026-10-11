"""Device resolution from configs/runtime.yaml, including the CPU fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gde import runtime


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime.ENV_OVERRIDE, raising=False)


def _config(tmp_path: Path, device: str) -> Path:
    path = tmp_path / "runtime.yaml"
    path.write_text(f"device: {device}\n", encoding="utf-8")
    return path


def test_repo_config_is_valid() -> None:
    assert runtime.load_runtime_config()["device"] in runtime.DEVICE_CHOICES


def test_cpu_is_always_cpu(tmp_path: Path) -> None:
    assert runtime.resolve_device(config_path=_config(tmp_path, "cpu")).type == "cpu"


def test_auto_follows_cuda_availability(tmp_path: Path) -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert runtime.resolve_device(config_path=_config(tmp_path, "auto")).type == expected


def test_cuda_falls_back_to_cpu_when_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="falling back to cpu"):
        device = runtime.resolve_device(config_path=_config(tmp_path, "cuda"))
    assert device.type == "cpu"


def test_auto_without_cuda_is_cpu_silently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runtime.resolve_device(config_path=_config(tmp_path, "auto")).type == "cpu"


def test_env_override_beats_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runtime.ENV_OVERRIDE, "cpu")
    assert runtime.resolve_device(config_path=_config(tmp_path, "cuda")).type == "cpu"


def test_invalid_device_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        runtime.resolve_device(config_path=_config(tmp_path, "tpu"))
