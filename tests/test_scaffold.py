"""P0 gate: package imports, configs parse, task runner exposes the documented targets."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

import pytest
import yaml

from gde import paths

SUBPACKAGES = [
    "ingestion", "simulator", "decisions", "features", "baselines",
    "labeling", "models", "evaluation", "api",
]


def test_package_version() -> None:
    import gde

    assert gde.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name: str) -> None:
    importlib.import_module(f"gde.{name}")


def test_objectives_config() -> None:
    cfg = yaml.safe_load((paths.CONFIGS / "objectives.yaml").read_text(encoding="utf-8"))
    assert set(cfg["presets"]) == {"balanced", "latency_critical", "cost_saver"}
    for weights in cfg["presets"].values():
        assert set(weights) == {"w_cost", "w_deadline", "w_completion", "w_frag"}
        assert all(w >= 0 for w in weights.values())


def test_cluster_small_config() -> None:
    cfg = yaml.safe_load((paths.CONFIGS / "cluster_small.yaml").read_text(encoding="utf-8"))
    assert sum(g["count"] for g in cfg["gpus"]) == 8
    for g in cfg["gpus"]:
        assert g["p_peak_w"] >= g["p_idle_w"] >= 0
        assert g["vram_gb"] > 0 and g["compute_factor"] > 0


def _load_tasks():
    spec = importlib.util.spec_from_file_location("tasks", paths.REPO_ROOT / "tasks.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_runner_targets() -> None:
    tasks = _load_tasks()
    documented = {"test", "data", "sim-smoke", "baselines", "label", "train-v1",
                  "bench", "train-v2", "api", "setup", "hwcheck"}
    assert documented <= set(tasks.TASKS)


def test_unimplemented_target_fails_loudly() -> None:
    r = subprocess.run([sys.executable, str(paths.REPO_ROOT / "tasks.py"), "bench"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "P8" in r.stderr


def test_python_version_supported() -> None:
    assert sys.version_info[:2] in {(3, 11), (3, 12)}
