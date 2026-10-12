"""Qwen backbone source resolution (PLAN.md §6, V2 scorer).

The backbone is pinned by exact id + revision in configs/backbone.yaml. `python tasks.py
download-qwen` materializes that snapshot under data/models/<name>/ with checksums; once
present, every load (probe, training, API) prefers the local snapshot so runs are offline
and reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gde import paths


def load_backbone_config(path: Path | None = None) -> dict[str, Any]:
    path = path or paths.CONFIGS / "backbone.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("model_id", "revision"):
        if not cfg.get(key):
            raise ValueError(f"{path} is missing required key {key!r}")
    return cfg


def backbone_local_dir(model_id: str) -> Path:
    return paths.DATA / "models" / model_id.split("/")[-1]


def local_snapshot_exists(model_id: str) -> bool:
    return (backbone_local_dir(model_id) / "config.json").is_file()


def resolve_backbone_source(model_id: str, revision: str) -> tuple[str, str | None]:
    """Return (from_pretrained source, revision), preferring the pinned local snapshot.

    A revision must not be passed for a local directory (transformers rejects it); the
    snapshot is already pinned — its checksums live in SHA256SUMS next to it.
    """
    if local_snapshot_exists(model_id):
        return str(backbone_local_dir(model_id)), None
    return model_id, revision
