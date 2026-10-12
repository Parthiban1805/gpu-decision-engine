"""Backbone snapshot resolution: prefer the pinned local copy over the Hub."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gde.models import backbone

MODEL = "Qwen/Qwen3-0.6B-Base"


def test_backbone_config_pins_id_and_full_revision() -> None:
    cfg = backbone.load_backbone_config()
    assert "/" in cfg["model_id"]
    assert len(cfg["revision"]) == 40  # full commit hash, never a moving branch


def test_hub_source_when_no_local_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backbone.paths, "DATA", tmp_path)
    assert backbone.resolve_backbone_source(MODEL, "deadbeef") == (MODEL, "deadbeef")


def test_local_snapshot_preferred_over_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backbone.paths, "DATA", tmp_path)  # patch BEFORE creating any files
    local = backbone.backbone_local_dir(MODEL)
    local.mkdir(parents=True)
    (local / "config.json").write_text("{}", encoding="utf-8")
    source, revision = backbone.resolve_backbone_source(MODEL, "deadbeef")
    assert source == str(local)
    assert revision is None  # a revision must not be passed for a local directory


LOCAL_SNAPSHOT = backbone.backbone_local_dir(backbone.load_backbone_config()["model_id"])


@pytest.mark.skipif(not (LOCAL_SNAPSHOT / "SHA256SUMS").is_file(),
                    reason="backbone not downloaded yet (python tasks.py download-qwen)")
def test_local_snapshot_checksums_match() -> None:
    """The downloaded weights are exactly the recorded pinned snapshot."""
    recorded = {}
    for line in (LOCAL_SNAPSHOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    assert recorded, "SHA256SUMS is empty"
    for name, digest in recorded.items():
        assert hashlib.sha256((LOCAL_SNAPSHOT / name).read_bytes()).hexdigest() == digest, name
