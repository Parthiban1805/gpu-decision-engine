"""Repository-relative paths, so no module hard-codes absolute locations."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
DATA = REPO_ROOT / "data"
DOCS = REPO_ROOT / "docs"
REPORTS = REPO_ROOT / "reports"
