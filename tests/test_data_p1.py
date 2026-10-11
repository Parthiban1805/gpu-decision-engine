"""P1: README parsing / safe extraction (offline) and the data gate (skipped if data not downloaded)."""

from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from gde import paths
from gde.ingestion import download as dl

RAW = paths.DATA / "raw"


# ------------------------------------------------------------ offline unit tests

def test_parse_sha256_lines() -> None:
    text = "```bash\n$sha256sum -c << EOF\n" + "a" * 64 + "  x.tar.gz\n" + "b" * 64 + "  y.csv\nEOF\n```"
    assert dl.parse_sha256_lines(text) == {"x.tar.gz": "a" * 64, "y.csv": "b" * 64}


def test_parse_markdown_links_strips_markup() -> None:
    text = "- [**pai_job_table**](https://h/p/pai_job_table.tar.gz): x\n[a](./csv/b.csv)"
    assert dl.parse_markdown_links(text) == [("pai_job_table", "https://h/p/pai_job_table.tar.gz"), ("a", "./csv/b.csv")]


def test_parse_tree_block_includes_root() -> None:
    tree = "📦data\n ┣ 📂Earth\n ┃ ┣ 📜a.csv\n ┃ ┗ 📜b.csv\n ┗ 📂Venus\n ┃ ┗ 📜a.csv\n"
    assert dl.parse_tree_block(tree) == ["data/Earth/a.csv", "data/Earth/b.csv", "data/Venus/a.csv"]


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"hi"))
    with pytest.raises(dl.DownloadError, match="unsafe path"):
        dl.safe_extract_tar(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_lfs_pointer_is_rejected(tmp_path: Path) -> None:
    pointer = tmp_path / "trace.tar.gz"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(dl.DownloadError, match="Git LFS pointer"):
        dl.http_download(pointer.as_uri(), tmp_path / "out" / "trace.tar.gz")


def test_checksum_mismatch_is_flagged() -> None:
    rec = dl.FileRecord("f", "u", "f", 1, "a" * 64, upstream_sha256="b" * 64)
    assert rec.check == "MISMATCH"


# ------------------------------------------------------------ P1 gate (needs downloaded data)

needs_alibaba = pytest.mark.skipif(not (RAW / "alibaba2020" / "SOURCES.md").exists(),
                                   reason="alibaba2020 not downloaded (python tasks.py data)")


def _sources_rows(sources: Path) -> list[tuple[str, int, str, str]]:
    rows = []
    for line in sources.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| `([^`]+)` \| [^|]+ \| (\d+) \| `([0-9a-f]{64})` \| ([^|]+) \|", line)
        if m:
            rows.append((m.group(1), int(m.group(2)), m.group(3), m.group(4).strip()))
    return rows


@needs_alibaba
def test_gate_alibaba2020_loaded_with_readme_columns() -> None:
    d = RAW / "alibaba2020"
    assert "Status: **downloaded**" in (d / "SOURCES.md").read_text(encoding="utf-8")
    cols = (d / "pai_job_table.header").read_text(encoding="utf-8").strip().split(",")
    df = pd.read_csv(d / "pai_job_table.csv", header=None, names=cols, nrows=1000)
    assert list(df.columns) == ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
    assert len(df) == 1000 and df["end_time"].dropna().ge(0).all()


@pytest.mark.parametrize("name", ["alibaba2020", "alibaba2023", "helios"])
def test_gate_files_match_sources_checksums(name: str) -> None:
    sources = RAW / name / "SOURCES.md"
    if not sources.exists() or "Status: **downloaded**" not in sources.read_text(encoding="utf-8"):
        pytest.skip(f"{name} not downloaded")
    rows = _sources_rows(sources)
    assert rows, "SOURCES.md lists no files"
    for rel, size, sha, check in rows:
        path = RAW / name / rel
        assert path.stat().st_size == size, rel
        assert dl.sha256_file(path) == sha, rel
        assert check in {"OK", "no upstream checksum"}, (rel, check)
    assert "License:" in sources.read_text(encoding="utf-8")


@needs_alibaba
def test_gate_every_configured_dataset_has_a_logged_status() -> None:
    cfgs = yaml.safe_load((paths.CONFIGS / "datasets.yaml").read_text(encoding="utf-8"))["datasets"]
    status = (RAW / "DOWNLOADS.md").read_text(encoding="utf-8")
    for name in cfgs:
        assert re.search(rf"^\| {name} \| \w+ \|", status, re.MULTILINE), f"{name} missing from DOWNLOADS.md"


@needs_alibaba
def test_gate_large_dataset_not_downloaded_without_confirmation() -> None:
    philly = RAW / "philly"
    if (philly / "SOURCES.md").exists() and "downloaded**" in (philly / "SOURCES.md").read_text(encoding="utf-8"):
        pytest.skip("philly was downloaded after confirmation")
    assert not (philly / "archives").exists()
