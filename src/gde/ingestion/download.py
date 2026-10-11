"""README-driven dataset acquisition (PLAN.md §2.1, P1).

Each dataset's README is fetched first; download URLs, upstream checksums and the expected file
layout are parsed from it. Nothing beyond the repository entry point in configs/datasets.yaml is
hard-coded. Failures are logged and the run continues with the next dataset.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

USER_AGENT = "gde-download/0.1 (+research; github.com)"
LFS_POINTER_PREFIX = b"version https://git-lfs"
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    pass


@dataclass
class FileRecord:
    name: str
    url: str
    path: str
    bytes: int
    sha256: str
    upstream_sha256: str | None = None
    kind: str = "download"  # download | extracted | metadata

    @property
    def check(self) -> str:
        if self.upstream_sha256 is None:
            return "no upstream checksum"
        return "OK" if self.upstream_sha256 == self.sha256 else "MISMATCH"


@dataclass
class DatasetResult:
    name: str
    status: str  # downloaded | failed | skipped | awaiting_confirmation
    readme_url: str | None = None
    access_path: str = ""
    retrieved_utc: str = ""
    files: list[FileRecord] = field(default_factory=list)
    license_name: str | None = None
    license_text: str | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------- README parsing

_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
_SHA_LINE_RE = re.compile(r"^\s*([0-9a-f]{64})\s+\*?(\S+)\s*$", re.MULTILINE)


def parse_markdown_links(text: str) -> list[tuple[str, str]]:
    """All [label](url) links, in order, labels stripped of markup."""
    return [(re.sub(r"[*`]", "", label).strip(), url) for label, url in _LINK_RE.findall(text)]


def parse_sha256_lines(text: str) -> dict[str, str]:
    """`<sha256>  <filename>` lines (sha256sum format) anywhere in the text."""
    return {name: digest for digest, name in _SHA_LINE_RE.findall(text)}


def parse_tree_block(text: str) -> list[str]:
    """Relative file paths from an emoji directory tree (📦root / 📂dir / 📜file)."""
    paths: list[str] = []
    root: str | None = None
    current_dir: str | None = None
    for line in text.splitlines():
        if "📦" in line:
            root, current_dir = line.split("📦", 1)[1].strip(), None
        elif "📂" in line:
            current_dir = line.split("📂", 1)[1].strip()
        elif "📜" in line:
            name = line.split("📜", 1)[1].strip()
            paths.append("/".join(p for p in (root, current_dir, name) if p))
    return paths


def detect_license_name(text: str) -> str:
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if "Attribution 4.0 International" in text[:500]:
        return "Creative Commons Attribution 4.0 International (CC BY 4.0)"
    return first[:120] or "unknown"


# ---------------------------------------------------------------- HTTP helpers

def _request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT}
    if "api.github.com" in url:
        headers["Accept"] = "application/vnd.github+json"
    return urllib.request.Request(url, headers=headers)


def http_text(url: str, retries: int = 3, timeout: int = 60) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001 - network errors of many types; retried then reported
            last = exc
            time.sleep(2 * (attempt + 1))
    raise DownloadError(f"GET {url} failed: {last}")


def http_download(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> tuple[int, str]:
    """Stream url to dest; return (bytes, sha256). Rejects Git LFS pointer files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(_request(url), timeout=timeout) as resp, tmp.open("wb") as fh:
                while chunk := resp.read(CHUNK):
                    if size == 0 and chunk.startswith(LFS_POINTER_PREFIX):
                        raise DownloadError(
                            f"{url} is a Git LFS pointer, not the data file; it must be fetched through Git LFS")
                    digest.update(chunk)
                    fh.write(chunk)
                    size += len(chunk)
            tmp.replace(dest)
            return size, digest.hexdigest()
        except DownloadError:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            tmp.unlink(missing_ok=True)
            time.sleep(3 * (attempt + 1))
    raise DownloadError(f"download {url} failed after {retries} attempts: {last}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_tar(archive: Path, out: Path) -> list[Path]:
    """Extract regular files only, refusing paths that escape `out` (tarfile filter needs 3.11.4+)."""
    root = out.resolve()
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for m in members:
            if root not in (root / m.name).resolve().parents:
                raise DownloadError(f"unsafe path in {archive.name}: {m.name}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(out, members=members, filter="data")
        else:
            tar.extractall(out, members=members)
    return [out / m.name for m in members]


def github_contents(repo: str, path: str, ref: str) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{repo}/contents/{path}".rstrip("/") + f"?ref={ref}"
    data = json.loads(http_text(url))
    if not isinstance(data, list):
        raise DownloadError(f"unexpected GitHub contents response for {repo}/{path}: {str(data)[:200]}")
    return data


def raw_base(gh: dict[str, str]) -> str:
    path = gh.get("path", "").strip("/")
    return f"https://raw.githubusercontent.com/{gh['repo']}/{gh['ref']}/" + (f"{path}/" if path else "")


# ---------------------------------------------------------------- download methods

class Fetcher:
    def __init__(self, name: str, cfg: dict[str, Any], out_dir: Path, force: bool = False) -> None:
        self.name, self.cfg, self.out, self.force = name, cfg, out_dir, force
        self.gh = cfg["github"]
        self.base = raw_base(self.gh)
        self.result = DatasetResult(name=name, status="failed", readme_url=self.base + "README.md")

    def fetch(self, url: str, dest: Path, upstream_sha: str | None = None) -> FileRecord:
        if dest.exists() and not self.force and (upstream_sha is None or sha256_file(dest) == upstream_sha):
            size, digest = dest.stat().st_size, sha256_file(dest)
        else:
            size, digest = http_download(url, dest)
        rec = FileRecord(dest.name, url, dest.relative_to(self.out).as_posix(), size, digest, upstream_sha)
        if rec.check == "MISMATCH":
            raise DownloadError(f"{dest.name}: sha256 {digest} != README {upstream_sha}")
        self.result.files.append(rec)
        return rec

    def record_extracted(self, path: Path, upstream_sha: str | None = None) -> FileRecord:
        rec = FileRecord(path.name, "(extracted)", path.relative_to(self.out).as_posix(),
                         path.stat().st_size, sha256_file(path), upstream_sha, kind="extracted")
        if rec.check == "MISMATCH":
            raise DownloadError(f"{path.name}: extracted sha256 {rec.sha256} != README {upstream_sha}")
        self.result.files.append(rec)
        return rec

    def run(self, tables: list[str] | None = None) -> DatasetResult:
        self.out.mkdir(parents=True, exist_ok=True)
        readme = http_text(self.result.readme_url)
        (self.out / "README.upstream.md").write_text(readme, encoding="utf-8")
        self.result.retrieved_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self._license()
        method = getattr(self, f"_{self.cfg['method']}")
        method(readme, tables)
        self.result.status = "downloaded"
        return self.result

    def _license(self) -> None:
        lic = self.cfg.get("license_file")
        if not lic:
            listing = github_contents(self.gh["repo"], self.gh.get("path", ""), self.gh["ref"])
            names = [e["name"] for e in listing if e["type"] == "file"]
            self.result.license_name = "none found"
            self.result.notes.append(
                f"No license file in the dataset folder (files: {', '.join(names)}) and the repository "
                "declares no license on GitHub. Usage terms are therefore unspecified; treat as "
                "research use with citation, and re-check before redistribution.")
            return
        text = http_text(self.base + lic)
        (self.out / f"LICENSE.upstream{Path(lic).suffix or '.txt'}").write_text(text, encoding="utf-8")
        self.result.license_name, self.result.license_text = detect_license_name(text), text

    # README links absolute .tar.gz archives + sha256sum blocks for archives, headers and CSVs.
    def _readme_tarballs(self, readme: str, tables: list[str] | None) -> None:
        wanted = tables or self.cfg["tables"]
        shas = parse_sha256_lines(readme)
        links = {Path(urllib.parse.urlparse(u).path).name: u for _, u in parse_markdown_links(readme)
                 if u.startswith("http") and u.endswith(".tar.gz")}
        available = sorted(n.removesuffix(".tar.gz") for n in links)
        missing = [t for t in wanted if f"{t}.tar.gz" not in links]
        if missing:
            raise DownloadError(f"tables {missing} not linked in README (README links: {available})")
        alt = [u for label, u in parse_markdown_links(readme) if "github.com" in u and "data" in label.lower()]
        self.result.access_path = (
            "README section 2 links each table as a .tar.gz on Aliyun OSS; README gives sha256 for "
            "archives, .header files and extracted CSVs. Header files are in the repo's "
            f"`{self.cfg['header_dir']}/` folder." + (f" README alternative: {alt[0]} (not used)." if alt else ""))
        skipped = [t for t in available if t not in wanted]
        if skipped:
            self.result.notes.append(f"README tables not downloaded (opt-in via --tables): {', '.join(skipped)}.")

        header_dir = f"{self.gh['path']}/{self.cfg['header_dir']}".strip("/")
        header_urls = {e["name"]: e["download_url"] for e in github_contents(self.gh["repo"], header_dir, self.gh["ref"])
                       if e["name"].endswith(".header")}
        for table in wanted:
            archive = f"{table}.tar.gz"
            self.fetch(links[archive], self.out / "archives" / archive, shas.get(archive))
            hdr = f"{table}.header"
            if hdr not in header_urls:
                raise DownloadError(f"{hdr} not found in {header_dir}")
            self.fetch(header_urls[hdr], self.out / hdr, shas.get(hdr))
            for p in safe_extract_tar(self.out / "archives" / archive, self.out):
                self.record_extracted(p, shas.get(p.name))

    # README links data files with relative paths (e.g. ./csv/x.csv); no upstream checksums.
    def _readme_relative_links(self, readme: str, tables: list[str] | None) -> None:
        pattern = re.compile(self.cfg["link_pattern"])
        rel = list(dict.fromkeys(u for _, u in parse_markdown_links(readme) if pattern.match(u)))
        if not rel:
            raise DownloadError(f"no README links match {pattern.pattern}")
        self.result.access_path = (
            f"README links {len(rel)} CSV files relative to the dataset folder; fetched from the "
            "GitHub raw URL of that folder. README provides no checksums.")
        for link in rel:
            relpath = link.removeprefix("./")
            self.fetch(self.base + relpath, self.out / relpath)

    # Data shipped as one archive in the repo; README documents the extracted layout.
    def _github_repo_archive(self, readme: str, tables: list[str] | None) -> None:
        archive = self.cfg["archive"]
        listing = github_contents(self.gh["repo"], self.gh.get("path", ""), self.gh["ref"])
        entry = next((e for e in listing if e["name"] == archive), None)
        if entry is None:
            raise DownloadError(f"{archive} not in repo listing: {[e['name'] for e in listing]}")
        self.result.access_path = (
            f"README describes the data layout; the data is the `{archive}` file in the repository "
            f"(GitHub contents API download_url, {entry.get('size')} bytes listed).")
        rec = self.fetch(entry["download_url"], self.out / "archives" / archive)
        expected = parse_tree_block(readme)
        extracted = self._extract(self.out / rec.path)
        for p in extracted:
            self.record_extracted(p)
        if expected:
            rel = {p.relative_to(self.out).as_posix() for p in extracted}
            missing = [e for e in expected if e not in rel]
            if missing:
                raise DownloadError(f"README-described files missing after extraction: {missing}")
            self.result.notes.append(f"All {len(expected)} files in the README's dataset tree are present.")

    def _extract(self, archive: Path) -> list[Path]:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/") and "__MACOSX" not in n]
                for n in names:
                    target = (self.out / n).resolve()
                    if self.out.resolve() not in target.parents:
                        raise DownloadError(f"unsafe path in zip: {n}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(n) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                return [self.out / n for n in names]
        return safe_extract_tar(archive, self.out)


# ---------------------------------------------------------------- reporting

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return str(n)


def write_sources_md(res: DatasetResult, cfg: dict[str, Any], out_dir: Path) -> Path:
    lines = [
        f"# {res.name} — sources",
        "",
        f"- Status: **{res.status}**" + (f" — {res.error}" if res.error else ""),
        f"- README read first: {res.readme_url}" if res.readme_url else "- README: n/a",
        f"- Access path (from README): {res.access_path}" if res.access_path else "",
        f"- Retrieved (UTC): {res.retrieved_utc}" if res.retrieved_utc else "",
        f"- Citation: {cfg.get('citation', 'n/a')}",
        f"- License: {res.license_name or 'n/a'}",
        "- Generated by `python tasks.py data` (`scripts/download_data.py`).",
        "",
    ]
    for kind, title in (("download", "Downloaded files"), ("extracted", "Extracted files")):
        recs = [r for r in res.files if r.kind == kind]
        if not recs:
            continue
        lines += [f"## {title}", "", "| File | Size | Bytes | SHA-256 | Upstream checksum | URL |", "|---|---|---|---|---|---|"]
        lines += [f"| `{r.path}` | {_fmt_bytes(r.bytes)} | {r.bytes} | `{r.sha256}` | {r.check} | {r.url} |" for r in recs]
        lines.append("")
    if res.notes:
        lines += ["## Notes", "", *[f"- {n}" for n in res.notes], ""]
    if res.license_text:
        lines += ["## License text (verbatim from upstream)", "", "```text", res.license_text.rstrip(), "```", ""]
    path = out_dir / "SOURCES.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ln for ln in lines if ln is not None), encoding="utf-8")
    return path


def append_log(log_path: Path, res: DatasetResult) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = asdict(res) | {"logged_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    entry.pop("license_text", None)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
