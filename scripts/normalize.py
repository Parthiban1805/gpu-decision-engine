"""CLI for P2 normalization: raw traces -> unified Parquet + cleaning report.

    python tasks.py normalize
    python scripts/normalize.py --datasets alibaba2023 helios_earth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gde.ingestion.normalize import run  # noqa: E402

ALL = ["alibaba2020", "alibaba2023", "helios_earth", "helios_saturn", "helios_uranus", "helios_venus"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=ALL, default=None, help="default: all present")
    args = ap.parse_args()
    results = run(datasets=args.datasets)
    skipped = [r.dataset for r in results if r.skipped]
    bad = [r.dataset for r in results if r.violations]
    if skipped:
        print(f"skipped (no raw data): {', '.join(skipped)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
