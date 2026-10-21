"""Fetch the NanoBench CSVs into ``data/nanobench``.

The files stay gitignored. This checks out only ``datasets/dataset`` from
https://github.com/syediu/nanobench.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLONE = REPO_ROOT / "data" / "nanobench"
DATASET = CLONE / "datasets" / "dataset"
REMOTE = "https://github.com/syediu/nanobench.git"


def fetch() -> Path:
    if DATASET.is_dir() and any(DATASET.glob("*.csv")):
        count = len(list(DATASET.glob("*.csv")))
        print(f"NanoBench already present: {count} csv files in {DATASET}")
        return DATASET
    CLONE.parent.mkdir(parents=True, exist_ok=True)
    if not CLONE.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                REMOTE,
                str(CLONE),
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(CLONE), "sparse-checkout", "set", "datasets/dataset"],
        check=True,
    )
    csvs = list(DATASET.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"sparse checkout finished but {DATASET} has no csv files")
    print(f"fetched {len(csvs)} NanoBench csv files into {DATASET}")
    return DATASET


def main() -> None:
    fetch()


if __name__ == "__main__":
    main()
