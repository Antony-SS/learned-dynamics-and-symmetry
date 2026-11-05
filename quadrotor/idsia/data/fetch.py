"""Fetch the IDSIA nano-drone benchmark and its Git LFS data."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from quadrotor.idsia.data.load import DEFAULT_ROOT

REMOTE = "https://github.com/idsia-robotics/nanodrone-sysid-benchmark.git"


def fetch() -> Path:
    expected = DEFAULT_ROOT / "data" / "train" / "chirp_20251017_run1.csv"
    if expected.is_file() and expected.stat().st_size > 100_000:
        print(f"IDSIA benchmark already present under {DEFAULT_ROOT}")
        return DEFAULT_ROOT
    if shutil.which("git-lfs") is None:
        raise RuntimeError(
            "git-lfs is required. On macOS run `brew install git-lfs && git lfs install`."
        )
    DEFAULT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_ROOT.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REMOTE, str(DEFAULT_ROOT)],
            check=True,
        )
    subprocess.run(["git", "-C", str(DEFAULT_ROOT), "lfs", "pull"], check=True)
    if not expected.is_file() or expected.stat().st_size <= 100_000:
        raise FileNotFoundError("Git LFS did not materialize the IDSIA CSV files")
    print(f"fetched IDSIA benchmark under {DEFAULT_ROOT}")
    return DEFAULT_ROOT


if __name__ == "__main__":
    fetch()
