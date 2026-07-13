"""Backward-compatible entry point for the renamed fair stress-test experiment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("'challenging' is retained as a legacy name. Running the fair sparse-pilot stress test instead.")
    command = [sys.executable, str(ROOT / "scripts" / "run_full_experiment.py"), "--config", "configs/final_experiment.json"]
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    main()
