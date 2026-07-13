from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_readme_pdf.py"), "--input", "report.md", "--output", "report.pdf"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
