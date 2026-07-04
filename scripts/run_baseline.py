from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_baselines
from src.plots import plot_ber
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate traditional OFDM receiver baselines.")
    parser.add_argument("--config", default="configs/default_config.json", help="Path to JSON config.")
    parser.add_argument("--num-test-frames", type=int, default=None, help="Frames per SNR point.")
    parser.add_argument("--seed", type=int, default=None, help="Evaluation random seed.")
    parser.add_argument("--results-dir", default="results", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    df = evaluate_baselines(args.config, num_frames=args.num_test_frames, seed=args.seed)
    csv_path = results_dir / "baseline_ber.csv"
    png_path = results_dir / "baseline_ber_vs_snr.png"
    df.to_csv(csv_path, index=False)
    plot_ber(df, png_path, title="Traditional OFDM Receiver Baselines")
    print(df.to_string(index=False))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
