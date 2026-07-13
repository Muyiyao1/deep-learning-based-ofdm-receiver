from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluate import evaluate_across_seeds
from src.experiment import save_evaluation_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fair conventional OFDM channel-estimation baselines.")
    parser.add_argument("--config", default="configs/default_config.json")
    parser.add_argument("--num-test-frames", type=int, default=None, help="Fixed frames per SNR and seed.")
    parser.add_argument("--results-dir", default="results/sanity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    frames = cfg.num_eval_frames if args.num_test_frames is None else args.num_test_frames
    raw, summary, constellation = evaluate_across_seeds(cfg, num_frames=frames)
    save_evaluation_artifacts(raw, summary, constellation, args.results_dir, "Sanity-check conventional baselines")
    # Compatibility filename now contains seed-aggregated results.
    summary.to_csv(Path(args.results_dir) / "baseline_ber.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved baseline artifacts to: {args.results_dir}")


if __name__ == "__main__":
    main()
