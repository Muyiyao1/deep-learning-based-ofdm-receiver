from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluate import evaluate_across_seeds, load_cnn_model
from src.experiment import save_evaluation_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all OFDM estimators on identical Monte Carlo streams.")
    parser.add_argument("--config", default="configs/final_experiment.json")
    parser.add_argument("--checkpoint", default="checkpoints/residual_cnn_stress.pt")
    parser.add_argument("--num-test-frames", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", default="results/final/matched")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model, device, _ = load_cnn_model(args.checkpoint, device=args.device, expected_cfg=cfg)
    frames = cfg.num_eval_frames if args.num_test_frames is None else args.num_test_frames
    raw, summary, constellation = evaluate_across_seeds(cfg, num_frames=frames, cnn_model=model, cnn_device=device)
    save_evaluation_artifacts(raw, summary, constellation, args.results_dir, "Fair sparse-pilot stress test", cnn_model=model)
    print(summary.to_string(index=False))
    print(f"Saved evaluation artifacts to: {args.results_dir}")


if __name__ == "__main__":
    main()
