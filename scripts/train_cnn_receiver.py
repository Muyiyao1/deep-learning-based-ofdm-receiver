from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import train_cnn_channel_estimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN OFDM channel estimator.")
    parser.add_argument("--config", default="configs/default_config.json", help="Path to JSON config.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--num-train-samples", type=int, default=5000, help="Training samples per epoch.")
    parser.add_argument("--num-val-samples", type=int, default=1000, help="Validation samples.")
    parser.add_argument("--checkpoint", default="checkpoints/cnn_channel_estimator.pt", help="Checkpoint path.")
    parser.add_argument("--results-dir", default="results", help="Output directory.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Keep 0 on Windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_cnn_channel_estimator(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
        num_workers=args.num_workers,
    )
    print(f"Best validation loss: {result['best_val_loss']:.6f}")
    print(f"Saved checkpoint: {result['checkpoint_path']}")
    print(f"Device: {result['device']}")


if __name__ == "__main__":
    main()
