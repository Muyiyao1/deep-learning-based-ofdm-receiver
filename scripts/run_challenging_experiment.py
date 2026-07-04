from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluate import evaluate_receiver_methods, load_cnn_model
from src.plots import plot_ber, plot_channel_mse, plot_constellation, plot_metric, plot_relative_gain
from src.train import train_cnn_channel_estimator
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the harder 16QAM sparse-pilot OFDM experiment."
    )
    parser.add_argument("--config", default="configs/challenging_config.json", help="Challenging config path.")
    parser.add_argument("--checkpoint", default="checkpoints/cnn_channel_estimator_challenging.pt")
    parser.add_argument("--results-dir", default="results/challenging")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-train-samples", type=int, default=3000)
    parser.add_argument("--num-val-samples", type=int, default=600)
    parser.add_argument("--num-test-frames", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if checkpoint exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    results_dir = ensure_dir(args.results_dir)

    if args.force_train or not checkpoint.exists():
        train_cnn_channel_estimator(
            config_path=args.config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            num_train_samples=args.num_train_samples,
            num_val_samples=args.num_val_samples,
            checkpoint_path=checkpoint,
            results_dir=results_dir,
        )

    cfg = load_config(args.config)
    model, device, _ = load_cnn_model(checkpoint, device=args.device)
    df, constellation = evaluate_receiver_methods(
        cfg,
        cfg.snr_db_values,
        num_frames=args.num_test_frames,
        seed=cfg.random_seed + 50_000,
        cnn_model=model,
        cnn_device=device,
        constellation_snr_db=20.0,
    )

    df.to_csv(results_dir / "ber_comparison.csv", index=False)
    plot_ber(df, results_dir / "ber_comparison.png", title="Challenging 16QAM OFDM BER Comparison")
    plot_metric(
        df,
        "ser",
        results_dir / "ser_comparison.png",
        title="Challenging 16QAM OFDM SER Comparison",
        ylabel="Symbol Error Rate",
        log_y=True,
    )
    plot_metric(
        df,
        "evm_rms",
        results_dir / "evm_vs_snr.png",
        title="Challenging 16QAM Equalized EVM vs SNR",
        ylabel="RMS EVM",
        log_y=True,
    )
    mse_df = (
        df.assign(method=df["method"].map(lambda x: "CNN estimator" if x.startswith("CNN") else "LS interpolation"))
        .groupby(["snr_db", "method"], as_index=False)["channel_mse"]
        .mean()
    )
    mse_df.to_csv(results_dir / "channel_mse.csv", index=False)
    plot_channel_mse(mse_df, results_dir / "channel_mse_vs_snr.png")
    plot_relative_gain(df, results_dir / "cnn_gain_vs_snr.png")
    if constellation:
        plot_constellation(constellation, results_dir / "constellation_example.png")

    print(df.to_string(index=False))
    print(f"Saved challenging results to: {results_dir}")


if __name__ == "__main__":
    main()
