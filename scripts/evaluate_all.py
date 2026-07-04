from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluate import evaluate_receiver_methods, load_cnn_model
from src.plots import plot_ber, plot_channel_mse, plot_constellation, plot_metric, plot_relative_gain
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate traditional and CNN-based OFDM receivers.")
    parser.add_argument("--config", default="configs/default_config.json", help="Path to JSON config.")
    parser.add_argument("--checkpoint", default="checkpoints/cnn_channel_estimator.pt", help="CNN checkpoint path.")
    parser.add_argument("--num-test-frames", type=int, default=None, help="Frames per SNR point.")
    parser.add_argument("--device", default="auto", help="Device for CNN inference.")
    parser.add_argument("--seed", type=int, default=None, help="Evaluation random seed.")
    parser.add_argument("--results-dir", default="results", help="Output directory.")
    parser.add_argument("--constellation-snr-db", type=float, default=20.0, help="SNR used for constellation example.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Run scripts/train_cnn_receiver.py first."
        )

    results_dir = ensure_dir(args.results_dir)
    model, device, _ = load_cnn_model(checkpoint, device=args.device)
    num_frames = cfg.num_eval_frames if args.num_test_frames is None else int(args.num_test_frames)
    seed = cfg.random_seed + 40_000 if args.seed is None else int(args.seed)

    df, constellation = evaluate_receiver_methods(
        cfg,
        cfg.snr_db_values,
        num_frames=num_frames,
        seed=seed,
        cnn_model=model,
        cnn_device=device,
        constellation_snr_db=args.constellation_snr_db,
    )

    comparison_csv = results_dir / "ber_comparison.csv"
    df.to_csv(comparison_csv, index=False)
    plot_ber(df, results_dir / "ber_comparison.png", title="OFDM Receiver BER Comparison")
    plot_metric(
        df,
        "ser",
        results_dir / "ser_comparison.png",
        title="OFDM Receiver SER Comparison",
        ylabel="Symbol Error Rate",
        log_y=True,
    )
    plot_metric(
        df,
        "evm_rms",
        results_dir / "evm_vs_snr.png",
        title="Equalized Data EVM vs SNR",
        ylabel="RMS EVM",
        log_y=True,
    )
    plot_relative_gain(df, results_dir / "cnn_gain_vs_snr.png")

    mse_df = (
        df.assign(method=np.where(df["method"].str.startswith("CNN"), "CNN estimator", "LS interpolation"))
        .groupby(["snr_db", "method"], as_index=False)["channel_mse"]
        .mean()
    )
    mse_df.to_csv(results_dir / "channel_mse.csv", index=False)
    plot_channel_mse(mse_df, results_dir / "channel_mse_vs_snr.png")

    if constellation:
        plot_constellation(constellation, results_dir / "constellation_example.png")

    print(df.to_string(index=False))
    print(f"Saved CSV: {comparison_csv}")
    print(f"Saved BER plot: {results_dir / 'ber_comparison.png'}")
    print(f"Saved SER plot: {results_dir / 'ser_comparison.png'}")
    print(f"Saved EVM plot: {results_dir / 'evm_vs_snr.png'}")
    print(f"Saved CNN gain plot: {results_dir / 'cnn_gain_vs_snr.png'}")
    print(f"Saved channel MSE plot: {results_dir / 'channel_mse_vs_snr.png'}")
    if constellation:
        print(f"Saved constellation plot: {results_dir / 'constellation_example.png'}")


if __name__ == "__main__":
    main()
