"""Controlled multi-training-seed ablation of residual and delay-domain priors."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluate import aggregate_seed_results, evaluate_cnn_only_across_seeds, load_cnn_model
from src.plots import plot_ablation
from src.train import train_cnn_channel_estimator
from src.utils import ensure_dir


METRICS = ["ber", "ser", "evm_rms", "evm_db", "channel_mse", "channel_nmse", "pilot_nmse", "data_nmse", "deep_fade_nmse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a matched-budget multi-seed delay-prior ablation.")
    parser.add_argument("--config", default="configs/multiseed_config.json")
    parser.add_argument("--results-dir", default="results/final/ablation")
    parser.add_argument("--checkpoint-dir", default="checkpoints/ablation")
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def _load_or_train(cfg, config_path: str, checkpoint: Path, result_dir: Path, device: str, variant: str, skip_train: bool):
    if checkpoint.exists():
        try:
            return load_cnn_model(checkpoint, device=device, expected_cfg=cfg)
        except (KeyError, RuntimeError):
            if skip_train:
                raise
    if skip_train:
        raise FileNotFoundError(f"Missing compatible checkpoint: {checkpoint}")
    train_cnn_channel_estimator(
        config_path=config_path,
        config=cfg,
        checkpoint_path=checkpoint,
        results_dir=result_dir,
        device=device,
        model_variant=variant,
    )
    return load_cnn_model(checkpoint, device=device, expected_cfg=cfg)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    eval_cfg = replace(
        cfg,
        snr_db_values=[int(args.snr_db)],
        evaluation_seeds=list(cfg.multiseed_test_seeds),
        num_eval_frames=int(cfg.multiseed_num_eval_frames),
    )
    results_dir = ensure_dir(args.results_dir)
    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    variants = {
        "plain": "Plain CNN",
        "residual": "Residual CNN",
        "residual_hard_delay": "Residual CNN + hard delay projection",
        "residual_soft_delay": "Residual CNN + soft delay regularization",
    }
    raw_rows: list[pd.DataFrame] = []
    test_summaries: list[pd.DataFrame] = []
    for variant, label in variants.items():
        for training_seed in cfg.multiseed_training_seeds:
            train_cfg = replace(cfg, random_seed=int(training_seed))
            checkpoint = checkpoint_dir / variant / f"seed_{training_seed}" / f"{variant}.pt"
            model, device, _ = _load_or_train(
                train_cfg,
                args.config,
                checkpoint,
                results_dir / "training" / variant / f"seed_{training_seed}",
                args.device,
                variant,
                args.skip_train,
            )
            raw, _, _ = evaluate_cnn_only_across_seeds(
                eval_cfg,
                num_frames=eval_cfg.num_eval_frames,
                seeds=eval_cfg.evaluation_seeds,
                cnn_model=model,
                cnn_device=device,
            )
            raw = raw.rename(columns={"seed": "test_seed"})
            raw.insert(0, "variant", label)
            raw.insert(0, "training_seed", int(training_seed))
            raw_rows.append(raw)
            summary = aggregate_seed_results(raw.rename(columns={"test_seed": "seed"}))
            summary.insert(0, "variant", label)
            summary.insert(0, "training_seed", int(training_seed))
            test_summaries.append(summary)

    raw_result = pd.concat(raw_rows, ignore_index=True)
    test_summary = pd.concat(test_summaries, ignore_index=True)
    model_rows: list[dict[str, float | int | str]] = []
    for (variant, snr_db), group in test_summary.groupby(["variant", "snr_db"], sort=False):
        n = int(group["training_seed"].nunique())
        critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
        row: dict[str, float | int | str] = {"variant": variant, "snr_db": float(snr_db), "num_training_seeds": n}
        for metric in METRICS:
            values = group[f"{metric}_mean"].to_numpy(dtype=float)
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = float(critical * std / np.sqrt(n))
        model_rows.append(row)
    model_summary = pd.DataFrame(model_rows)
    raw_result.to_csv(results_dir / "ablation_training_seed_test_seed_results.csv", index=False)
    test_summary.to_csv(results_dir / "ablation_test_monte_carlo_summary.csv", index=False)
    model_summary.to_csv(results_dir / "ablation_model_seed_summary.csv", index=False)
    plot_ablation(model_summary, results_dir / "ablation_ber_nmse.png")
    (results_dir / "ablation_manifest.json").write_text(
        json.dumps(
            {
                "training_seeds": cfg.multiseed_training_seeds,
                "test_seeds": cfg.multiseed_test_seeds,
                "same_training_budget": {
                    "epochs": cfg.training_epochs,
                    "samples_per_epoch": cfg.training_samples_per_epoch,
                    "validation_samples": cfg.validation_samples,
                    "optimizer": "Adam",
                    "learning_rate": cfg.training_learning_rate,
                },
                "ci_method": "two-sided Student-t across independent training seeds",
                "variants": variants,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(model_summary.to_string(index=False))
    print(f"Saved ablation artifacts to: {results_dir}")


if __name__ == "__main__":
    main()
