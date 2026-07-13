"""Train independent CNN seeds and compare them with practical cached LMMSE."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import benchmark_cached_estimators
from src.config import OFDMConfig, load_config
from src.evaluate import (
    aggregate_seed_results,
    build_lmmse_priors,
    evaluate_across_seeds,
    evaluate_cnn_only_across_seeds,
    load_cnn_model,
)
from src.ofdm import generate_pilot_mask, pilot_pattern_statistics
from src.plots import plot_ber, plot_channel_nmse, plot_complexity_benchmark, plot_pilot_pattern, plot_training_seed_variability
from src.train import train_cnn_channel_estimator
from src.utils import ensure_dir


METRICS = ["ber", "ser", "evm_rms", "evm_db", "channel_mse", "channel_nmse", "pilot_nmse", "data_nmse", "deep_fade_nmse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run independent training-seed, practical-LMMSE, and domain-randomization experiments.")
    parser.add_argument("--config", default="configs/multiseed_config.json")
    parser.add_argument("--domain-config", default="configs/domain_randomized_config.json")
    parser.add_argument("--mismatch-config", default="configs/mismatch_config.json")
    parser.add_argument("--unseen-config", default="configs/unseen_generalization_config.json")
    parser.add_argument("--results-dir", default="results/final/multiseed")
    parser.add_argument("--checkpoint-dir", default="checkpoints/multiseed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--single-regime-only", action="store_true", help="Skip domain-randomized training for a faster diagnostic run.")
    return parser.parse_args()


def _train_or_load(cfg: OFDMConfig, config_path: str, checkpoint: Path, result_dir: Path, device: str, skip_train: bool):
    """Load a fingerprint-compatible checkpoint or train one independent seed."""

    if checkpoint.exists():
        try:
            start = perf_counter()
            model, device_obj, _ = load_cnn_model(checkpoint, device=device, expected_cfg=cfg)
            return model, device_obj, 1e3 * (perf_counter() - start)
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
        model_variant="residual_hard_delay",
    )
    start = perf_counter()
    model, device_obj, _ = load_cnn_model(checkpoint, device=device, expected_cfg=cfg)
    return model, device_obj, 1e3 * (perf_counter() - start)


def _test_seed_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fixed test streams for each trained model using Student-t CI."""

    tables: list[pd.DataFrame] = []
    for (regime, training_seed, scenario), group in raw.groupby(["training_regime", "training_seed", "scenario"], sort=False):
        summary = aggregate_seed_results(group.rename(columns={"test_seed": "seed"}))
        summary.insert(0, "scenario", scenario)
        summary.insert(0, "training_seed", int(training_seed))
        summary.insert(0, "training_regime", regime)
        tables.append(summary)
    return pd.concat(tables, ignore_index=True)


def _model_seed_summary(test_summary: pd.DataFrame) -> pd.DataFrame:
    """Separate model/training uncertainty from test Monte Carlo uncertainty."""

    source = test_summary[test_summary["method"] == "ResidualCNN+ZF"].copy()
    rows: list[dict[str, float | int | str]] = []
    for (regime, scenario, snr_db), group in source.groupby(["training_regime", "scenario", "snr_db"], sort=False):
        n = int(group["training_seed"].nunique())
        critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
        row: dict[str, float | int | str] = {
            "training_regime": regime,
            "scenario": scenario,
            "snr_db": float(snr_db),
            "method": f"ResidualCNN-{regime}+ZF",
            "estimator": f"ResidualCNN-{regime}",
            "equalizer": "ZF",
            "num_training_seeds": n,
            "test_seed_count": int(group["num_seeds"].iloc[0]),
            "frames_per_test_seed": int(group["frames_per_seed"].iloc[0]),
        }
        for metric in METRICS:
            values = group[f"{metric}_mean"].to_numpy(dtype=float)
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = float(critical * std / np.sqrt(n))
        rows.append(row)
    return pd.DataFrame(rows)


def _practical_comparison(test_summary: pd.DataFrame, model_summary: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Combine a single fixed-baseline summary with training-seed CNN uncertainty."""

    sample_methods = sorted(
        (method for method in test_summary["method"].unique() if method.startswith("LMMSE-sample-") and method.endswith("+ZF")),
        key=lambda method: int(method.removeprefix("LMMSE-sample-").removesuffix("+ZF")),
    )
    if not sample_methods:
        raise KeyError("No sample-covariance LMMSE method found in test summary.")
    baseline = test_summary[
        (test_summary["scenario"] == scenario)
        & (test_summary["training_regime"] == "baseline")
        & (test_summary["training_seed"] == 0)
        & (test_summary["method"].isin(["Frame-LS-linear+ZF", "DFT-LS+ZF", "LMMSE-oracle+ZF", "LMMSE-train-prior+ZF", sample_methods[-1]]))
    ].copy()
    cnn = model_summary[model_summary["scenario"] == scenario].copy()
    baseline = baseline.drop(columns=["training_regime", "training_seed", "scenario"], errors="ignore")
    return pd.concat([baseline, cnn], ignore_index=True, sort=False)


def _prior_metadata(priors: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": prior.name,
                "source": prior.source,
                "sample_count": prior.sample_count,
                "covariance_build_time_ms": prior.covariance_build_time_ms,
                "diagonal_loading": prior.diagonal_loading,
                "minimum_eigenvalue": prior.minimum_eigenvalue,
                "prior_storage_bytes": prior.storage_bytes,
            }
            for prior in priors.values()
        ]
    )


def main() -> None:
    args = parse_args()
    static_cfg = load_config(args.config)
    domain_cfg = load_config(args.domain_config)
    scenario_cfgs: list[tuple[str, OFDMConfig, bool]] = [
        ("matched_12tap_exponential", static_cfg, False),
        ("mismatch_8tap_uniform", load_config(args.mismatch_config), True),
        ("unseen_10tap_steep_exponential", load_config(args.unseen_config), True),
    ]
    results_dir = ensure_dir(args.results_dir)
    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    regimes = [("single", static_cfg, args.config)]
    if not args.single_regime_only:
        regimes.append(("domain_randomized", domain_cfg, args.domain_config))

    raw_tables: list[pd.DataFrame] = []
    scenario_priors: dict[str, dict] = {}
    evaluation_configs: dict[str, OFDMConfig] = {}
    for scenario, scenario_cfg, include_train_prior in scenario_cfgs:
        priors = build_lmmse_priors(
            scenario_cfg,
            include_assumed_lmmse=include_train_prior,
            sample_covariance_sizes=static_cfg.sample_covariance_sizes,
            sample_prior_cfg=static_cfg,
        )
        scenario_priors[scenario] = priors
        eval_cfg = replace(
            scenario_cfg,
            snr_db_values=list(static_cfg.multiseed_snr_db_values),
            evaluation_seeds=list(static_cfg.multiseed_test_seeds),
            num_eval_frames=int(static_cfg.multiseed_num_eval_frames),
        )
        evaluation_configs[scenario] = eval_cfg
        baseline_raw, _, _ = evaluate_across_seeds(
            eval_cfg,
            num_frames=eval_cfg.num_eval_frames,
            seeds=eval_cfg.evaluation_seeds,
            include_assumed_lmmse=include_train_prior,
            lmmse_priors=priors,
        )
        baseline_raw = baseline_raw.rename(columns={"seed": "test_seed"})
        baseline_raw.insert(0, "scenario", scenario)
        baseline_raw.insert(0, "training_seed", 0)
        baseline_raw.insert(0, "training_regime", "baseline")
        raw_tables.append(baseline_raw)
    first_model = None
    first_load_ms = 0.0
    for regime, base_train_cfg, config_path in regimes:
        for training_seed in base_train_cfg.multiseed_training_seeds:
            train_cfg = replace(base_train_cfg, random_seed=int(training_seed))
            checkpoint = checkpoint_dir / regime / f"seed_{training_seed}" / "residual_hard_delay.pt"
            model, device, load_ms = _train_or_load(
                train_cfg,
                config_path,
                checkpoint,
                results_dir / "training" / regime / f"seed_{training_seed}",
                args.device,
                args.skip_train,
            )
            if first_model is None:
                first_model, first_load_ms, first_checkpoint = model, load_ms, checkpoint
            for scenario, _, _ in scenario_cfgs:
                eval_cfg = evaluation_configs[scenario]
                raw, _, _ = evaluate_cnn_only_across_seeds(
                    eval_cfg,
                    num_frames=eval_cfg.num_eval_frames,
                    seeds=eval_cfg.evaluation_seeds,
                    cnn_model=model,
                    cnn_device=device,
                )
                raw = raw.rename(columns={"seed": "test_seed"})
                raw.insert(0, "scenario", scenario)
                raw.insert(0, "training_seed", int(training_seed))
                raw.insert(0, "training_regime", regime)
                raw_tables.append(raw)

    raw_results = pd.concat(raw_tables, ignore_index=True)
    test_summary = _test_seed_summary(raw_results)
    model_summary = _model_seed_summary(test_summary)
    raw_results.to_csv(results_dir / "training_seed_results.csv", index=False)
    test_summary.to_csv(results_dir / "test_monte_carlo_summary.csv", index=False)
    model_summary.to_csv(results_dir / "model_seed_summary.csv", index=False)

    matched_priors = scenario_priors["matched_12tap_exponential"]
    _prior_metadata(matched_priors).to_csv(results_dir / "practical_lmmse_prior_metadata.csv", index=False)
    sample_prior_names = sorted(
        (name for name in matched_priors if name.startswith("LMMSE-sample-")),
        key=lambda name: int(name.removeprefix("LMMSE-sample-")),
    )
    benchmark_prior_names = {"LMMSE-oracle"}
    if sample_prior_names:
        benchmark_prior_names.add(sample_prior_names[-1])
    complexity = benchmark_cached_estimators(
        static_cfg,
        first_model,
        device,
        {name: prior for name, prior in matched_priors.items() if name in benchmark_prior_names},
        checkpoint_path=first_checkpoint,
        checkpoint_load_ms=first_load_ms,
    )
    complexity.to_csv(results_dir / "complexity_benchmark.csv", index=False)
    plot_complexity_benchmark(complexity, results_dir / "complexity_benchmark.png")

    pilot_stats = pd.DataFrame([pilot_pattern_statistics(static_cfg)])
    pilot_stats.to_csv(results_dir / "pilot_pattern_statistics.csv", index=False)
    plot_pilot_pattern(generate_pilot_mask(static_cfg), results_dir / "pilot_pattern.png", "Staggered pilots: frame-level observation fusion")

    for scenario, _, _ in scenario_cfgs:
        comparison = _practical_comparison(test_summary, model_summary, scenario)
        comparison.to_csv(results_dir / f"practical_comparison_{scenario}.csv", index=False)
        methods = [
            "Frame-LS-linear+ZF",
            "DFT-LS+ZF",
            "LMMSE-oracle+ZF",
            "LMMSE-train-prior+ZF",
            *[f"{name}+ZF" for name in sample_prior_names[-1:]],
            "ResidualCNN-single+ZF",
            "ResidualCNN-domain_randomized+ZF",
        ]
        plot_ber(comparison, results_dir / f"practical_ber_{scenario}.png", f"{scenario}: BER (Student-t 95% CI)", methods=methods)
        estimators = ["Frame-LS-linear", "DFT-LS", "LMMSE-oracle", "LMMSE-train-prior", *sample_prior_names[-1:], "ResidualCNN-single", "ResidualCNN-domain_randomized"]
        plot_channel_nmse(comparison, results_dir / f"practical_nmse_{scenario}.png", f"{scenario}: channel NMSE (Student-t 95% CI)", estimators=estimators)

    plot_training_seed_variability(
        model_summary[model_summary["scenario"] == "matched_12tap_exponential"],
        results_dir / "training_seed_variability_matched.png",
    )
    run_manifest = {
        "training_seeds": static_cfg.multiseed_training_seeds,
        "test_seeds": static_cfg.multiseed_test_seeds,
        "test_snr_db_values": static_cfg.multiseed_snr_db_values,
        "frames_per_test_seed": static_cfg.multiseed_num_eval_frames,
        "training_regimes": [name for name, _, _ in regimes],
        "scenarios": [name for name, _, _ in scenario_cfgs],
        "ci_method": "two-sided Student-t across independent seeds",
        "sample_covariance_sizes": static_cfg.sample_covariance_sizes,
    }
    (results_dir / "multiseed_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(model_summary.to_string(index=False))
    print(f"Saved multiseed artifacts to: {results_dir}")


if __name__ == "__main__":
    main()
