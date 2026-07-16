"""Regenerate every CSV-derived figure without rerunning training or simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.experiment import preferred_methods
from src.ofdm import generate_pilot_mask
from src.plots import (
    plot_ablation,
    plot_ber,
    plot_channel_nmse,
    plot_complexity_benchmark,
    plot_pilot_pattern,
    plot_training_loss,
    plot_training_seed_variability,
)


SCENARIO_TITLES = {
    "matched_12tap_exponential": "Matched 12-tap exponential PDP",
    "mismatch_8tap_uniform": "Mismatched 8-tap uniform PDP",
    "unseen_10tap_steep_exponential": "Unseen 10-tap steep exponential PDP",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required result CSV is missing: {path}")
    return pd.read_csv(path)


def _regenerate_matched_or_mismatch(result_dir: Path, title: str, include_train_prior: bool) -> list[Path]:
    summary = _read_csv(result_dir / "summary_results.csv")
    methods = preferred_methods(summary, include_train_prior=include_train_prior)
    estimators = [method.rsplit("+", 1)[0] for method in methods if method.endswith("+ZF") and not method.startswith("PerfectCSI")]
    outputs = [result_dir / "ber_vs_snr.png", result_dir / "channel_nmse_vs_snr.png"]
    plot_ber(summary, outputs[0], f"{title}: BER (mean with 95% Student-t CI)", methods=methods)
    plot_channel_nmse(summary, outputs[1], f"{title}: channel NMSE (mean with 95% Student-t CI)", estimators=estimators)
    return outputs


def _sample_prior_methods(comparison: pd.DataFrame) -> list[str]:
    names = [method for method in comparison["method"].unique() if method.startswith("LMMSE-sample-") and method.endswith("+ZF")]
    return sorted(names, key=lambda method: int(method.removeprefix("LMMSE-sample-").removesuffix("+ZF")))[-1:]


def _regenerate_practical_comparisons(results_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for csv_path in sorted(results_dir.glob("practical_comparison_*.csv")):
        scenario = csv_path.stem.removeprefix("practical_comparison_")
        scenario_title = SCENARIO_TITLES.get(scenario, scenario.replace("_", " ").title())
        comparison = _read_csv(csv_path)
        sample_methods = _sample_prior_methods(comparison)
        methods = [
            "Frame-LS-linear+ZF",
            "DFT-LS+ZF",
            "LMMSE-oracle+ZF",
            "LMMSE-train-prior+ZF",
            *sample_methods,
            "ResidualCNN-single+ZF",
            "ResidualCNN-domain_randomized+ZF",
        ]
        estimators = [method.removesuffix("+ZF") for method in methods]
        ber_path = results_dir / f"practical_ber_{scenario}.png"
        nmse_path = results_dir / f"practical_nmse_{scenario}.png"
        plot_ber(comparison, ber_path, f"{scenario_title}: BER (95% Student-t CI)", methods=methods)
        plot_channel_nmse(
            comparison,
            nmse_path,
            f"{scenario_title}: channel NMSE (95% Student-t CI)",
            estimators=estimators,
        )
        outputs.extend([ber_path, nmse_path])
    return outputs


def _regenerate_training_curves(results_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for history_path in sorted(results_dir.glob("**/training_history.csv")):
        output_path = history_path.with_name("training_loss.png")
        plot_training_loss(_read_csv(history_path).to_dict("records"), output_path)
        outputs.append(output_path)
    return outputs


def regenerate(results_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    outputs.extend(_regenerate_matched_or_mismatch(results_dir / "matched", "Fair sparse-pilot stress test", include_train_prior=False))
    outputs.extend(_regenerate_matched_or_mismatch(results_dir / "mismatch", "PDP/tap-count generalization test", include_train_prior=True))

    ablation = _read_csv(results_dir / "ablation" / "ablation_model_seed_summary.csv")
    ablation_path = results_dir / "ablation" / "ablation_ber_nmse.png"
    plot_ablation(ablation, ablation_path)
    outputs.append(ablation_path)

    multiseed_dir = results_dir / "multiseed"
    complexity_path = multiseed_dir / "complexity_benchmark.png"
    plot_complexity_benchmark(_read_csv(multiseed_dir / "complexity_benchmark.csv"), complexity_path)
    outputs.append(complexity_path)

    pilot_path = multiseed_dir / "pilot_pattern.png"
    plot_pilot_pattern(
        generate_pilot_mask(load_config("configs/multiseed_config.json")),
        pilot_path,
        "Staggered pilots: frame-level observation fusion",
    )
    outputs.append(pilot_path)
    outputs.extend(_regenerate_practical_comparisons(multiseed_dir))

    model_summary = _read_csv(multiseed_dir / "model_seed_summary.csv")
    seed_variability_path = multiseed_dir / "training_seed_variability_matched.png"
    plot_training_seed_variability(
        model_summary[model_summary["scenario"] == "matched_12tap_exponential"],
        seed_variability_path,
    )
    outputs.append(seed_variability_path)
    outputs.extend(_regenerate_training_curves(results_dir))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate result figures from existing CSV files only.")
    parser.add_argument("--results-dir", default="results/final")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = regenerate(Path(args.results_dir))
    print(f"Regenerated {len(outputs)} CSV-derived figures from {args.results_dir}.")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
