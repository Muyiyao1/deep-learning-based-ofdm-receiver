from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluate import evaluate_across_seeds, load_cnn_model
from src.experiment import save_evaluation_artifacts
from src.train import train_cnn_channel_estimator
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command train, matched evaluation, mismatch evaluation, and reporting.")
    parser.add_argument("--config", default="configs/final_experiment.json")
    parser.add_argument("--mismatch-config", default="configs/mismatch_config.json")
    parser.add_argument("--multiseed-config", default="configs/multiseed_config.json")
    parser.add_argument("--domain-config", default="configs/domain_randomized_config.json")
    parser.add_argument("--unseen-config", default="configs/unseen_generalization_config.json")
    parser.add_argument("--checkpoint", default="checkpoints/residual_cnn_stress.pt")
    parser.add_argument("--ablation-checkpoint-dir", default=None, help="Optional checkpoint directory for delay-prior ablations.")
    parser.add_argument("--multiseed-checkpoint-dir", default=None, help="Optional checkpoint directory for multi-seed CNN experiments.")
    parser.add_argument("--results-dir", default="results/final")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-extended", action="store_true", help="Skip multi-training-seed/practical-LMMSE/domain-randomization results.")
    parser.add_argument("--skip-report", action="store_true", help="Do not rewrite report.md/report.pdf (useful for quick mode).")
    parser.add_argument("--skip-pdf", action="store_true")
    return parser.parse_args()


def _valid_checkpoint(path: Path, cfg, device: str) -> bool:
    if not path.exists():
        return False
    try:
        load_cnn_model(path, device=device, expected_cfg=cfg)
    except (RuntimeError, KeyError):
        return False
    return True


def _auxiliary_checkpoint_dirs(results_dir: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    """Keep quick/scratch checkpoints separate from formal experiment caches."""

    formal_results = (ROOT / "results" / "final").resolve()
    if results_dir.resolve() == formal_results:
        default_ablation = Path("checkpoints/ablation")
        default_multiseed = Path("checkpoints/multiseed")
    else:
        tag = results_dir.name
        default_ablation = Path("checkpoints") / tag / "ablation"
        default_multiseed = Path("checkpoints") / tag / "multiseed"
    return (
        Path(args.ablation_checkpoint_dir) if args.ablation_checkpoint_dir else default_ablation,
        Path(args.multiseed_checkpoint_dir) if args.multiseed_checkpoint_dir else default_multiseed,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = ensure_dir(args.results_dir)
    checkpoint = Path(args.checkpoint)
    ablation_checkpoint_dir, multiseed_checkpoint_dir = _auxiliary_checkpoint_dirs(results_dir, args)
    if not args.skip_train and not _valid_checkpoint(checkpoint, cfg, args.device):
        train_cnn_channel_estimator(
            config_path=args.config,
            device=args.device,
            checkpoint_path=checkpoint,
            results_dir=results_dir / "training",
        )
    model, device, _ = load_cnn_model(checkpoint, device=args.device, expected_cfg=cfg)

    raw, summary, constellation = evaluate_across_seeds(cfg, cfg.num_eval_frames, cnn_model=model, cnn_device=device)
    save_evaluation_artifacts(raw, summary, constellation, results_dir / "matched", "Fair sparse-pilot stress test", cnn_model=model)

    mismatch_cfg = load_config(args.mismatch_config)
    mismatch_raw, mismatch_summary, mismatch_constellation = evaluate_across_seeds(
        mismatch_cfg,
        mismatch_cfg.num_eval_frames,
        cnn_model=model,
        cnn_device=device,
        include_assumed_lmmse=True,
    )
    save_evaluation_artifacts(
        mismatch_raw,
        mismatch_summary,
        mismatch_constellation,
        results_dir / "mismatch",
        "PDP/tap-count generalization test",
        cnn_model=model,
        include_train_prior=True,
    )

    if not args.skip_ablation:
        ablation_command = [
            sys.executable,
            str(ROOT / "scripts" / "run_ablation.py"),
            "--config", args.multiseed_config,
            "--results-dir", str(results_dir / "ablation"),
            "--checkpoint-dir", str(ablation_checkpoint_dir),
            "--device", args.device,
        ]
        if args.skip_train:
            ablation_command.append("--skip-train")
        subprocess.run(ablation_command, cwd=ROOT, check=True)
    if not args.skip_extended:
        multiseed_command = [
            sys.executable,
            str(ROOT / "scripts" / "run_multiseed_experiment.py"),
            "--config", args.multiseed_config,
            "--domain-config", args.domain_config,
            "--mismatch-config", args.mismatch_config,
            "--unseen-config", args.unseen_config,
            "--results-dir", str(results_dir / "multiseed"),
            "--checkpoint-dir", str(multiseed_checkpoint_dir),
            "--device", args.device,
        ]
        if args.skip_train:
            multiseed_command.append("--skip-train")
        subprocess.run(multiseed_command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_experiment_summary.py"), "--results-dir", str(results_dir)], cwd=ROOT, check=True)
    if not args.skip_report:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_report_source.py"), "--results-dir", str(results_dir)], cwd=ROOT, check=True)
        if not args.skip_pdf:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "build_report_pdf.py")], cwd=ROOT, check=True)
    print(f"Completed full experiment: {results_dir}")


if __name__ == "__main__":
    main()
