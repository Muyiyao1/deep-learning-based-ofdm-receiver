"""Training loop for the physics-informed residual OFDM estimator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import OFDMConfig, load_config
from .dataset import OFDMFrameDataset
from .models import CNNChannelEstimator, delay_domain_tail_energy, pilot_consistency_loss
from .plots import plot_training_loss
from .utils import ensure_dir, set_seed


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` to CUDA where available, otherwise CPU."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def build_model(cfg: OFDMConfig, variant: str = "residual_delay") -> CNNChannelEstimator:
    """Construct the documented residual estimator from configuration."""

    variants = {
        "plain": {"residual": False, "delay_projection": False, "delay_regularization": False},
        "residual": {"residual": True, "delay_projection": False, "delay_regularization": False},
        "residual_soft_delay": {"residual": True, "delay_projection": False, "delay_regularization": True},
        "residual_hard_delay": {"residual": True, "delay_projection": True, "delay_regularization": True},
        "residual_delay": {"residual": True, "delay_projection": True, "delay_regularization": True},
    }
    if variant not in variants:
        raise ValueError(f"Unsupported model variant: {variant}")
    return CNNChannelEstimator(
        hidden_channels=cfg.model_hidden_channels,
        num_blocks=cfg.model_num_blocks,
        channel_length=cfg.effective_dft_taps,
        **variants[variant],
    )


def _loss_terms(model: CNNChannelEstimator, x: torch.Tensor, target: torch.Tensor, cfg: OFDMConfig) -> tuple[torch.Tensor, dict[str, float]]:
    predicted, raw = model(x, return_raw=True)
    complex_mse = F.mse_loss(predicted, target)
    delay_loss = delay_domain_tail_energy(raw, cfg.effective_dft_taps)
    pilot_loss = pilot_consistency_loss(predicted, x)
    delay_weight = cfg.delay_loss_weight if model.delay_regularization else 0.0
    total = complex_mse + delay_weight * delay_loss + cfg.pilot_loss_weight * pilot_loss
    return total, {
        "complex_mse": float(complex_mse.detach().cpu()),
        "delay_loss": float(delay_loss.detach().cpu()),
        "pilot_loss": float(pilot_loss.detach().cpu()),
    }


def _run_epoch(
    model: CNNChannelEstimator,
    loader: DataLoader,
    cfg: OFDMConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "complex_mse": 0.0, "delay_loss": 0.0, "pilot_loss": 0.0, "samples": 0}
    iterator = tqdm(loader, leave=False, desc="train" if training else "valid")
    with torch.set_grad_enabled(training):
        for batch in iterator:
            x = batch["cnn_input"].to(device, non_blocking=True)
            target = batch["channel_target"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, terms = _loss_terms(model, x, target, cfg)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            batch_size = int(x.shape[0])
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["samples"] += batch_size
            for key, value in terms.items():
                totals[key] += value * batch_size
            iterator.set_postfix(loss=totals["loss"] / max(totals["samples"], 1))
    samples = max(int(totals.pop("samples")), 1)
    return {key: value / samples for key, value in totals.items()}


def train_cnn_channel_estimator(
    config_path: str | Path = "configs/default_config.json",
    config: OFDMConfig | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    device: str = "auto",
    num_train_samples: int | None = None,
    num_val_samples: int | None = None,
    checkpoint_path: str | Path = "checkpoints/cnn_channel_estimator.pt",
    results_dir: str | Path = "results",
    num_workers: int = 0,
    model_variant: str = "residual_delay",
) -> dict[str, Any]:
    """Train on independent on-the-fly frames and save the best validation model."""

    cfg = load_config(config_path) if config is None else config
    cfg.validate()
    set_seed(cfg.random_seed)
    device_obj = resolve_device(device)
    total_epochs = cfg.training_epochs if epochs is None else int(epochs)
    actual_batch_size = cfg.training_batch_size if batch_size is None else int(batch_size)
    actual_lr = cfg.training_learning_rate if learning_rate is None else float(learning_rate)
    train_samples = cfg.training_samples_per_epoch if num_train_samples is None else int(num_train_samples)
    val_samples = cfg.validation_samples if num_val_samples is None else int(num_val_samples)
    checkpoint_path = Path(checkpoint_path)
    results_dir = ensure_dir(results_dir)
    ensure_dir(checkpoint_path.parent)

    train_ds = OFDMFrameDataset(cfg, num_samples=train_samples, seed=cfg.random_seed + 10_000_000)
    val_ds = OFDMFrameDataset(cfg, num_samples=val_samples, seed=cfg.random_seed + 20_000_000)
    loader_generator = torch.Generator().manual_seed(cfg.random_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=actual_batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
    )

    model = build_model(cfg, variant=model_variant).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=actual_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=cfg.scheduler_patience,
    )
    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, total_epochs + 1):
        train_ds.set_epoch(epoch)
        train_terms = _run_epoch(model, train_loader, cfg, device_obj, optimizer=optimizer)
        val_terms = _run_epoch(model, val_loader, cfg, device_obj)
        scheduler.step(val_terms["loss"])
        row = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_terms["loss"],
            "val_loss": val_terms["loss"],
            "train_complex_mse": train_terms["complex_mse"],
            "val_complex_mse": val_terms["complex_mse"],
            "val_delay_loss": val_terms["delay_loss"],
            "val_pilot_loss": val_terms["pilot_loss"],
        }
        history.append(row)
        print(f"Epoch {epoch:03d}: train={row['train_loss']:.6f}, val={row['val_loss']:.6f}, lr={row['learning_rate']:.2e}")
        if val_terms["loss"] < best_val_loss - 1e-8:
            best_val_loss = val_terms["loss"]
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": 3,
                    "model_state_dict": model.state_dict(),
                    "model_spec": model.model_spec(),
                    "config": cfg.to_dict(),
                    "config_fingerprint": cfg.fingerprint(),
                    "model_variant": model_variant,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.early_stopping_patience:
                print(f"Early stopping after epoch {epoch}; no validation improvement for {stale_epochs} epochs.")
                break

    pd.DataFrame(history).to_csv(results_dir / "training_history.csv", index=False)
    plot_training_loss(history, results_dir / "training_loss.png")
    manifest = {
        "config_path": str(config_path),
        "config_fingerprint": cfg.fingerprint(),
        "model_spec": model.model_spec(),
        "model_variant": model_variant,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device_obj),
        "train_samples_per_epoch": train_samples,
        "validation_samples": val_samples,
        "epochs_requested": total_epochs,
        "epochs_completed": len(history),
        "best_validation_loss": best_val_loss,
        "train_snr_db_range": cfg.train_snr_db_range,
        "split_seeds": {"train": cfg.random_seed + 10_000_000, "validation": cfg.random_seed + 20_000_000},
    }
    (results_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"best_val_loss": best_val_loss, "history": history, "checkpoint_path": str(checkpoint_path), "device": str(device_obj), **manifest}
