"""Training loop for the CNN channel estimator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import OFDMConfig, load_config
from .dataset import OFDMFrameDataset
from .models import CNNChannelEstimator
from .plots import plot_training_loss
from .utils import ensure_dir, set_seed


def resolve_device(device: str) -> torch.device:
    """Resolve an explicit device string or 'auto'."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    iterator = tqdm(loader, leave=False, desc="train" if training else "valid")
    with torch.set_grad_enabled(training):
        for batch in iterator:
            x = batch["cnn_input"].to(device, non_blocking=True)
            y = batch["channel_target"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = x.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            iterator.set_postfix(loss=total_loss / max(total_samples, 1))
    return total_loss / max(total_samples, 1)


def train_cnn_channel_estimator(
    config_path: str | Path = "configs/default_config.json",
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "auto",
    num_train_samples: int = 5000,
    num_val_samples: int = 1000,
    checkpoint_path: str | Path = "checkpoints/cnn_channel_estimator.pt",
    results_dir: str | Path = "results",
    num_workers: int = 0,
) -> dict[str, Any]:
    """Train the CNN channel estimator and save the best checkpoint."""

    cfg = load_config(config_path)
    set_seed(cfg.random_seed)
    device_obj = resolve_device(device)
    ensure_dir(Path(checkpoint_path).parent)
    ensure_dir(results_dir)

    train_ds = OFDMFrameDataset(cfg, num_samples=num_train_samples, seed=cfg.random_seed)
    val_ds = OFDMFrameDataset(cfg, num_samples=num_val_samples, seed=cfg.random_seed + 1_000_000)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
    )

    model = CNNChannelEstimator().to(device_obj)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        train_loss = _run_epoch(model, train_loader, criterion, device_obj, optimizer=optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device_obj, optimizer=None)
        row = {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.to_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

    history_path = Path(results_dir) / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    plot_training_loss(history, Path(results_dir) / "training_loss.png")
    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device_obj),
    }
