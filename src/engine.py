"""Training and inference loops."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import compute_metrics


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return `(logits, targets)`.

    Logits rather than predictions so the caller can compute top-k accuracy and
    calibration without a second pass.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_logits), np.concatenate(all_targets)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float | None,
    epoch: int,
    log_every: int = 100,
) -> dict:
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0
    started = time.time()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            # Must unscale first, or the threshold applies to loss-scaled
            # gradients and does nothing useful.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        batch = targets.size(0)
        running_loss += loss.item() * batch
        correct += (logits.argmax(1) == targets).sum().item()
        seen += batch

        if step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  epoch {epoch} step {step}/{len(loader)} "
                f"loss {running_loss / seen:.4f} acc {correct / seen:.4f} lr {lr:.2e}",
                flush=True,
            )

    return {
        "train_loss": running_loss / seen,
        "train_accuracy": correct / seen,
        "seconds": time.time() - started,
    }


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    class_names: list[str],
    device: torch.device,
    epochs: int,
    max_lr: float,
    weight_decay: float,
    grad_clip: float | None,
    label_smoothing: float,
    checkpoint_path: Path,
    history_path: Path,
    patience: int,
) -> dict:
    """Train with OneCycle scheduling, keeping the best macro-F1 checkpoint.

    Selection on macro F1 rather than accuracy: with this class imbalance,
    accuracy keeps picking checkpoints that are good at the common classes and
    indifferent to the rare ones.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(loaders["train"]),
        pct_start=0.25,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler,
            scaler, device, grad_clip, epoch,
        )

        val_logits, val_targets = predict(model, loaders["val"], device)
        val_metrics = compute_metrics(val_logits, val_targets, class_names)

        record = {
            "epoch": epoch,
            **stats,
            # Same label smoothing as training, otherwise the two loss curves
            # sit on different scales and their gap reads as overfitting.
            "val_loss": float(
                nn.functional.cross_entropy(
                    torch.from_numpy(val_logits),
                    torch.from_numpy(val_targets),
                    label_smoothing=label_smoothing,
                )
            ),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        history.append(record)

        print(
            f"epoch {epoch}/{epochs}  "
            f"train_loss {record['train_loss']:.4f}  "
            f"val_acc {record['val_accuracy']:.4f}  "
            f"val_macro_f1 {record['val_macro_f1']:.4f}  "
            f"({record['seconds']:.0f}s)",
            flush=True,
        )

        # Written before the early-stop check so the final epoch is never lost.
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "class_names": class_names,
                },
                checkpoint_path,
            )
            print(f"  new best (macro F1 {best_f1:.4f}) -> {checkpoint_path}", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stop: no improvement for {patience} epochs", flush=True)
                break

    return {"history": history, "best_epoch": best_epoch, "best_val_macro_f1": best_f1}
