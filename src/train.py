"""Train a plant-disease classifier on PlantVillage and evaluate it in-domain.

Cross-dataset evaluation lives in cross_eval.py, so PlantDoc plays no part in
model selection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data import build_loaders
from engine import fit, get_device, predict
from metrics import compute_metrics, format_summary
from models import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet18",
                    choices=["resnet9", "resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--scratch", action="store_true",
                    help="Skip ImageNet initialisation (always true for resnet9).")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out-root", type=Path, default=Path("results"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--limit-train-batches", type=int, default=None,
                    help="Smoke-test escape hatch: stop each epoch after N batches.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or (
        f"{args.model}{'_scratch' if args.scratch else ''}"
    )
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}  run: {run_name}")

    loaders, class_names = build_loaders(
        args.data_root / "plantvillage",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    for split, loader in loaders.items():
        print(f"  {split}: {len(loader.dataset)} images / {len(loader)} batches")

    if args.limit_train_batches:
        loaders = dict(loaders)
        loaders["train"] = _truncate(loaders["train"], args.limit_train_batches)

    pretrained = not args.scratch
    model = build_model(args.model, len(class_names), pretrained=pretrained).to(device)
    print(f"model: {args.model} (pretrained={pretrained and args.model != 'resnet9'}) "
          f"{count_parameters(model):,} trainable params")

    started = time.time()
    train_result = fit(
        model,
        loaders,
        class_names,
        device,
        epochs=args.epochs,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        checkpoint_path=out_dir / "best.pt",
        history_path=out_dir / "history.json",
        patience=args.patience,
    )
    train_seconds = time.time() - started

    # The final epoch is not necessarily the best one.
    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"\nloaded best checkpoint from epoch {checkpoint['epoch']}")

    test_logits, test_targets = predict(model, loaders["test"], device)
    test_metrics = compute_metrics(test_logits, test_targets, class_names)

    print()
    print(format_summary("PlantVillage held-out test", test_metrics))
    print()
    print(test_metrics["sklearn_report"])

    payload = {
        "run_name": run_name,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "class_names": class_names,
        "trainable_parameters": count_parameters(model),
        "train_seconds": train_seconds,
        "best_epoch": train_result["best_epoch"],
        "best_val_macro_f1": train_result["best_val_macro_f1"],
        "history": train_result["history"],
        "test": test_metrics,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Saved so the plotting code can reuse this pass instead of re-running
    # inference over 8000 images.
    torch.save(
        {"logits": test_logits, "targets": test_targets, "class_names": class_names},
        out_dir / "test_predictions.pt",
    )
    print(f"\nwrote {out_dir / 'test_metrics.json'}")


def _truncate(loader, n_batches: int):
    """Wrap a DataLoader so iteration stops early, for smoke tests."""

    class _Truncated:
        def __init__(self, inner, limit):
            self.inner = inner
            self.limit = limit
            self.dataset = inner.dataset

        def __iter__(self):
            for i, batch in enumerate(self.inner):
                if i >= self.limit:
                    return
                yield batch

        def __len__(self):
            return min(self.limit, len(self.inner))

    return _Truncated(loader, n_batches)


if __name__ == "__main__":
    main()
