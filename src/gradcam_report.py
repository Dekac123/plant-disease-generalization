"""Generate Grad-CAM figures and background-reliance statistics.

Produces, for both the in-domain and out-of-domain datasets, a qualitative grid
of saliency overlays plus a quantitative summary of how much attention mass
falls outside the centre of the frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import (
    FixedVocabImageFolder,
    denormalize,
    eval_transform,
    read_vocabulary,
)
import i18n
from engine import get_device
from gradcam import GradCAM, background_mass
from models import build_model, target_layer_for_gradcam

SEED = 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--n-samples", type=int, default=8,
                    help="Images per dataset shown in the qualitative figure.")
    ap.add_argument("--n-stats", type=int, default=384,
                    help="Images per dataset used for the background-mass statistic.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lang", default="en", choices=sorted(i18n.STRINGS))
    return ap.parse_args()


def _short(name: str) -> str:
    """Shorten a class name so it fits under a subplot."""
    name = name.replace("___", " / ").replace("_", " ")
    return name if len(name) <= 26 else name[:25] + "…"


def _subset_loader(dataset, n: int, batch_size: int, seed: int) -> DataLoader:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    return DataLoader(Subset(dataset, idx.tolist()), batch_size=batch_size, shuffle=False)


def collect(cam: GradCAM, loader: DataLoader, device: torch.device):
    """Run Grad-CAM over a loader, returning images, maps, preds and targets."""
    images, maps, preds, targets = [], [], [], []

    for batch_images, batch_targets in loader:
        batch_images = batch_images.to(device)
        heatmaps, logits = cam(batch_images, class_indices=None)

        images.append(denormalize(batch_images).cpu().numpy())
        maps.append(heatmaps)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(batch_targets.numpy())

    return (
        np.concatenate(images),
        np.concatenate(maps),
        np.concatenate(preds),
        np.concatenate(targets),
    )


def plot_grid(images, maps, preds, targets, vocabulary, title, out_path: Path, t: dict) -> None:
    n = len(images)
    # Four across rather than eight: the per-panel captions carry full class
    # names, and narrower panels make them collide.
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    # Each sample takes two stacked cells (image, then overlay); the spacing
    # keeps per-cell titles off the image above them.
    fig, axes = plt.subplots(
        rows * 2, cols,
        figsize=(3.1 * cols, 7.2 * rows),
        gridspec_kw={"hspace": 0.32, "wspace": 0.06},
    )
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax_img = axes[r * 2][c]
        ax_cam = axes[r * 2 + 1][c]

        if i >= n:
            ax_img.axis("off")
            ax_cam.axis("off")
            continue

        img = images[i].transpose(1, 2, 0)
        correct = preds[i] == targets[i]

        ax_img.imshow(img)
        ax_img.set_title(
            f"{t['gc_true']}: {_short(vocabulary[targets[i]])}", fontsize=8.5,
        )
        ax_img.axis("off")

        ax_cam.imshow(img)
        ax_cam.imshow(maps[i], cmap="jet", alpha=0.45)
        ax_cam.set_title(
            f"{t['gc_pred']}: {_short(vocabulary[preds[i]])}",
            fontsize=8.5,
            color="green" if correct else "red",
        )
        ax_cam.axis("off")

    fig.suptitle(title, fontsize=13)
    # No tight_layout: it overrides the gridspec spacing above.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    args = parse_args()
    device = get_device()
    vocabulary = read_vocabulary(args.data_root / "plantvillage")

    model = build_model(args.model, len(vocabulary), pretrained=False).to(device)
    checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    layer = target_layer_for_gradcam(model, args.model)

    datasets = {
        "plantvillage": FixedVocabImageFolder(
            args.data_root / "plantvillage" / "test", vocabulary, transform=eval_transform()
        ),
        "plantdoc": FixedVocabImageFolder(
            args.data_root / "plantdoc", vocabulary, transform=eval_transform()
        ),
    }

    t = i18n.get(args.lang)
    titles = {
        "plantvillage": t["gradcam_lab"],
        "plantdoc": t["gradcam_field"],
    }

    stats: dict[str, dict] = {}
    figures_dir = args.run_dir / (
        "figures" if args.lang == "en" else f"figures_{args.lang}"
    )

    with GradCAM(model, layer) as cam:
        for key, dataset in datasets.items():
            # Qualitative grid.
            loader = _subset_loader(dataset, args.n_samples, args.batch_size, SEED)
            images, maps, preds, targets = collect(cam, loader, device)
            plot_grid(
                images, maps, preds, targets, vocabulary,
                titles[key], figures_dir / f"gradcam_{key}.png", t,
            )

            # Quantitative background reliance over a larger sample.
            stat_loader = _subset_loader(dataset, args.n_stats, args.batch_size, SEED + 1)
            _, stat_maps, stat_preds, stat_targets = collect(cam, stat_loader, device)
            border = np.array([background_mass(m) for m in stat_maps])

            stats[key] = {
                "n": int(len(border)),
                "mean_border_attention": float(np.nanmean(border)),
                "median_border_attention": float(np.nanmedian(border)),
                "accuracy_on_sample": float(np.mean(stat_preds == stat_targets)),
            }
            print(
                f"{key}: border attention mean={stats[key]['mean_border_attention']:.3f} "
                f"median={stats[key]['median_border_attention']:.3f} "
                f"(n={stats[key]['n']}, sample acc={stats[key]['accuracy_on_sample']:.3f})"
            )

    # A uniform heatmap puts exactly this much mass in the border, so values
    # near it mean attention is spread everywhere.
    uniform_baseline = 1.0 - (1 - 2 * 0.25) ** 2
    stats["uniform_baseline_border_attention"] = uniform_baseline
    print(f"uniform-attention baseline = {uniform_baseline:.3f}")

    out_path = args.run_dir / f"gradcam_stats{'' if args.lang == 'en' else '_' + args.lang}.json"
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
