"""Figure generation for the report.

Blue is always in-domain and orange always out-of-domain, across every chart.
The confusion matrices encode magnitude, so they use a single-hue ramp rather
than the categorical pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- palette -----------------------------------------------------------
SERIES_1 = "#2a78d6"   # blue   -- in-domain / PlantVillage
SERIES_2 = "#eb6834"   # orange -- out-of-domain / PlantDoc
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Single-hue sequential ramp (blue 100 -> 700) for magnitude encodings.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": SECONDARY_INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "legend.frameon": False,
        }
    )


def training_curves(history: list[dict], out_path: Path) -> None:
    """Loss and validation quality across epochs.

    Two panels rather than twin y-axes: loss and F1 have unrelated scales, and
    overlaying them lets the choice of scale imply a relationship.
    """
    epochs = [h["epoch"] for h in history]

    fig, (ax_loss, ax_quality) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history],
                 color=SERIES_1, linewidth=2, marker="o", markersize=4, label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history],
                 color=SERIES_2, linewidth=2, marker="s", markersize=4, label="validation")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_quality.plot(epochs, [h["val_accuracy"] for h in history],
                    color=SERIES_1, linewidth=2, marker="o", markersize=4, label="accuracy")
    ax_quality.plot(epochs, [h["val_macro_f1"] for h in history],
                    color=SERIES_2, linewidth=2, marker="s", markersize=4, label="macro F1")
    ax_quality.set_xlabel("epoch")
    ax_quality.set_ylabel("validation score")
    ax_quality.set_ylim(0, 1.02)
    ax_quality.set_title("Validation quality")
    ax_quality.legend()

    # Label the final value directly rather than every point.
    for ax, series, colour in (
        (ax_quality, [h["val_accuracy"] for h in history], SERIES_1),
        (ax_quality, [h["val_macro_f1"] for h in history], SERIES_2),
    ):
        ax.annotate(f"{series[-1]:.3f}", (epochs[-1], series[-1]),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=8, color=colour, va="center")

    fig.suptitle("Training history", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_path)


def confusion_heatmap(metrics: dict, title: str, out_path: Path) -> None:
    """Row-normalised confusion matrix (i.e. per-class recall on the diagonal)."""
    matrix = np.array(metrics["confusion_matrix"], dtype=float)
    labels = [l.replace("___", " / ").replace("_", " ") for l in metrics["confusion_labels"]]

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    n = len(labels)
    size = max(7.0, n * 0.32)
    fig, ax = plt.subplots(figsize=(size + 3, size))

    im = ax.imshow(normalised, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="equal")
    ax.grid(False)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=12, pad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("fraction of true class", color=SECONDARY_INK, fontsize=8)
    cbar.outline.set_visible(False)

    _save(fig, out_path)


def domain_gap_chart(cross_eval: dict, out_path: Path) -> None:
    """Headline comparison: the same model on lab images vs field images."""
    in_domain = cross_eval["results"]["plantvillage_test_shared"]
    out_domain = cross_eval["results"]["plantdoc_open"]

    measures = [
        ("Accuracy", "accuracy"),
        ("Balanced\naccuracy", "balanced_accuracy"),
        ("Macro F1", "macro_f1"),
        ("Top-5\naccuracy", "top5_accuracy"),
        ("Crop\nidentification", "crop_accuracy"),
        ("Healthy vs\ndiseased", "healthy_vs_diseased_accuracy"),
    ]

    labels = [m[0] for m in measures]
    lab_values = [in_domain[m[1]] for m in measures]
    field_values = [out_domain[m[1]] for m in measures]

    x = np.arange(len(labels))
    width = 0.38
    gap = 0.012

    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars_lab = ax.bar(x - width / 2 - gap, lab_values, width,
                      label="PlantVillage (lab)", color=SERIES_1)
    bars_field = ax.bar(x + width / 2 + gap, field_values, width,
                        label="PlantDoc (field)", color=SERIES_2)

    for bars, values in ((bars_lab, lab_values), (bars_field, field_values)):
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}",
                        (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color=SECONDARY_INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Same model, same 28 classes — lab photographs vs field photographs",
        fontsize=12,
        pad=30,
    )
    # Outside the axes: the lab bars reach the top and would sit under it.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2)

    _save(fig, out_path)


def per_class_f1_chart(cross_eval: dict, out_path: Path) -> None:
    """Per-class F1 in both domains, so the collapse can be read class by class."""
    in_domain = {r["class"]: r for r in
                 cross_eval["results"]["plantvillage_test_shared"]["per_class"]}
    out_domain = {r["class"]: r for r in
                  cross_eval["results"]["plantdoc_open"]["per_class"]}

    shared = [c for c in cross_eval["shared_classes"] if c in in_domain and c in out_domain]
    shared.sort(key=lambda c: out_domain[c]["f1"], reverse=True)

    labels = [c.replace("___", " / ").replace("_", " ") for c in shared]
    y = np.arange(len(shared))
    height = 0.38
    gap = 0.012

    fig, ax = plt.subplots(figsize=(9, max(5.0, len(shared) * 0.34)))
    ax.barh(y + height / 2 + gap, [in_domain[c]["f1"] for c in shared], height,
            label="PlantVillage (lab)", color=SERIES_1)
    ax.barh(y - height / 2 - gap, [out_domain[c]["f1"] for c in shared], height,
            label="PlantDoc (field)", color=SERIES_2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1.02)
    ax.set_title("Per-class F1, ordered by field performance", fontsize=12)
    ax.legend(loc="lower right")

    _save(fig, out_path)


def calibration_chart(cross_eval: dict, out_path: Path) -> None:
    """Reliability diagram. On the diagonal is well calibrated; below it means
    the model claims more certainty than it earns."""
    panels = [
        ("PlantVillage (lab)", "plantvillage_test_shared", SERIES_1),
        ("PlantDoc (field)", "plantdoc_open", SERIES_2),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)

    for ax, (title, key, colour) in zip(axes, panels):
        cal = cross_eval["results"][key]["calibration"]
        bins = [b for b in cal["bins"] if b["count"] > 0]

        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1,
                linestyle="--", label="perfect calibration")
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins],
                color=colour, linewidth=2, marker="o", markersize=6, label="observed")

        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("predicted confidence")
        ax.set_title(
            f"{title}\nECE {cal['expected_calibration_error']:.3f}  ·  "
            f"mean confidence {cal['mean_confidence']:.2f}  ·  "
            f"accuracy {cal['accuracy']:.2f}",
            fontsize=9.5,
        )
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("observed accuracy")
    fig.suptitle("Reliability diagrams — confidence vs correctness", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_path)


def model_comparison_chart(runs: dict[str, dict], out_path: Path) -> None:
    """Lab vs field accuracy for each architecture.

    Deliberately not a table of in-domain metrics: the finding worth showing is
    that the two models are indistinguishable on the benchmark and far apart in
    the field, so benchmark rank does not predict field rank.
    """
    names = list(runs)
    lab = [runs[n]["cross"]["results"]["plantvillage_test_shared"]["accuracy"] for n in names]
    field = [runs[n]["cross"]["results"]["plantdoc_open"]["accuracy"] for n in names]

    x = np.arange(len(names))
    width = 0.34
    gap = 0.012

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars_lab = ax.bar(x - width / 2 - gap, lab, width,
                      label="PlantVillage (lab)", color=SERIES_1)
    bars_field = ax.bar(x + width / 2 + gap, field, width,
                        label="PlantDoc (field)", color=SERIES_2)

    for bars, values in ((bars_lab, lab), (bars_field, field)):
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.3f}",
                        (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8.5, color=SECONDARY_INK)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Benchmark rank does not predict field rank",
        fontsize=12, pad=30,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2)

    _save(fig, out_path)


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--compare-dirs", type=Path, nargs="*", default=[],
                    help="Additional run directories to include in the architecture chart.")
    args = ap.parse_args()

    apply_style()
    figures = args.run_dir / "figures"

    test_payload = json.loads((args.run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    training_curves(test_payload["history"], figures / "training_curves.png")
    confusion_heatmap(
        test_payload["test"],
        "PlantVillage held-out test — row-normalised confusion matrix",
        figures / "confusion_plantvillage.png",
    )

    cross_path = args.run_dir / "cross_eval.json"
    if cross_path.exists():
        cross_eval = json.loads(cross_path.read_text(encoding="utf-8"))
        domain_gap_chart(cross_eval, figures / "domain_gap.png")
        per_class_f1_chart(cross_eval, figures / "per_class_f1.png")
        calibration_chart(cross_eval, figures / "calibration.png")
        confusion_heatmap(
            cross_eval["results"]["plantdoc_open"],
            "PlantDoc (field photographs) — row-normalised confusion matrix",
            figures / "confusion_plantdoc.png",
        )
    else:
        print(f"skipping cross-dataset figures: {cross_path} not found")

    if args.compare_dirs:
        runs = {}
        for d in [args.run_dir, *args.compare_dirs]:
            metrics_path = d / "test_metrics.json"
            cross_path = d / "cross_eval.json"
            if not (metrics_path.exists() and cross_path.exists()):
                print(f"skipping {d.name} in comparison: needs both "
                      "test_metrics.json and cross_eval.json")
                continue
            runs[d.name] = {
                "test": json.loads(metrics_path.read_text(encoding="utf-8")),
                "cross": json.loads(cross_path.read_text(encoding="utf-8")),
            }
        if len(runs) >= 2:
            model_comparison_chart(runs, figures / "model_comparison.png")
        else:
            print("need at least two complete runs for the comparison chart")


if __name__ == "__main__":
    main()
