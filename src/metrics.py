"""Metric computation.

PlantVillage is imbalanced enough (~5000 images in the largest class against
~150 in the smallest) that plain accuracy hides poor performance on the rare
diseases, so macro averages and a per-class table are reported alongside it.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from labels import crop_of, is_healthy


def top_k_accuracy(logits: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    """Fraction of samples whose true class is among the k highest scores."""
    k = min(k, logits.shape[1])
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([t in row for row, t in zip(topk, targets)]))


def compute_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    present_only: bool = False,
) -> dict:
    """Full metric bundle for one evaluation pass.

    `present_only` restricts averaging to classes occurring in `targets`.
    PlantDoc covers only 28 of the 39 training classes, and averaging over all
    39 would penalise the model for 11 classes that cannot appear.
    """
    preds = logits.argmax(axis=1)

    if present_only:
        label_ids = sorted(set(targets.tolist()))
    else:
        label_ids = list(range(len(class_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=label_ids, average=None, zero_division=0
    )

    macro = precision_recall_fscore_support(
        targets, preds, labels=label_ids, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        targets, preds, labels=label_ids, average="weighted", zero_division=0
    )

    per_class = [
        {
            "class": class_names[cid],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cid in enumerate(label_ids)
    ]

    return {
        "accuracy": float(np.mean(preds == targets)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "top5_accuracy": top_k_accuracy(logits, targets, k=5),
        "cohen_kappa": float(cohen_kappa_score(targets, preds)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "n_samples": int(len(targets)),
        "n_classes_evaluated": len(label_ids),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(targets, preds, labels=label_ids).tolist(),
        "confusion_labels": [class_names[c] for c in label_ids],
        "sklearn_report": classification_report(
            targets,
            preds,
            labels=label_ids,
            target_names=[class_names[c] for c in label_ids],
            zero_division=0,
            digits=4,
        ),
    }


def relaxed_metrics(
    logits: np.ndarray, targets: np.ndarray, class_names: list[str]
) -> dict:
    """Two easier questions than exact classification.

    Crop accuracy asks whether the model identifies the plant at all while
    getting the disease wrong; the binary score asks whether it can tell sick
    from well. Together they show *how* a collapse happened.
    """
    preds = logits.argmax(axis=1)

    pred_crop = np.array([crop_of(class_names[p]) for p in preds])
    true_crop = np.array([crop_of(class_names[t]) for t in targets])

    pred_healthy = np.array([is_healthy(class_names[p]) for p in preds])
    true_healthy = np.array([is_healthy(class_names[t]) for t in targets])

    tp = int(np.sum(~pred_healthy & ~true_healthy))
    fp = int(np.sum(~pred_healthy & true_healthy))
    fn = int(np.sum(pred_healthy & ~true_healthy))

    return {
        "crop_accuracy": float(np.mean(pred_crop == true_crop)),
        "healthy_vs_diseased_accuracy": float(np.mean(pred_healthy == true_healthy)),
        # Missing a sick plant costs more than a false alarm, so recall on
        # "diseased" is the agronomically relevant half.
        "diseased_recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "diseased_precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    # -inf entries (from vocabulary masking) become exactly zero probability.
    exp = np.exp(shifted, where=np.isfinite(shifted), out=np.zeros_like(shifted))
    return exp / exp.sum(axis=1, keepdims=True)


def calibration_metrics(
    logits: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> dict:
    """Expected calibration error plus confidence broken down by correctness.

    A well-calibrated classifier reporting 90% confidence should be right about
    90% of the time. The failure mode that matters in the field is confident
    error: a wrong diagnosis with no signal that the model is unsure.
    """
    probabilities = _softmax(logits.astype(np.float64))
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets

    # ECE = accuracy/confidence gap averaged over equal-width bins, weighted by
    # bin population.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"lower": float(lo), "upper": float(hi), "count": 0,
                         "accuracy": None, "confidence": None})
            continue

        bin_accuracy = float(correct[in_bin].mean())
        bin_confidence = float(confidence[in_bin].mean())
        ece += (count / len(confidence)) * abs(bin_accuracy - bin_confidence)
        bins.append({"lower": float(lo), "upper": float(hi), "count": count,
                     "accuracy": bin_accuracy, "confidence": bin_confidence})

    return {
        "expected_calibration_error": float(ece),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "mean_confidence_when_correct": (
            float(confidence[correct].mean()) if correct.any() else float("nan")
        ),
        "mean_confidence_when_wrong": (
            float(confidence[~correct].mean()) if (~correct).any() else float("nan")
        ),
        "overconfidence": float(confidence.mean() - correct.mean()),
        # The number that matters: how confident is it when it is wrong?
        "fraction_above_90_confidence": float((confidence > 0.9).mean()),
        "accuracy_above_90_confidence": (
            float(correct[confidence > 0.9].mean())
            if (confidence > 0.9).any() else float("nan")
        ),
        "bins": bins,
    }


def format_summary(name: str, metrics: dict) -> str:
    """One compact human-readable block per evaluation."""
    lines = [
        f"--- {name} ---",
        f"  samples              {metrics['n_samples']}"
        f"  (classes: {metrics['n_classes_evaluated']})",
        f"  accuracy             {metrics['accuracy']:.4f}",
        f"  balanced accuracy    {metrics['balanced_accuracy']:.4f}",
        f"  top-5 accuracy       {metrics['top5_accuracy']:.4f}",
        f"  macro  P / R / F1    {metrics['macro_precision']:.4f} / "
        f"{metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}",
        f"  weighted P / R / F1  {metrics['weighted_precision']:.4f} / "
        f"{metrics['weighted_recall']:.4f} / {metrics['weighted_f1']:.4f}",
        f"  Cohen's kappa        {metrics['cohen_kappa']:.4f}",
    ]
    return "\n".join(lines)
