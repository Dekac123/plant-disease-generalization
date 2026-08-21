"""Print a report-ready summary of a finished run.

Reads back the JSON written by train.py, cross_eval.py and gradcam_report.py,
so reported numbers are never transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row(label: str, value: float | None, width: int = 26) -> str:
    if value is None:
        return f"  {label:<{width}} --"
    return f"  {label:<{width}} {value:.4f}"


def in_domain_section(payload: dict) -> None:
    test = payload["test"]
    print("=" * 72)
    print(f"IN-DOMAIN — PlantVillage held-out test ({test['n_samples']} images, "
          f"{test['n_classes_evaluated']} classes)")
    print("=" * 72)
    print(f"  best epoch                 {payload['best_epoch']}")
    print(f"  training time              {payload['train_seconds'] / 60:.1f} min")
    print(f"  trainable parameters       {payload['trainable_parameters']:,}")
    for label, key in [
        ("accuracy", "accuracy"),
        ("balanced accuracy", "balanced_accuracy"),
        ("top-5 accuracy", "top5_accuracy"),
        ("macro precision", "macro_precision"),
        ("macro recall", "macro_recall"),
        ("macro F1", "macro_f1"),
        ("weighted F1", "weighted_f1"),
        ("Cohen's kappa", "cohen_kappa"),
    ]:
        print(_row(label, test[key]))

    weak = sorted(test["per_class"], key=lambda r: r["f1"])[:5]
    print("\n  weakest classes:")
    for r in weak:
        print(f"    F1 {r['f1']:.4f}  P {r['precision']:.3f}  R {r['recall']:.3f}  "
              f"n={r['support']:<5d} {r['class']}")


def cross_domain_section(cross: dict) -> None:
    gap = cross["generalization_gap"]
    print()
    print("=" * 72)
    print("CROSS-DATASET — the same model on real field photographs")
    print("=" * 72)

    header = f"  {'metric':<28}{'lab':>10}{'field':>10}{'retained':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    lab = cross["results"]["plantvillage_test_shared"]
    field = cross["results"]["plantdoc_open"]

    for label, key in [
        ("accuracy", "accuracy"),
        ("balanced accuracy", "balanced_accuracy"),
        ("macro precision", "macro_precision"),
        ("macro recall", "macro_recall"),
        ("macro F1", "macro_f1"),
        ("top-5 accuracy", "top5_accuracy"),
        ("Cohen's kappa", "cohen_kappa"),
        ("crop identification", "crop_accuracy"),
        ("healthy vs diseased", "healthy_vs_diseased_accuracy"),
        ("  diseased precision", "diseased_precision"),
        ("  diseased recall", "diseased_recall"),
    ]:
        lab_value, field_value = lab[key], field[key]
        retained = f"{field_value / lab_value * 100:9.1f}%" if lab_value else "        --"
        print(f"  {label:<28}{lab_value:>10.4f}{field_value:>10.4f}{retained:>11}")

    control = cross["results"].get("plantvillage_mirror")
    if control:
        print(f"\n  control (independent PlantVillage mirror, {control['n_samples']} images): "
              f"accuracy {control['accuracy']:.4f}, macro F1 {control['macro_f1']:.4f}")
        print("    a high score here confirms the evaluation pipeline is sound,")
        print("    so the field collapse is domain shift rather than a bug")

    print(f"\n  restricted-vocabulary accuracy "
          f"{cross['results']['plantdoc_restricted']['accuracy']:.4f} "
          f"(field, predictions limited to the 28 possible classes)")
    print(f"  absolute drop  {gap['absolute_drop']:.4f}")
    print(f"  relative drop  {gap['relative_drop'] * 100:.1f}%")

    print("\n  calibration:")
    for name, block in (("lab", lab), ("field", field)):
        cal = block.get("calibration")
        if not cal:
            continue
        print(f"    {name:<6} ECE {cal['expected_calibration_error']:.4f}  "
              f"mean confidence {cal['mean_confidence']:.4f}  "
              f"confidence when wrong {cal['mean_confidence_when_wrong']:.4f}")
        print(f"           {cal['fraction_above_90_confidence'] * 100:.1f}% of predictions "
              f"claim >90% confidence; {cal['accuracy_above_90_confidence'] * 100:.1f}% "
              f"of those are correct")


def gradcam_section(stats: dict) -> None:
    print()
    print("=" * 72)
    print("GRAD-CAM — where the attention falls")
    print("=" * 72)
    baseline = stats.get("uniform_baseline_border_attention")
    for key in ("plantvillage", "plantdoc"):
        if key not in stats:
            continue
        s = stats[key]
        print(f"  {key:<14} border attention {s['mean_border_attention']:.3f} "
              f"(median {s['median_border_attention']:.3f}, n={s['n']}, "
              f"sample accuracy {s['accuracy_on_sample']:.3f})")
    if baseline:
        print(f"  uniform-attention baseline {baseline:.3f} "
              "— higher means attention is more spread out")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()

    payload = _load(args.run_dir / "test_metrics.json")
    if payload is None:
        raise SystemExit(f"{args.run_dir / 'test_metrics.json'} not found")
    in_domain_section(payload)

    cross = _load(args.run_dir / "cross_eval.json")
    if cross:
        cross_domain_section(cross)

    stats = _load(args.run_dir / "gradcam_stats.json")
    if stats:
        gradcam_section(stats)


if __name__ == "__main__":
    main()
