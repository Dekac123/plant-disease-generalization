"""Cross-dataset evaluation: PlantVillage-trained model vs PlantDoc field photos.

PlantVillage images are studio shots -- one detached leaf, centred, uniform
background. PlantDoc is the same diseases photographed in real fields. A model
that learnt pathology should transfer; one that learnt which backdrop goes with
which label should not.

Four evaluations, because a raw in-domain vs out-of-domain gap conflates
several effects:

  plantvillage_test_shared  in-domain, restricted to the 28 classes PlantDoc
                            covers, so the comparison is not also a change of
                            label set
  plantdoc_open             field images over the full 39-way head, i.e. what
                            deployment looks like
  plantdoc_restricted       field images with logits masked to the 28 shared
                            classes; separates impossible predictions from
                            genuine confusion
  plantvillage_mirror       control on an independently packaged PlantVillage
                            derivative, to show the pipeline itself is sound
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import build_crosseval_loader, build_loaders, read_vocabulary
from engine import get_device, predict
from labels import PLANTDOC_TO_PLANTVILLAGE, assert_mapping_targets_exist
from metrics import (
    calibration_metrics,
    compute_metrics,
    format_summary,
    relaxed_metrics,
)
from models import build_model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Directory holding best.pt, as written by train.py.")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=224)
    return ap.parse_args()


def _restrict_logits(logits: np.ndarray, allowed: list[int]) -> np.ndarray:
    """Mask every class outside `allowed` to -inf so it can never be predicted."""
    masked = np.full_like(logits, -np.inf)
    masked[:, allowed] = logits[:, allowed]
    return masked


def main() -> None:
    args = parse_args()
    device = get_device()

    vocabulary = read_vocabulary(args.data_root / "plantvillage")
    assert_mapping_targets_exist(vocabulary)

    shared_names = sorted(set(PLANTDOC_TO_PLANTVILLAGE.values()))
    shared_ids = sorted(vocabulary.index(n) for n in shared_names)
    print(f"{len(shared_ids)} classes shared between PlantVillage and PlantDoc")

    model = build_model(args.model, len(vocabulary), pretrained=False).to(device)
    checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"loaded {args.run_dir / 'best.pt'} (epoch {checkpoint['epoch']})")

    results: dict[str, dict] = {}

    # --- 1. In-domain, restricted to the shared classes -------------------
    loaders, _ = build_loaders(
        args.data_root / "plantvillage",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        splits=("test",),
    )
    pv_logits, pv_targets = predict(model, loaders["test"], device)

    keep = np.isin(pv_targets, shared_ids)
    results["plantvillage_test_shared"] = compute_metrics(
        pv_logits[keep], pv_targets[keep], vocabulary, present_only=True
    )
    results["plantvillage_test_shared"].update(
        relaxed_metrics(pv_logits[keep], pv_targets[keep], vocabulary)
    )
    results["plantvillage_test_shared"]["calibration"] = calibration_metrics(
        pv_logits[keep], pv_targets[keep]
    )

    # --- 2 & 3. Out-of-domain -------------------------------------------
    pd_loader = build_crosseval_loader(
        args.data_root / "plantdoc",
        vocabulary,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    pd_logits, pd_targets = predict(model, pd_loader, device)

    results["plantdoc_open"] = compute_metrics(
        pd_logits, pd_targets, vocabulary, present_only=True
    )
    results["plantdoc_open"].update(relaxed_metrics(pd_logits, pd_targets, vocabulary))
    results["plantdoc_open"]["calibration"] = calibration_metrics(pd_logits, pd_targets)

    pd_restricted = _restrict_logits(pd_logits, shared_ids)
    results["plantdoc_restricted"] = compute_metrics(
        pd_restricted, pd_targets, vocabulary, present_only=True
    )
    results["plantdoc_restricted"].update(
        relaxed_metrics(pd_restricted, pd_targets, vocabulary)
    )

    # --- 4. Control: an independently packaged PlantVillage mirror --------
    mirror_root = args.data_root / "plantvillage_mirror"
    if mirror_root.is_dir():
        mirror_loader = build_crosseval_loader(
            mirror_root,
            vocabulary,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )
        mr_logits, mr_targets = predict(model, mirror_loader, device)
        results["plantvillage_mirror"] = compute_metrics(
            mr_logits, mr_targets, vocabulary, present_only=True
        )
        results["plantvillage_mirror"].update(
            relaxed_metrics(mr_logits, mr_targets, vocabulary)
        )
        results["plantvillage_mirror"]["calibration"] = calibration_metrics(
            mr_logits, mr_targets
        )
    else:
        print(f"no control set at {mirror_root}; skipping")

    for name in ("plantvillage_test_shared", "plantdoc_open", "plantdoc_restricted",
                 "plantvillage_mirror"):
        if name not in results:
            continue
        m = results[name]
        print()
        print(format_summary(name, m))
        print(f"  crop accuracy        {m['crop_accuracy']:.4f}")
        print(f"  healthy vs diseased  {m['healthy_vs_diseased_accuracy']:.4f}")
        print(f"  diseased recall      {m['diseased_recall']:.4f}")
        if "calibration" in m:
            cal = m["calibration"]
            print(f"  mean confidence      {cal['mean_confidence']:.4f} "
                  f"(ECE {cal['expected_calibration_error']:.4f})")
            print(f"  confidence when wrong {cal['mean_confidence_when_wrong']:.4f}")
            print(f"  >90% confident       {cal['fraction_above_90_confidence']:.4f} "
                  f"of samples, {cal['accuracy_above_90_confidence']:.4f} of them correct")

    in_domain = results["plantvillage_test_shared"]["accuracy"]
    out_domain = results["plantdoc_open"]["accuracy"]
    gap = {
        "in_domain_accuracy": in_domain,
        "out_of_domain_accuracy": out_domain,
        "absolute_drop": in_domain - out_domain,
        "relative_drop": (in_domain - out_domain) / in_domain if in_domain else float("nan"),
        "retained_fraction": out_domain / in_domain if in_domain else float("nan"),
    }
    print(
        f"\ngeneralization gap: {in_domain:.4f} -> {out_domain:.4f} "
        f"({gap['relative_drop'] * 100:.1f}% relative drop)"
    )

    payload = {
        "run_dir": str(args.run_dir),
        "checkpoint_epoch": checkpoint["epoch"],
        "shared_classes": shared_names,
        "generalization_gap": gap,
        "results": results,
    }
    out_path = args.run_dir / "cross_eval.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    np.savez(
        args.run_dir / "plantdoc_predictions.npz",
        logits=pd_logits,
        targets=pd_targets,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
