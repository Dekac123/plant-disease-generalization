"""Materialise the Hugging Face parquet datasets into an on-disk image tree.

The `datasets` image column decodes through PIL on every access, so training
straight off parquet pays the decode cost inside the training loop every epoch.
A plain folder tree lets ImageFolder decode in worker processes instead.

All sources are parquet-only; `trust_remote_code` is never set, so nothing from
the Hub executes locally.
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image

from labels import (
    BACKGROUND_CLASS,
    PLANTDOC_TO_PLANTVILLAGE,
    assert_mapping_targets_exist,
    normalize_mirror_class,
)

PLANTVILLAGE_REPO = "Project-AgML/plant_village_classification"
PLANTDOC_REPO = "Project-AgML/plant_doc_classification"
# An independently packaged PlantVillage derivative, used as a control.
MIRROR_REPO = "Hemg/new-plant-diseases-dataset"

# Stored at 256px on the short side, random-cropped to 224 during training.
STORE_SIZE = 256

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def _safe_name(name: str) -> str:
    """Make a class name usable as a directory name on Windows."""
    out = name.strip().replace(" ", "_")
    for ch in '<>:"/\\|?*':
        out = out.replace(ch, "-")
    return out


def _write_image(raw: bytes, dest: Path) -> bool:
    """Write one image to `dest`, downscaling if larger than STORE_SIZE.

    Returns False on unreadable input so a corrupt row is counted and skipped
    rather than aborting a 55k-image run.
    """
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return False

    if img.mode != "RGB":
        img = img.convert("RGB")

    short_side = min(img.size)
    if short_side > STORE_SIZE:
        scale = STORE_SIZE / short_side
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new_size, Image.BILINEAR)

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="JPEG", quality=92)
    return True


def _write_all(jobs, workers: int, total: int) -> int:
    """Run `(bytes, dest)` write jobs through a thread pool.

    Consumed lazily in chunks; materialising all 55k jobs up front would hold
    the better part of a gigabyte of encoded images in RAM.
    """
    failures = 0
    done = 0
    chunk_size = max(workers * 32, 512)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        it = iter(jobs)
        while True:
            chunk = []
            for _ in range(chunk_size):
                try:
                    chunk.append(next(it))
                except StopIteration:
                    break
            if not chunk:
                break

            for ok in pool.map(lambda job: _write_image(*job), chunk):
                if not ok:
                    failures += 1
            done += len(chunk)
            print(f"  {done}/{total} written", flush=True)

    return failures


def _stratified_split(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Assign every row to train/val/test, stratified within each class.

    PlantVillage ranges from ~5000 images per class down to ~150, and a uniform
    random split would leave the rare classes with too few test images for a
    stable per-class F1.
    """
    assignment = np.empty(len(labels), dtype=object)

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(round(n * SPLIT_FRACTIONS["train"]))
        n_val = int(round(n * SPLIT_FRACTIONS["val"]))
        # Remainder goes to test so the parts always sum to n.
        assignment[idx[:n_train]] = "train"
        assignment[idx[n_train : n_train + n_val]] = "val"
        assignment[idx[n_train + n_val :]] = "test"

    return assignment


def prepare_plantvillage(out_root: Path, workers: int) -> dict:
    print(f"Loading {PLANTVILLAGE_REPO} ...")
    ds = load_dataset(PLANTVILLAGE_REPO, split="train")
    class_names = ds.features["label"].names
    print(f"  {len(ds)} images, {len(class_names)} classes")

    labels = np.array(ds["label"])
    rng = np.random.default_rng(SEED)
    assignment = _stratified_split(labels, rng)

    # decode=False gives the original encoded bytes, skipping a decode/encode
    # round trip for images already at the target size.
    ds_raw = ds.cast_column("image", HFImage(decode=False))

    counts: Counter = Counter()

    def jobs():
        for i, row in enumerate(ds_raw):
            split = assignment[i]
            cls = _safe_name(class_names[row["label"]])
            counts[(split, cls)] += 1
            yield row["image"]["bytes"], out_root / split / cls / f"{i:06d}.jpg"

    failures = _write_all(jobs(), workers, len(ds_raw))
    print(f"  done ({failures} unreadable images skipped)")
    return {
        "classes": class_names,
        "splits": {
            s: sum(v for (sp, _), v in counts.items() if sp == s)
            for s in SPLIT_FRACTIONS
        },
        "failures": failures,
    }


def prepare_plantdoc(out_root: Path, workers: int, plantvillage_classes: list[str]) -> dict:
    """Write PlantDoc under PlantVillage class names.

    Remapping at write time lets the cross-dataset evaluation reuse the same
    ImageFolder machinery as the in-domain test set.
    """
    print(f"Loading {PLANTDOC_REPO} ...")
    ds = load_dataset(PLANTDOC_REPO, split="train")
    class_names = ds.features["label"].names
    print(f"  {len(ds)} images, {len(class_names)} classes")

    unmapped = [c for c in class_names if c not in PLANTDOC_TO_PLANTVILLAGE]
    if unmapped:
        raise SystemExit(
            f"PlantDoc classes with no PlantVillage counterpart: {unmapped}\n"
            "Update PLANTDOC_TO_PLANTVILLAGE in labels.py before continuing."
        )

    # Both directions must line up: every PlantDoc class needs a target, and
    # every target must be a class the model was trained on.
    assert_mapping_targets_exist(plantvillage_classes)

    ds_raw = ds.cast_column("image", HFImage(decode=False))

    counts: Counter = Counter()

    def jobs():
        for i, row in enumerate(ds_raw):
            cls = _safe_name(PLANTDOC_TO_PLANTVILLAGE[class_names[row["label"]]])
            counts[cls] += 1
            yield row["image"]["bytes"], out_root / cls / f"{i:05d}.jpg"

    failures = _write_all(jobs(), workers, len(ds_raw))
    print(f"  done ({failures} unreadable images skipped)")
    return {
        "source_classes": class_names,
        "mapped_classes": sorted(counts),
        "per_class_counts": dict(sorted(counts.items())),
        "failures": failures,
    }


def prepare_mirror(
    out_root: Path, workers: int, plantvillage_classes: list[str], sample: int
) -> dict:
    """Write a sample of an independent PlantVillage mirror as a control set.

    A pipeline check, not a generalization test: these images derive from the
    same source photographs as the training data, so a high score only shows
    the label mapping and evaluation code are sound.
    """
    print(f"Loading {MIRROR_REPO} ...")
    ds = load_dataset(MIRROR_REPO, split="train")
    class_names = ds.features["label"].names
    print(f"  {len(ds)} images, {len(class_names)} classes")

    known = set(plantvillage_classes)
    renamed = {}
    unknown = []
    for name in class_names:
        mapped = normalize_mirror_class(name)
        if mapped not in known:
            unknown.append(f"{name} -> {mapped}")
        renamed[name] = mapped

    if unknown:
        raise SystemExit(
            "mirror classes that do not normalise onto the PlantVillage "
            f"vocabulary:\n  " + "\n  ".join(unknown)
        )

    rng = np.random.default_rng(SEED)
    keep = rng.choice(len(ds), size=min(sample, len(ds)), replace=False)
    keep_set = set(keep.tolist())
    print(f"  sampling {len(keep_set)} images")

    ds_raw = ds.cast_column("image", HFImage(decode=False))
    counts: Counter = Counter()

    def jobs():
        for i, row in enumerate(ds_raw):
            if i not in keep_set:
                continue
            cls = _safe_name(renamed[class_names[row["label"]]])
            counts[cls] += 1
            yield row["image"]["bytes"], out_root / cls / f"{i:06d}.jpg"

    failures = _write_all(jobs(), workers, len(keep_set))
    print(f"  done ({failures} unreadable images skipped)")

    return {
        "source_repo": MIRROR_REPO,
        "source_classes": class_names,
        "renames": {k: v for k, v in renamed.items() if k != v},
        "per_class_counts": dict(sorted(counts.items())),
        "sampled": len(keep_set),
        "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Threads used for resize+save (I/O bound, so oversubscribing is fine).",
    )
    ap.add_argument(
        "--only",
        choices=["all", "plantvillage", "plantdoc", "mirror"],
        default="all",
        help="Re-prepare only one dataset, e.g. after fixing the label mapping.",
    )
    ap.add_argument(
        "--mirror-sample",
        type=int,
        default=8000,
        help="How many control images to keep (the full mirror is 70k).",
    )
    args = ap.parse_args()

    pv_root = args.data_root / "plantvillage"
    pd_root = args.data_root / "plantdoc"
    mirror_root = args.data_root / "plantvillage_mirror"
    manifest_path = args.data_root / "manifest.json"

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.only in {"all", "plantvillage"}:
        manifest["plantvillage"] = prepare_plantvillage(pv_root, args.workers)

    if args.only in {"all", "plantdoc", "mirror"}:
        # Read the vocabulary off disk, not from Hub metadata: the directory
        # names are what training and evaluation actually use.
        train_dir = pv_root / "train"
        if not train_dir.is_dir():
            raise SystemExit(f"{train_dir} missing; prepare PlantVillage first.")
        pv_classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())

    if args.only in {"all", "plantdoc"}:
        manifest["plantdoc"] = prepare_plantdoc(pd_root, args.workers, pv_classes)

    if args.only in {"all", "mirror"}:
        manifest["plantvillage_mirror"] = prepare_mirror(
            mirror_root, args.workers, pv_classes, args.mirror_sample
        )

    manifest.update(
        {"store_size": STORE_SIZE, "seed": SEED, "background_class": BACKGROUND_CLASS}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {manifest_path}")
    if "plantvillage" in manifest:
        print("PlantVillage splits:", manifest["plantvillage"]["splits"])
    if "plantdoc" in manifest:
        print("PlantDoc images:", sum(manifest["plantdoc"]["per_class_counts"].values()))


if __name__ == "__main__":
    main()
