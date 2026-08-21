"""Datasets, transforms and loaders."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Applied to the from-scratch model too, so both architectures see identically
# normalised inputs.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224


def train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Augmentation for training.

    Deliberately aggressive: PlantVillage images are uniform enough that a
    network will otherwise latch onto lighting and background regularities.
    Flips and rotations are safe because a leaf has no canonical orientation.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )


def eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic transform for validation, test and cross-dataset eval."""
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo `Normalize`, for turning a model input back into a viewable image."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


class FixedVocabImageFolder(ImageFolder):
    """An ImageFolder pinned to an externally supplied class vocabulary.

    Directories map to their index in `vocabulary`, not to their position in
    the local sorted listing. PlantDoc covers only 28 of the 39 classes, and a
    stock ImageFolder would build a fresh 0..27 index where every predicted id
    means something different than it did during training.
    """

    def __init__(self, root: str | Path, vocabulary: list[str], **kwargs) -> None:
        self._vocabulary = list(vocabulary)
        super().__init__(str(root), **kwargs)
        # find_classes only returns the directories present; restore the full
        # vocabulary so dataset.classes[label] is always valid.
        self.classes = list(self._vocabulary)

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        vocab_index = {name: i for i, name in enumerate(self._vocabulary)}
        present = sorted(d.name for d in Path(directory).iterdir() if d.is_dir())

        unknown = [d for d in present if d not in vocab_index]
        if unknown:
            raise ValueError(
                f"{directory} contains directories absent from the class "
                f"vocabulary: {unknown}"
            )

        return present, {name: vocab_index[name] for name in present}


def read_vocabulary(plantvillage_root: Path) -> list[str]:
    """The canonical class list: the sorted training-split directory names."""
    train_dir = Path(plantvillage_root) / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"{train_dir} not found -- run prepare_data.py before training."
        )
    return sorted(d.name for d in train_dir.iterdir() if d.is_dir())


def build_loaders(
    plantvillage_root: Path,
    batch_size: int,
    num_workers: int,
    image_size: int = IMAGE_SIZE,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> tuple[dict[str, DataLoader], list[str]]:
    plantvillage_root = Path(plantvillage_root)
    vocabulary = read_vocabulary(plantvillage_root)

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        is_train = split == "train"
        dataset = FixedVocabImageFolder(
            plantvillage_root / split,
            vocabulary,
            transform=train_transform(image_size) if is_train else eval_transform(image_size),
        )
        loaders[split] = _make_loader(dataset, batch_size, num_workers, shuffle=is_train)

    return loaders, vocabulary


def build_crosseval_loader(
    plantdoc_root: Path,
    vocabulary: list[str],
    batch_size: int,
    num_workers: int,
    image_size: int = IMAGE_SIZE,
) -> DataLoader:
    dataset = FixedVocabImageFolder(
        Path(plantdoc_root), vocabulary, transform=eval_transform(image_size)
    )
    return _make_loader(dataset, batch_size, num_workers, shuffle=False)


def _make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        # Workers are process-spawned on Windows; keeping them alive avoids
        # paying startup cost every epoch.
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )
