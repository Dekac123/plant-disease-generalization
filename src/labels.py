"""Class vocabularies and the mapping between datasets."""

from __future__ import annotations

import re

# PlantDoc names its classes differently from PlantVillage, and a bare
# "<crop> leaf" label means a healthy leaf of that crop. Keyed by name rather
# than index because index order depends on how `datasets` sorts the
# ClassLabel feature, and drifting indices would corrupt every reported number.
PLANTDOC_TO_PLANTVILLAGE: dict[str, str] = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry___healthy",
    "Corn Gray leaf spot": "Corn___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn leaf blight": "Corn___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn___Common_rust",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites_Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

# PlantVillage's "no leaf in frame" class. A valid training target, but it has
# no PlantDoc counterpart.
BACKGROUND_CLASS = "Background_without_leaves"


def crop_of(plantvillage_class: str) -> str:
    """"Tomato___Late_blight" -> "Tomato"."""
    if "___" in plantvillage_class:
        return plantvillage_class.split("___", 1)[0]
    return plantvillage_class


def is_healthy(plantvillage_class: str) -> bool:
    return plantvillage_class.endswith("_healthy")


def normalize_mirror_class(name: str) -> str:
    """Normalise a class name from another PlantVillage packaging to ours.

    Mirrors differ only cosmetically: parenthesised crop qualifiers, spaces
    where we use underscores, a stray trailing underscore.
    """
    name = name.replace("_(maize)", "").replace("_(including_sour)", "")
    name = name.replace(" ", "_")
    return re.sub(r"_+$", "", name)


def assert_mapping_targets_exist(plantvillage_classes: list[str]) -> None:
    """Fail if any mapping target is not a real PlantVillage class.

    A typo here produces a directory no trained model has seen, and the
    cross-dataset accuracy silently reads as zero -- which looks exactly like
    the generalization failure we are trying to measure.
    """
    known = set(plantvillage_classes)
    bad = {pd: pv for pd, pv in PLANTDOC_TO_PLANTVILLAGE.items() if pv not in known}
    if bad:
        raise SystemExit(
            f"PLANTDOC_TO_PLANTVILLAGE targets that are not PlantVillage classes:\n"
            f"  {bad}\nKnown classes:\n  {sorted(known)}"
        )


def build_crosseval_mapping(
    plantvillage_classes: list[str],
    plantdoc_classes: list[str],
) -> tuple[dict[int, int], list[str]]:
    """Map PlantDoc label indices onto PlantVillage label indices.

    Returns the mapping plus any PlantDoc class names that could not be
    resolved; a non-empty second element should be treated as fatal.
    """
    pv_index = {name: i for i, name in enumerate(plantvillage_classes)}

    mapping: dict[int, int] = {}
    unmapped: list[str] = []
    for pd_idx, pd_name in enumerate(plantdoc_classes):
        pv_name = PLANTDOC_TO_PLANTVILLAGE.get(pd_name)
        if pv_name is None or pv_name not in pv_index:
            unmapped.append(pd_name)
            continue
        mapping[pd_idx] = pv_index[pv_name]

    return mapping, unmapped
