"""Figure text in English and Serbian.

Charts are rendered twice, once per language, so the Serbian report does not
end up with English axis labels. Class names are deliberately left untranslated
-- they are dataset identifiers, and renaming them would break the link between
a figure and the JSON it came from.
"""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "train": "train",
        "validation": "validation",
        "epoch": "epoch",
        "loss_axis": "cross-entropy loss",
        "loss_panel": "Loss",
        "accuracy": "accuracy",
        "macro_f1": "macro F1",
        "val_score_axis": "validation score",
        "val_panel": "Validation quality",
        "training_history": "Training history",
        "predicted": "predicted",
        "true": "true",
        "cm_colorbar": "fraction of true class",
        "lab_series": "PlantVillage (lab)",
        "field_series": "PlantDoc (field)",
        "score_axis": "score",
        "f1_axis": "F1 score",
        "domain_gap_title": "Same model, same 28 classes — lab photographs vs field photographs",
        "per_class_title": "Per-class F1, ordered by field performance",
        "comparison_title": "Benchmark rank does not predict field rank",
        "cm_lab_title": "PlantVillage held-out test — row-normalised confusion matrix",
        "cm_field_title": "PlantDoc (field photographs) — row-normalised confusion matrix",
        "confidence_axis": "predicted confidence",
        "observed_axis": "observed accuracy",
        "perfect_calibration": "perfect calibration",
        "observed": "observed",
        "calibration_title": "Reliability diagrams — confidence vs correctness",
        "lab_panel": "PlantVillage (lab)",
        "field_panel": "PlantDoc (field)",
        "ece": "ECE",
        "mean_confidence": "mean confidence",
        "m_accuracy": "Accuracy",
        "m_balanced": "Balanced\naccuracy",
        "m_macro_f1": "Macro F1",
        "m_top5": "Top-5\naccuracy",
        "m_crop": "Crop\nidentification",
        "m_binary": "Healthy vs\ndiseased",
        "gradcam_lab": "Grad-CAM — PlantVillage held-out test (lab conditions)",
        "gradcam_field": "Grad-CAM — PlantDoc (real field photographs)",
        "gc_true": "true",
        "gc_pred": "pred",
    },
    "sr": {
        "train": "obuka",
        "validation": "validacija",
        "epoch": "epoha",
        "loss_axis": "gubitak (unakrsna entropija)",
        "loss_panel": "Funkcija gubitka",
        "accuracy": "tačnost",
        "macro_f1": "makro F1",
        "val_score_axis": "vrednost na validaciji",
        "val_panel": "Kvalitet na validaciji",
        "training_history": "Tok obuke",
        "predicted": "predviđeno",
        "true": "stvarno",
        "cm_colorbar": "udeo stvarne klase",
        "lab_series": "PlantVillage (laboratorija)",
        "field_series": "PlantDoc (teren)",
        "score_axis": "vrednost",
        "f1_axis": "F1 mera",
        "domain_gap_title": "Isti model, istih 28 klasa — laboratorijski naspram terenskih snimaka",
        "per_class_title": "F1 mera po klasama, sortirano po uspehu na terenu",
        "comparison_title": "Rang na referentnom skupu ne predviđa rang na terenu",
        "cm_lab_title": "PlantVillage, test skup — matrica konfuzije (normalizovano po redovima)",
        "cm_field_title": "PlantDoc (terenski snimci) — matrica konfuzije (normalizovano po redovima)",
        "confidence_axis": "prijavljena pouzdanost",
        "observed_axis": "izmerena tačnost",
        "perfect_calibration": "savršena kalibracija",
        "observed": "izmereno",
        "calibration_title": "Dijagrami pouzdanosti — prijavljena pouzdanost naspram tačnosti",
        "lab_panel": "PlantVillage (laboratorija)",
        "field_panel": "PlantDoc (teren)",
        "ece": "ECE",
        "mean_confidence": "prosečna pouzdanost",
        "m_accuracy": "Tačnost",
        "m_balanced": "Balansirana\ntačnost",
        "m_macro_f1": "Makro F1",
        "m_top5": "Top-5\ntačnost",
        "m_crop": "Prepoznavanje\nbiljke",
        "m_binary": "Zdravo /\nbolesno",
        "gradcam_lab": "Grad-CAM — PlantVillage, test skup (laboratorijski uslovi)",
        "gradcam_field": "Grad-CAM — PlantDoc (stvarni terenski snimci)",
        "gc_true": "stvarno",
        "gc_pred": "predviđeno",
    },
}


def get(lang: str) -> dict[str, str]:
    if lang not in STRINGS:
        raise ValueError(f"no strings for language '{lang}'; have {sorted(STRINGS)}")
    return STRINGS[lang]
