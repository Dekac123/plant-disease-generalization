"""Build the report: substitute measured values, then inline the figures.

Two kinds of placeholder are resolved here.

`{{VAL:path:format}}` pulls a number straight out of the JSON written by
train.py and cross_eval.py. Every figure quoted in the report goes through it,
so the prose cannot drift away from the run that produced it -- retyping a
metric by hand is exactly how a report ends up claiming something the data no
longer says.

`{{FIG:name}}` inlines a rendered chart as a data URI, since the published page
must be self-contained. Grad-CAM grids are re-encoded as JPEG; they are an
order of magnitude larger than the charts as PNG and the loss is invisible
under a saliency overlay.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

from PIL import Image

# Figures that are photographs, and therefore worth JPEG-encoding.
PHOTOGRAPHIC = {"gradcam_plantvillage", "gradcam_plantdoc"}

JPEG_QUALITY = 82
MAX_WIDTH = 2000


def encode(path: Path) -> str:
    """Return a data URI for one figure, compressing photographs as JPEG."""
    image = Image.open(path)
    image.load()

    if image.width > MAX_WIDTH:
        scale = MAX_WIDTH / image.width
        image = image.resize(
            (MAX_WIDTH, max(1, round(image.height * scale))), Image.LANCZOS
        )

    buffer = io.BytesIO()
    if path.stem in PHOTOGRAPHIC:
        image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        mime = "image/jpeg"
    else:
        image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    print(f"  {path.stem:<28} {len(encoded) / 1024:8.0f} KB ({mime})")
    return f"data:{mime};base64,{encoded}"


def build_namespace(run_dir: Path) -> dict:
    """Flatten the run's JSON into the dotted names the template refers to."""
    test = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    cross = json.loads((run_dir / "cross_eval.json").read_text(encoding="utf-8"))
    results = cross["results"]

    namespace = {
        "test": test["test"],
        "meta": {
            "best_epoch": test["best_epoch"],
            "train_minutes": test["train_seconds"] / 60,
            "parameters": test["trainable_parameters"],
            "n_classes": len(test["class_names"]),
        },
        "lab": results["plantvillage_test_shared"],
        "field": results["plantdoc_open"],
        "restricted": results["plantdoc_restricted"],
        "gap": cross["generalization_gap"],
    }
    if "plantvillage_mirror" in results:
        namespace["control"] = results["plantvillage_mirror"]

    # Calibration lives one level down; expose it as lab_cal / field_cal so the
    # template does not need nested paths.
    for key in ("lab", "field", "control"):
        block = namespace.get(key)
        if block and "calibration" in block:
            namespace[f"{key}_cal"] = block["calibration"]

    return namespace


def _lookup(namespace: dict, path: str):
    value = namespace
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _format(value, spec: str) -> str:
    if spec == "pct":
        return f"{value * 100:.2f}%"
    if spec == "pct1":
        return f"{value * 100:.1f}%"
    if spec == "pct0":
        return f"{value * 100:.0f}%"
    if spec == "f4":
        return f"{value:.4f}"
    if spec == "f3":
        return f"{value:.3f}"
    if spec == "f1":
        return f"{value:.1f}"
    if spec == "int":
        return f"{int(value):,}"
    if spec == "raw":
        return str(value)
    raise ValueError(f"unknown format '{spec}'")


def substitute_values(html: str, namespace: dict) -> tuple[str, int]:
    pattern = re.compile(r"\{\{VAL:([a-z0-9_.]+):([a-z0-9]+)\}\}")
    missing: list[str] = []

    def replace(match: re.Match) -> str:
        path, spec = match.group(1), match.group(2)
        try:
            return _format(_lookup(namespace, path), spec)
        except KeyError:
            missing.append(path)
            return match.group(0)

    html, count = pattern.subn(replace, html)
    if missing:
        raise SystemExit(f"template refers to values not in the run JSON: {sorted(set(missing))}")
    return html, count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--figures", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    html = args.template.read_text(encoding="utf-8")

    html, n_values = substitute_values(html, build_namespace(args.run_dir))
    print(f"substituted {n_values} measured values")

    placeholders = sorted(set(re.findall(r"\{\{FIG:([a-z0-9_]+)\}\}", html)))
    print(f"embedding {len(placeholders)} figures")

    missing = []
    for name in placeholders:
        path = args.figures / f"{name}.png"
        if not path.exists():
            missing.append(name)
            continue
        html = html.replace(f"{{{{FIG:{name}}}}}", encode(path))

    if missing:
        raise SystemExit(f"missing figures: {missing}")

    args.out.write_text(html, encoding="utf-8")
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"\nwrote {args.out} ({size_mb:.1f} MB)")
    if size_mb > 16:
        raise SystemExit("report exceeds the 16 MB artifact limit")


if __name__ == "__main__":
    main()
