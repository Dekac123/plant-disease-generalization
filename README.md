# Plant disease classification — and where it breaks

A rebuild of a PlantVillage leaf-disease classifier, extended to answer a
question the usual version of this project never asks: **does a model that
scores ~99% on the benchmark actually work on a photograph taken in a field?**

Short answer: no, and not by a small margin.

## The experiment

Nearly every plant-disease project trains and tests on
[PlantVillage](https://huggingface.co/datasets/Project-AgML/plant_village_classification),
reports an accuracy in the high nineties, and stops. But every PlantVillage
image is a studio shot — a single detached leaf, centred, on a uniform
background, under even lighting. Train and test both come from that
distribution, so a high score does not establish that the model learnt anything
about plant pathology.

This project trains on PlantVillage and then evaluates the same model, unchanged,
on [PlantDoc](https://huggingface.co/datasets/Project-AgML/plant_doc_classification)
— the same diseases photographed in real fields, with cluttered backgrounds,
multiple overlapping leaves, variable light and awkward angles. All 28 PlantDoc
classes map onto PlantVillage classes, so the comparison is like-for-like.

Four evaluations are reported, because a bare in-domain vs out-of-domain
comparison would confound several different effects:

| Evaluation | What it isolates |
|---|---|
| `plantvillage_test_shared` | In-domain accuracy, restricted to the 28 shared classes — so the comparison is not contaminated by a differing label set |
| `plantdoc_open` | Field images scored over the full 39-way head; what deployment actually looks like |
| `plantdoc_restricted` | Field images with logits masked to the 28 possible classes; separates "predicted something impossible" from "genuinely confused" |
| `plantvillage_mirror` | **Control.** An independently packaged PlantVillage derivative (different uploader, different class naming) run through the identical code path |

The control matters. Without it, "the field accuracy collapsed" and "the label
mapping is broken" produce the same number, and this project hit exactly that
bug once already. Scoring 97.71% on a foreign packaging of the training
distribution shows the mapping and evaluation code work, which is what makes
the PlantDoc result attributable to domain shift.

Alongside accuracy the report includes **precision, recall and F1** (per class,
macro and weighted), balanced accuracy, Cohen's kappa, **calibration** (does the
model know when it is wrong?) and **Grad-CAM** saliency maps (what is it
actually looking at?).

## Results

| | Lab | Field | Retained |
|---|---|---|---|
| Accuracy | 0.9960 | 0.1483 | 14.9% |
| Balanced accuracy | 0.9948 | 0.1425 | 14.3% |
| Macro F1 | 0.9944 | 0.1602 | 16.1% |
| Top-5 accuracy | 1.0000 | 0.4352 | 43.5% |
| Cohen's kappa | 0.9958 | 0.1308 | 13.1% |
| Crop identification | 0.9986 | 0.2880 | 28.8% |

Control (independent PlantVillage mirror): **0.9771** accuracy.

Three findings worth more than the headline number:

- **34.9% of field photographs are classified `Background_without_leaves`** —
  the class meaning "no leaf in this image". The model learned "leaf" to mean a
  specimen isolated on a grey studio background.
- **Calibration error rises from 0.041 to 0.437.** On field images 12.7% of
  predictions claim over 90% confidence and only 22.2% of those are correct.
- **The healthy/diseased score is an artifact.** 74.15% looks usable until you
  notice 67.07% of PlantDoc is diseased, so a constant "diseased" answer scores
  67.07%. The model answers "diseased" 85.29% of the time.

## What was changed from the original approach

The starting point was a notebook that trained a from-scratch ResNet-style CNN
for one epoch and reported a single validation accuracy figure.

| | Original | This version |
|---|---|---|
| Evaluation split | validation only | proper stratified 70/15/15 train/val/test |
| Metrics | accuracy | accuracy, balanced accuracy, precision/recall/F1 (per-class + macro + weighted), Cohen's kappa, top-5, calibration/ECE |
| Augmentation | none | random resized crop, flips, rotation, colour jitter, random erasing |
| Normalisation | none | ImageNet channel statistics |
| Precision | fp32 | mixed precision (AMP) |
| Batch size | 8 | 64 |
| Model selection | last epoch | best validation macro-F1, with early stopping |
| Architectures | one scratch CNN | scratch CNN vs ImageNet-pretrained ResNet18 |
| Generalization | untested | cross-dataset evaluation on field photographs |
| Explainability | none | Grad-CAM saliency + a quantitative background-attention measure |
| Epoch time | ~90 min | ~70 s |

Selection is on **macro** F1 rather than accuracy: PlantVillage is heavily
imbalanced (~5000 images in the largest class against ~150 in the smallest), and
accuracy would keep choosing checkpoints that are strong on common classes and
indifferent to rare ones.

## A note on dataset safety

Both datasets are plain **parquet** files. Some Hugging Face datasets — including
the most popular PlantVillage mirror, `mohanty/PlantVillage` — ship a `.py`
loading script that only runs if you pass `trust_remote_code=True`, which
executes arbitrary code from the Hub on your machine. Nothing here does that;
the parquet-only sources were chosen deliberately.

## Setup

```bash
uv venv --python 3.12
```

```bash
uv pip install --torch-backend=auto torch torchvision
```

```bash
uv pip install datasets scikit-learn matplotlib seaborn pandas tqdm pillow
```

## Running it

Download all three datasets and materialise them as an on-disk image tree
(~3 GB download, ~66k images):

```bash
python src/prepare_data.py --data-root data
```

Train the pretrained ResNet18:

```bash
python src/train.py --model resnet18 --epochs 12 --batch-size 64 --run-name resnet18_pretrained
```

Train the original from-scratch architecture for comparison:

```bash
python src/train.py --model resnet9 --epochs 12 --batch-size 64 --run-name resnet9_scratch
```

Evaluate cross-dataset on field photographs:

```bash
python src/cross_eval.py --model resnet18 --run-dir results/resnet18_pretrained
```

Generate Grad-CAM saliency maps and background-attention statistics:

```bash
python src/gradcam_report.py --model resnet18 --run-dir results/resnet18_pretrained
```

Render all figures:

```bash
python src/figures.py --run-dir results/resnet18_pretrained --compare-dirs results/resnet9_scratch
```

Print every reported number, read back from the JSON:

```bash
python src/summarize.py --run-dir results/resnet18_pretrained
```

Build the written report. Measured values are substituted from the run's JSON
and the figures are inlined, so the page is self-contained and cannot quote a
number the data no longer supports:

```bash
python src/build_report.py --template results/report.template.html --figures results/resnet18_pretrained/figures --run-dir results/resnet18_pretrained --out results/report.html
```

## Layout

```
src/
  labels.py          class vocabularies + the PlantDoc -> PlantVillage bridge
  prepare_data.py    Hub parquet -> stratified on-disk image tree
  data.py            transforms, augmentation, fixed-vocabulary ImageFolder
  models.py          the original scratch CNN + pretrained baselines
  engine.py          training loop (AMP, OneCycle, early stopping) and inference
  metrics.py         accuracy / P / R / F1 / kappa / calibration
  cross_eval.py      the generalization experiment
  gradcam.py         Grad-CAM, implemented directly from the definition
  gradcam_report.py  saliency figures + background-attention statistic
  figures.py         all report figures
  summarize.py       report-ready tables read back from the JSON
  build_report.py    inlines figures into the HTML report
  check_report.py    tag-balance and theme-token checks on the report template
results/<run>/
  best.pt            best-macro-F1 checkpoint
  history.json       per-epoch training record
  test_metrics.json  full in-domain metrics
  cross_eval.json    the four-way generalization comparison
  gradcam_stats.json background-attention measurements
  figures/           rendered charts
results/
  report.html        the written report
  izvestaj.docx      two-page summary in Serbian
```

## Data sources

- PlantVillage — `Project-AgML/plant_village_classification` (55,448 images,
  39 classes), derived from Hughes & Salathé (2015).
- PlantDoc — `Project-AgML/plant_doc_classification` (2,569 images, 28 classes),
  CC-BY-SA-4.0, from Singh et al. (2020).
- Control — `Hemg/new-plant-diseases-dataset` (70,295 images, 38 classes; 8,000
  sampled), an independent packaging of the same PlantVillage source.
