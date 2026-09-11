# Reproduction notes

This repository contains source code only. Neither the imaging data (IRB-restricted) nor the trained
checkpoints (institutional network security policy) can be distributed, so the published metrics cannot
be reproduced from this repository alone. These notes describe the inputs the scripts expect so the
pipeline can be run on an equivalent local dataset.

## Expected data layout

```
data/
├── <dicom_root>/                  one directory per patient
│   └── <patient_id>/
│       ├── T1/    *.dcm           oblique coronal FS T1
│       └── T2/    *.dcm           oblique coronal STIR
├── reference/
│   ├── reference_stir.dcm         histogram-matching reference (STIR)
│   └── reference_fst1.dcm         histogram-matching reference (FS T1)
└── labels/
    ├── bbox_label.csv
    ├── bme_label.csv
    ├── combined_bme_labels.csv
    └── axspa_label.csv
```

Patient identifiers are zero-padded to 7 characters in the label files and the DICOM directory names.

## Annotation file schemas

**`bbox_label.csv`** — SIJ bounding boxes, one row per annotated slice.

| Column | Type | Description |
|---|---|---|
| `patient_id` | int/str | patient identifier |
| `slice_number` | int | slice index within the series |
| `slice_name` | str | source slice label |
| `left_box` | str | `[x1, y1, x2, y2]` in source-image pixels |
| `right_box` | str | `[x1, y1, x2, y2]` in source-image pixels |

Boxes are stored as bracketed strings and parsed with `ast.literal_eval`. Both joints are mapped to a
single `SIJ` class for detection.

**`bme_label.csv`** — patient-level BME per side (`left_bme`, `right_bme`; 0/1).

**`combined_bme_labels.csv`** — slice-level BME (`patient_id`, `slice_number`, `bme`; 0/1). A slice is
positive if either joint shows edema.

**`axspa_label.csv`** — patient-level reference standard (`patient_id`, `FDA`, `DGOL`). `FDA` is the
treating rheumatologist's clinical diagnosis of axSpA (0/1) and is the target for Stage 2.

## Pipeline

Paths are set at the top of each script; edit them to match the layout above before running.

### Stage 1a — SIJ localization

```bash
python src/stage1_sij_localization/prepare_dataset.py
python src/stage1_sij_localization/train.py
python src/stage1_sij_localization/evaluate.py
```

`prepare_dataset.py` reads the DICOM tree and `bbox_label.csv`, applies preprocessing, performs the
**patient-wise** 80/20 split (`random.seed(42)`), and writes a YOLO-format dataset with `data.yaml`.
Grouping by patient before splitting is what prevents slices of the same patient from appearing in
more than one subset.

`evaluate.py` reports mean IoU and mAP with bootstrap 95% confidence intervals. For each image it
selects the highest-confidence box on each side of the image midline and matches them against the
ground truth. Evaluate at the training resolution: inferring at a different `imgsz` degrades IoU.

### Stage 1b — BME classification

```bash
python src/stage1_bme_classification/train.py
python src/stage1_bme_classification/evaluate.py
python src/stage1_bme_classification/gradcam.py     # optional
```

ROIs are cropped from the preprocessed slice using the bounding boxes, resized to 512 × 512, and the
right-side crop is mirrored so both joints share an orientation. Each ROI is one sample, so a slice
contributes two.

### Stage 2 — patient-level axSpA classification

```bash
python src/stage2_axspa_classification/train.py
python src/stage2_axspa_classification/evaluate.py
python src/stage2_axspa_classification/gradcam.py   # optional
```

Stage 2 consumes the binary slice-level BME predictions from Stage 1 alongside the FS T1 and STIR
images, so Stage 1b must be run first. Slices are padded or trimmed to 12 per patient.

## Reproducibility caveats

- **Split membership.** The patient-wise split is seeded (`random.seed(42)`), but membership also
  depends on the set of patient directories present, so a different cohort yields a different split.
- **Checkpoint selection.** Stage 1 uses the Ultralytics validation fitness criterion; Stages 1b and 2
  select the lowest validation loss. Runs therefore stop at different epochs depending on the data.
- **Non-determinism.** cuDNN autotuning, multi-GPU reduction order, and dataloader worker scheduling
  are not pinned; metrics vary slightly between identical runs.
- **Hyperparameters were selected empirically** on the training and validation sets, without automated
  search. The values in `configs/training_config.yaml` are those used for the published results and are
  not guaranteed to be optimal on other cohorts.
- **Inference resolution must match training resolution** for the detector; this materially affects IoU.

## Environment

Python 3.8, PyTorch 2.4.1 (CUDA 11.8), Ultralytics 8.3.123, NVIDIA RTX A6000. Exact package versions are
pinned in `requirements.txt`.
