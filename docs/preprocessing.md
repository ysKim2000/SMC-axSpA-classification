# Preprocessing

Verified against source code:
`src/stage1_sij_localization/prepare_dataset.py` (SIJ localization),
`src/stage1_bme_classification/dataset.py` + `evaluate.py` (BME classification),
`src/stage2_axspa_classification/train.py` (patient-level axSpA classification).

---

## Summary (as reported in the paper)

> FS T1 and STIR images were intensity-normalized and processed using histogram matching and
> contrast-limited adaptive histogram equalization (CLAHE) to reduce scanner-related intensity variation
> and enhance lesion visibility (Supplementary Figure S2). Each DICOM slice was first converted to a
> single-channel image and linearly rescaled to an 8-bit range using its own minimum and maximum
> intensities (min–max normalization per slice); for the patient-level classification stage the stored
> DICOM window centre and width were applied before rescaling. Each image was then matched to the
> intensity histogram of a fixed reference slice from the training set
> (`skimage.exposure.match_histograms`), so that no test-set image contributed to the reference. A single
> mid-joint reference slice acquired on the Siemens MAGNETOM Skyra was used per sequence and per stage:
> one STIR reference for SIJ localization and BME classification, and sequence-specific FS T1 and STIR
> references for patient-level classification. CLAHE was then applied. For SIJ localization CLAHE was
> computed on the normalized image with a clip limit of 0.01 and a 64 × 64-pixel kernel
> (`skimage.exposure.equalize_adapthist`); for BME and patient-level classification CLAHE was computed on
> the 8-bit image with an 8 × 8 tile grid and a clip limit of 2.0 and 4.0, respectively
> (`cv2.createCLAHE`). Reconstructed matrix size varied across examinations, and images were resized
> according to the input requirements of each model stage: 512 × 512 pixels for SIJ localization and BME
> classification and 256 × 256 pixels for patient-level axSpA classification (bilinear interpolation).
> For BME classification, the left and right sacroiliac joint regions were cropped from the
> preprocessed slice using the bounding boxes and each crop was resized to 512 × 512 pixels, with the
> right-side crop mirrored horizontally so that both joints shared a common orientation. Images were
> replicated across three channels to match the ImageNet-pretrained backbones. For patient-level
> classification each slice was additionally rescaled to [0, 1] by its own minimum and maximum after
> conversion to a tensor and then normalized with the ImageNet channel statistics
> (mean 0.485/0.456/0.406, SD 0.229/0.224/0.225). All preprocessing parameters were fixed before
> evaluation and applied consistently across the training, validation, and test sets.

---

## Stage-by-stage parameters

| Step | SIJ localization | BME classification | Patient-level axSpA |
|---|---|---|---|
| DICOM windowing | not applied | not applied | `WindowCenter`/`WindowWidth` applied |
| Intensity normalization | per-slice min–max → 8-bit | per-slice min–max → 8-bit | per-slice min–max → 8-bit |
| Order of operations | resize → histogram match → rescale to [0,1] → CLAHE | resize → histogram match → CLAHE → crop | window → histogram match → CLAHE → resize |
| Histogram-matching reference | `reference_stir.dcm` (STIR) | `reference_stir.dcm` (STIR) (Skyra, 640×640) | T1: `reference_fst1.dcm` (FS T1); T2: `reference_stir.dcm` (STIR) (both Skyra) |
| Reference resized to | 512 × 512 | 512 × 512 | 256 × 256 |
| CLAHE implementation | `skimage.exposure.equalize_adapthist` | `cv2.createCLAHE` | `cv2.createCLAHE` |
| CLAHE clip limit | 0.01 (normalized units) | 2.0 | 4.0 |
| CLAHE kernel / tile grid | 64 × 64 px kernel | 8 × 8 tiles | 8 × 8 tiles |
| Final input size | 512 × 512 | 512 × 512 per joint crop | 256 × 256 |
| Channels | 1 (grayscale JPEG) | 3 (replicated) | 3 (replicated) |
| Additional steps | — | left/right ROI crop; right crop mirrored | per-slice rescale to [0,1] → ImageNet mean/SD normalization |

## Implementation notes

1. **The two CLAHE clip-limit numbers are not interchangeable.** The SIJ stage uses
   scikit-image, whose `clip_limit` is a normalized value in [0, 1] (0.01 used here), whereas the
   classification stages use OpenCV, whose `clipLimit` is an unnormalized histogram-count threshold
   (2.0 and 4.0). The library must therefore be named whenever these values are quoted together.
2. **No test data entered the reference.** All histogram-matching references are single slices from
   training-set patients (P-B, P-C, P-D), so the transform is fitted on training data only.
3. **Reference scanner.** All references come from the Siemens MAGNETOM Skyra, the most frequent platform
   in the cohort (143/291, 49.1%); histogram matching therefore maps Philips examinations onto the
   Siemens intensity distribution. This is the mechanism by which scanner-related intensity variation is reduced.
4. **Augmentation is separate.** This document covers preprocessing only. Training-time augmentation
   (mosaic, RandAugment, translation/scaling for detection; flips and affine transforms for
   classification) was applied to the training split only; see `configs/training_config.yaml`.
