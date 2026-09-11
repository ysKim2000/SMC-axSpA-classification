"""Aggregate MR acquisition metadata for the final 291-patient cohort (manuscript revision)."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

COHORT_ROOT = Path("bbox_detection/dataset_oneclass_holdout/images")
DICOM_ROOT = Path("data/axSpA_bbox_dataset")
OUT_DIR = Path("bbox_detection/result/dicom_metadata")

TAGS = [
    "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "SoftwareVersions",
    "ProtocolName", "SeriesDescription", "ScanningSequence", "SequenceVariant", "ScanOptions",
    "MRAcquisitionType", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle",
    "EchoTrainLength", "NumberOfAverages", "PixelBandwidth", "SliceThickness",
    "SpacingBetweenSlices", "PixelSpacing", "Rows", "Columns", "AcquisitionMatrix",
    "ReconstructionDiameter", "PercentPhaseFieldOfView", "BodyPartExamined",
    "ReceiveCoilName", "ImagedNucleus", "StudyDate", "InstitutionName",
]
NUMERIC = [
    "MagneticFieldStrength", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle",
    "EchoTrainLength", "NumberOfAverages", "PixelBandwidth", "SliceThickness",
    "SpacingBetweenSlices", "ReconstructionDiameter", "PercentPhaseFieldOfView",
    "Rows", "Columns", "InPlaneResolution",
]


def cohort_ids():
    ids = set()
    for split in ("train", "val", "test"):
        for f in (COHORT_ROOT / split).iterdir():
            ids.add(f.stem.rsplit("_", 1)[0])
    return sorted(ids)


# Patient <PATIENT_ID>: the series filed under T1/ is SeriesNumber 706 ("sT2 mDIXON COR FS", the water
# image of a T2 Dixon acquisition). The true FS T1 for this study is SeriesNumber 802.
SERIES_OVERRIDE = {
    ("<PATIENT_ID>", "T1"): Path("data/override/<patient>/T1_series802"),
}


def read_series(pid, seq):
    """One representative slice per patient/sequence (metadata is series-constant)."""
    seq_dir = SERIES_OVERRIDE.get((pid, seq), DICOM_ROOT / pid / seq)
    if not seq_dir.is_dir():
        return None
    for dcm in sorted(seq_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
        except Exception:
            continue
        row = {"patient_id": pid, "sequence": seq, "n_slices": len(list(seq_dir.glob("*.dcm")))}
        for tag in TAGS:
            v = getattr(ds, tag, None)
            if tag == "PixelSpacing" and v is not None:
                row["InPlaneResolution"] = round(float(v[0]), 4)
                v = f"{float(v[0]):.4f}\\{float(v[1]):.4f}"
            elif tag == "AcquisitionMatrix" and v is not None:
                v = "x".join(str(int(x)) for x in v if int(x) != 0)
            elif isinstance(v, pydicom.multival.MultiValue):
                v = "/".join(str(x) for x in v)
            elif v is not None:
                v = float(v) if tag in NUMERIC else str(v)
            row[tag] = v
        return row
    return None


def fmt(series, unit="", nd=1):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "n/a"
    if s.nunique() == 1:
        return f"{s.iloc[0]:.{nd}f}{unit}"
    return (f"median {s.median():.{nd}f}{unit} "
            f"(range {s.min():.{nd}f}–{s.max():.{nd}f})")


def describe(df, seq):
    d = df[df.sequence == seq]
    if d.empty:
        return {}
    return {
        "n_patients": int(d.patient_id.nunique()),
        "scanners": {f"{a} {b}": int(c) for (a, b), c in
                     d.groupby(["Manufacturer", "ManufacturerModelName"]).size().items()},
        "field_strength_T": {str(k): int(v) for k, v in
                             d.MagneticFieldStrength.value_counts().items()},
        "software_versions": {str(k): int(v) for k, v in
                              d.SoftwareVersions.value_counts().items()},
        "protocol_names": {str(k): int(v) for k, v in d.ProtocolName.value_counts().items()},
        "scanning_sequence": {str(k): int(v) for k, v in d.ScanningSequence.value_counts().items()},
        "sequence_variant": {str(k): int(v) for k, v in d.SequenceVariant.value_counts().items()},
        "scan_options": {str(k): int(v) for k, v in d.ScanOptions.value_counts().items()},
        "acquisition_type": {str(k): int(v) for k, v in d.MRAcquisitionType.value_counts().items()},
        "TR_ms": fmt(d.RepetitionTime),
        "TE_ms": fmt(d.EchoTime),
        "TI_ms": fmt(d.InversionTime),
        "flip_angle_deg": fmt(d.FlipAngle, nd=0),
        "echo_train_length": fmt(d.EchoTrainLength, nd=0),
        "NSA": fmt(d.NumberOfAverages, nd=0),
        "pixel_bandwidth_Hz": fmt(d.PixelBandwidth, nd=0),
        "slice_thickness_mm": fmt(d.SliceThickness),
        "slice_spacing_mm": fmt(d.SpacingBetweenSlices),
        "in_plane_resolution_mm": fmt(d.InPlaneResolution, nd=3),
        "FOV_mm": fmt(d.ReconstructionDiameter, nd=0),
        "matrix": {str(k): int(v) for k, v in d.AcquisitionMatrix.value_counts().items()},
        "reconstructed_size": {f"{r:.0f}x{c:.0f}": int(n) for (r, c), n in
                               d.groupby(["Rows", "Columns"]).size().items()},
        "slices_per_patient": fmt(d.n_slices, nd=0),
        "coil": {str(k): int(v) for k, v in d.ReceiveCoilName.value_counts().items()},
        "body_part": {str(k): int(v) for k, v in d.BodyPartExamined.value_counts().items()},
        "institution": {str(k): int(v) for k, v in d.InstitutionName.value_counts().items()},
    }


ids = cohort_ids()
rows = [r for pid in ids for seq in ("T1", "T2") if (r := read_series(pid, seq))]
df = pd.DataFrame(rows)

OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_DIR / "per_patient_metadata.csv", index=False)

summary = {
    "cohort_n_patients": len(ids),
    "patients_with_T1": int(df[df.sequence == "T1"].patient_id.nunique()),
    "patients_with_T2": int(df[df.sequence == "T2"].patient_id.nunique()),
    "missing_dicom": sorted(set(ids) - set(df.patient_id)),
    "T1": describe(df, "T1"),
    "T2": describe(df, "T2"),
}
(OUT_DIR / "acquisition_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print(f"cohort: {len(ids)} patients | T1 series: {summary['patients_with_T1']} | "
      f"T2 series: {summary['patients_with_T2']}")
if summary["missing_dicom"]:
    print("MISSING DICOM:", summary["missing_dicom"])
for seq in ("T1", "T2"):
    print(f"\n===== {seq} =====")
    for k, v in summary[seq].items():
        print(f"  {k}: {v}")
print(f"\nsaved -> {OUT_DIR}")
