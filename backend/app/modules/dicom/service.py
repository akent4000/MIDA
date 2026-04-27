"""DICOM parsing, anonymisation, and pixel extraction.

Supported formats:
    DICOM (.dcm or auto-detected by magic bytes)
    PNG / JPEG (via Pillow, converted to grayscale)
    NIfTI (.nii / .nii.gz, middle axial slice)

All pixel arrays are returned as float32 in [0, 1], shape (H, W).
The full DICOM VOI LUT + MONOCHROME1 inversion is applied automatically —
matching the load_dicom_array() function in ml/training/datasets.py.

CLI usage (headless test):
    python -m backend.app.modules.dicom.service path/to/image.dcm
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_voi_lut

from .types import Metadata, Study

Source = bytes | Path | str

# DICOM tags scrubbed on anonymisation (subset of DICOM PS 3.15 Annex E).
_ANONYMISE_TAGS: tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "ReferringPhysicianName",
    "InstitutionName",
    "InstitutionAddress",
    "StationName",
    "StudyID",
    "AccessionNumber",
    "RequestingPhysician",
    "PerformingPhysicianName",
    "OperatorsName",
)


class DicomService:
    """Load, inspect, and anonymise medical image files.

    Methods are intentionally stateless — instantiate once and reuse freely.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source: Source) -> Study:
        """Load a medical image from a file path, URL, or raw bytes.

        Format detection order:
            1. Explicit extension (.nii/.nii.gz → NIfTI; .png/.jpg/.jpeg → PIL)
            2. DICOM magic bytes (0x00 preamble or DICM at offset 128)
            3. Fallback: PIL (handles most bitmap formats)
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            suffix = path.suffix.lower()
            raw = path.read_bytes()
        else:
            raw = bytes(source)
            suffix = ""

        if suffix in (".nii",) or suffix == ".gz":
            return self._load_nifti(raw, suffix)
        if suffix in (".png", ".jpg", ".jpeg"):
            fmt = suffix.lstrip(".").replace("jpg", "jpeg")
            return self._load_pil(raw, fmt)
        if self._is_dicom(raw):
            return self._load_dicom(raw)
        # Best-effort PIL fallback for PNG/BMP/TIFF uploaded without extension
        return self._load_pil(raw, "png")

    def anonymize(self, study: Study) -> Study:
        """Return a copy of *study* with patient-identifying fields cleared."""
        anon = Metadata(
            modality=study.metadata.modality,
            rows=study.metadata.rows,
            columns=study.metadata.columns,
            pixel_spacing=study.metadata.pixel_spacing,
        )
        return Study(
            pixel_data=study.pixel_data.copy(),
            metadata=anon,
            file_format=study.file_format,
        )

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_dicom(self, raw: bytes) -> Study:
        ds = pydicom.dcmread(io.BytesIO(raw))
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)

        if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr

        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

        px_spacing: tuple[float, float] | None = None
        if hasattr(ds, "PixelSpacing"):
            with contextlib.suppress(IndexError, TypeError, ValueError):
                px_spacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))

        def _str(tag: str) -> str | None:
            val = ds.get(tag)
            return str(val) if val is not None else None

        meta = Metadata(
            patient_id=_str("PatientID"),
            patient_name=_str("PatientName"),
            patient_sex=_str("PatientSex"),
            patient_age=_str("PatientAge"),
            study_date=_str("StudyDate"),
            study_description=_str("StudyDescription"),
            series_description=_str("SeriesDescription"),
            modality=_str("Modality"),
            rows=int(getattr(ds, "Rows", arr.shape[0])),
            columns=int(getattr(ds, "Columns", arr.shape[1])),
            pixel_spacing=px_spacing,
            manufacturer=_str("Manufacturer"),
        )
        return Study(pixel_data=arr, metadata=meta, file_format="dicom")

    def _load_pil(self, raw: bytes, fmt: str) -> Study:
        from PIL import Image

        img = Image.open(io.BytesIO(raw)).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        h, w = arr.shape
        return Study(
            pixel_data=arr,
            metadata=Metadata(rows=h, columns=w),
            file_format=fmt,
        )

    def _load_nifti(self, raw: bytes, suffix: str) -> Study:
        import nibabel as nib  # optional dep — only needed for NIfTI

        ext = ".nii.gz" if suffix == ".gz" else ".nii"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            img = nib.load(tmp)
            data = np.asarray(img.dataobj, dtype=np.float32)
            # 3-D volume → take middle axial slice
            if data.ndim == 3:
                data = data[:, :, data.shape[2] // 2]
            elif data.ndim == 4:
                data = data[:, :, data.shape[2] // 2, 0]
            lo, hi = float(data.min()), float(data.max())
            data = (data - lo) / (hi - lo) if hi > lo else np.zeros_like(data)
            h, w = data.shape
            return Study(
                pixel_data=data,
                metadata=Metadata(rows=h, columns=w),
                file_format="nifti",
            )
        finally:
            os.unlink(tmp)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_dicom(raw: bytes) -> bool:
        if len(raw) >= 132:
            return raw[128:132] == b"DICM"
        # Implicit VR DICOM with no preamble starts with a group-0 tag
        return len(raw) >= 4 and raw[0:2] in (b"\x08\x00", b"\x00\x08")


# ---------------------------------------------------------------------------
# CLI — headless smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.app.modules.dicom.service <file>")
        sys.exit(1)

    svc = DicomService()
    study = svc.load(sys.argv[1])
    m = study.metadata
    print(f"Format : {study.file_format}")
    print(f"Shape  : {study.pixel_data.shape}  dtype={study.pixel_data.dtype}")
    print(f"Range  : [{study.pixel_data.min():.3f}, {study.pixel_data.max():.3f}]")
    print(f"Modality  : {m.modality}")
    print(f"Patient   : {m.patient_id}  {m.patient_name}")
    print(f"Study date: {m.study_date}")
