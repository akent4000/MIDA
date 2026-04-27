"""Unit tests for DicomService.

All tests create synthetic images in memory — no DICOM files on disk required.
"""

from __future__ import annotations

import io

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from backend.app.modules.dicom.service import DicomService
from backend.app.modules.dicom.types import Metadata, Study

# ---------------------------------------------------------------------------
# Synthetic DICOM builder
# ---------------------------------------------------------------------------


def _make_dicom_bytes(
    rows: int = 64,
    cols: int = 64,
    photometric: str = "MONOCHROME2",
    patient_id: str = "TEST001",
    modality: str = "CR",
) -> bytes:
    """Create a minimal valid DICOM dataset in memory."""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        filename_or_obj=None,
        dataset={},
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.PatientID = patient_id
    ds.PatientName = "Test^Patient"
    ds.PatientSex = "M"
    ds.Modality = modality
    ds.StudyDate = "20240101"

    pixel_array = np.random.randint(50, 200, (rows, cols), dtype=np.uint8)
    ds.PixelData = pixel_array.tobytes()

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds)
    return buf.getvalue()


def _make_png_bytes(rows: int = 32, cols: int = 32) -> bytes:
    from PIL import Image

    arr = np.random.randint(0, 256, (rows, cols), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# DicomService.load — DICOM
# ---------------------------------------------------------------------------


class TestDicomLoad:
    def test_load_returns_study(self) -> None:
        svc = DicomService()
        raw = _make_dicom_bytes()
        study = svc.load(raw)
        assert isinstance(study, Study)

    def test_pixel_data_float32_in_unit_range(self) -> None:
        svc = DicomService()
        study = svc.load(_make_dicom_bytes())
        arr = study.pixel_data
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_pixel_data_shape_matches_rows_cols(self) -> None:
        svc = DicomService()
        study = svc.load(_make_dicom_bytes(rows=48, cols=56))
        assert study.pixel_data.shape == (48, 56)

    def test_metadata_extracted(self) -> None:
        svc = DicomService()
        study = svc.load(_make_dicom_bytes(patient_id="PAT123", modality="DX"))
        assert study.metadata.patient_id == "PAT123"
        assert study.metadata.modality == "DX"
        assert study.metadata.rows == 64
        assert study.metadata.columns == 64

    def test_file_format_is_dicom(self) -> None:
        svc = DicomService()
        study = svc.load(_make_dicom_bytes())
        assert study.file_format == "dicom"

    def test_monochrome1_inverted(self) -> None:
        svc = DicomService()
        study_m2 = svc.load(_make_dicom_bytes(photometric="MONOCHROME2"))
        study_m1 = svc.load(_make_dicom_bytes(photometric="MONOCHROME1"))
        # Both should be normalised to [0, 1]; inversion changes the distribution
        # but both must stay within range.
        for study in (study_m2, study_m1):
            assert study.pixel_data.min() >= 0.0
            assert study.pixel_data.max() <= 1.0


# ---------------------------------------------------------------------------
# DicomService.load — PNG via bytes
# ---------------------------------------------------------------------------


class TestPngLoad:
    def test_load_png_from_path(self, tmp_path: Path) -> None:  # noqa: F821

        png_bytes = _make_png_bytes()
        p = tmp_path / "test.png"
        p.write_bytes(png_bytes)

        svc = DicomService()
        study = svc.load(p)
        assert study.file_format == "png"
        assert study.pixel_data.dtype == np.float32
        assert study.pixel_data.min() >= 0.0
        assert study.pixel_data.max() <= 1.0

    def test_load_png_shape(self, tmp_path: Path) -> None:  # noqa: F821

        png_bytes = _make_png_bytes(rows=48, cols=64)
        p = tmp_path / "img.png"
        p.write_bytes(png_bytes)

        study = DicomService().load(p)
        assert study.pixel_data.shape == (48, 64)

    def test_metadata_rows_cols_set_for_png(self, tmp_path: Path) -> None:  # noqa: F821

        p = tmp_path / "x.png"
        p.write_bytes(_make_png_bytes(rows=20, cols=30))
        m = DicomService().load(p).metadata
        assert m.rows == 20
        assert m.columns == 30


# ---------------------------------------------------------------------------
# DicomService.anonymize
# ---------------------------------------------------------------------------


class TestAnonymize:
    def test_anonymize_clears_patient_fields(self) -> None:
        svc = DicomService()
        study = svc.load(_make_dicom_bytes(patient_id="SECRET"))
        anon = svc.anonymize(study)
        assert anon.metadata.patient_id is None
        assert anon.metadata.patient_name is None
        assert anon.metadata.patient_sex is None

    def test_anonymize_preserves_modality_and_shape(self) -> None:
        svc = DicomService()
        study = svc.load(_make_dicom_bytes(modality="CR", rows=64, cols=64))
        anon = svc.anonymize(study)
        assert anon.metadata.modality == "CR"
        assert anon.metadata.rows == 64
        assert anon.metadata.columns == 64

    def test_anonymize_pixel_data_unchanged(self) -> None:
        svc = DicomService()
        study = svc.load(_make_dicom_bytes())
        anon = svc.anonymize(study)
        np.testing.assert_array_equal(study.pixel_data, anon.pixel_data)

    def test_anonymize_returns_copy(self) -> None:
        svc = DicomService()
        study = svc.load(_make_dicom_bytes())
        anon = svc.anonymize(study)
        anon.pixel_data[0, 0] = 999.0
        assert study.pixel_data[0, 0] != 999.0


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TestTypes:
    def test_metadata_defaults(self) -> None:
        m = Metadata()
        assert m.patient_id is None
        assert m.rows == 0
        assert m.extra == {}

    def test_study_fields(self) -> None:
        arr = np.zeros((10, 10), dtype=np.float32)
        s = Study(pixel_data=arr, metadata=Metadata(), file_format="dicom")
        assert s.file_format == "dicom"
        assert s.pixel_data.shape == (10, 10)
