"""DICOM module data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Metadata:
    """Key DICOM / image metadata extracted from a loaded study."""

    patient_id: str | None = None
    patient_name: str | None = None
    patient_sex: str | None = None
    patient_age: str | None = None
    study_date: str | None = None
    study_description: str | None = None
    series_description: str | None = None
    modality: str | None = None
    rows: int = 0
    columns: int = 0
    pixel_spacing: tuple[float, float] | None = None  # (row_spacing, col_spacing) in mm
    manufacturer: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Study:
    """A loaded medical image study ready for preprocessing."""

    pixel_data: np.ndarray  # (H, W) float32 in [0, 1]
    metadata: Metadata
    file_format: str  # "dicom" | "png" | "jpeg" | "nifti"
