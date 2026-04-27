"""DICOM module — medical image loading, anonymisation, pixel extraction."""

from .service import DicomService
from .types import Metadata, Study

__all__ = ["DicomService", "Metadata", "Study"]
