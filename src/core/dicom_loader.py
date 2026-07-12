"""
DICOM Loader - Handles DICOM series loading using SimpleITK

CRITICAL IMPLEMENTATION NOTE:
Always use SimpleITK.ImageSeriesReader.GetGDCMSeriesFileNames() for proper
slice ordering. Manual sorting by filename/instance number can fail with
various DICOM sources.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import SimpleITK as sitk
import pydicom


class DicomLoader:
    """
    Handles loading of DICOM CT series with proper slice ordering.

    Usage:
        loader = DicomLoader()
        image = loader.load_series("/path/to/dicom/folder")
        metadata = loader.get_metadata()
    """

    def __init__(self):
        self._image: Optional[sitk.Image] = None
        self._file_names: List[str] = []
        self._metadata: Dict[str, Any] = {}
        self._series_ids: List[str] = []

    def scan_directory(self, directory: str) -> List[str]:
        """
        Scan directory for DICOM series.

        Args:
            directory: Path to directory containing DICOM files

        Returns:
            List of Series Instance UIDs found in the directory
        """
        reader = sitk.ImageSeriesReader()
        self._series_ids = list(reader.GetGDCMSeriesIDs(directory) or [])
        return list(self._series_ids)

    def get_series_summaries(self, directory: str) -> List[Dict[str, Any]]:
        """Return lightweight metadata used to choose among DICOM series."""
        reader = sitk.ImageSeriesReader()
        series_ids = list(reader.GetGDCMSeriesIDs(directory) or [])
        summaries: List[Dict[str, Any]] = []

        for series_id in series_ids:
            file_names = list(reader.GetGDCMSeriesFileNames(directory, series_id))
            summary: Dict[str, Any] = {
                "series_id": series_id,
                "modality": "Unknown",
                "description": "Unnamed series",
                "series_number": "?",
                "num_files": len(file_names),
            }
            if file_names:
                try:
                    ds = pydicom.dcmread(
                        file_names[0],
                        stop_before_pixels=True,
                        specific_tags=[
                            "Modality",
                            "SeriesDescription",
                            "ProtocolName",
                            "SeriesNumber",
                        ],
                    )
                    description = getattr(ds, "SeriesDescription", None)
                    if not description:
                        description = getattr(ds, "ProtocolName", None)
                    summary.update(
                        {
                            "modality": str(getattr(ds, "Modality", "Unknown")),
                            "description": str(description or "Unnamed series"),
                            "series_number": str(
                                getattr(ds, "SeriesNumber", "?")
                            ),
                        }
                    )
                except Exception:
                    pass
            summaries.append(summary)

        return summaries

    def load_series(
        self,
        directory: str,
        series_id: Optional[str] = None
    ) -> sitk.Image:
        """
        Load a DICOM series from directory.

        CRITICAL: Uses GetGDCMSeriesFileNames for correct slice ordering.
        This handles various DICOM quirks like:
        - Non-sequential instance numbers
        - Inconsistent naming conventions
        - Multi-frame images

        Args:
            directory: Path to DICOM directory
            series_id: Optional Series Instance UID (uses first if not specified)

        Returns:
            sitk.Image: Loaded 3D volume

        Raises:
            ValueError: If no DICOM series found
            RuntimeError: If loading fails
        """
        reader = sitk.ImageSeriesReader()

        # Get all series IDs
        series_ids = reader.GetGDCMSeriesIDs(directory)
        if not series_ids:
            raise ValueError(f"No DICOM series found in: {directory}")

        # Select series
        if series_id is None:
            summaries = self.get_series_summaries(directory)
            if summaries:
                series_id = max(
                    summaries,
                    key=lambda item: (
                        item.get("modality") == "CT",
                        int(item.get("num_files", 0)),
                    ),
                )["series_id"]
            else:
                series_id = series_ids[0]
        elif series_id not in series_ids:
            raise ValueError(f"Series ID {series_id} not found in directory")

        # CRITICAL: Use GetGDCMSeriesFileNames for proper ordering
        self._file_names = reader.GetGDCMSeriesFileNames(directory, series_id)

        if not self._file_names:
            raise ValueError(f"No files found for series: {series_id}")

        # Configure reader
        reader.SetFileNames(self._file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        # Load the image
        self._image = reader.Execute()

        # Extract metadata
        self._extract_metadata(reader)

        return self._image

    def _extract_metadata(self, reader: sitk.ImageSeriesReader) -> None:
        """Extract relevant metadata from DICOM headers."""
        if not self._file_names:
            return

        # Read first file for header info
        try:
            ds = pydicom.dcmread(self._file_names[0], stop_before_pixels=True)

            self._metadata = {
                "patient_id": getattr(ds, "PatientID", "Unknown"),
                "patient_name": str(getattr(ds, "PatientName", "Unknown")),
                "study_date": getattr(ds, "StudyDate", "Unknown"),
                "modality": getattr(ds, "Modality", "Unknown"),
                "manufacturer": getattr(ds, "Manufacturer", "Unknown"),
                "slice_thickness": float(getattr(ds, "SliceThickness", 0)),
                "kvp": float(getattr(ds, "KVP", 0)),
                "body_part": getattr(ds, "BodyPartExamined", "Unknown"),
                "num_slices": len(self._file_names),
            }
        except Exception as e:
            self._metadata = {"error": str(e)}

        # Add SimpleITK computed properties
        if self._image:
            self._metadata.update({
                "size": self._image.GetSize(),
                "spacing": self._image.GetSpacing(),
                "origin": self._image.GetOrigin(),
                "direction": self._image.GetDirection(),
            })

    def get_metadata(self) -> Dict[str, Any]:
        """Return extracted metadata."""
        return self._metadata.copy()

    def get_image(self) -> Optional[sitk.Image]:
        """Return loaded SimpleITK image."""
        return self._image

    def get_volume_bounds(self) -> Optional[Tuple[float, ...]]:
        """
        Get the physical bounds of the loaded volume.

        Returns:
            (x_min, x_max, y_min, y_max, z_min, z_max) or None
        """
        if self._image is None:
            return None

        size = self._image.GetSize()
        spacing = self._image.GetSpacing()
        origin = self._image.GetOrigin()

        x_max = origin[0] + (size[0] - 1) * spacing[0]
        y_max = origin[1] + (size[1] - 1) * spacing[1]
        z_max = origin[2] + (size[2] - 1) * spacing[2]

        return (origin[0], x_max, origin[1], y_max, origin[2], z_max)

    def get_center(self) -> Optional[Tuple[float, float, float]]:
        """
        Get the center point of the loaded volume in world coordinates.

        Returns:
            (x, y, z) center point or None
        """
        if self._image is None:
            return None

        size = self._image.GetSize()
        spacing = self._image.GetSpacing()
        origin = self._image.GetOrigin()

        center_x = origin[0] + (size[0] - 1) * spacing[0] / 2
        center_y = origin[1] + (size[1] - 1) * spacing[1] / 2
        center_z = origin[2] + (size[2] - 1) * spacing[2] / 2

        return (center_x, center_y, center_z)


def load_dicom_directory(directory: str) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Convenience function to load DICOM directory.

    Args:
        directory: Path to DICOM folder

    Returns:
        (image, metadata) tuple
    """
    loader = DicomLoader()
    image = loader.load_series(directory)
    metadata = loader.get_metadata()
    return image, metadata
