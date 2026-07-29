from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class SegmentMetadata(BaseModel):
    """Metadata for a video segment file.

    Attributes:
        start_frame: First frame number in the segment
        end_frame: Last frame number in the segment
        start_time: Timestamp of first frame (optional)
        end_time: Timestamp of last frame (optional)
        fps: Frames per second for this segment
        actual_frames: Actual frame count from ffprobe (optional)
        duration: Actual duration in seconds (optional)

    """

    start_frame: int
    end_frame: int
    start_time: datetime | None = None
    end_time: datetime | None = None
    fps: float
    actual_frames: int | None = None
    duration: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary with ISO format timestamps."""
        data = self.model_dump()
        if data.get("start_time"):
            data["start_time"] = data["start_time"].isoformat()
        if data.get("end_time"):
            data["end_time"] = data["end_time"].isoformat()
        return data

    @classmethod
    def load(cls, path: Path) -> SegmentMetadata | None:
        """Load metadata from a JSON file.

        Args:
            path: Path to the .meta.json file

        Returns:
            SegmentMetadata object or None if file doesn't exist or is invalid

        """
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            if data.get("start_time"):
                data["start_time"] = datetime.fromisoformat(data["start_time"])
            if data.get("end_time"):
                data["end_time"] = datetime.fromisoformat(data["end_time"])

            return cls.model_validate(data)
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    def save(self, path: Path) -> None:
        """Save metadata to a JSON file.

        Args:
            path: Path to write the .meta.json file

        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def get_metadata_path(segment_path: Path) -> Path:
        """Get the metadata file path for a segment.

        Args:
            segment_path: Path to the segment video file

        Returns:
            Path to the corresponding .meta.json file

        """
        return segment_path.with_suffix(segment_path.suffix + ".meta.json")

    @staticmethod
    def get_segment_path(metadata_path: Path) -> Path:
        """Get the segment file path from a metadata file path.

        Args:
            metadata_path: Path to the .meta.json file

        Returns:
            Path to the corresponding segment video file

        """
        if not metadata_path.name.endswith(".meta.json"):
            raise ValueError(f"Invalid metadata path: {metadata_path}")
        # Strip ".meta.json" suffix to get base segment path (e.g., segment_0000.mp4)
        return Path(str(metadata_path)[: -len(".meta.json")])
