import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wildcamtools.cli.watch import find_segments_for_framerange
from wildcamtools.lib.segment_metadata import SegmentMetadata


@pytest.fixture
def sample_metadata() -> SegmentMetadata:
    """Create sample metadata for testing."""
    return SegmentMetadata(
        start_frame=100,
        end_frame=250,
        start_time=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        end_time=datetime(2024, 1, 15, 10, 30, 15, tzinfo=UTC),
        fps=10.0,
    )


@pytest.fixture
def temp_metadata_file(tmp_path: Path, sample_metadata: SegmentMetadata) -> Path:
    """Create a temporary metadata file."""
    metadata_path = tmp_path / "seg_2024_01_15__10_30_00_0001.meta.json"
    sample_metadata.save(metadata_path)
    return metadata_path


class TestSegmentMetadata:
    """Tests for SegmentMetadata model."""

    def test_create_metadata(self, sample_metadata: SegmentMetadata) -> None:
        """Test creating metadata object."""
        assert sample_metadata.start_frame == 100
        assert sample_metadata.end_frame == 250
        assert sample_metadata.fps == 10.0
        assert sample_metadata.start_time is not None
        assert sample_metadata.end_time is not None

    def test_save_and_load_metadata(self, temp_metadata_file: Path) -> None:
        """Test saving and loading metadata."""
        loaded = SegmentMetadata.load(temp_metadata_file)
        assert loaded is not None
        assert loaded.start_frame == 100
        assert loaded.end_frame == 250
        assert loaded.fps == 10.0

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading from nonexistent file."""
        metadata_path = tmp_path / "nonexistent.meta.json"
        result = SegmentMetadata.load(metadata_path)
        assert result is None

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON file."""
        metadata_path = tmp_path / "invalid.meta.json"
        metadata_path.write_text("not valid json")
        result = SegmentMetadata.load(metadata_path)
        assert result is None

    def test_get_metadata_path(self) -> None:
        """Test getting metadata path from segment path."""
        segment_path = Path("/segments/seg_2024_01_15__10_30_00_0001.mp4")
        metadata_path = SegmentMetadata.get_metadata_path(segment_path)
        assert metadata_path.name == "seg_2024_01_15__10_30_00_0001.mp4.meta.json"

    def test_get_segment_path(self) -> None:
        """Test getting segment path from metadata path."""
        metadata_path = Path("/segments/seg_2024_01_15__10_30_00_0001.mp4.meta.json")
        segment_path = SegmentMetadata.get_segment_path(metadata_path)
        assert segment_path.name == "seg_2024_01_15__10_30_00_0001.mp4"

    def test_get_segment_path_invalid(self) -> None:
        """Test getting segment path from invalid metadata path."""
        metadata_path = Path("/segments/invalid.json")
        with pytest.raises(ValueError, match="Invalid metadata path"):
            SegmentMetadata.get_segment_path(metadata_path)

    def test_model_dump_json(self, sample_metadata: SegmentMetadata) -> None:
        """Test JSON serialization."""
        json_str = sample_metadata.model_dump_json(indent=2)
        data = json.loads(json_str)
        assert data["start_frame"] == 100
        assert data["end_frame"] == 250
        assert data["fps"] == 10.0
        assert "start_time" in data
        assert "end_time" in data

    def test_metadata_without_timestamps(self) -> None:
        """Test metadata with None timestamps."""
        metadata = SegmentMetadata(
            start_frame=0,
            end_frame=150,
            fps=15.0,
        )
        assert metadata.start_time is None
        assert metadata.end_time is None
        json_str = metadata.model_dump_json()
        data = json.loads(json_str)
        assert data["start_time"] is None
        assert data["end_time"] is None


@pytest.fixture
def setup_segments_dir(tmp_path: Path) -> Path:
    """Create a directory with segment metadata files."""
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    # Create metadata files for 3 segments
    segments_data = [
        (0, 150, "seg_2024_01_15__10_30_00_0001.mp4"),
        (151, 300, "seg_2024_01_15__10_30_15_0002.mp4"),
        (301, 450, "seg_2024_01_15__10_30_30_0003.mp4"),
    ]

    for start_frame, end_frame, segment_name in segments_data:
        metadata = SegmentMetadata(
            start_frame=start_frame,
            end_frame=end_frame,
            fps=10.0,
        )
        metadata_path = segments_dir / f"{segment_name}.meta.json"
        metadata.save(metadata_path)

        # Create dummy segment file
        segment_path = segments_dir / segment_name
        segment_path.write_bytes(b"dummy video content")

        # Also create the .meta file that the test is looking for
        meta_file = segments_dir / f"{segment_name}.meta"
        meta_file.write_bytes(b"dummy meta")

    return segments_dir


class TestFindSegmentsForFramerange:
    """Tests for find_segments_for_framerange function."""

    def test_find_segments_for_framerange_basic(self, setup_segments_dir: Path) -> None:
        """Test finding segments for a frame range."""
        # Range 100-200 overlaps with segment 1 (0-150) and segment 2 (151-300)
        result = find_segments_for_framerange(100, 200, setup_segments_dir)
        assert result is not None
        assert len(result) == 2
        assert "seg_2024_01_15__10_30_00_0001.mp4" in str(result[0])
        assert "seg_2024_01_15__10_30_15_0002.mp4" in str(result[1])

    def test_find_segments_spans_multiple_files(self, setup_segments_dir: Path) -> None:
        """Test finding segments that span multiple files."""
        result = find_segments_for_framerange(100, 350, setup_segments_dir)
        assert result is not None
        assert len(result) == 3

    def test_find_segments_no_overlap(self, setup_segments_dir: Path) -> None:
        """Test finding segments with no overlap."""
        result = find_segments_for_framerange(500, 600, setup_segments_dir)
        assert result is not None
        assert len(result) == 0

    def test_find_segments_incomplete_range(self, setup_segments_dir: Path) -> None:
        """Test finding segments when range extends beyond available segments."""
        # Request range that extends beyond last segment
        result = find_segments_for_framerange(400, 500, setup_segments_dir)
        assert result is None  # Should wait for more segments

    def test_find_segments_no_metadata_files(self, tmp_path: Path) -> None:
        """Test finding segments when no metadata files exist."""
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()

        result = find_segments_for_framerange(0, 100, segments_dir)
        assert result is None

    def test_find_segments_missing_segment_file(self, tmp_path: Path) -> None:
        """Test finding segments when segment file is missing."""
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()

        # Create metadata but not segment file
        metadata = SegmentMetadata(
            start_frame=0,
            end_frame=150,
            fps=10.0,
        )
        metadata_path = segments_dir / "seg_2024_01_15__10_30_00_0001.mp4.meta.json"
        metadata.save(metadata_path)

        result = find_segments_for_framerange(0, 100, segments_dir)
        assert result is not None
        assert len(result) == 0

    def test_find_segments_exact_boundary(self, setup_segments_dir: Path) -> None:
        """Test finding segments at exact boundaries."""
        # Exact start of first segment
        result = find_segments_for_framerange(0, 50, setup_segments_dir)
        assert result is not None
        assert len(result) == 1

        # Exact end of last segment
        result = find_segments_for_framerange(400, 450, setup_segments_dir)
        assert result is not None
        assert len(result) == 1

    def test_find_segments_single_frame(self, setup_segments_dir: Path) -> None:
        """Test finding segment for a single frame."""
        result = find_segments_for_framerange(200, 200, setup_segments_dir)
        assert result is not None
        assert len(result) == 1
