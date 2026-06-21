from pathlib import Path

import pytest

from wildcamtools.lib.segment import create_segment_process
from wildcamtools.lib.segment_metadata import SegmentMetadata


@pytest.fixture(name="temp_output_dir")
def fixture_temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for segment output."""
    return tmp_path / "segments"


def test_segment_process_creates_metadata_files(temp_output_dir: Path, video_path: Path) -> None:
    """Test that segmentation creates metadata sidecar files."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=2.0,
    )

    process.wait(timeout=10.0)
    assert process.returncode == 0

    # Check that metadata files were created
    metadata_files = list(temp_output_dir.glob("*.meta.json"))
    assert len(metadata_files) > 0

    # Check that each metadata file has a corresponding segment file
    for metadata_path in metadata_files:
        segment_path = SegmentMetadata.get_segment_path(metadata_path)
        assert segment_path.exists(), f"Segment file missing: {segment_path}"


def test_segment_metadata_contains_frame_tracking(temp_output_dir: Path, video_path: Path) -> None:
    """Test that metadata files contain frame tracking information."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=2.0,
    )

    process.wait(timeout=10.0)

    metadata_files = sorted(temp_output_dir.glob("*.meta.json"))
    assert len(metadata_files) > 0

    for metadata_path in metadata_files:
        metadata = SegmentMetadata.load(metadata_path)
        assert metadata is not None
        assert metadata.start_frame >= 0
        assert metadata.end_frame >= metadata.start_frame
        assert metadata.fps > 0


def test_segment_metadata_frame_continuity(temp_output_dir: Path, video_path: Path) -> None:
    """Test that frame numbers are continuous across segments."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=1.0,
    )

    process.wait(timeout=10.0)

    metadata_files = sorted(temp_output_dir.glob("*.meta.json"))
    assert len(metadata_files) > 0

    prev_end_frame = -1
    for metadata_path in metadata_files:
        metadata = SegmentMetadata.load(metadata_path)
        assert metadata is not None

        # Check continuity (allowing for small gaps due to frame timing)
        if prev_end_frame >= 0:
            assert metadata.start_frame <= prev_end_frame + 5, (
                f"Frame gap too large: {prev_end_frame} to {metadata.start_frame}"
            )

        prev_end_frame = metadata.end_frame


def test_segment_metadata_fps_detection(temp_output_dir: Path, video_path: Path) -> None:
    """Test that FPS is correctly detected and stored."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=2.0,
    )

    process.wait(timeout=10.0)

    metadata_files = list(temp_output_dir.glob("*.meta.json"))
    assert len(metadata_files) > 0

    # All segments should have the same FPS
    fps_values = set()
    for metadata_path in metadata_files:
        metadata = SegmentMetadata.load(metadata_path)
        assert metadata is not None
        fps_values.add(metadata.fps)

    assert len(fps_values) == 1, f"Inconsistent FPS values: {fps_values}"
    fps = fps_values.pop()
    assert fps > 0


def test_segment_metadata_temporal_consistency(temp_output_dir: Path, video_path: Path) -> None:
    """Test that timestamps are consistent with frame counts."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=2.0,
    )

    process.wait(timeout=10.0)

    metadata_files = sorted(temp_output_dir.glob("*.meta.json"))
    assert len(metadata_files) > 0

    for metadata_path in metadata_files:
        metadata = SegmentMetadata.load(metadata_path)
        assert metadata is not None

        # Check temporal consistency
        if metadata.start_time and metadata.end_time:
            duration = (metadata.end_time - metadata.start_time).total_seconds()
            frame_duration = (metadata.end_frame - metadata.start_frame) / metadata.fps

            # Allow 10% tolerance for timing differences
            assert abs(duration - frame_duration) < max(0.5, duration * 0.1), (
                f"Duration mismatch: {duration}s vs {frame_duration}s"
            )


def test_segment_process_short_video(temp_output_dir: Path) -> None:
    """Test segmentation with a video shorter than segment duration."""
    short_video = Path("tests/data/short.mp4")
    assert short_video.exists()

    process = create_segment_process(
        input_=short_video,
        output=temp_output_dir,
        duration=5.0,
    )

    process.wait(timeout=10.0)

    assert process.returncode == 0
    metadata_files = list(temp_output_dir.glob("*.meta.json"))
    assert len(metadata_files) == 1

    metadata = SegmentMetadata.load(metadata_files[0])
    assert metadata is not None
    assert metadata.start_frame >= 0
    assert metadata.end_frame >= metadata.start_frame


def test_frame_based_naming_for_files(temp_output_dir: Path, video_path: Path) -> None:
    """Test that file inputs use frame-based segment naming."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=1.0,
    )

    process.wait(timeout=10.0)

    segments = sorted(temp_output_dir.glob("*.mp4"))
    assert len(segments) > 0

    # Verify frame-based naming pattern
    for segment in segments:
        assert segment.name.startswith("seg_frame"), f"Expected frame-based naming, got {segment.name}"
        assert segment.name.endswith(".mp4")


def test_timestamp_based_naming_for_streams(temp_output_dir: Path) -> None:
    """Test that stream inputs use timestamp-based segment naming."""
    process = create_segment_process(
        input_="rtsp://localhost:8554/stream",
        output=temp_output_dir,
        duration=1.0,
    )

    # Don't wait for completion (RTSP stream won't complete in test)
    process.wait(timeout=0.5)

    # Check that any segments created use timestamp-based naming
    segments = list(temp_output_dir.glob("*.mp4"))
    for segment in segments:
        # Should match pattern: seg_YYYY_MM_DD__HH_MM_SS_NNNN.mp4
        assert segment.name.startswith("seg_"), f"Expected timestamp-based naming, got {segment.name}"
        assert "_" in segment.name[4:], f"Expected timestamp in segment name, got {segment.name}"


def test_output_clip_metadata_model() -> None:
    """Test OutputClipMetadata model serialization."""
    from datetime import UTC, datetime

    from wildcamtools.cli.watch import ClipMetadata, OutputClipMetadata
    from wildcamtools.lib.states import MotionWindow, WatcherTransitionMetrics
    from wildcamtools.lib.watch_config import WatchConfig

    config = WatchConfig(rtsp_stream="rtsp://localhost:8554/stream")
    motion_window = MotionWindow(
        start_frame=100,
        start_time=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        end_frame=250,
        end_time=datetime(2024, 1, 15, 10, 30, 15, tzinfo=UTC),
        transition_metrics=WatcherTransitionMetrics(),
        transition_window_metrics={},
        config=config,
        source_fps=30.0,
    )

    # Test with timestamps (stream)
    clip_metadata_stream = ClipMetadata(
        start_frame=100,
        end_frame=250,
        start_time=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        end_time=datetime(2024, 1, 15, 10, 30, 15, tzinfo=UTC),
        motion_window=motion_window,
    )
    metadata_stream = OutputClipMetadata(
        clip=clip_metadata_stream,
        config=config,
    )

    json_str = metadata_stream.model_dump_json(indent=2)
    assert "clip" in json_str
    assert "config" in json_str
    assert "start_frame" in json_str
    assert "motion_window" in json_str

    # Test without timestamps (file)
    clip_metadata_file = ClipMetadata(
        start_frame=100,
        end_frame=250,
        motion_window=motion_window,
    )
    metadata_file = OutputClipMetadata(
        clip=clip_metadata_file,
        config=config,
    )

    json_str = metadata_file.model_dump_json(indent=2)
    assert "clip" in json_str
    assert "config" in json_str
    assert "start_frame" in json_str
    assert "motion_window" in json_str
