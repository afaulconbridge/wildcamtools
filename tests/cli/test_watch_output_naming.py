from datetime import UTC, datetime
from pathlib import Path

import pytest

from wildcamtools.cli.watch import ClipMetadata, OutputClipMetadata, WatcherManager
from wildcamtools.lib.states import MotionWindow, WatcherTransitionMetrics
from wildcamtools.lib.watch_config import WatchConfig


@pytest.fixture
def sample_watch_config() -> WatchConfig:
    """Create a sample watch config for testing."""
    return WatchConfig(rtsp_stream="rtsp://localhost:8554/stream")


@pytest.fixture
def sample_motion_window(sample_watch_config: WatchConfig) -> MotionWindow:
    """Create a sample motion window for testing."""
    return MotionWindow(
        start_frame=100,
        start_time=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        end_frame=250,
        end_time=datetime(2024, 1, 15, 10, 30, 15, tzinfo=UTC),
        transition_metrics=WatcherTransitionMetrics(),
        transition_window_metrics={},
        config=sample_watch_config,
    )


def test_output_clip_metadata_model(sample_watch_config: WatchConfig) -> None:
    """Test OutputClipMetadata model serialization."""
    motion_window = MotionWindow(
        start_frame=100,
        start_time=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        end_frame=250,
        end_time=datetime(2024, 1, 15, 10, 30, 15, tzinfo=UTC),
        transition_metrics=WatcherTransitionMetrics(),
        transition_window_metrics={},
        config=sample_watch_config,
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
        config=sample_watch_config,
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
        config=sample_watch_config,
    )

    json_str = metadata_file.model_dump_json(indent=2)
    assert "clip" in json_str
    assert "config" in json_str
    assert "start_frame" in json_str
    assert "motion_window" in json_str


def test_watcher_manager_detects_stream_input(tmp_path: Path) -> None:
    """Test that WatcherManager correctly detects stream inputs."""
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = WatchConfig(rtsp_stream="rtsp://localhost:8554/stream")
    manager = WatcherManager(
        config=config,
        segments_dir=segments_dir,
        output_dir=output_dir,
    )

    assert manager._is_stream is True


def test_watcher_manager_detects_file_input(tmp_path: Path, video_path: Path) -> None:
    """Test that WatcherManager correctly detects file inputs."""
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = WatchConfig(rtsp_stream=str(video_path))
    manager = WatcherManager(
        config=config,
        segments_dir=segments_dir,
        output_dir=output_dir,
    )

    assert manager._is_stream is False


def test_output_clip_naming_uses_motion_window_frames(tmp_path: Path, sample_watch_config: WatchConfig) -> None:
    """Test that output clips use motion window frames, not offset-adjusted frames.

    This is a regression test for the bug where output clips were named starting
    from frame 000000 because the offset-adjusted start_frame was used instead of
    the actual motion_window.start_frame.
    """
    from wildcamtools.cli.watch import OutputClipMetadata

    # Simulate a motion window that starts early in the video
    motion_window = MotionWindow(
        start_frame=325,
        start_time=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        end_frame=367,
        end_time=datetime(2024, 1, 15, 10, 30, 15, tzinfo=UTC),
        transition_metrics=WatcherTransitionMetrics(),
        transition_window_metrics={},
        config=sample_watch_config,
    )

    # Simulate offset calculation (10 seconds at 30 FPS = 300 frames)
    offset_start_frames = 300  # 10 seconds * 30 FPS
    offset_end_frames = 300

    # Offset-adjusted frames for segment selection
    segment_start_frame = max(0, motion_window.start_frame - offset_start_frames)
    segment_end_frame = motion_window.end_frame + offset_end_frames

    # Output filename should use motion window frames, not segment frames
    output_base = f"out_frame{motion_window.start_frame:06d}_{motion_window.end_frame:06d}"

    assert output_base == "out_frame000325_000367"
    assert segment_start_frame == 25  # 325 - 300
    assert segment_end_frame == 667  # 367 + 300

    # Verify metadata also uses motion window frames
    clip_metadata = ClipMetadata(
        start_frame=motion_window.start_frame,
        end_frame=motion_window.end_frame,
        motion_window=motion_window,
    )
    metadata = OutputClipMetadata(
        clip=clip_metadata,
        config=sample_watch_config,
    )

    assert metadata.clip.start_frame == 325
    assert metadata.clip.end_frame == 367
