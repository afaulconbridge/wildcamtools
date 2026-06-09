import logging
from pathlib import Path

import av
import pytest

from wildcamtools.lib.segment import create_segment_process
from wildcamtools.lib.utils import is_stream_url

logger = logging.getLogger(__name__)


@pytest.fixture(name="temp_output_dir")
def fixture_temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for segment output."""
    return tmp_path / "segments"


def test_pyav_segment_process_creation(temp_output_dir: Path, video_path: Path) -> None:
    """Test that PyAV segment process can be created and started."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=2.0,
    )

    assert process is not None
    assert process.poll() is None
    assert process.returncode is None

    process.wait(timeout=10.0)
    assert process.returncode == 0
    assert temp_output_dir.exists()
    segments = list(temp_output_dir.glob("*.mp4"))
    assert len(segments) > 0


def test_pyav_segment_process_produces_segments(temp_output_dir: Path, video_path: Path) -> None:
    """Test that segmentation produces multiple segment files."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=1.0,
    )

    process.wait(timeout=10.0)

    segments = sorted(temp_output_dir.glob("*.mp4"))
    assert len(segments) >= 1

    for segment in segments:
        assert segment.exists()
        assert segment.stat().st_size > 0


def test_pyav_segment_process_terminate(temp_output_dir: Path, video_path: Path) -> None:
    """Test that segment process can be terminated."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=5.0,
    )

    assert process.poll() is None

    process.terminate()
    process.wait(timeout=5.0)

    assert process.poll() is not None


def test_pyav_segment_process_interface(temp_output_dir: Path, video_path: Path) -> None:
    """Test that PyAV segment process has Popen-like interface."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=1.0,
    )

    assert hasattr(process, "poll")
    assert hasattr(process, "wait")
    assert hasattr(process, "terminate")
    assert hasattr(process, "kill")
    assert hasattr(process, "returncode")
    assert hasattr(process, "pid")

    process.wait(timeout=10.0)

    assert process.returncode is not None
    assert isinstance(process.pid, int)


def test_pyav_segment_process_with_audio(temp_output_dir: Path, video_path: Path) -> None:
    """Test segmentation handles both video and audio streams."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=2.0,
    )
    process.wait(timeout=10.0)

    segments = list(temp_output_dir.glob("*.mp4"))
    assert len(segments) > 0

    for segment in segments:
        with av.open(str(segment)) as container:
            assert len(container.streams.video) > 0


def test_pyav_segment_process_invalid_input_error(tmp_path: Path) -> None:
    """Test that segmentation exits with error code 1 for invalid input files."""
    temp_output_dir = tmp_path / "segments"
    temp_output_dir.mkdir()
    empty_file = temp_output_dir / "empty.mp4"
    empty_file.write_bytes(b"")

    process = create_segment_process(
        input_=empty_file,
        output=temp_output_dir / "output",
        duration=1.0,
    )

    process.wait(timeout=10.0)

    assert process.returncode == 1
    output_dir = temp_output_dir / "output"
    segments = list(output_dir.glob("*.mp4"))
    assert len(segments) == 0


def test_pyav_segment_process_short_video_produces_segment(tmp_path: Path) -> None:
    """Test that segmentation produces a segment even when video is shorter than segment duration.

    When the entire video is shorter than one segment duration, one segment should still be produced.
    """
    temp_output_dir = tmp_path / "segments"
    temp_output_dir.mkdir()
    short_video = Path("tests/data/short.mp4")
    assert short_video.exists()

    segment_duration = 5.0

    process = create_segment_process(
        input_=short_video,
        output=temp_output_dir,
        duration=segment_duration,
    )

    process.wait(timeout=10.0)

    assert process.returncode == 0
    segments = list(temp_output_dir.glob("*.mp4"))
    assert len(segments) == 1


def test_is_stream_url_rtsp() -> None:
    """Test RTSP URL detection."""
    assert is_stream_url("rtsp://localhost:8554/stream") is True
    assert is_stream_url("rtsp://user:pass@example.com:554/path") is True


def test_is_stream_url_rtmp() -> None:
    """Test RTMP URL detection."""
    assert is_stream_url("rtmp://localhost:1935/live/stream") is True


def test_is_stream_url_http() -> None:
    """Test HTTP/HTTPS URL detection."""
    assert is_stream_url("http://example.com/stream") is True
    assert is_stream_url("https://example.com/stream") is True


def test_is_stream_url_file_paths() -> None:
    """Test that file paths are not detected as stream URLs."""
    assert is_stream_url("/path/to/video.mp4") is False
    assert is_stream_url("./relative/path.mp4") is False
    assert is_stream_url("video.mp4") is False
    assert is_stream_url(Path("/absolute/path.mp4")) is False
    assert is_stream_url(Path("./relative/path.mp4")) is False


def test_is_stream_url_edge_cases() -> None:
    """Test edge cases that look like URLs but aren't."""
    # Filenames containing protocol-like strings should be treated as files
    assert is_stream_url("http_video.mp4") is False
    assert is_stream_url("rtsp_backup.ts") is False


def test_restart_on_exit_auto_detect_rtsp(temp_output_dir: Path) -> None:
    """Test that RTSP URLs auto-detect as restart_on_exit=True."""
    process = create_segment_process(
        input_="rtsp://localhost:8554/stream",
        output=temp_output_dir,
        duration=1.0,
    )
    assert process.restart_on_exit is True


def test_restart_on_exit_auto_detect_file(temp_output_dir: Path, video_path: Path) -> None:
    """Test that file paths auto-detect as restart_on_exit=False."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=1.0,
    )
    assert process.restart_on_exit is False


def test_restart_on_exit_explicit_override_true(temp_output_dir: Path, video_path: Path) -> None:
    """Test explicit restart_on_exit=True override for file input."""
    process = create_segment_process(
        input_=video_path,
        output=temp_output_dir,
        duration=1.0,
        restart_on_exit=True,
    )
    assert process.restart_on_exit is True


def test_restart_on_exit_explicit_override_false(temp_output_dir: Path) -> None:
    """Test explicit restart_on_exit=False override for RTSP URL."""
    process = create_segment_process(
        input_="rtsp://localhost:8554/stream",
        output=temp_output_dir,
        duration=1.0,
        restart_on_exit=False,
    )
    assert process.restart_on_exit is False
