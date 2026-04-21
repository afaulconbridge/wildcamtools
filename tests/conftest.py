import logging
import os
from collections.abc import Generator
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from wildcamtools.lib import Frame
from wildcamtools.lib.rtsp import BackgroundFFMPEGBroadcast, BackgroundMediaMTX
from wildcamtools.lib.vidio import FrameSourceFFMPEG


@pytest.fixture(name="logging", scope="session", autouse=True)
def fixture_logging() -> None:
    logging.basicConfig(level=logging.DEBUG, force=True)


@pytest.fixture(name="data_directory", scope="session")
def fixture_data_directory() -> Path:
    data_path = Path(os.path.dirname(os.path.realpath(__file__))) / "data"
    assert data_path.exists()
    assert data_path.is_dir()
    return data_path


@pytest.fixture(name="video_path", scope="session")
def fixture_video_path(data_directory: Path) -> Path:
    video_path = data_directory / "test.mp4"
    assert video_path.exists()
    assert video_path.is_file()
    return video_path


@pytest.fixture(name="video_frame_generator")
def fixture_video_frame_generator(video_path: Path) -> Generator[Generator[Frame]]:
    def internal_generator() -> Generator[Frame]:
        with FrameSourceFFMPEG(video_path) as video_source:
            yield from video_source

    yield internal_generator


@pytest.fixture(name="rtsp_server", scope="session")
def fixture_rtsp_server(video_path: Path) -> Generator[str]:
    with BackgroundMediaMTX(), BackgroundFFMPEGBroadcast("tests/data/test.mp4"):
        yield "rtsp://localhost:8554/stream"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_dirs(tmp_path):
    segments_dir = tmp_path / "segments"
    output_dir = tmp_path / "output"
    segments_dir.mkdir()
    output_dir.mkdir()
    return segments_dir, output_dir


@pytest.fixture
def dummy_segments(temp_dirs):
    segments_dir, _ = temp_dirs
    start_time = datetime(2026, 4, 21, 10, 0, 0)
    for i in range(5):
        t = start_time + timedelta(seconds=i * 15)
        fname = t.strftime("seg_%Y_%m_%d__%H_%M_%S.mp4")
        (segments_dir / fname).touch()
    return segments_dir


@pytest.fixture(autouse=True)
def mock_ffmpeg():
    with patch("wildcamtools.lib.concat.concat_ffmpeg") as mock_concat:
        yield mock_concat


@pytest.fixture(autouse=True)
def mock_subprocess():
    with patch("subprocess.Popen") as mock_popen:
        yield mock_popen
