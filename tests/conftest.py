import logging
import os
from collections.abc import Generator
from pathlib import Path

import pytest

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
    # video_path = data_directory / "04-51-08.mp4"
    assert video_path.exists()
    assert video_path.is_file()
    return video_path


@pytest.fixture(name="video_frame_generator")
def fixture_video_frame_generator(video_path: Path) -> Generator[Generator[Frame]]:

    def internal_generator() -> Generator[Frame]:
        yield from FrameSourceFFMPEG(video_path)

    yield internal_generator


@pytest.fixture(name="rtsp_server", scope="session")
def fixture_rtsp_server(video_path: Path) -> Generator[str]:
    with BackgroundMediaMTX(), BackgroundFFMPEGBroadcast("tests/data/test.mp4"):
        yield "rtsp://localhost:8554/stream"
