from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from wildcamtools.lib import Frame
from wildcamtools.lib.vidio import FrameSourceFFMPEG, FrameWriterFFMPEG


@pytest.mark.skip(reason="Test may hang due to video length")
def test_frame_source_ffmpeg(video_path: Path) -> None:
    with FrameSourceFFMPEG(video_path) as frame_source:
        for frame in frame_source:
            frame_no = frame.frame_no
            array = frame.raw
            assert frame_no >= 0
            assert frame_no < (5 + 1) * 30  # expect just over 5 frames at 30 fps
            assert isinstance(array, np.ndarray)
            assert array.ndim == 3
            assert array.shape == (2160, 3840, 3)  # 4k colour


def test_frame_source_ffmpeg_rtsp(rtsp_server: str) -> None:
    frame_no = 0
    with FrameSourceFFMPEG(rtsp_server, 3840, 2160) as frame_source:
        for frame in frame_source:
            if frame.frame_no > 150:
                break
            frame_no = frame.frame_no
            array = frame.raw
            assert frame_no >= 0
            assert frame_no < 181  # expect 5 seconds at 30 fps
            assert isinstance(array, np.ndarray)
            assert array.ndim == 3
            assert array.shape == (2160, 3840, 3)  # 4k colour
        assert frame_no >= 150
        assert frame_no < 181  # expect 5 seconds at 30 fps


@pytest.mark.skip(reason="Test may hang due to video length")
def test_frame_writer_ffmpeg(video_frame_generator: Generator[Frame], tmp_path: Path) -> None:
    with FrameWriterFFMPEG(tmp_path / "out.mp4", fps=30.0) as writer:
        for frame in video_frame_generator():
            writer.write(frame.raw)

    # now read what was written back to check its valid
    with FrameSourceFFMPEG(tmp_path / "out.mp4", 3840, 2160) as frame_source:
        for frame in frame_source:
            frame_no = frame.frame_no
            array = frame.raw
            assert frame_no >= 0
            assert frame_no < 181  # expect 5 seconds at 30 fps
            assert isinstance(array, np.ndarray)
            assert array.ndim == 3
            assert array.shape == (2160, 3840, 3)  # 4k colour
