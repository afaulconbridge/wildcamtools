from collections.abc import Generator
from pathlib import Path

import numpy as np

from wildcamtools.lib import Frame
from wildcamtools.lib.vidio import FrameSourceFFMPEG, FrameWriterFFMPEG


def test_frame_source_ffmpeg(video_path: Path) -> None:
    for frame in FrameSourceFFMPEG(video_path):
        frame_no = frame.frame_no
        array = frame.raw
        assert frame_no >= 0
        assert frame_no < (5 + 1) * 30  # expect just over 5 frames at 30 fps
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3
        assert array.shape == (2160, 3840, 3)  # 4k colour


def test_frame_source_ffmpeg_rtsp(rtsp_server: str) -> None:
    for frame in FrameSourceFFMPEG(rtsp_server, 3840, 2160):
        frame_no = frame.frame_no
        array = frame.raw
        assert frame_no >= 0
        assert frame_no < (5 + 1) * 30  # expect just over 5 frames at 30 fps
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3
        assert array.shape == (2160, 3840, 3)  # 4k colour


def test_frame_writer_ffmpeg(video_frame_generator: Generator[Frame], tmp_path: Path) -> None:
    with FrameWriterFFMPEG(tmp_path / "out.mp4", fps=30.0) as writer:
        for frame in video_frame_generator():
            writer.write(frame)

    # now read what was written back to check its valid
    for frame in FrameSourceFFMPEG(tmp_path / "out.mp4", 3840, 2160):
        frame_no = frame.frame_no
        array = frame.raw
        assert frame_no >= 0
        assert frame_no < (5 + 1) * 30  # expect just over 5 frames at 30 fps
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3
        assert array.shape == (2160, 3840, 3)  # 4k colour
