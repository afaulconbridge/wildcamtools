from collections.abc import Generator
from pathlib import Path

import numpy as np

from wildcamtools.lib import Frame
from wildcamtools.lib.vidio import VideoReader, VideoWriter


def test_video_reader(video_path: Path) -> None:
    with VideoReader(video_path) as video_reader:
        for frame in video_reader:
            frame_no = frame.frame_no
            array = frame.raw
            assert frame_no >= 0
            assert frame_no < (5 + 1) * 30  # expect just over 5 frames at 30 fps
            assert isinstance(array, np.ndarray)
            assert array.ndim == 3
            assert array.shape == (2160, 3840, 3)  # 4k colour


def test_video_reader_rtsp(rtsp_server: str) -> None:
    frame_no = 0
    with VideoReader(rtsp_server, 3840, 2160) as video_reader:
        for frame in video_reader:
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


def test_video_writer(video_frame_generator: Generator[Frame], tmp_path: Path) -> None:
    with VideoWriter(tmp_path / "out.mp4", fps=30.0) as writer:
        for frame in video_frame_generator():
            writer.write(frame.raw)

    with VideoReader(tmp_path / "out.mp4", 3840, 2160) as video_reader:
        for frame in video_reader:
            frame_no = frame.frame_no
            array = frame.raw
            assert frame_no >= 0
            assert frame_no < 181  # expect 5 seconds at 30 fps
            assert isinstance(array, np.ndarray)
            assert array.ndim == 3
            assert array.shape == (2160, 3840, 3)  # 4k colour
