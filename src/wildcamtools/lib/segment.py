import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

import av
import av.container
import av.stream

from wildcamtools.lib import Frame
from wildcamtools.lib.errors.core import (
    VideoNotInContextError,
    translate_av_error,
)

logger = logging.getLogger(__name__)


class VideoSegmenter:
    """
    FrameSource that segments video input while emitting frames.

    Uses two separate PyAV containers:
    - Container 1: Segment muxer that writes segment files to disk
    - Container 2: Frame decoder that emits Frame objects

    This dual-container architecture allows decoding once while muxing
    to both outputs simultaneously.

    Parameters:
        input_: Path to input video file or RTSP URL
        segment_dir: Directory to write segment files
        segment_duration: Duration of each segment in seconds
        segment_pattern: Filename pattern for segments (default: seg_%Y_%m_%d__%H_%M_%S.mp4)
        format_options: Optional dict of format options for segment muxer

    TODO: Optimize to use single container with custom output callback
          to avoid maintaining two separate container instances.
    """

    input_: str | Path
    segment_dir: Path
    segment_duration: float
    segment_pattern: str
    format_options: dict[str, str] | None

    _input_container: av.container.InputContainer | None
    _segment_container: av.container.OutputContainer | None
    _video_stream: av.VideoStream | None
    _frame_no: int
    _segment_count: int
    _last_segment_time: float
    _segment_file: Path | None

    def __init__(
        self,
        input_: str | Path,
        segment_dir: str | Path,
        segment_duration: float,
        segment_pattern: str = "seg_%Y_%m_%d__%H_%M_%S.mp4",
        format_options: dict[str, str] | None = None,
    ) -> None:
        self.input_ = input_
        self.segment_dir = Path(segment_dir)
        self.segment_duration = segment_duration
        self.segment_pattern = segment_pattern
        self.format_options = format_options

        self._input_container = None
        self._segment_container = None
        self._video_stream = None
        self._frame_no = 0
        self._segment_count = 0
        self._last_segment_time = 0.0
        self._segment_file = None

    def __iter__(self) -> Self:
        return self

    def __enter__(self) -> Self:
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._input_container = av.open(str(self.input_), mode="r")
            self._video_stream = self._input_container.streams.video[0]
        except Exception as e:
            raise translate_av_error(e, str(self.input_), "opening input container") from e
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> Literal[False]:
        self._close_segment_container()
        if self._input_container:
            try:
                self._input_container.close()
            except Exception:
                logger.exception("Error closing input container")
            self._input_container = None
        self._video_stream = None
        return False

    def _close_segment_container(self) -> None:
        if self._segment_container:
            try:
                self._segment_container.close()
            except Exception:
                logger.exception("Error closing segment container")
            self._segment_container = None

    def _create_segment_container(self) -> None:
        if not self._video_stream:
            raise VideoNotInContextError()

        segment_path = str(self.segment_dir / self.segment_pattern)
        segment_path = datetime.now().strftime(segment_path)

        try:
            self._segment_container = av.open(segment_path, mode="w")
            output_stream = self._segment_container.add_stream(
                codec_name="libx264",
                rate=self._video_stream.average_rate or self._video_stream.base_rate,
            )
            output_stream.width = self._video_stream.width
            output_stream.height = self._video_stream.height
            output_stream.pix_fmt = "yuv420p"
            output_stream.time_base = self._video_stream.time_base

            if self.format_options:
                for key, value in self.format_options.items():
                    self._segment_container.options[key] = value
        except Exception as e:
            raise translate_av_error(e, segment_path, "creating segment container") from e

    def _write_frame_to_segment(self, frame: av.VideoFrame) -> None:
        if not self._segment_container or not self._video_stream:
            return

        try:
            stream = self._segment_container.streams.video[0]
            for packet in stream.encode(frame):
                self._segment_container.mux(packet)
        except Exception as e:
            raise translate_av_error(e, str(self._segment_container.file), "muxing frame to segment") from e

    def _maybe_rotate_segment(self, frame_time: float) -> None:
        if self._segment_container is None:
            self._create_segment_container()
            self._last_segment_time = frame_time
            return

        if frame_time - self._last_segment_time >= self.segment_duration:
            self._close_segment_container()
            self._segment_count += 1
            self._create_segment_container()
            self._last_segment_time = frame_time

    def __next__(self) -> Frame:
        if not self._input_container or not self._video_stream:
            raise VideoNotInContextError()

        for packet in self._input_container.demux(self._video_stream):
            try:
                for frame in packet.decode():
                    if not isinstance(frame, av.VideoFrame):
                        continue

                    frame_time = float(frame.time)
                    self._maybe_rotate_segment(frame_time)
                    self._write_frame_to_segment(frame)

                    rgb_frame = frame.to_rgb().to_ndarray()
                    result = Frame(raw=rgb_frame, frame_no=self._frame_no)
                    self._frame_no += 1
                    return result
            except Exception as e:
                raise translate_av_error(e, str(self.input_), "reading frame") from e

        raise StopIteration

    @property
    def segment_count(self) -> int:
        return self._segment_count

    @property
    def frame_count(self) -> int:
        return self._frame_no
