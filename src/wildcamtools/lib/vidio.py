import logging
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Self

import av
import av.container
import cv2
import numpy as np

from wildcamtools.lib import Frame
from wildcamtools.lib.errors import VideoNotInContextError, VideoSizeNotSetError, VideoWriteError, translate_av_error
from wildcamtools.lib.errors.core import StreamNotFoundError

logger = logging.getLogger(__name__)


class FrameSource:
    frame_no: int

    def __init__(self) -> None:
        self.frame_no = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Frame:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> Literal[False]:
        return False


class FileFrameSourceCV2(FrameSource):
    filename: str
    cap: cv2.VideoCapture | None = None

    def __init__(self, filename: str):
        super()
        self.filename = filename

    def __enter__(self) -> Self:
        self.cap = cv2.VideoCapture(self.filename)
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> Literal[False]:
        if self.cap:
            self.cap.release()
        self.cap = None
        return False

    def __next__(self) -> Frame:
        if not self.cap:
            raise VideoNotInContextError()

        ret, raw = self.cap.read()
        if ret:
            frame = Frame(raw=raw, frame_no=self.frame_no)
            self.frame_no += 1
            return frame
        else:
            raise StopIteration


class VideoReader(FrameSource):
    """PyAV-based video reader with frame scaling and FPS control.

    Uses PyAV for direct library access without subprocess or named pipes.
    """

    filename: Path | str
    width: int | None
    height: int | None
    scale: float
    fps: float
    frame_no: int
    hwaccel: str | None

    _container: av.container.InputContainer | None
    _stream: av.VideoStream | None
    _target_fps: float | None
    _last_pts: float | None
    _decoded_frames: list[av.VideoFrame]

    def __init__(
        self,
        filename: Path | str,
        width: int | None = None,
        height: int | None = None,
        scale: float = 1.0,
        fps: float = -1.0,
        hwaccel: str | None = None,
    ):
        super().__init__()
        self.filename = filename
        self.width = width
        self.height = height
        self.scale = scale
        self.fps = fps
        self.hwaccel = hwaccel
        self.frame_no = 0
        self._container = None
        self._stream = None
        self._target_fps = None
        self._last_pts = None
        self._decoded_frames: list[av.VideoFrame] = []

        if hwaccel is not None:
            logger.warning("Hardware acceleration (hwaccel) is not yet implemented")

    def __enter__(self) -> Self:
        try:
            self._container = av.open(self.filename)
            self._stream = self._container.streams.video[0]
        except av.error.FFmpegError as e:
            raise translate_av_error(e, str(self.filename), "open") from e
        except (KeyError, IndexError) as e:
            raise StreamNotFoundError(str(self.filename), "video") from e

        if self.width is None or self.height is None:
            self._detect_dimensions()

        if self.fps > 0.0:
            self._target_fps = self.fps
        else:
            self._target_fps = None

        self._last_pts = None
        self.frame_no = 0
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> Literal[False]:
        if self._container:
            self._container.close()
        self._container = None
        self._stream = None
        self._target_fps = None
        self._last_pts = None
        return False

    def _detect_dimensions(self) -> None:
        if not self._stream:
            raise VideoNotInContextError()
        self.width = int(self._stream.width * self.scale)
        self.height = int(self._stream.height * self.scale)

    def __next__(self) -> Frame:
        if not self._container or not self._stream:
            raise VideoNotInContextError()

        if not self.width or not self.height:
            raise VideoSizeNotSetError()

        try:
            while True:
                if self._decoded_frames:
                    return self._process_next_buffered_frame()

                self._decode_next_packet()

                if not self._decoded_frames:
                    raise StopIteration
        except (av.error.EOFError, EOFError) as e:
            raise StopIteration from e
        except av.error.FFmpegError as e:
            raise translate_av_error(e, str(self.filename), "read frame") from e

    def _decode_next_packet(self) -> None:
        """Decode the next packet and buffer all video frames."""
        if not self._container or not self._stream:
            raise VideoNotInContextError()
        for packet in self._container.demux(self._stream):
            decoded = packet.decode()
            for frame in decoded:
                if isinstance(frame, av.VideoFrame):
                    self._decoded_frames.append(frame)
            if self._decoded_frames:
                break

    def _process_next_buffered_frame(self) -> Frame:
        """Process and return the next buffered frame."""
        frame = self._decoded_frames.pop(0)
        if self._should_drop_frame(frame):
            return self.__next__()

        rgb_frame = frame.to_ndarray(format="rgb24")

        if self.scale != 1.0 and self.width and self.height:
            rgb_frame = cv2.resize(  # type: ignore[assignment]
                rgb_frame,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )

        result = Frame(raw=rgb_frame, frame_no=self.frame_no)
        self.frame_no += 1
        return result

    def _should_drop_frame(self, frame: av.VideoFrame) -> bool:
        if self._target_fps is None:
            return False

        if not self._stream or not self._stream.time_base:
            return False

        if frame.time is not None:
            current_pts = frame.time
        elif frame.pts is not None:
            current_pts = frame.pts * float(self._stream.time_base)
        else:
            return False

        if self._last_pts is None:
            self._last_pts = current_pts
            return False

        min_interval = 1.0 / self._target_fps
        elapsed = current_pts - self._last_pts

        if elapsed < min_interval:
            return True

        self._last_pts = current_pts
        return False


class VideoWriter:
    """
    Context-managed video writer using PyAV for direct library access.
    Accepts numpy arrays (HxWx3 RGB or HxWx4 RGBA) and writes to video files.
    """

    # TODO: Add hardware acceleration support (e.g., h264_nvenc, h264_vaapi)

    def __init__(
        self,
        out_filename: Path | str,
        fps: float,
        codec: str = "libx264",
        crf: int = 23,
        preset: str = "medium",
        pix_fmt: str = "yuv420p",
    ):
        self.out_filename = Path(out_filename)
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.pix_fmt = pix_fmt

        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._started = False
        self._frame_count = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> Literal[False]:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
            logger.exception("Error during cleanup while handling another exception")
        return False

    def write(self, frame: np.ndarray) -> None:
        """
        Write a single frame to the output video.
        Infers dimensions from the first frame.
        """
        if frame.ndim == 2:
            frame = np.stack((frame,) * 3, axis=-1)

        h, w, ch = frame.shape
        if ch == 4:
            frame = frame[:, :, :3]

        if not self._started:
            self._width = w
            self._height = h
            self._open_container()

        if frame.shape[0] != (self._height or 0) or frame.shape[1] != (self._width or 0):
            frame = cv2.resize(frame, (int(self._width or 0), int(self._height or 0)), interpolation=cv2.INTER_LINEAR)

        self._write_frame(frame)

    def _open_container(self) -> None:
        if self._width is None or self._height is None:
            raise VideoSizeNotSetError()

        logger.debug("Opening video writer for %s (codec=%s, fps=%s)", self.out_filename, self.codec, self.fps)
        self._container = av.open(str(self.out_filename), mode="w")
        stream = self._container.add_stream(self.codec, rate=Fraction(str(self.fps)))
        if not isinstance(stream, av.video.stream.VideoStream):
            raise VideoWriteError(str(self.out_filename), "opening container", "Expected video stream")
        self._stream = stream
        self._stream.width = self._width
        self._stream.height = self._height
        self._stream.pix_fmt = self.pix_fmt
        self._stream.options = {"crf": str(self.crf), "preset": self.preset}
        self._started = True
        logger.debug("Video writer opened: %dx%d", self._width, self._height)

    def _write_frame(self, frame: np.ndarray) -> None:
        if not self._container or not self._stream:
            raise VideoWriteError(str(self.out_filename), "writing frame", "Container or stream not initialized")

        try:
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            av_frame.pts = self._frame_count
            self._frame_count += 1

            packets = self._stream.encode(av_frame)
            for packet in packets:
                self._container.mux(packet)
        except Exception as e:
            logger.exception("Failed to write frame %d to %s", self._frame_count, self.out_filename)
            raise translate_av_error(e, str(self.out_filename), "writing frame") from e

    def close(self) -> None:
        """Flush encoder and close container."""
        if not self._container or not self._stream:
            return

        try:
            packets = self._stream.encode(None)
            for packet in packets:
                self._container.mux(packet)

            self._container.close()
        except Exception as e:
            raise translate_av_error(e, str(self.out_filename), "closing writer") from e
        finally:
            self._container = None
            self._stream = None
            self._started = False
            self._width = None
            self._height = None
            self._frame_count = 0
