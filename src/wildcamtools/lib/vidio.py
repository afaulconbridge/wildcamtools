import logging
import os
import tempfile
import time
from io import BufferedReader
from pathlib import Path
from subprocess import Popen
from typing import Any, Literal, Self

import av
import cv2
import ffmpeg
import ffmpeg.codecs.encoders
import ffmpeg.formats.muxers
import numpy as np

from wildcamtools.lib import Frame
from wildcamtools.lib.errors import (
    FFmpegPipeClosedError,
    ProcessTypeMismatchError,
    VideoNotInContextError,
    VideoProbeError,
    VideoSizeNotSetError,
    VideoWriteError,
    translate_av_error,
)

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


class FrameSourceFFMPEG(FrameSource):
    filename: Path | str
    reader: Popen | None = None
    width: int | None
    height: int | None
    scale: float
    fps: float
    frame_no: int
    hwaccel: str | None
    cumulative_time: float = 0.0

    _named_pipe: Path | None = None
    _named_pipe_reader: BufferedReader | None = None
    _temporary_dir: Path | None = None

    def __init__(
        self,
        filename: Path | str,
        width: int | None = None,
        height: int | None = None,
        scale: float = 1.0,
        fps: float = -1.0,
        hwaccel: str | None = None,
    ):
        super()
        self.filename = filename
        self.width = width
        self.height = height
        self.scale = scale
        self.fps = fps
        self.hwaccel = hwaccel
        self.frame_no = 0

    def _detect_width_height(self) -> None:
        probe = ffmpeg.probe(self.filename)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        if video_stream is None:
            raise VideoProbeError()
        self.width = int(video_stream["width"] * self.scale)
        self.height = int(video_stream["height"] * self.scale)

    def _create_ffmpeg_proc(self) -> Popen[bytes]:
        if not self._named_pipe:
            raise VideoNotInContextError()
        f_in: ffmpeg.AVStream | ffmpeg.VideoStream = ffmpeg.input(
            self.filename,
            hwaccel=self.hwaccel,  # see https://trac.ffmpeg.org/wiki/HWAccelIntro
        )
        # change fps if appropriate
        # change fps first to save rescaling frames that will be dropped
        if self.fps > 0.0:
            f_in = f_in.fps(fps=f"{self.fps}")

        # apply scaling if appropriate
        if self.scale != 1.0:
            # iw = input width, ih=input height
            f_in = f_in.scale(
                w=f"{self.scale:f}*iw",
                h=f"{self.scale:f}*ih",
            )

        # using stdout from ffmpeg is unstable
        # use a named pipe (FIFO) instead - with a context manager
        f_out = f_in.output(
            filename=self._named_pipe,
            f="rawvideo",
            pix_fmt="rgb24",
        )
        res = (
            f_out.global_args(hide_banner=True, loglevel="error")
            .overwrite_output()
            .run_async(pipe_stdout=True, quiet=True)
        )
        if not isinstance(res, Popen):
            raise ProcessTypeMismatchError()
        return res

    def __enter__(self) -> Self:
        self._temporary_dir = Path(tempfile.mkdtemp(prefix="wildcamtools_"))
        self._named_pipe = self._temporary_dir / "pipe.mp4"
        logger.debug(f"Creating pipe {self._named_pipe}")
        os.mkfifo(self._named_pipe)
        logger.debug(f"Created pipe {self._named_pipe}")

        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> Literal[False]:
        if self._named_pipe_reader:
            self._named_pipe_reader.close()
        self._named_pipe_reader = None

        if self._named_pipe:
            os.unlink(self._named_pipe)
        self._named_pipe = None

        if self._temporary_dir:
            os.rmdir(self._temporary_dir)
        self._temporary_dir = None
        return False

    def __next__(self) -> Frame:
        start = time.time()

        if not self.reader:
            if not self.width or not self.height:
                self._detect_width_height()
            self.reader = self._create_ffmpeg_proc()

        if self.reader.poll() is not None:
            raise StopIteration

        if not self._named_pipe_reader:
            logger.debug(f"Opening pipe {self._named_pipe}")
            # have to open the pipe _after_ ffmpeg has started writing into the pipe
            # otherwise it hangs
            if self._named_pipe:
                self._named_pipe_reader = open(self._named_pipe, "rb")  # noqa: SIM115
            logger.debug(f"Opened pipe {self._named_pipe}")

        in_bytes = (
            self._named_pipe_reader.read(self.width * self.height * 3)
            if self._named_pipe_reader and self.width and self.height
            else b""
        )

        if not in_bytes:
            self.reader.wait()
            self.reader = None
            self.width = None
            self.height = None
            raise StopIteration

        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height or 0, self.width or 0, 3])
        frame = Frame(raw=in_frame, frame_no=self.frame_no)
        self.frame_no += 1

        end = time.time()
        self.cumulative_time += end - start

        return frame


class FrameSourceFFMPEGSegmenter(FrameSource):
    segment_dir: str | Path

    def __init__(
        self,
        filename: Path | str,
        segment_dir: str | Path,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.segment_dir = segment_dir
        super().__init__()
        self.filename = filename
        self.width = width
        self.height = height

    def _create_ffmpeg_proc(self) -> Popen:
        input_split = ffmpeg.input(self.filename).split(outputs=2)
        input_0: ffmpeg.VideoStream = input_split.video(0)
        input_1: ffmpeg.VideoStream = input_split.video(1)
        output_0 = input_0.output(
            codec="copy",
            f="segment",
            muxer_options=ffmpeg.formats.muxers.segment(
                segment_time=str(15),  # every N seconds
                segment_format="mp4",
                segment_format_options="movflags=+faststart",
                reset_timestamps=True,
                strftime=True,
            ),
            filename="ffmpeg/out/out_%Y_%m_%d__%H_%M_%S.mp4",
        )
        output_1 = input_1.output(filename="pipe:", f="rawvideo", pix_fmt="rgb24")
        return (
            ffmpeg.merge_outputs(output_0, output_1)
            .global_args(hide_banner=True, loglevel="error")
            .overwrite_output()
            .run_async(pipe_stdout=True, quiet=True)
        )


class FrameWriterFFMPEG:
    """
    Context-managed writer that accepts Frame objects via .write(frame)
    and writes them to a video file using ffmpeg-python. Width/height are
    inferred from the first frame. Assumes Frame.raw is an HxWx3 (RGB)
    or HxWx4 (RGBA) numpy array.
    """

    def __init__(
        self,
        out_filename: Path | str,
        fps: float,
        pix_fmt: str = "rgb24",
        crf: int = 23,
        preset: str = "medium",
    ):
        self.out_filename = out_filename
        self.fps = fps
        self.pix_fmt = pix_fmt
        self.crf = crf
        self.preset = preset

        self._proc: Popen[bytes] | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._started = False

    def _start_process(self) -> None:
        if self._width is None or self._height is None:
            raise VideoSizeNotSetError()
        self._proc = (
            ffmpeg.input(
                "pipe:",
                f="rawvideo",
                pix_fmt=self.pix_fmt,
                s=f"{self._width}x{self._height}",
                readrate=self.fps,
            )
            .output(
                filename=str(self.out_filename),
                pix_fmt="yuv420p",
                r=self.fps,
                encoder_options=ffmpeg.codecs.encoders.libx264(
                    crf=self.crf,
                    preset=self.preset,
                ),
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        self._started = True

    def write(self, frame: np.ndarray) -> None:
        """
        Write a single Frame to the output. Starts the ffmpeg process on the
        first call by inferring width/height from frame.raw.
        """

        # Handle boolean mask input HxW (single channel)
        if frame.ndim == 2:
            frame = np.stack((frame,) * 3, axis=-1)  # HxWx3 uint8

        h, w, ch = frame.shape
        if ch == 4:
            frame = frame[:, :, :3]

        if not self._started:
            self._width = w
            self._height = h
            self._start_process()

        # resize if needed (all frames must match initial dims)
        if frame.shape[0] != (self._height or 0) or frame.shape[1] != (self._width or 0):
            frame = cv2.resize(frame, (int(self._width or 0), int(self._height or 0)), interpolation=cv2.INTER_LINEAR)

        try:
            if self._proc and self._proc.stdin:
                self._proc.stdin.write(frame.astype(np.uint8).tobytes())
        except BrokenPipeError as err:
            # allow ffmpeg to fail silently or raise a clearer error
            raise FFmpegPipeClosedError() from err

    def close(self) -> None:
        """Finish writing and close the ffmpeg process."""
        if self._proc and self._proc.stdin:
            self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
            self._started = False
            self._width = None
            self._height = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> Literal[False]:
        self.close()
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
        self.close()
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

        logger.debug(f"Opening video writer for {self.out_filename} (codec={self.codec}, fps={self.fps})")
        self._container = av.open(str(self.out_filename), mode="w")
        stream = self._container.add_stream(self.codec, rate=int(self.fps))
        if not isinstance(stream, av.video.stream.VideoStream):
            raise VideoWriteError(str(self.out_filename), "opening container", "Expected video stream")
        self._stream = stream
        self._stream.width = self._width
        self._stream.height = self._height
        self._stream.pix_fmt = self.pix_fmt
        self._stream.options = {"crf": str(self.crf), "preset": self.preset}
        self._started = True
        logger.debug(f"Video writer opened: {self._width}x{self._height}")

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
            logger.warning(f"Error closing video writer: {e}")
        finally:
            self._container = None
            self._stream = None
            self._started = False
            self._width = None
            self._height = None
            self._frame_count = 0
