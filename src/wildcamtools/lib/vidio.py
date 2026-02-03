import logging
import os
import tempfile
import time
from io import BufferedReader
from pathlib import Path
from subprocess import Popen
from typing import Any, Self

import cv2
import numpy as np

import ffmpeg
from wildcamtools.lib import Frame

logger = logging.getLogger(__name__)


class FrameSource:
    frame_no: int

    def __init__(self):
        self.frame_no = 0

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> bool:
        return False


class FileFrameSourceCV2(FrameSource):
    filename: str
    cap: cv2.VideoCapture | None = None

    def __init__(self, filename: str):
        super()
        self.filename = filename

    def __enter__(self) -> Self:
        self.cap = cv2.VideoCapture(self.filename)

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> bool:
        self.cap.release()
        self.cap = None
        return False

    def __next__(self) -> Frame:
        if not self.cap:
            raise RuntimeError("Must be used in context")

        ret, frame = self.cap.read()
        if ret:
            frame = Frame(raw=frame, frame_no=self.frame_no)
            self.frame_no += 1
            return frame
        else:
            raise StopIteration


class FrameSourceFFMPEG(FrameSource):
    filename: str
    reader: Popen | None = None
    width: int | None
    height: int | None
    scale: float
    frame_no: int
    hwaccel: str | None
    cumulative_time: int = 0

    _named_pipe: Path | None = None
    _named_pipe_reader: BufferedReader | None = None
    _temporary_dir: Path | None = None

    def __init__(
        self,
        filename: str,
        width: int | None = None,
        height: int | None = None,
        scale: float = 1.0,
        hwaccel: str | None = None,
    ):
        super()
        self.filename = filename
        self.width = width
        self.height = height
        self.scale = scale
        self.hwaccel = hwaccel
        self.frame_no = 0

    def _detect_width_height(self) -> None:
        probe = ffmpeg.probe(self.filename)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        self.width = int(video_stream["width"] * self.scale)
        self.height = int(video_stream["height"] * self.scale)

    def _create_ffmpeg_proc(self) -> Popen[bytes]:
        if not self._named_pipe:
            raise RuntimeError("Must be used in context")
        f_in = ffmpeg.input(
            self.filename,
            hwaccel=self.hwaccel,  # see https://trac.ffmpeg.org/wiki/HWAccelIntro
        )
        # apply scaling if appropriate
        if self.scale != 1.0:
            # iw = input width, ih=input height
            f_in = f_in.scale(w=f"{self.scale:f}*iw", h=f"{self.scale:f}*ih")

        # using stdout from ffmpeg is unstable
        # use a named pipe (FIFO) instead - with a context manager
        f_out = f_in.output(
            filename=self._named_pipe,
            f="rawvideo",
            pix_fmt="rgb24",
        )
        return (
            f_out.global_args(hide_banner=True, loglevel="error")
            .overwrite_output()
            .run_async(pipe_stdout=True, quiet=True)
        )

    def __enter__(self) -> Self:
        self._temporary_dir = Path(tempfile.mkdtemp(prefix="wildcamtools_"))
        self._named_pipe = self._temporary_dir / "pipe.mp4"
        logger.debug(f"Creating pipe {self._named_pipe}")
        os.mkfifo(self._named_pipe)
        logger.debug(f"Created pipe {self._named_pipe}")

        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> bool:
        self._named_pipe_reader.close()
        self._named_pipe_reader = None

        os.unlink(self._named_pipe)
        self._named_pipe = None

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
            self._named_pipe_reader = open(self._named_pipe, "rb")
            logger.debug(f"Opened pipe {self._named_pipe}")

        in_bytes = self._named_pipe_reader.read(self.width * self.height * 3)

        if not in_bytes:
            self.reader.wait()
            self.reader = None
            self.width = None
            self.height = None
            raise StopIteration

        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
        frame = Frame(raw=in_frame, frame_no=self.frame_no)
        self.frame_no += 1

        end = time.time()
        self.cumulative_time += end - start

        return frame


class FrameSourceFFMPEGSegmenter(FrameSource):
    segment_dir: str | Path

    def __init__(self, filename: str, segment_dir: str | Path, width: int | None = None, height: int | None = None):
        self.segment_dir = segment_dir
        super().__init__(filename=filename, width=width, height=height)

    def _create_ffmpeg_proc(self) -> Popen:
        input_split: ffmpeg.VideoStream = ffmpeg.input(self.filename).split(outputs=2)
        input_0: ffmpeg.VideoStream = input_split.video(0)
        input_1: ffmpeg.VideoStream = input_split.video(1)
        output_0 = input_0.output(
            codec="copy",
            f="segment",
            muxer_options=ffmpeg.formats.muxers.segment(
                segment_time=15,  # every N seconds
                segment_format="mp4",
                segment_format_options="movflags=+faststart",
                reset_timestamps=1,
                strftime=1,
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
        out_filename: str,
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

        self._proc = None
        self._width: int | None = None
        self._height: int | None = None
        self._started = False

    def _start_process(self) -> None:
        if self._width is None or self._height is None:
            raise ValueError("must have size")
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
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)

        try:
            self._proc.stdin.write(frame.astype(np.uint8).tobytes())
        except BrokenPipeError:
            # allow ffmpeg to fail silently or raise a clearer error
            raise RuntimeError("FFmpeg process pipe is closed")

    def close(self) -> None:
        """Finish writing and close the ffmpeg process."""
        if self._proc:
            self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
            self._started = False
            self._width = None
            self._height = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
