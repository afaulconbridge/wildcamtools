import time
from subprocess import Popen

import cv2
import numpy as np

import ffmpeg
from wildcamtools.lib import Frame


class FrameSource:
    frame_no: int

    def __init__(self):
        self.frame_no = 0

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        raise NotImplementedError


class FileFrameSourceCV2(FrameSource):
    filename: str
    cap: cv2.VideoCapture | None = None

    def __init__(self, filename: str):
        super()
        self.filename = filename

    def __next__(self) -> Frame:
        if not self.cap:
            self.cap = cv2.VideoCapture(self.filename)

        ret, frame = self.cap.read()
        if ret:
            frame = Frame(raw=frame, frame_no=self.frame_no)
            self.frame_no += 1
            return frame
        else:
            self.cap.release()
            self.cap = None
            raise StopIteration


class FrameSourceFFMPEG(FrameSource):
    filename: str
    reader: Popen | None = None
    width: int | None
    height: int | None
    frame_no: int
    cumulative_time: int = 0

    def __init__(self, filename: str, width: int | None = None, height: int | None = None):
        super()
        self.filename = filename
        self.width = width
        self.height = height
        self.frame_no = 0

    def _detect_width_height(self) -> None:
        probe = ffmpeg.probe(self.filename)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        self.width = int(video_stream["width"])
        self.height = int(video_stream["height"])

    def __next__(self) -> Frame:
        start = time.time()

        if not self.reader:
            if not self.width or not self.height:
                self._detect_width_height()
            self.reader = (
                ffmpeg.input(self.filename)
                .output(filename="pipe:", f="rawvideo", pix_fmt="rgb24")
                .run_async(pipe_stdout=True, quiet=True)
            )

        if self.reader.poll() is not None:
            raise StopIteration
        in_bytes = self.reader.stdout.read(self.width * self.height * 3)

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
