import contextlib
import threading
import time
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np

from wildcamtools.lib.errors import CannotSeekVideoError, RTSPCloseTimeoutError, RTSPOpenError
from wildcamtools.lib.stats import Colourspace, VideoFileStats


def generate_frames_cv2(filename: str | Path, *, start_ms: float | None = None) -> Generator[tuple[int, np.ndarray]]:
    """Generates video frames with OpenCV.

    Args:
        filename (str | Path): Path to the video file
        start_ms (float | None): Optional start time in milliseconds

    Yields:
        tuple[int, np.ndarray]: A tuple containing (frame_number, frame_array)

    Raises:
        FrameError: If unable to seek to the requested position
    """
    video_capture = cv2.VideoCapture(str(filename), cv2.CAP_ANY)
    try:
        # cv2.CAP_PROP_POS_FRAMES is unreliable - see https://github.com/opencv/opencv/issues/9053
        if start_ms is not None:
            set_success = video_capture.set(cv2.CAP_PROP_POS_MSEC, start_ms)
            if not set_success:
                raise CannotSeekVideoError()

        success = True
        while success:
            success, array = video_capture.read()
            frame_no = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            if success:
                yield frame_no, array
    finally:
        video_capture.release()


def generate_frames_cv2_rtsp(
    rtsp_url: str,
) -> Generator[np.ndarray]:
    """
    Generates video frames from an RTSP stream using OpenCV.

    Args:
        rtsp_url (str): The RTSP stream URL to connect to.

    Yields:
        np.ndarray: frame as an array

    Raises:
        RTSPError: If unable to read from the RTSP stream.
    """

    video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_ANY)
    try:
        if not video_capture.isOpened():
            raise RTSPOpenError(rtsp_url)

        success = True
        while success:
            success, array = video_capture.read()
            if success:
                yield array
    finally:
        video_capture.release()


def _grab_frame(
    video_capture: cv2.VideoCapture,
    lock: threading.Lock,
    terminator: threading.Event,
    sleep: float | None,
) -> None:
    """Grab frames from a video capture object while respecting threading constraints.

    This function continuously grabs frames from the provided VideoCapture object
    in a thread-safe manner using the provided lock. The termination event can be set
    to stop grabbing frames gracefully.
    An optional sleep interval can be specified to control the frame rate. If this is
    None then the lock may never be freed enough for aother thread to retrieve the video
    frames.

    Note that OpenCV uses "grab" to mean capture an image which is disctinct from
    "retrieve" which is to receive and decode the last image captured.

    Args:
        video_capture: OpenCV VideoCapture object to grab frames from
        lock: Threading lock to ensure thread-safe access to the video capture
        terminator: Threading Event that signals when to stop frame grabbing
        sleep: Optional sleep duration between frame grabs to control frame rate
    """
    success = True
    while success and not terminator.is_set():
        with lock:
            success = video_capture.grab()
        # share the lock more nicely
        if sleep:
            time.sleep(sleep)


def generate_latest_frames_cv2_rtsp(
    rtsp_url: str, termination_timeout: float = 5, grab_sleep_interval: float = 0.01
) -> Generator[np.ndarray]:
    """
    Generator that always returns the freshest frame.

    This uses a thread in the background to fetch frames from the camera, and might skip frames
    if the consumer can't keep up.
    """

    video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_ANY)
    try:
        if not video_capture.isOpened():
            raise RTSPOpenError(rtsp_url)

        terminator = threading.Event()
        lock = threading.Lock()
        frame_thread = threading.Thread(
            target=_grab_frame,
            args=[video_capture, lock, terminator, grab_sleep_interval],
            daemon=True,
        )
        try:
            frame_thread.start()
            while frame_thread.is_alive():
                with lock:
                    success, array = video_capture.retrieve()
                    if success:
                        yield array
                    else:
                        break
        finally:
            terminator.set()
            frame_thread.join(termination_timeout)
            if frame_thread.is_alive():
                raise RTSPCloseTimeoutError(rtsp_url)
    finally:
        video_capture.release()


@contextlib.contextmanager
def save_video(output_path: Path | str, stats: VideoFileStats, codex: str = "mp4v") -> Generator[cv2.VideoWriter]:
    """
    Saves a video by creating a VideoWriter object and yielding it as a context manager.

    Note: VideoWriter expects BGR format frames

    Args:
        output_path (Path | str): Path to the output video file.
        stats (VideoFileStats): Statistics of the input video file (FPS, dimensions).
        codex (str, optional): Codec to use for writing the output video. Defaults to "mp4v".

    Yields:
        cv2.VideoWriter: A VideoWriter object that can be used to write frames to the output video.
    """
    fourcc = cv2.VideoWriter_fourcc(*codex)  # type: ignore
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        stats.fps,
        (stats.x, stats.y),
        isColor=stats.colourspace in {Colourspace.RGB},
    )
    try:
        yield out
    finally:
        out.release()
