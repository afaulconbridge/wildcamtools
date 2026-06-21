from enum import StrEnum
from pathlib import Path

import cv2
from pydantic import BaseModel


class Colourspace(StrEnum):
    """Enum representing different color spaces."""

    RGB = "rgb"
    greyscale = "greyscale"
    boolean = "boolean"  # mask


class VideoStats(BaseModel):
    """Contains metadata about a video source, typically a file.

    Attributes:
        fps (float): Frames per second of the video
        frame_count (int): Total number of frames in the video
        x (int): Width of the video frames in pixels
        y (int): Height of the video frames in pixels
        colourspace (Colourspace): The color space of the video frames

    Properties:
        shape (tuple[int, int, int]): Shape of the video frames as (height, width, channels)
        nbytes (int): Total bytes required to store one frame
        frame_duration (int): Duration of a single frame in milliseconds

    """

    # special pydantic configs
    class Config:
        use_enum_values = True

    fps: float
    frame_count: int
    x: int
    y: int
    colourspace: Colourspace

    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns the shape of the video frames as (height, width, channels)."""
        # Y,X to match openCV
        return (self.y, self.x, 1 if self.colourspace == Colourspace.greyscale else 3)

    @property
    def nbytes(self) -> int:
        """Calculates and returns the total bytes required to store one frame."""
        match self.colourspace:
            case Colourspace.greyscale:
                return self.x * self.y
            case Colourspace.RGB:
                return self.x * self.y * 3
            case _:
                raise NotImplementedError("Unimplemented colourspace")

    @property
    def frame_duration(self) -> int:
        """Return the duration of a single video frame in milliseconds

        Note: this rounds down
        """
        return int(1000 / self.fps)

    @property
    def duration_in_sconds(self) -> float:
        """Return the total duration of the video in seconds"""
        return self.frame_count / self.fps


def get_video_stats(filename: str | Path) -> VideoStats:
    """Retrieves metadata about a video file.

    Uses PyAV for reliable FPS detection, as OpenCV's CAP_PROP_FPS can return
    incorrect values for certain video codecs.

    Args:
        filename (str | Path): Path to the video file

    Returns:
        VideoFileStats: Object containing video metadata

    Raises:
        RuntimeError: If unable to read frames from the video

    """
    import av

    # Use PyAV for reliable FPS and frame count
    container = None
    video_capture = None
    try:
        container = av.open(str(filename))
        video_stream = container.streams.video[0]

        # Get FPS from codec's base rate (real frame rate)
        # This is more reliable for frame number calculations than average_rate
        # which accounts for variable frame rate playback
        if video_stream.codec_context.framerate:
            fps = float(video_stream.codec_context.framerate)
        elif video_stream.average_rate:
            fps = float(video_stream.average_rate)
        elif video_stream.guessed_rate:
            fps = float(video_stream.guessed_rate)
        else:
            fps = 30.0  # Fallback

        # Get frame count from stream (prefer frames from codec, then duration-based)
        if video_stream.frames:
            frame_count = video_stream.frames
        elif video_stream.duration is not None and video_stream.time_base:
            duration_seconds = float(video_stream.duration * video_stream.time_base)
            frame_count = int(duration_seconds * fps)
        else:
            frame_count = 0

        # OpenCV for frame dimensions (more reliable for actual pixel data)
        video_capture = cv2.VideoCapture(str(filename), cv2.CAP_ANY)
        (success, frame) = video_capture.read()
        if not success:
            msg = f"Unable to read frame from {filename}"
            raise RuntimeError(msg)
        y = frame.shape[0]
        x = frame.shape[1]
        # this doesn't work, always read as colour
        colourspace = Colourspace.greyscale if frame.shape[2] == 1 else Colourspace.RGB

        return VideoStats(
            fps=fps,
            frame_count=frame_count,
            x=x,
            y=y,
            colourspace=colourspace,
        )
    finally:
        if video_capture:
            video_capture.release()
            video_capture = None
        if container:
            container.close()
