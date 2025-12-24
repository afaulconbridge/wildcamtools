from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2


class Colourspace(Enum):
    """Enum representing different color spaces."""

    RGB = "rgb"
    greyscale = "greyscale"
    boolean = "boolean"  # mask


@dataclass(frozen=True)
class VideoStats:
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
        """
        Return the duration of a single video frame in milliseconds

        Note: this rounds down
        """
        return int(1000 / self.fps)

    @property
    def duration_in_sconds(self) -> float:
        """
        Return the total duration of the video in seconds
        """
        return self.frame_count / self.fps


def get_video_stats(filename: str | Path) -> VideoStats:
    """Retrieves metadata about a video file.

    Args:
        filename (str | Path): Path to the video file

    Returns:
        VideoFileStats: Object containing video metadata

    Raises:
        RuntimeError: If unable to read frames from the video
    """
    video_capture = None
    try:
        video_capture = cv2.VideoCapture(str(filename), cv2.CAP_ANY)
        fps = float(video_capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        # frame details are more reliable than capture properties
        # x = cv2.CAP_PROP_FRAME_WIDTH
        # y = cv2.CAP_PROP_FRAME_HEIGHT
        # bw = cv2.CAP_PROP_MONOCHROME
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
