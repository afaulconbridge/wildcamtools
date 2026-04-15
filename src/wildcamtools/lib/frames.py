from pathlib import Path

import cv2
from skimage.metrics import structural_similarity

from wildcamtools.lib.stats import VideoStats

from . import Frame, FrameHandler


class FrameImageWriter(FrameHandler):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        super().__init__()

    def handle(self, frame: Frame) -> Frame | None:
        filename = self.output_dir / f"frame_{frame.frame_no:05d}.jpg"
        cv2.imwrite(str(filename), frame.raw)
        return frame


class Rescaler(FrameHandler):
    stats: VideoStats
    x: int
    y: int
    xy: tuple[int, int]
    fps: float
    source_frametime: float  # milliseconds
    target_frametime: float  # milliseconds
    now: float = 0.0

    def __init__(
        self,
        stats: VideoStats,
        x: int | None = None,
        y: int | None = None,
        fps: float | None = None,
    ):
        self.stats = stats
        self.x = stats.x if x is None else x
        self.y = stats.y if y is None else y
        self.xy = (self.x, self.y)
        self.fps = stats.fps if fps is None else fps
        self.source_frametime = stats.frame_duration
        self.target_frametime = 1000.0 / self.fps
        super().__init__()

    def handle(self, frame: Frame) -> Frame | None:
        self.now += self.source_frametime
        if self.now >= self.target_frametime:
            # were going to return this frame, so rescale it
            frame_rescaled = cv2.resize(frame.raw, self.xy, interpolation=cv2.INTER_LINEAR)
            self.now -= self.target_frametime
            return Frame(raw=frame_rescaled, frame_no=frame.frame_no)
        else:
            # skip this frame
            return None


class FilterSSIM(FrameHandler):
    """Structural Similarity Image Metric based frame filter"""

    similarity_minimum: float
    frame_previous_interesting: Frame | None = None

    def __init__(self, similarity_minimum: float = 0.9):
        self.similarity_minimum = similarity_minimum
        super().__init__()

    def handle(self, frame: Frame) -> Frame | None:
        # no previous frame of interest, so this frame is interesting by default
        if self.frame_previous_interesting is None:
            self.frame_previous_interesting = frame
            return frame

        # this calculation is pretty slow
        similarity = float(
            structural_similarity(self.frame_previous_interesting.raw, frame.raw, data_range=255, channel_axis=2)
        )

        if similarity > self.similarity_minimum:
            # frame is too similar to the previous interesting frame, skip it
            return None
        else:
            self.frame_previous_interesting = frame
            return frame
