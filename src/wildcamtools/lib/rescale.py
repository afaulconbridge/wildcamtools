import cv2

from wildcamtools.lib import Frame, FrameHandler
from wildcamtools.lib.stats import VideoFileStats


class Rescaler(FrameHandler):
    stats: VideoFileStats
    x: int
    y: int
    xy: tuple[int, int]
    fps: float
    source_frametime: float  # milliseconds
    target_frametime: float  # milliseconds
    now: float = 0.0

    def __init__(
        self,
        stats: VideoFileStats,
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
