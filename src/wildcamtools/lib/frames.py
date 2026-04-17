from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from wildcamtools.lib.errors.core import InvalidAlphaError, InvalidMaxMagnitudeError
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


class MotionFlowHighlighter(FrameHandler):
    alpha: float
    max_magnitude: float
    prev_gray: cv2.typing.MatLike | None = None
    flow_magnitude: cv2.typing.MatLike | None = None
    flow_angle: cv2.typing.MatLike | None = None

    def __init__(self, alpha: float = 0.5, max_magnitude: float = 10.0):
        if alpha < 0.0 or 1.0 < alpha:  # noqa: SIM300
            raise InvalidAlphaError(alpha)
        if max_magnitude <= 0.0:
            raise InvalidMaxMagnitudeError(max_magnitude)

        self.alpha = alpha
        self.max_magnitude = max_magnitude
        super().__init__()

    def handle(self, frame: Frame) -> Frame | None:
        curr_gray = cv2.cvtColor(frame.raw, cv2.COLOR_RGB2GRAY)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return frame

        # Calculate dense optical flow
        # note: upstream typing is overspecific for optionals
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # type: ignore[call-overload]

        # Compute magnitude and angle of 2D vectors
        self.flow_magnitude, self.flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Create HSV image
        hsv = np.zeros_like(frame.raw)
        # Set saturation to maximum
        hsv[..., 1] = 255
        # Set hue based on angle (0 to 180 in OpenCV HSV)
        hsv[..., 0] = self.flow_angle * 180 / np.pi / 2
        # Set value based on magnitude (normalized to 0-255 using fixed max_magnitude)
        hsv[..., 2] = np.clip((self.flow_magnitude / self.max_magnitude) * 255, 0, 255).astype(np.uint8)

        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        highlighted = cv2.addWeighted(frame.raw, 1 - self.alpha, flow_bgr, self.alpha, 0)

        self.prev_gray = curr_gray
        return Frame(raw=highlighted, frame_no=frame.frame_no)
