import logging
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from wildcamtools.lib.errors.core import (
    ExpansionValueError,
    InertiaValueError,
    InvalidAlphaError,
    InvalidMaxMagnitudeError,
)
from wildcamtools.lib.motion import MotionHandler
from wildcamtools.lib.stats import VideoStats

from . import BBox, Frame, FrameHandler

logger = logging.getLogger(__name__)


def resize_with_aspect_ratio(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resizes image to fit within target_size while maintaining aspect ratio.

    Args:
        image: The input image as a numpy array.
        target_size: A tuple of (width, height).

    Returns:
        The resized numpy array.

    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # Calculate scaling factor
    ratio_w = target_w / float(w)
    ratio_h = target_h / float(h)
    scale = min(ratio_w, ratio_h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def pad_to_size(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Pads an image with black borders to reach the exact target_size.

    Args:
        image: The input image (usually after being resized).
        target_size: A tuple of (width, height).

    Returns:
        The padded numpy array.

    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # Calculate required padding
    delta_w = target_w - w
    delta_h = target_h - h

    # Distribute padding evenly on both sides
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def match_image_sizes(
    img1: np.ndarray,
    img2: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Main coordinator function that uses resize and pad to make two images identical in size.

    Args:
        img1: First input image.
        img2: Second input image.
        target_size: Optional target (width, height). If None, uses the max of both images.

    Returns:
        A tuple containing the two processed numpy arrays.

    """
    if target_size is None:
        max_w = max(img1.shape[1], img2.shape[1])
        max_h = max(img1.shape[0], img2.shape[0])
        target_size = (max_w, max_h)

    # Process Image 1: Resize -> Pad
    rescaled1 = resize_with_aspect_ratio(img1, target_size)
    padded1 = pad_to_size(rescaled1, target_size)

    # Process Image 2: Resize -> Pad
    rescaled2 = resize_with_aspect_ratio(img2, target_size)
    padded2 = pad_to_size(rescaled2, target_size)

    return padded1, padded2


class FrameImageWriter(FrameHandler):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        super().__init__()

    def handle(self, frame: Frame) -> Frame:
        if frame.filter_keep:
            filename = self.output_dir / f"frame_{frame.frame_no:05d}.jpg"
            cv2.imwrite(str(filename), frame.output)
            logger.debug("Writing %s -> %s", frame.frame_no, filename)
        return frame


class Rescaler(FrameHandler):
    stats: VideoStats
    x: int
    y: int
    fps: float
    preserve_aspect: bool
    source_frametime: float  # milliseconds
    target_frametime: float  # milliseconds
    now: float = 0.0

    def __init__(
        self,
        stats: VideoStats,
        x: int | None = None,
        y: int | None = None,
        fps: float | None = None,
        preserve_aspect: bool = True,
    ):
        self.stats = stats
        self.x = stats.x if x is None else x
        self.y = stats.y if y is None else y
        self.fps = stats.fps if fps is None else fps
        self.preserve_aspect = preserve_aspect
        self.source_frametime = stats.frame_duration
        self.target_frametime = 1000.0 / self.fps
        self.now = self.target_frametime  # ensure first frame is kept
        super().__init__()

    def handle(self, frame: Frame) -> Frame:
        self.now += self.source_frametime

        # count filtered frames for frametime but don't do anything else with them
        if not frame.filter_keep:
            return frame

        if self.now >= self.target_frametime:
            # were going to return this frame, so rescale it
            # if multiple frames have not been kept this could be significantly high
            if self.preserve_aspect:
                frame.rescale = resize_with_aspect_ratio(frame.output, (self.x, self.y))
            else:
                frame.rescale = cv2.resize(frame.output, (self.x, self.y), interpolation=cv2.INTER_LINEAR)

            while self.now >= self.target_frametime:
                self.now -= self.target_frametime

            logger.debug("Rescaled %s", frame.frame_no)
            return frame
        # skip this frame
        frame.filter_keep = False
        return frame


class FilterSSIM(FrameHandler):
    """Structural Similarity Image Metric based frame filter"""

    similarity_minimum: float
    frame_previous_interesting: np.ndarray | None = None

    def __init__(self, similarity_minimum: float = 0.9):
        self.similarity_minimum = similarity_minimum
        super().__init__()

    def handle(self, frame: Frame) -> Frame:
        # skip frames filtered out already
        if not frame.filter_keep:
            return frame

        # no previous frame of interest, so this frame is interesting by default
        if self.frame_previous_interesting is None:
            self.frame_previous_interesting = frame.output.copy()
            return frame

        frame_current = frame.output
        frame_previous = self.frame_previous_interesting

        # to calculate ssim, images must be identical sizes so rescale+pad if necessary
        frame_y, frame_x = frame_current.shape[:2]
        frame_previous_y, frame_previous_x = frame_previous.shape[:2]
        if frame_x != frame_previous_x or frame_y != frame_previous_y:
            frame_previous_resized, frame_resized = match_image_sizes(frame_previous, frame_current)
        else:
            frame_previous_resized, frame_resized = (frame_previous, frame_current)

        # this calculation is pretty slow
        similarity = float(structural_similarity(frame_previous_resized, frame_resized, data_range=255, channel_axis=2))
        logger.debug("SSIM (%s) = %04f", frame.frame_no, similarity)

        if similarity > self.similarity_minimum:
            # frame is too similar to the previous interesting frame, skip it
            frame.filter_keep = False
        else:
            self.frame_previous_interesting = frame.output.copy()
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

    def handle(self, frame: Frame) -> Frame:
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
        frame.raw = highlighted
        return frame


class CropPanHandler(FrameHandler):
    motion_handler: MotionHandler
    window: BBox | None = None
    expansion: float
    inertia: float

    def __init__(self, motion_handler: MotionHandler, expansion: float = 0.75, inertia: float = 10.0) -> None:
        super().__init__()
        self.motion_handler = motion_handler
        self.expansion = expansion
        if inertia < 0.0:
            raise InertiaValueError()
        if expansion < 0.0:
            raise ExpansionValueError()
        self.inertia = inertia

    def handle(self, frame: Frame) -> Frame:
        frame_mask = self.motion_handler.handle(frame)
        frame_y, frame_x = frame.raw.shape[:2]
        # some motion was detected
        if self.motion_handler.motion_mask is not None and (bboxes := self.motion_handler.get_contour_bboxes()):
            # get minimum box and maximum box
            x1 = min(b.x1 for b in bboxes)
            x2 = max(b.x2 for b in bboxes)
            y1 = min(b.y1 for b in bboxes)
            y2 = max(b.y2 for b in bboxes)
            # enlarge crop box within screen
            w = x2 - x1
            x1 = max(0, x1 - int(w * self.expansion))
            x2 = min(frame_x, x2 + int(w * self.expansion))
            h = y2 - y1
            y1 = max(0, y1 - int(h * self.expansion))
            y2 = min(frame_y, y2 + int(h * self.expansion))
        else:
            # no motion, use full frame
            x1 = 0
            y1 = 0
            x2 = frame_x
            y2 = frame_y

        # initialize or move crop window
        if self.window is None:
            self.window = BBox(x1, y1, x2, y2)
        else:
            self.window = BBox(
                self.window.x1 + int((x1 - self.window.x1) / self.inertia),
                self.window.y1 + int((y1 - self.window.y1) / self.inertia),
                self.window.x2 + int((x2 - self.window.x2) / self.inertia),
                self.window.y2 + int((y2 - self.window.y2) / self.inertia),
            )

        crop = frame.raw[self.window.y1 : self.window.y2, self.window.x1 : self.window.x2]
        logger.debug(
            "Crop %s to (%s,%s)-(%s,%s)",
            frame.frame_no,
            self.window.x1,
            self.window.y1,
            self.window.x2,
            self.window.y2,
        )
        frame.crop = crop
        frame.crop_bbox = self.window
        frame.motion_proportion = frame_mask.motion_proportion
        return frame
