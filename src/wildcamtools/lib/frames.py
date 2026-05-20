import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from wildcamtools.lib.errors.core import (
    ExpansionValueError,
    InertiaValueError,
    InvalidAlphaError,
    InvalidMaxMagnitudeError,
)
from wildcamtools.lib.motion import FlowMotion, MogMotion, MotionHandler
from wildcamtools.lib.stats import VideoStats, get_video_stats
from wildcamtools.lib.vidio import VideoReader

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
    outputs: list[Path]
    output_dir: Path

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.outputs = []
        super().__init__()

    def handle(self, frame: Frame) -> Frame:
        # write a whole frame image
        if frame.filter_keep:
            if frame.tiles is not None and frame.tiling_rows is not None and frame.tiling_cols is not None:
                for row in range(frame.tiling_rows):
                    for col in range(frame.tiling_cols):
                        tile = frame.get_tile(col, row)
                        if tile is not None:
                            filename = self.output_dir / f"frame_{frame.frame_no:05d}_tile_{row}_{col}.jpg"
                            cv2.imwrite(str(filename), tile)
                            logger.debug(
                                "Writing tile %d,%d (%dx%d) -> %s",
                                row,
                                col,
                                tile.shape[0],
                                tile.shape[1],
                                filename,
                            )
                            self.outputs.append(filename)
                logger.info(
                    "Wrote %d tiles for frame %s",
                    len(frame.tiles),
                    frame.frame_no,
                )
            # write a whole frame image
            filename = self.output_dir / f"frame_{frame.frame_no:05d}.jpg"
            cv2.imwrite(str(filename), frame.output)
            logger.debug("Writing %s -> %s", frame.frame_no, filename)
            self.outputs.append(filename)
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


class FrameTiler(FrameHandler):
    """Splits frames into a grid of tiles for parallel motion detection.

    When image dimensions don't divide evenly, extra pixels are distributed
    to edge tiles (right and bottom edges are larger).
    """

    cols: int
    rows: int
    overlap: float

    def __init__(self, cols: int = 2, rows: int = 2, overlap: float = 0.0) -> None:
        if cols < 1:
            raise ValueError(f"cols must be at least 1, got {cols}")
        if rows < 1:
            raise ValueError(f"rows must be at least 1, got {rows}")
        if overlap < 0.0 or overlap >= 1.0:
            raise ValueError(f"overlap must be between 0.0 and 1.0, got {overlap}")
        self.cols = cols
        self.rows = rows
        self.overlap = overlap
        super().__init__()

    def handle(self, frame: Frame) -> Frame:
        img = frame.output
        img_h, img_w = img.shape[:2]

        if self.overlap > 0:
            tile_w = img_w / (1 + (self.cols - 1) * (1 - self.overlap))
            tile_h = img_h / (1 + (self.rows - 1) * (1 - self.overlap))
        else:
            tile_w = img_w / self.cols
            tile_h = img_h / self.rows

        tile_w_int = int(tile_w)
        tile_h_int = int(tile_h)

        logger.info(
            "Tiling frame %s: %dx%d -> %dx%d tiles (%d cols x %d rows, overlap %.2f%%)",
            frame.frame_no,
            img_w,
            img_h,
            tile_w_int,
            tile_h_int,
            self.cols,
            self.rows,
            self.overlap * 100,
        )

        tiles: list[cv2.typing.MatLike] = []
        for row in range(self.rows):
            for col in range(self.cols):
                x_start = int(col * tile_w * (1 - self.overlap))
                y_start = int(row * tile_h * (1 - self.overlap))

                if col == self.cols - 1:
                    x_start = img_w - tile_w_int
                if row == self.rows - 1:
                    y_start = img_h - tile_h_int

                x_end = x_start + tile_w_int
                y_end = y_start + tile_h_int

                tile = img[y_start:y_end, x_start:x_end]
                tiles.append(tile)
                logger.debug(
                    "Tile (%d,%d): %dx%d pixels",
                    row,
                    col,
                    tile.shape[1],
                    tile.shape[0],
                )

        frame.tiles = tiles
        frame.tiling_cols = self.cols
        frame.tiling_rows = self.rows
        frame.tiling_width = tile_w_int
        frame.tiling_height = tile_h_int
        return frame


@dataclass(kw_only=True)
class CropPanConfig:
    """Configuration for crop and pan features."""

    expansion: float = 0.75
    inertia: float = 10.0
    motion_type: str = "mog"
    history: int = 30
    threshold: float = 16.0
    kernel_size: float = 0.02

    def __post_init__(self) -> None:
        if self.inertia < 0.0:
            raise InertiaValueError()
        if self.expansion < 0.0:
            raise ExpansionValueError()


@dataclass(kw_only=True)
class TilingConfig:
    """Configuration for tiling features."""

    cols: int = 2
    rows: int = 2
    overlap: float = 0.0

    def __post_init__(self) -> None:
        if self.cols < 1:
            raise ValueError(f"cols must be at least 1, got {self.cols}")
        if self.rows < 1:
            raise ValueError(f"rows must be at least 1, got {self.rows}")
        if self.overlap < 0.0 or self.overlap >= 1.0:
            raise ValueError(f"overlap must be between 0.0 and 1.0, got {self.overlap}")


@dataclass(kw_only=True)
class FrameCreation:
    filename: Path
    video_directory: Path
    tmpdir: Path
    x: int | None = None
    y: int | None = None
    fps: float | None = None
    similarity_minimum: float | None = None
    crop_pan: CropPanConfig | None = None
    tiling: TilingConfig | None = None

    @classmethod
    def with_crop_pan(
        cls,
        filename: Path,
        video_directory: Path,
        tmpdir: Path,
        *,
        expansion: float = 0.75,
        inertia: float = 10.0,
        motion_type: str = "mog",
        history: int = 30,
        threshold: float = 16.0,
        kernel_size: float = 0.02,
        x: int | None = None,
        y: int | None = None,
        fps: float | None = None,
        similarity_minimum: float | None = None,
    ) -> Self:
        """Create a FrameCreation instance with crop and pan enabled."""
        return cls(
            filename=filename,
            video_directory=video_directory,
            tmpdir=tmpdir,
            x=x,
            y=y,
            fps=fps,
            similarity_minimum=similarity_minimum,
            crop_pan=CropPanConfig(
                expansion=expansion,
                inertia=inertia,
                motion_type=motion_type,
                history=history,
                threshold=threshold,
                kernel_size=kernel_size,
            ),
        )

    @classmethod
    def with_tiling(
        cls,
        filename: Path,
        video_directory: Path,
        tmpdir: Path,
        *,
        cols: int = 2,
        rows: int = 2,
        overlap: float = 0.0,
        x: int | None = None,
        y: int | None = None,
        fps: float | None = None,
        similarity_minimum: float | None = None,
    ) -> Self:
        """Create a FrameCreation instance with tiling enabled."""
        return cls(
            filename=filename,
            video_directory=video_directory,
            tmpdir=tmpdir,
            x=x,
            y=y,
            fps=fps,
            similarity_minimum=similarity_minimum,
            tiling=TilingConfig(
                cols=cols,
                rows=rows,
                overlap=overlap,
            ),
        )

    @classmethod
    def with_crop_pan_and_tiling(
        cls,
        filename: Path,
        video_directory: Path,
        tmpdir: Path,
        *,
        expansion: float = 0.75,
        inertia: float = 10.0,
        motion_type: str = "mog",
        history: int = 30,
        threshold: float = 16.0,
        kernel_size: float = 0.02,
        cols: int = 2,
        rows: int = 2,
        overlap: float = 0.0,
        x: int | None = None,
        y: int | None = None,
        fps: float | None = None,
        similarity_minimum: float | None = None,
    ) -> Self:
        """Create a FrameCreation instance with both crop/pan and tiling enabled."""
        return cls(
            filename=filename,
            video_directory=video_directory,
            tmpdir=tmpdir,
            x=x,
            y=y,
            fps=fps,
            similarity_minimum=similarity_minimum,
            crop_pan=CropPanConfig(
                expansion=expansion,
                inertia=inertia,
                motion_type=motion_type,
                history=history,
                threshold=threshold,
                kernel_size=kernel_size,
            ),
            tiling=TilingConfig(
                cols=cols,
                rows=rows,
                overlap=overlap,
            ),
        )


@dataclass(kw_only=True)
class FrameCreationResult(FrameCreation):
    frame_count: int

    @classmethod
    def from_creation(cls, creation: FrameCreation, *, frame_count: int) -> Self:
        return cls(
            filename=creation.filename,
            video_directory=creation.video_directory,
            tmpdir=creation.tmpdir,
            x=creation.x,
            y=creation.y,
            fps=creation.fps,
            similarity_minimum=creation.similarity_minimum,
            crop_pan=creation.crop_pan,
            tiling=creation.tiling,
            frame_count=frame_count,
        )


def create_frames(
    frame_creation: FrameCreation,
) -> FrameCreationResult:
    stats = get_video_stats(frame_creation.video_directory / frame_creation.filename)
    handlers: list[FrameHandler] = []

    if frame_creation.crop_pan is not None:
        motion_handler: FlowMotion | MogMotion
        if frame_creation.crop_pan.motion_type == "flow":
            motion_handler = FlowMotion(
                history=frame_creation.crop_pan.history,
                threshold=frame_creation.crop_pan.threshold,
                kernel_size=frame_creation.crop_pan.kernel_size,
            )
        else:
            motion_handler = MogMotion(
                history=frame_creation.crop_pan.history,
                threshold=int(frame_creation.crop_pan.threshold),
                kernel_size=frame_creation.crop_pan.kernel_size,
            )
        handlers.append(
            CropPanHandler(
                motion_handler=motion_handler,
                expansion=frame_creation.crop_pan.expansion,
                inertia=frame_creation.crop_pan.inertia,
            )
        )

    if frame_creation.similarity_minimum is not None:
        handlers.append(FilterSSIM(similarity_minimum=frame_creation.similarity_minimum))

    if frame_creation.x or frame_creation.y or frame_creation.fps:
        handlers.append(
            Rescaler(
                stats=stats,
                x=frame_creation.x,
                y=frame_creation.y,
                fps=frame_creation.fps,
            )
        )

    if frame_creation.tiling is not None:
        handlers.append(
            FrameTiler(
                cols=frame_creation.tiling.cols,
                rows=frame_creation.tiling.rows,
                overlap=frame_creation.tiling.overlap,
            )
        )

    handlers.append(FrameImageWriter(frame_creation.tmpdir))

    frame_count = 0
    with VideoReader(frame_creation.video_directory / frame_creation.filename) as video_input:
        for _frame in video_input:
            frame_count += 1
            for handler in handlers:
                _frame = handler.handle(_frame)

    return FrameCreationResult.from_creation(frame_creation, frame_count=frame_count)


class FrameImageRecreator:
    def __init__(self, raw_images: Sequence[Path], rescale_images: Sequence[Path]) -> None:
        if len(rescale_images) != len(raw_images):
            raise ValueError("image lengths must match")
        self.raw_images = raw_images
        self.rescale_images = rescale_images
        self.i = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Frame:
        if self.i >= len(self.raw_images):
            raise StopIteration()

        raw = cv2.imread(self.raw_images[self.i])
        if raw is None:
            raise ValueError(f"Unable to read {self.raw_images[self.i]}")

        rescale = cv2.imread(self.rescale_images[self.i])
        if rescale is None:
            raise ValueError(f"Unable to read {self.rescale_images[self.i]}")

        frame = Frame(raw, self.i, rescale=rescale)
        self.i += 1
        return frame
