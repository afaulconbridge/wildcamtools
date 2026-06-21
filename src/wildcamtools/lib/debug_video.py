"""Debug video output handlers for motion detection visualization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import cv2
import numpy as np

from wildcamtools.lib import Frame, FrameHandler
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.vidio import VideoWriter

if TYPE_CHECKING:
    from wildcamtools.lib.states import Watcher

logger = logging.getLogger(__name__)


class DebugVideoOverlay(FrameHandler):
    """Adds debug overlays to frames showing motion detection state and contours.

    Overlays are drawn on the full-resolution frame (frame.raw), with motion
    contours scaled up from the motion detection resolution to match.
    """

    watcher: Watcher
    motion_handler: MogMotion
    state_colors: dict[str, tuple[int, int, int]]
    _contour_scale: tuple[float, float] | None

    def __init__(
        self,
        watcher: Watcher,
        motion_handler: MogMotion,
    ) -> None:
        from wildcamtools.lib.states import WatcherStateEnum

        self.watcher = watcher
        self.motion_handler = motion_handler
        self.state_colors = {
            WatcherStateEnum.PREPARING: (128, 128, 128),
            WatcherStateEnum.GREEN: (0, 255, 0),
            WatcherStateEnum.AMBER: (255, 195, 0),
            WatcherStateEnum.RED: (255, 0, 0),
            WatcherStateEnum.RED_AMBER: (255, 140, 0),
            WatcherStateEnum.DISABLED: (64, 64, 64),
        }
        self._contour_scale = None
        logger.debug("DebugVideoOverlay initialized for watcher state machine")

    def _contour_scale_cached(self, frame: Frame) -> tuple[float, float]:
        """Compute and cache contour scale factor.

        Assumes frame resolution is constant throughout the video.
        The scale factor is computed once on first call and cached.
        """
        if self._contour_scale is not None:
            return self._contour_scale

        self._contour_scale = self._compute_contour_scale(frame)
        return self._contour_scale

    def _compute_contour_scale(self, frame: Frame) -> tuple[float, float]:
        """Compute scale factor from motion mask resolution to full frame resolution.

        Returns (1.0, 1.0) if motion mask is not yet available.
        """
        if self.motion_handler.motion_mask is None:
            return (1.0, 1.0)

        mask_h, mask_w = self.motion_handler.motion_mask.shape
        frame_h, frame_w = frame.raw.shape[:2]

        scale_x = frame_w / mask_w
        scale_y = frame_h / mask_h

        logger.debug(
            "Contour scale: mask %dx%d -> frame %dx%d (scale %.2f, %.2f)",
            mask_w,
            mask_h,
            frame_w,
            frame_h,
            scale_x,
            scale_y,
        )

        return (scale_x, scale_y)

    def _scale_contours(
        self,
        contours: list[np.ndarray],
        scale_x: float,
        scale_y: float,
    ) -> list[np.ndarray]:
        """Scale contour coordinates from motion resolution to full resolution."""
        scaled_contours: list[np.ndarray] = []
        for contour in contours:
            scaled = contour.copy()
            scaled[:, :, 0] = (scaled[:, :, 0] * scale_x).astype(np.int32)
            scaled[:, :, 1] = (scaled[:, :, 1] * scale_y).astype(np.int32)
            scaled_contours.append(scaled)
        return scaled_contours

    def handle(self, frame: Frame) -> Frame:
        output = frame.raw.copy()
        height, width = output.shape[:2]
        state = self.watcher.state
        color = self.state_colors.get(state, (128, 128, 128))

        cv2.rectangle(output, (0, 0), (width - 1, height - 1), color, 3)

        text_scale = 0.5
        text_thickness = 2
        text_y_base = 50
        text_spacing = 40

        motion_text = f"Motion: {frame.motion_proportion:.4f}"
        (text_w, _), _ = cv2.getTextSize(
            motion_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_thickness,
        )
        cv2.putText(
            output,
            motion_text,
            (width - text_w - 10, text_y_base),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

        frame_number_text = f"Frame: {frame.frame_no:06d}"
        (text_w, _), _ = cv2.getTextSize(
            frame_number_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_thickness,
        )
        cv2.putText(
            output,
            frame_number_text,
            (width - text_w - 10, text_y_base + text_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

        timestamp = frame.timestamp if frame.timestamp is not None else frame.frame_no / 30.0
        timestamp_text = f"Time: {timestamp:010.4f}"
        (text_w, _), _ = cv2.getTextSize(
            timestamp_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_thickness,
        )
        cv2.putText(
            output,
            timestamp_text,
            (width - text_w - 10, text_y_base + text_spacing * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

        if self.motion_handler.motion_mask is not None:
            scale_x, scale_y = self._contour_scale_cached(frame)

            contours_raw, _ = cv2.findContours(
                self.motion_handler.motion_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            contours = list(contours_raw)
            scaled_contours = self._scale_contours(contours, scale_x, scale_y)
            cv2.drawContours(output, scaled_contours, -1, (0, 0, 255), 2)

        return Frame(
            raw=output,
            frame_no=frame.frame_no,
            motion_proportion=frame.motion_proportion,
            filter_keep=frame.filter_keep,
            crop=frame.crop,
            rescale=frame.rescale,
            crop_bbox=frame.crop_bbox,
            tiles=frame.tiles,
            tiling_cols=frame.tiling_cols,
            tiling_rows=frame.tiling_rows,
            tiling_width=frame.tiling_width,
            tiling_height=frame.tiling_height,
            timestamp=frame.timestamp,
        )


class DebugVideoWriter(FrameHandler):
    """Writes debug video frames to a file at full resolution."""

    writer: VideoWriter
    width: int
    height: int
    _closed: bool

    def __init__(self, output_path: Path, width: int, height: int, fps: float) -> None:
        self.writer = VideoWriter(
            out_filename=output_path,
            fps=fps,
            codec="libx264",
            crf=23,
            preset="medium",
            pix_fmt="yuv420p",
        )
        self.width = width
        self.height = height
        self._closed = False
        logger.debug("DebugVideoWriter initialized for %s (%dx%d @ %.2f fps)", output_path, width, height, fps)

    def __enter__(self) -> Self:
        self.writer.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> Literal[False]:
        self.close()
        return False

    def handle(self, frame: Frame) -> Frame:
        if frame.filter_keep:
            self.writer.write(frame.raw)
            logger.debug("Wrote debug frame %d (%dx%d)", frame.frame_no, frame.raw.shape[1], frame.raw.shape[0])
        else:
            logger.debug("Skipping debug frame %d (filter_keep=False)", frame.frame_no)
        return frame

    def close(self) -> None:
        if not self._closed:
            self.writer.close()
            self._closed = True
            logger.debug("DebugVideoWriter closed")
