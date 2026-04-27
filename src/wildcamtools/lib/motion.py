import logging
from abc import abstractmethod
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
from cv2.typing import MatLike

from wildcamtools.lib import BBox, Frame, FrameHandler
from wildcamtools.lib.errors.core import MotionMaskNotCreatedError

logger = logging.getLogger(__name__)


class MotionHandler(FrameHandler):
    history: int
    kernel_size: float
    kernel: np.ndarray | None = None
    background_subtractor: cv2.BackgroundSubtractor
    motion_mask: np.ndarray | None = None

    def __init__(
        self,
        history: int,
        kernel_size: float = 0.005,
        motion_mask: np.ndarray | None = None,
    ):
        self.history = history
        self.kernel_size = kernel_size
        self.motion_mask = motion_mask

    def handle(self, frame: Frame) -> Frame:
        frame_out = self.update_background(frame.raw)
        # only set if we've gone through history
        proportion = self.get_motion_proportion(frame_out) if frame.frame_no > self.history else -1.0
        logger.debug("Motion %d: %03f", frame.frame_no, proportion)
        return Frame(raw=frame_out, frame_no=frame.frame_no, motion_proportion=proportion)

    def _despeckle(self, mask: MatLike) -> MatLike:
        if self.kernel_size > 0:
            if self.kernel is None:
                self.kernel = self._compute_kernel(mask)
            # mask = cv2.erode(mask, self.kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _compute_kernel(self, frame_raw: MatLike) -> np.ndarray:
        # Calculate kernel size based on longest dimension
        # Round down to nearest odd number, minimum 3
        # TODO make kernel size based on area not length
        max_dim = max(frame_raw.shape[:2])
        k_size = int(max_dim * self.kernel_size)
        k_size = k_size if k_size % 2 != 0 else k_size - 1
        k_size = max(3, k_size)
        return np.ones((k_size, k_size), np.uint8)

    def _get_contours(self, mask: MatLike) -> Sequence[MatLike]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.motion_mask is None:
            return contours
        else:
            # Filter out contours whose lowest point (max Y coordinate) is within a masked area.
            # If a contour has multiple points with the same maximum Y coordinate,
            # we consider it masked if ANY of those points are in a masked area.
            # This prevents motion in specifically excluded regions (like moving foliage at the top)
            # from triggering a motion detection.
            mh, mw = self.motion_mask.shape
            filtered_contours = []
            for cnt in contours:
                # Find the maximum Y coordinate among all points in the contour
                # cnt shape is (N, 1, 2)
                max_y = np.max(cnt[:, 0, 1])

                # Find all points that have this maximum Y coordinate
                lowest_points = cnt[cnt[:, 0, 1] == max_y]

                # The contour is removed if ANY of its lowest points are in a masked area
                is_masked = False
                # Extract coordinates of points at the lowest Y level
                pts = [(int(p[0, 0]), int(p[0, 1])) for p in lowest_points]

                # Check the points themselves
                for x, y in pts:
                    if 0 <= y < mh and 0 <= x < mw and self.motion_mask[y, x] != 0:
                        is_masked = True
                        break

                # For CHAIN_APPROX_SIMPLE, a straight bottom edge is represented by only its endpoints.
                # We check the segment between the leftmost and rightmost points at max_y.
                if not is_masked and len(pts) >= 2:
                    # Sort by X coordinate
                    pts.sort()
                    x_start, y_start = pts[0]
                    x_end, _ = pts[-1]
                    # Scan along the horizontal segment at max_y
                    for x_sample in range(x_start, x_end + 1):
                        if 0 <= y_start < mh and 0 <= x_sample < mw and self.motion_mask[y_start, x_sample] != 0:
                            is_masked = True
                            break

                if not is_masked:
                    filtered_contours.append(cnt)
            return filtered_contours

    def get_motion_proportion(self, frame: MatLike) -> float:
        if self.motion_mask is None:
            return cv2.countNonZero(frame) / (float(frame.shape[0]) * float(frame.shape[1]))
        else:
            areas = (cv2.contourArea(cnt) for cnt in self._get_contours(frame))
            area_total = sum(areas)
            area_propotion = area_total / (float(frame.shape[0]) * float(frame.shape[1]))
            return area_propotion

    @abstractmethod
    def update_background(self, frame: MatLike) -> MatLike: ...

    def get_contour_bboxes(self) -> tuple[BBox, ...]:
        if self.motion_mask is None:
            raise MotionMaskNotCreatedError()

        contours = self._get_contours(self.motion_mask)
        rects = [cv2.boundingRect(cv2.approxPolyDP(contour, 3, True)) for contour in contours]
        # Convert cv2 bounding boxes (x, y, w, h) to BBox (x1, y1, x2, y2)
        merged_rects = [BBox(r[0], r[1], r[0] + r[2], r[1] + r[3]) for r in rects]

        # merge until cannot merge any more
        has_merged = True
        while has_merged:
            has_merged = False
            new_rects: list[BBox] = []

            while merged_rects:
                current = merged_rects.pop(0)
                merged = False

                for i in range(len(new_rects)):
                    if current.overlaps(new_rects[i]):
                        # Merge 'current' into the existing rect in new_rects
                        new_rects[i] = current.merge_with(new_rects[i])
                        merged = True
                        has_merged = True
                        break

                if not merged:
                    new_rects.append(current)

            merged_rects = new_rects

        return tuple(merged_rects)


class FlowMotion(MotionHandler):
    threshold: float
    prev_gray: MatLike | None = None
    average_flow: MatLike | None = None
    history_count: int = 1

    def __init__(
        self,
        history: int = 1,
        threshold: float = 1.0,
        kernel_size: float = 0.005,
        motion_mask: np.ndarray | None = None,
    ):
        super().__init__(history=history, kernel_size=kernel_size, motion_mask=motion_mask)
        self.threshold = threshold

    def update_background(self, frame: MatLike) -> MatLike:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Calculate dense optical flow
        flow = cast(MatLike, cv2.calcOpticalFlowFarneback(self.prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0))  # type: ignore[call-overload]
        self.prev_gray = curr_gray

        # average motion over time
        if self.history > 1:
            # create average on first frame
            if self.average_flow is None:
                self.average_flow = np.full((flow.shape[0], flow.shape[1], 2), 0.0, np.float32)

            # apply part of the difference between the image and the average to the average
            self.average_flow[..., 0] = self.average_flow[..., 0] + (
                (flow[..., 0] - self.average_flow[..., 0]) / float(self.history_count)
            )
            self.average_flow[..., 1] = self.average_flow[..., 1] + (
                (flow[..., 1] - self.average_flow[..., 1]) / float(self.history_count)
            )
            # over time, new frames change it less up to a limit
            self.history_count = self.history_count + 1 if self.history_count < self.history else self.history_count
            flow = self.average_flow

        # Compute magnitude
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Create a mask where magnitude > threshold
        mask = (magnitude > self.threshold).astype(np.uint8) * 255

        mask = cast(MatLike, mask)  # type: ignore[assignment]

        self.motion_mask = self._despeckle(mask)
        return self.motion_mask


class MogMotion(MotionHandler):
    threshold: int
    detect_shadows: bool

    def __init__(
        self,
        history: int = 500,
        threshold: int = 16,
        detect_shadows: bool = False,
        kernel_size: float = 0.005,
        motion_mask: np.ndarray | None = None,
    ):
        super().__init__(history=history, kernel_size=kernel_size, motion_mask=motion_mask)
        self.threshold = threshold
        self.detect_shadows = detect_shadows

        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            detectShadows=self.detect_shadows,
            varThreshold=self.threshold,
        )

    def update_background(self, frame: MatLike) -> MatLike:
        mask = self.background_subtractor.apply(frame)
        if mask is None:
            # Return a zero-filled array of the same size as input
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        self.motion_mask = self._despeckle(mask)
        return self.motion_mask


class AvgMotion(MotionHandler):
    threshold: int
    average: MatLike | None = None
    count: int = 1

    def __init__(
        self,
        history: int = 500,
        threshold: int = 16,
        kernel_size: float = 0.005,
        motion_mask: np.ndarray | None = None,
    ):
        super().__init__(history=history, kernel_size=kernel_size, motion_mask=motion_mask)
        self.threshold = threshold

    def update_background(self, frame: MatLike) -> MatLike:
        # create average frame on first frame
        if self.average is None:
            self.average = np.full((frame.shape[0], frame.shape[1]), 128.0, np.float32)

        # convert image to greyscale floating point for calculations
        frame_float = frame.astype(np.float32)
        frame_float = np.mean(frame_float, axis=2)

        frame_difference = frame_float - self.average

        # now update the background too
        # apply part of the difference between the image and the average to the average
        self.average = self.average + (frame_difference / float(self.count))
        # over time, new frames change it less up to a limit
        self.count = self.count + 1 if self.count < self.history else self.count

        # calculate a binary mask for deviations from the average
        frame_difference[frame_difference > self.threshold] = 255.0
        frame_difference[frame_difference < -self.threshold] = 255.0

        # output should be uint8 with values of 255 or 0
        motion_mask = frame_difference.astype(np.uint8)

        self.motion_mask = self._despeckle(motion_mask)
        return self.motion_mask
