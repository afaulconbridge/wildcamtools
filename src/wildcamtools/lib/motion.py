from abc import abstractmethod
from typing import cast

import cv2
import numpy as np
from cv2.typing import MatLike

from wildcamtools.lib import Frame, FrameHandler


class MotionHandler(FrameHandler):
    history: int
    kernel_size: int
    background_subtractor: cv2.BackgroundSubtractor
    motion_mask: np.ndarray | None = None

    def __init__(
        self,
        history: int,
        kernel_size: int = 3,
        motion_mask: np.ndarray | None = None,
    ):
        self.history = history
        self.kernel_size = kernel_size
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        self.motion_mask = motion_mask

    def handle(self, frame: Frame) -> Frame:
        frame_out = self.update_background(frame.raw)
        # despeckle if appropriate
        if self.kernel_size:
            frame_out = cv2.morphologyEx(frame_out, cv2.MORPH_OPEN, self.kernel)
            frame_out = cv2.morphologyEx(frame_out, cv2.MORPH_CLOSE, self.kernel)
        # only set if we've gone through history
        proportion = self.get_motion_proportion(frame_out) if frame.frame_no > self.history else -1.0
        return Frame(raw=frame_out, frame_no=frame.frame_no, motion_proportion=proportion)

    def get_motion_proportion(self, frame: MatLike) -> float:
        if self.motion_mask is None:
            return cv2.countNonZero(frame) / (float(frame.shape[0]) * float(frame.shape[1]))
        else:
            contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out contours whose lowest point (max Y coordinate) is within a masked area.
            # If a contour has multiple points with the same maximum Y coordinate,
            # we consider it masked if ANY of those points are in a masked area.
            # This prevents motion in specifically excluded regions (like moving foliage at the top)
            # from triggering a motion detection.
            filtered_contours = []
            for cnt in contours:
                # Find the maximum Y coordinate among all points in the contour
                # cnt shape is (N, 1, 2)
                max_y = np.max(cnt[:, 0, 1])

                # Find all points that have this maximum Y coordinate
                lowest_points = cnt[cnt[:, 0, 1] == max_y]

                # The contour is removed if ANY of its lowest points are in a masked area
                is_masked = False
                if self.motion_mask is not None:
                    mh, mw = self.motion_mask.shape
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

            areas = (cv2.contourArea(cnt) for cnt in filtered_contours)
            area_total = sum(areas)
            area_propotion = area_total / (float(frame.shape[0]) * float(frame.shape[1]))
            return area_propotion

    @abstractmethod
    def update_background(self, frame: MatLike) -> MatLike:
        raise NotImplementedError


class MogMotion(MotionHandler):
    threshold: int
    detect_shadows: bool
    kernel: np.ndarray

    def __init__(
        self,
        history: int = 500,
        threshold: int = 16,
        detect_shadows: bool = False,
        kernel_size: int = 3,
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
        frame_out = self.background_subtractor.apply(frame)
        if frame_out is None:
            # Return a zero-filled array of the same size as input
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        return frame_out


class AvgMotion(MotionHandler):
    threshold: int
    average: MatLike | None = None
    count: int = 1

    def __init__(
        self,
        history: int = 500,
        threshold: int = 16,
        kernel_size: int = 3,
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

        # ensure we return the appropriate type because Pandas typing is complicated
        return cast(MatLike, motion_mask)
