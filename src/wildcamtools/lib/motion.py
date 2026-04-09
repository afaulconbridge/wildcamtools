from abc import abstractmethod

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
    ):
        self.history = history
        self.kernel_size = kernel_size
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

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
            # TODO filter contours remove those whose lowest point is in the masked areas
            areas = (cv2.contourArea(cnt) for cnt in contours)
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
    ):
        super().__init__(history=history, kernel_size=kernel_size)
        self.threshold = threshold
        self.detect_shadows = detect_shadows

        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            detectShadows=self.detect_shadows,
            varThreshold=self.threshold,
        )

    def update_background(self, frame: MatLike) -> MatLike:
        frame_out = self.background_subtractor.apply(frame)
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
    ):
        super().__init__(history=history, kernel_size=kernel_size)
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

        return motion_mask
