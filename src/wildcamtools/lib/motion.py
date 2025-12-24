import cv2
import numpy as np

from wildcamtools.lib import Frame, FrameHandler


class MogMotion(FrameHandler):
    history: int
    threshold: int
    detect_shadows: bool
    kernel_size: int
    background_subtractor: cv2.BackgroundSubtractorMOG2
    kernel: np.ndarray

    def __init__(
        self,
        history: int = 500,
        threshold: int = 16,
        detect_shadows: bool = False,
        kernel_size: int = 3,
    ):
        self.history = history
        self.threshold = threshold
        self.detect_shadows = detect_shadows
        self.kernel_size = kernel_size

        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            detectShadows=self.detect_shadows,
            varThreshold=self.threshold,
        )
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

    def handle(self, frame: Frame) -> Frame:
        frame_out = self.background_subtractor.apply(frame.raw)
        # despekle if appropriate
        if self.kernel_size:
            frame_out = cv2.morphologyEx(frame_out, cv2.MORPH_OPEN, self.kernel)
            frame_out = cv2.morphologyEx(frame_out, cv2.MORPH_CLOSE, self.kernel)
        # only set if we've gone through history
        proportion = self.get_motion_proportion(frame_out) if frame.frame_no > self.history else -1.0
        return Frame(raw=frame_out, frame_no=frame.frame_no, motion_proportion=proportion)

    def get_motion_proportion(self, frame: Frame) -> float:
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = (cv2.contourArea(cnt) for cnt in contours)
        area_total = sum(areas)
        area_propotion = area_total / (float(frame.shape[0]) * float(frame.shape[1]))
        return area_propotion
