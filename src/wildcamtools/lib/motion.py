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
    last_frame: np.ndarray | None = None

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
        # TODO only set this if we've gone through history
        # TODO move to parent class
        self.last_frame = frame_out
        return Frame(raw=frame_out, frame_no=frame.frame_no)

    def get_motion_proportion(self) -> float:
        if self.last_frame is None:
            raise RuntimeError("Cannot get motion proportion before processing frames")

        contours, _ = cv2.findContours(self.last_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = (cv2.contourArea(cnt) for cnt in contours)
        area_total = sum(areas)
        area_propotion = area_total / (float(self.last_frame.shape[0]) * float(self.last_frame.shape[1]))
        return area_propotion
