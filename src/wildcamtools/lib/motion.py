import cv2
import numpy as np

from wildcamtools.lib import FrameHandler


class MogMotion(FrameHandler):
    history: int
    threshld: int
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

    def handle(self, frame: np.ndarray) -> np.ndarray:
        frame_out = self.background_subtractor.apply(frame)
        # despekle if appropriate
        if self.kernel_size:
            frame_out = cv2.morphologyEx(frame_out, cv2.MORPH_OPEN, self.kernel)
            frame_out = cv2.morphologyEx(frame_out, cv2.MORPH_CLOSE, self.kernel)
        return frame_out
