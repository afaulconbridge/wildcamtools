import abc
from dataclasses import dataclass

import cv2.typing


@dataclass
class Frame:
    raw: cv2.typing.MatLike
    frame_no: int

    @property
    def width(self) -> float:
        return self.raw.shape[1]

    @property
    def height(self) -> float:
        return self.raw.shape[0]


class FrameHandler(abc.ABC):
    @abc.abstractmethod
    def handle(self, frame: Frame) -> Frame | None: ...
