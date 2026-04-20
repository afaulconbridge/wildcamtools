import abc
from dataclasses import dataclass
from typing import Self

import cv2.typing

from wildcamtools.lib.errors.core import BoundingBoxHeightError, BoundingBoxWidthError


@dataclass
class Frame:
    raw: cv2.typing.MatLike
    frame_no: int
    motion_proportion: float = -1.0

    @property
    def width(self) -> float:
        return float(self.raw.shape[1])

    @property
    def height(self) -> float:
        return float(self.raw.shape[0])


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 <= self.x1:
            raise BoundingBoxWidthError()
        if self.y2 <= self.y1:
            raise BoundingBoxHeightError()

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def overlaps(self, r: Self) -> bool:
        return not (self.x1 > r.x2 or self.x2 < r.x1 or self.y1 > r.y2 or self.y2 < r.y1)

    def merge_with(self, r: Self) -> Self:
        x1 = min(self.x1, r.x1)
        y1 = min(self.y1, r.y1)
        x2 = max(self.x2, r.x2)
        y2 = max(self.y2, r.y2)
        return self.__class__(x1, y1, x2, y2)


class FrameHandler(abc.ABC):
    @abc.abstractmethod
    def handle(self, frame: Frame) -> Frame | None: ...
