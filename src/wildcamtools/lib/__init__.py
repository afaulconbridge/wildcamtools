import abc

import numpy as np


class FrameHandler(abc.ABC):
    @abc.abstractmethod
    def handle(self, frame: np.ndarray) -> np.ndarray | None: ...
