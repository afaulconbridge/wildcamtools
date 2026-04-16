import abc
import base64
from collections.abc import Iterable
from pathlib import Path


class AbstractAnalyser(abc.ABC):
    @abc.abstractmethod
    def analyze_video(self, images: Iterable[Path]) -> str: ...

    @staticmethod
    def path_jpeg_to_base64(input_: Path) -> str:
        with input_.open("rb") as infile:
            return base64.b64encode(infile.read()).decode()
