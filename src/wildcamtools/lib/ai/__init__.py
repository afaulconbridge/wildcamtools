import abc
import base64
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field


class Backend(StrEnum):
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"


class FrameResult(BaseModel):
    frame_no: int
    left: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    right: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    top: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    bottom: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]


class Result(BaseModel):
    species_name: str
    frames: list[FrameResult]


class ResultList(BaseModel):
    results: list[Result]


class AbstractAnalyser(abc.ABC):
    model: str
    backend: Backend
    url: str
    api_key: str | None = None
    message: str | None = None

    @abc.abstractmethod
    def analyze_video(self, images: Iterable[Path]) -> str: ...

    @abc.abstractmethod
    def detect(self, images: Iterable[Path]) -> ResultList: ...

    @staticmethod
    def path_jpeg_to_base64(input_: Path) -> str:
        with input_.open("rb") as infile:
            return base64.b64encode(infile.read()).decode()
