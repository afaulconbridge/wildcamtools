import abc
import base64
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, Field


@dataclass(kw_only=True)
class AIEvaluation:
    frame_directory: Path
    label: str
    backend: Any
    url: str
    model: str
    api_key: str = "API_KEY"
    prompt: str | None = None


@dataclass(kw_only=True)
class AIEvaluationResult(AIEvaluation):
    correct: bool
    raw_result: str

    @classmethod
    def from_evaluation(cls, evaluation: AIEvaluation, *, correct: bool, raw_result: str) -> Self:
        return cls(
            frame_directory=evaluation.frame_directory,
            label=evaluation.label,
            backend=evaluation.backend,
            url=evaluation.url,
            model=evaluation.model,
            api_key=evaluation.api_key,
            prompt=evaluation.prompt,
            correct=correct,
            raw_result=raw_result,
        )


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
