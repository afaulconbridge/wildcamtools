from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, Field


@dataclass(kw_only=True)
class AIEvaluation:
    frame_directory: Any
    label: str
    backend: Any
    url: str
    model: str
    api_key: str = "API_KEY"
    prompt: str


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


class BoolResponse(BaseModel):
    answer: bool


class StringResponse(BaseModel):
    message: str


class SpeciesResult(BaseModel):
    species_name: str
