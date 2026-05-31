from enum import StrEnum
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


class BoolResponse(BaseModel):
    answer: bool


class StringResponse(BaseModel):
    message: str


class SpeciesResult(BaseModel):
    species_name: str


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationResult(BaseModel):
    species_name: str
    confidence: ConfidenceLevel
    verified: bool


class ResultClassification(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"
    NO_ANIMAL = "no_animal"
