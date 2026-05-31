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
    answer: Annotated[
        bool,
        Field(
            description="Boolean response to a yes/no question. Return false when the condition is not met, "
            "when nothing is detected, or when you cannot confidently confirm.",
            examples=[True, False],
        ),
    ]


class StringResponse(BaseModel):
    message: Annotated[
        str,
        Field(
            description="Text response that can express uncertainty. Use 'unknown' when you cannot confidently "
            "identify something, 'no animal' when nothing is present, or provide a specific identification.",
            examples=["unknown", "no animal", "European Badger"],
        ),
    ]


class SpeciesResult(BaseModel):
    species_name: Annotated[
        str,
        Field(
            description="Identified wildlife species from camera trap images. Return a specific species name when "
            "confident (e.g., 'European Badger', 'Red Fox', 'Wood Pigeon'). Return 'unknown' when you cannot "
            "confidently identify the species. Return 'no animal' when no animal is present in the image.",
            examples=["European Badger", "Red Fox", "unknown", "no animal"],
        ),
    ]


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationResult(BaseModel):
    species_name: Annotated[
        str,
        Field(
            description="Verified species identification. Use the confirmed species name when verified is true, "
            "or 'unknown' when verification fails or confidence is too low.",
            examples=["European Badger", "unknown"],
        ),
    ]
    confidence: Annotated[
        ConfidenceLevel,
        Field(
            description="Your confidence level in this verification: 'high' for >80% confidence, "
            "'medium' for 50-80% confidence with some uncertainty, 'low' for <50% confidence or significant doubts.",
            examples=["high", "medium", "low"],
        ),
    ]
    verified: Annotated[
        bool,
        Field(
            description="Whether you confirm the initial classification is correct. Set to false when the "
            "classification is incorrect, when confidence is low, or when you cannot verify it.",
            examples=[True, False],
        ),
    ]


class ResultClassification(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"


__all__ = [
    "Backend",
    "BoolResponse",
    "ConfidenceLevel",
    "FrameResult",
    "Result",
    "ResultClassification",
    "ResultList",
    "SpeciesResult",
    "StringResponse",
    "VerificationResult",
]
