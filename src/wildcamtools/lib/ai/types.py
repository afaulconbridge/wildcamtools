from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field

ABSENCE_MARKERS = {"", "none", "no animal", "missing", "empty", "no", "no detection"}
UNKNOWN_MARKERS = {"unknown", "uncertain", "unsure"}


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
            description="Whether the initial classification is verified as correct.",
            examples=[True, False],
        ),
    ]


class RichResult(BaseModel):
    is_animal_present: Annotated[
        bool, Field(description="Flag if any animal is visible in these images.", examples=["true", "false"])
    ]
    is_animal_unknown: Annotated[
        bool,
        Field(
            description="Flag if there is an animal visible in these images but you cannot identify what it is. "
            "If no animal is visible, use 'false'. If an animal is visible and you can identify what species it is, "
            "use 'false'. Use 'true' only if you are confident there is an animal visible and you are not confident "
            "what species it is.",
            examples=["true", "false"],
        ),
    ]

    defining_features: Annotated[
        str,
        Field(
            description="If there is an animal visible and you can confidently identify "
            "what species it is, then the important features used for this identification should be described here. "
            "Return 'unknown' when you cannot confidently identify the species. Return 'no animal' when no animal "
            "is present in the image.",
            examples=[
                "short fur, pointy ears, long tail",
                "small size, black feathers, yellow beak",
                "unknown",
                "no animal",
            ],
        ),
    ]

    species_name: Annotated[
        str,
        Field(
            description="Identified wildlife species from camera trap images. Return a specific species name when "
            "confident (e.g., 'European Badger', 'Red Fox', 'Wood Pigeon'). Return 'unknown' when you cannot "
            "confidently identify the species. Return 'no animal' when no animal is present in the image.",
            examples=["European Badger", "Red Fox", "unknown", "no animal"],
        ),
    ]
    confidence: Annotated[
        ConfidenceLevel,
        Field(
            description="Your confidence level in this result: 'high' for >80% confidence, "
            "'medium' for 50-80% confidence with some uncertainty, 'low' for <50% confidence or significant doubts.",
            examples=["high", "medium", "low"],
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
    "RichResult",
    "VerificationResult",
]
