import abc
import base64
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from wildcamtools.lib.ai.types import Backend, RichResult

T = TypeVar("T", bound=BaseModel)

DEFAULT_SYSTEM_MESSAGE = (
    "You must respond with valid JSON that matches the provided schema exactly. "
    "Do not include any markdown formatting, code blocks, or explanations. "
    "Only output the raw JSON object.\n\n"
    "Expected JSON schema:\n{schema}"
)


class AbstractLlm(abc.ABC):
    model: str
    backend: Backend
    url: str
    api_key: str | None = None

    @abc.abstractmethod
    def message_with_schema(
        self,
        message: str,
        images: Sequence[Path] = (),
        # mypy limitation with generic type defaults
        response_class: type[T] = RichResult,  # type: ignore[assignment]
    ) -> T: ...

    @staticmethod
    def path_jpeg_to_base64(input_: Path) -> str:
        with input_.open("rb") as infile:
            return base64.b64encode(infile.read()).decode()
