import abc
import base64
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from wildcamtools.lib.ai import Backend, StringResponse

T = TypeVar("T", bound=BaseModel)


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
        response_class: type[T] = StringResponse,  # type: ignore[assignment]
    ) -> T: ...

    @staticmethod
    def path_jpeg_to_base64(input_: Path) -> str:
        with input_.open("rb") as infile:
            return base64.b64encode(infile.read()).decode()
