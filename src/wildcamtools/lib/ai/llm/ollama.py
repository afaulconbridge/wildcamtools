import json
import logging
from collections.abc import Sequence
from pathlib import Path

import ollama as ollama_lib
from pydantic import ValidationError

from wildcamtools.lib.ai.llm.abstract import DEFAULT_SYSTEM_MESSAGE, AbstractLlm, T
from wildcamtools.lib.ai.types import Backend, StringResponse

logger = logging.getLogger(__name__)


class OllamaLlm(AbstractLlm):
    client: ollama_lib.Client
    model: str
    backend: Backend = Backend.OLLAMA
    url: str
    api_key: str | None = None
    system_message: str

    def __init__(
        self,
        model: str = "qwen3.5:cloud",
        api_key: str | None = None,
        host: str = "https://ollama.com",
        system_message: str | None = None,
    ) -> None:
        self.client = ollama_lib.Client(
            host=host,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )
        self.model = model
        self.url = host
        self.api_key = api_key
        self.system_message = system_message or DEFAULT_SYSTEM_MESSAGE

    def message_with_schema(
        self,
        message: str,
        images: Sequence[Path] = (),
        # mypy limitation with generic type defaults
        response_class: type[T] = StringResponse,  # type: ignore[assignment]
    ) -> T:
        image_bytes = [image.read_bytes() for image in images]
        logger.info("Loaded %d images for message", len(image_bytes))

        schema_json = json.dumps(response_class.model_json_schema(), indent=2)
        system_content = self.system_message.format(schema=schema_json)
        logger.debug("System message: %s", system_content)

        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": message,
                "images": image_bytes,
            },
        ]
        logger.info("Sending %d images to %s model with schema", len(image_bytes), self.model)
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={"seed": 42, "temperature": 0},
            keep_alive=60.0,
            format=response_class.model_json_schema(),
        )

        logger.info("Schema response: %s", response.message.content)

        json_string = response.message.content or "{}"
        if json_string.startswith("```json\n") and json_string.endswith("\n```"):
            json_string = json_string.removeprefix("```json\n")
            json_string = json_string.removesuffix("\n```")

        try:
            return response_class.model_validate_json(json_string)
        except ValidationError:
            logger.exception("Failed to validate response against schema")
            raise
