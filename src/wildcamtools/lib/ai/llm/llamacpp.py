import json
import logging
from collections.abc import Sequence
from pathlib import Path

import openai as openai_lib
from pydantic import ValidationError

from wildcamtools.lib.ai.llm.abstract import DEFAULT_SYSTEM_MESSAGE, AbstractLlm, T
from wildcamtools.lib.ai.types import Backend, RichResult

logger = logging.getLogger(__name__)


class LlamaCppLlm(AbstractLlm):
    client: openai_lib.Client
    model: str
    backend: Backend = Backend.LLAMACPP
    url: str
    api_key: str | None = None
    system_message: str

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        system_message: str | None = None,
    ) -> None:
        self.client = openai_lib.OpenAI(
            api_key=api_key or "dummy",
            base_url=base_url,
        )
        self.model = model
        self.url = base_url
        self.api_key = api_key
        self.system_message = system_message or DEFAULT_SYSTEM_MESSAGE

    def message_with_schema(
        self,
        message: str,
        images: Sequence[Path] = (),
        # mypy limitation with generic type defaults
        response_class: type[T] = RichResult,  # type: ignore[assignment]
    ) -> T:
        image_bytes = [self.path_jpeg_to_base64(image) for image in images]
        logger.info("Loaded %d images for message", len(image_bytes))

        schema_json = json.dumps(response_class.model_json_schema(), indent=2)
        system_content = self.system_message.format(schema=schema_json)
        logger.debug("System message: %s", system_content)

        content_items: list[dict[str, str | dict[str, str]]] = [
            {
                "type": "text",
                "text": message,
            },
        ]

        for image_base64 in image_bytes:
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            })

        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": content_items,
            },
        ]

        logger.info("Sending %d images to %s model with schema", len(image_bytes), self.model)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore [arg-type]
            temperature=0,
        )

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty response from API")

        json_string = response.choices[0].message.content
        logger.info("Schema response: %s", json_string)
        if json_string.startswith("```json\n") and json_string.endswith("\n```"):
            json_string = json_string.removeprefix("```json\n")
            json_string = json_string.removesuffix("\n```")

        try:
            return response_class.model_validate_json(json_string)
        except ValidationError:
            logger.exception("Failed to validate response against schema")
            raise
