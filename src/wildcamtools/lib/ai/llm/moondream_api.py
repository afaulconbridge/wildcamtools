import base64
import logging
from collections.abc import Sequence
from pathlib import Path

import httpx
from pydantic import ValidationError

from wildcamtools.lib.ai.llm.abstract import AbstractLlm, T
from wildcamtools.lib.ai.types import Backend, StringResponse

logger = logging.getLogger(__name__)


class MoondreamApiLlm(AbstractLlm):
    model: str = "moondream-3"
    backend: Backend = Backend.MOONDREAM_CLOUD
    url: str = "https://api.moondream.ai/v1/"
    api_key: str | None = None
    system_message: str

    def __init__(
        self,
        model: str = "moondream-3",
        api_key: str | None = None,
        url: str = "https://api.moondream.ai/v1/",
        system_message: str | None = None,
    ) -> None:
        self.model = model
        self.url = url
        self.api_key = api_key
        self.system_message = system_message or ""

    def message_with_schema(
        self,
        message: str,
        images: Sequence[Path] = (),
        response_class: type[T] = StringResponse,  # type: ignore[assignment]
    ) -> T:
        if len(images) != 1:
            raise ValueError("Moondream API only supports one image at a time")

        system_content = """You must respond with valid JSON that matches the example below:
{
  "species_name": "",
}

This is a video image from a UK garden near a river. Is there an animal in this image, if so what? Respond with the species_name set to the species_name. If there is no animal, respond with species_name set to "no animal". If there is an animal but you are not certain what it is, respond with species_name set to "unknown"
"""

        image_data = self._encode_image_to_base64(images[0])
        image_url = f"data:image/jpeg;base64,{image_data}"

        payload = {
            "image_url": image_url,
            "question": system_content,
        }

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["X-Moondream-Auth"] = self.api_key

        logger.info("Sending 1 image to %s model with schema", self.model)

        with httpx.Client() as client:
            response = client.post(
                f"{self.url}query",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

        if not result or "answer" not in result:
            raise ValueError("Empty or invalid response from Moondream API")

        json_string = result.get("answer", "{}")
        logger.debug("Schema response: %s", json_string)

        if json_string.startswith("```json\n") and json_string.endswith("\n```"):
            json_string = json_string.removeprefix("```json\n")
            json_string = json_string.removesuffix("\n```")

        try:
            return response_class.model_validate_json(json_string)
        except ValidationError:
            logger.exception("Failed to validate response against schema")
            raise

    @staticmethod
    def _encode_image_to_base64(image_path: Path) -> str:
        with image_path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
