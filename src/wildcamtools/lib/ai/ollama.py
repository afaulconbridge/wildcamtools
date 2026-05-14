import logging
from collections.abc import Iterable
from pathlib import Path

import ollama as ollama_lib
from pydantic import ValidationError

from wildcamtools.lib.ai import AbstractAnalyser, Backend, ResultList

logger = logging.getLogger(__name__)


class OllamaAnalyser(AbstractAnalyser):
    message: str = """This is a video image from a UK garden near a river. Is there an animal in this image, if so what? answer only 'no' or its common name.
For example: 'deer', 'bird', 'fox', 'mouse', 'no', 'hedgehog', 'otter', etc."""
    client: ollama_lib.Client
    model: str
    backend: Backend = Backend.OLLAMA
    url: str
    api_key: str | None = None

    def __init__(
        self,
        model: str = "qwen3.5:cloud",
        api_key: str | None = None,
        host: str = "https://ollama.com",
        message: str | None = None,
    ) -> None:
        if message is not None:
            self.message = message
        self.client = ollama_lib.Client(
            host=host,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )
        self.model = model
        self.url = host
        self.api_key = api_key

    def analyze_video(self, images: Iterable[Path]) -> str:
        image_bytes = [image.read_bytes() for image in images]
        logger.info("Loaded %d images for analysis", len(image_bytes))

        messages = [
            {
                "role": "user",
                "content": self.message,
                "images": image_bytes,
            },
        ]
        logger.info("Sending %d images to %s model", len(image_bytes), self.model)
        response = self.client.chat(
            self.model,
            messages=messages,
            options={"seed": 42, "temperature": 0},
            keep_alive=60.0,
        )
        logger.info("Analysis result: %s", response.message.content)
        return response.message.content or ""

    def detect(self, images: Iterable[Path]) -> ResultList:
        images_sorted = sorted(images)
        image_bytes = [image.read_bytes() for image in images_sorted]
        logger.info("Loaded %d images for analysis", len(image_bytes))

        messages = []
        messages.append({
            "role": "user",
            "content": """These are image from a video taken in a UK garden near a river.
            Identify any animals you are highly confident of in the image.
            Return JSON only with this exact structure:
            {"results": [{"species_name": "string", "frames": [{"frame_no":0,"left": 0.0, "right": 1.0, "top": 1.0, "bottom": 0.0}]}]}
            Note that the bounding box coordinates are proportional to the dimension and therefore must be between 0.0 and 1.0.
            If no animals are detected, return {"results": []}.""",
            "images": image_bytes,
        })
        logger.info("Sending %d images to %s model", len(image_bytes), self.model)
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={"seed": 42, "temperature": 0},
            keep_alive=60.0,
            format=ResultList.model_json_schema(),
        )

        logger.info("Analysis result: %s", response.message.content)

        json_string = response.message.content or "{}"
        if json_string.startswith("```json\n") and json_string.endswith("\n```"):
            json_string = json_string.removeprefix("```json\n")
            json_string = json_string.removesuffix("\n```")

        try:
            result_list = ResultList.model_validate_json(json_string)
        except ValidationError:
            logger.exception("Unable to validate")
            return ResultList(results=[])
            # TODO check if its using pixel-like coordinates and manually correct them

        return result_list
