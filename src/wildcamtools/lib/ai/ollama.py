import logging
from collections.abc import Iterable
from pathlib import Path

import ollama as ollama_lib

from wildcamtools.lib.ai import AbstractAnalyser

logger = logging.getLogger(__name__)


class OllamaAnalyser(AbstractAnalyser):
    message: str = "This is a video image from a UK garden near a river. Is there an animal in this image, if so what? answer only 'no' or the common English name of the species."
    client: ollama_lib.Client
    model: str

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
