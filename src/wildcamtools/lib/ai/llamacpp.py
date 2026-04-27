import logging
from collections.abc import Iterable
from pathlib import Path

import openai as openai_lib

from wildcamtools.lib.ai import AbstractAnalyser

logger = logging.getLogger(__name__)


class LlamaCppAnalyser(AbstractAnalyser):
    message: str = """This is a video image from a UK garden near a river. Is there an animal in this image, if so what? answer only 'no' or its common name.
For example: 'deer', 'bird', 'fox', 'mouse', 'hedgehog', 'otter', etc"""
    client: openai_lib.Client
    model: str

    def __init__(
        self,
        model: str,
        base_url: str,
        message: str | None = None,
    ) -> None:
        if message is not None:
            self.message = message
        self.client = openai_lib.OpenAI(
            api_key="dummy",
            base_url=base_url,
        )
        self.model = model

    def analyze_video(self, images: Iterable[Path]) -> str:
        # create messages for the conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.message,
                    }
                ],
            },
        ]

        # Add each image as base64-encoded content
        for image in images:
            image_base64 = self.path_jpeg_to_base64(image)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    }
                ],
            })
        logger.info("Sending %d images to %s model", len(messages) - 1, self.model)

        # stream it back to minimize chance of timeouts
        collected = ""
        completion_stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore [arg-type]
            temperature=0,
            stream=True,
        )
        for chunk in completion_stream:
            delta = chunk.choices[0].delta  # type: ignore [union-attr]
            if not delta:
                continue
            text = getattr(delta, "content", "")
            if text:
                collected += text
                logger.info("Streaming chunk: %s", text)
        logger.info("Analysis result: %s", collected)
        return collected
