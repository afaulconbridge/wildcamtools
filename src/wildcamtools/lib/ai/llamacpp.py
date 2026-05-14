import logging
from collections.abc import Iterable
from pathlib import Path

import openai as openai_lib

from wildcamtools.lib.ai import AbstractAnalyser, Backend, ResultList

logger = logging.getLogger(__name__)


class LlamaCppAnalyser(AbstractAnalyser):
    message: str = """This is a video image from a UK garden near a river. Is there an animal in this image, if so what? answer only 'no' or its common name.
For example: 'deer', 'bird', 'fox', 'mouse', 'hedgehog', 'otter', etc"""
    client: openai_lib.Client
    model: str
    backend: Backend = Backend.LLAMACPP
    url: str
    api_key: str | None = None

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
        self.url = base_url

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

    def detect(self, images: Iterable[Path]) -> ResultList:
        images_sorted = sorted(images)
        messages = [
            {
                "role": "system",
                "content": "You are a wildlife detection assistant. Analyze video frames and return JSON with detected animals.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """These are frames from a video taken in a UK garden near a river.
Identify any animals you are highly confident of in the images.
Return JSON only with this exact structure:
{"results": [{"species_name": "string", "frames": [{"frame_no":0,"left": 0.0, "right": 1.0, "top": 1.0, "bottom": 0.0}]}]}
Note that the bounding box coordinates are proportional to the image dimensions.
If no animals are detected, return {"results": []}.""",
                    }
                ],
            },
        ]

        for image in images_sorted:
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

        logger.info("Sending %d images to %s model for detection", len(images_sorted), self.model)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore [arg-type]
            temperature=0,
        )

        json_string = response.choices[0].message.content or "{}"
        if json_string.startswith("```json\n") and json_string.endswith("\n```"):
            json_string = json_string.removeprefix("```json\n")
            json_string = json_string.removesuffix("\n```")

        result_list = ResultList.model_validate_json(json_string)
        return result_list
