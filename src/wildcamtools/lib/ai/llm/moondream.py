import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pydantic import ValidationError
from transformers import AutoModelForCausalLM

from wildcamtools.lib.ai.llm.abstract import DEFAULT_SYSTEM_MESSAGE, AbstractLlm, T
from wildcamtools.lib.ai.types import Backend, StringResponse

logger = logging.getLogger(__name__)


class MoondreamLlm(AbstractLlm):
    model_instance: Any
    backend: Backend = Backend.MOONDREAM
    url: str = "local"
    api_key: str | None = None
    reasoning: bool = True

    def __init__(
        self,
        model: str = "moondream/moondream3-preview",
        device: str = "cpu",
        dtype: str = "bfloat16",
        system_message: str | None = None,
    ) -> None:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        selected_dtype = dtype_map.get(dtype)
        if selected_dtype is None:
            logger.warning("Invalid dtype '%s', defaulting to bfloat16", dtype)
            selected_dtype = torch.bfloat16

        logger.info("Loading Moondream model: %s on %s with dtype %s", model, device, dtype)
        self.model_instance = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            dtype=selected_dtype,
            device_map={"": device},
        )
        logger.info("Compiling Moondream model for fast decoding")
        try:
            self.model_instance.compile()
        except Exception as e:
            logger.warning("Model compilation failed, continuing without optimization: %s", e)
        self.model = model
        self.system_message = system_message or DEFAULT_SYSTEM_MESSAGE

    def message_with_schema(
        self,
        message: str,
        images: Sequence[Path] = (),
        response_class: type[T] = StringResponse,  # type: ignore[assignment]
        max_retries: int = 3,
    ) -> T:
        """Send message with schema validation and retry on failure.

        Args:
            message: The user message/prompt
            images: Sequence of image paths (Moondream requires exactly 1)
            response_class: Pydantic model class for response validation
            max_retries: Maximum retry attempts after validation failure (default: 3)

        Returns:
            Validated response object

        Raises:
            ValidationError: If all retries fail validation
            ValueError: If model returns empty/invalid response or wrong image count
        """
        return self._retry_with_schema_validation(
            message=message,
            images=images,
            response_class=response_class,
            max_retries=max_retries,
        )

    def _retry_with_schema_validation(
        self,
        message: str,
        images: Sequence[Path],
        response_class: type[T],
        max_retries: int,
    ) -> T:
        """
        Retry orchestration logic.

        Manages the retry loop, tracks errors, and delegates to
        _build_attempt_prompt() for prompt construction.

        Args:
            message: The original user message
            images: Sequence of image paths
            response_class: Pydantic model class for validation
            max_retries: Maximum number of retry attempts

        Returns:
            Validated response object

        Raises:
            ValidationError: If all retry attempts fail
        """
        schema_json = json.dumps(response_class.model_json_schema(), indent=2)

        last_error: ValidationError | None = None
        last_response: str = ""

        for attempt in range(1, max_retries + 1):
            logger.info("Schema validation attempt %d/%d", attempt, max_retries)

            prompt = self._build_attempt_prompt(
                original_message=message,
                invalid_response=last_response,
                error=last_error,
                schema_json=schema_json,
                attempt_number=attempt,
            )

            response = self._send_query(prompt, images)
            logger.debug("Attempt %d raw response: %s", attempt, response)

            try:
                result = response_class.model_validate_json(response)
            except ValidationError as e:
                last_error = e
                last_response = response
                logger.warning(
                    "Schema validation failed (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    str(e),
                )

                if attempt == max_retries:
                    logger.exception(
                        "Schema validation failed after %d attempts",
                        max_retries,
                    )
                    raise
                else:
                    continue

            if attempt > 1:
                logger.info("Schema validation succeeded on attempt %d/%d", attempt, max_retries)
            return result

        raise RuntimeError("Unreachable")  # pragma: no cover

    def _build_attempt_prompt(
        self,
        original_message: str,
        invalid_response: str,
        error: ValidationError | None,
        schema_json: str,
        attempt_number: int,
    ) -> str:
        """
        Build prompt for a specific attempt.

        First attempt: Standard schema prompt
        Retry attempts: Include error context and correction request

        Args:
            original_message: The original user message
            invalid_response: The previous invalid response (empty on first attempt)
            error: The previous validation error (None on first attempt)
            schema_json: The JSON schema string
            attempt_number: Current attempt number (1-indexed)

        Returns:
            Complete prompt string for this attempt
        """
        if attempt_number == 1:
            system_content = self.system_message.format(schema=schema_json)
            return f"{system_content}\n\n{original_message}"

        error_details = error.errors() if error else "Unknown validation error"

        return (
            "Your previous response failed JSON schema validation.\n\n"
            f"Original request: {original_message}\n\n"
            f"Your invalid response: {invalid_response}\n\n"
            f"Validation error: {error_details}\n\n"
            f"Expected JSON schema:\n{schema_json}\n\n"
            "Please correct your response to match the schema exactly. "
            "Output only the valid JSON object, no explanations or markdown formatting."
        )

    def _send_query(
        self,
        prompt: str,
        images: Sequence[Path],
    ) -> str:
        """
        Backend-specific message sending.

        Handles image loading, Moondream API call, and response extraction.
        Returns raw response string (not yet validated).

        Args:
            prompt: The prompt/question to send
            images: Sequence of image paths (Moondream requires exactly 1)

        Returns:
            Raw response string from the model

        Raises:
            ValueError: If image count is not exactly 1 or response is empty
        """
        if len(images) != 1:
            logger.error("Moondream requires exactly 1 image, got %d", len(images))
            raise ValueError("Moondream only supports one image at a time")

        logger.debug("Loading image for query")
        with Image.open(images[0]) as img:
            pil_image = img.copy()

        logger.info("Sending query to %s model", self.model)
        result = self.model_instance.query(
            image=pil_image,
            question=prompt,
            reasoning=self.reasoning,
        )

        if not result or "answer" not in result:
            logger.error("Empty or invalid response from Moondream model: %s", result)
            raise ValueError("Empty or invalid response from Moondream model")

        json_string: str = result.get("answer", "{}")

        if json_string.startswith("```json\n") and json_string.endswith("\n```"):
            logger.debug("Stripping markdown code blocks from response")
            json_string = json_string.removeprefix("```json\n")
            json_string = json_string.removesuffix("\n```")
        elif json_string.startswith("```\n") and json_string.endswith("\n```"):
            logger.debug("Stripping plain markdown code blocks from response")
            json_string = json_string.removeprefix("```\n")
            json_string = json_string.removesuffix("\n```")

        return json_string
