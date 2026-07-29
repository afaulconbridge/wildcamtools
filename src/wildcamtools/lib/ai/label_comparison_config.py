from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, Field

from wildcamtools.lib.ai.label_comparison import (
    AbstractLabelComparator,
    ExactLabelComparator,
    LLMLabelComparator,
)
from wildcamtools.lib.ai.pipeline_config import LlmConfig


class LabelComparisonType(StrEnum):
    EXACT = "exact"
    LLM = "llm"


class LabelComparisonConfig(BaseModel):
    """Configuration for label comparison in evaluation harness.

    Args:
        comparator_type: Type of comparator to use (exact or llm).
        llm: Optional LLM configuration for LLM-based comparison.
            If not provided and comparator_type is 'llm', uses default settings.
        prompt: Custom prompt for LLM comparison. Uses sensible default if not provided.
        cache_enabled: Whether to cache comparison results. Defaults to True.

    Example:
        ```json
        {
            "comparator_type": "llm",
            "llm": {
                "model": "llama3.2:1b",
                "backend": "ollama",
                "url": "http://localhost:8080/v1"
            },
            "prompt": "Does '{result}' qualify as a type of '{label}'?",
            "cache_enabled": true
        }
        ```

    """

    comparator_type: LabelComparisonType = LabelComparisonType.EXACT
    llm: LlmConfig | None = None
    prompt: Annotated[
        str | None,
        Field(
            description="Custom prompt for LLM comparison. Uses {result} and {label} placeholders.",
            min_length=1,
        ),
    ] = None
    cache_enabled: bool = True

    @classmethod
    def from_json(cls, path: Path) -> Self:
        """Load configuration from JSON file."""
        content = path.read_text()
        return cls.model_validate_json(content)

    def to_json(self, path: Path, indent: int = 2) -> None:
        """Save configuration to JSON file."""
        json_str = self.model_dump_json(indent=indent)
        path.write_text(json_str)

    def create_comparator(self) -> AbstractLabelComparator:
        """Create a comparator instance based on the configuration.

        Returns:
            AbstractLabelComparator: The configured comparator instance.

        Raises:
            ValueError: If comparator_type is 'llm' but no LLM config is provided.
            NotImplementedError: If the comparator_type is not supported.

        """
        match self.comparator_type:
            case LabelComparisonType.EXACT:
                return ExactLabelComparator()

            case LabelComparisonType.LLM:
                if self.llm is None:
                    raise ValueError(
                        "LLM configuration is required when comparator_type is 'llm'. "
                        "Please provide 'llm' configuration in the label comparison config.",
                    )

                llm_instance = self.llm.create_llm()
                return LLMLabelComparator(
                    llm=llm_instance,
                    prompt=self.prompt,
                    cache_enabled=self.cache_enabled,
                )

            case _:
                raise NotImplementedError(f"Unsupported comparator type: {self.comparator_type}")


LabelComparisonConfig.model_rebuild()
