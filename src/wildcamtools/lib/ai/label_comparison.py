import logging
from abc import ABC, abstractmethod

from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.pipeline import CONFIDENCE_ORDER
from wildcamtools.lib.ai.types import (
    ABSENCE_MARKERS,
    UNKNOWN_MARKERS,
    BoolResponse,
    ConfidenceLevel,
    ResultClassification,
    RichResult,
)

logger = logging.getLogger(__name__)


class AbstractLabelComparator(ABC):
    """Abstract base class for label comparison strategies."""

    @staticmethod
    def _check_special_cases(result: str, label: str) -> ResultClassification | None:
        """Check for special cases (unknown markers, absence markers).

        Args:
            result: The raw result string from the pipeline.
            label: The ground truth label.

        Returns:
            Classification if a special case is detected, None otherwise.

        """
        result_lower = result.lower()
        label_lower = label.lower()

        if result_lower in UNKNOWN_MARKERS:
            return ResultClassification.UNKNOWN

        if result_lower in ABSENCE_MARKERS:
            if label_lower in ABSENCE_MARKERS:
                return ResultClassification.CORRECT
            return ResultClassification.INCORRECT

        if result_lower == label_lower:
            return ResultClassification.CORRECT

        return None

    @abstractmethod
    def compare(self, result: RichResult, label: str) -> ResultClassification:
        """Compare a pipeline result against a ground truth label.

        Args:
            result: The result from the pipeline.
            label: The ground truth label.

        Returns:
            The classification type (correct, incorrect, unknown).
            The correctness can be inferred from the classification value.

        """
        ...

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of the comparison method for logging/output."""
        ...


class ExactLabelComparator(AbstractLabelComparator):
    """Exact string matching comparator (case-insensitive)."""

    def compare(self, result: RichResult, label: str) -> ResultClassification:
        if result.is_animal_unknown:
            return ResultClassification.UNKNOWN

        if CONFIDENCE_ORDER[result.confidence] < CONFIDENCE_ORDER[ConfidenceLevel.HIGH]:
            return ResultClassification.UNKNOWN

        if not result.is_animal_present:
            if label.lower() in ABSENCE_MARKERS:
                return ResultClassification.CORRECT
            return ResultClassification.INCORRECT

        special_case = self._check_special_cases(result.species_name, label)
        if special_case is not None:
            return special_case

        # only get here if its not an exact match
        return ResultClassification.INCORRECT

    @property
    def method_name(self) -> str:
        return "exact"


class LLMLabelComparator(AbstractLabelComparator):
    """LLM-based semantic label comparator.

    Uses an LLM to determine if a result semantically matches a label.
    The rule is: result can be a subset/specific instance of label, but not vice versa.

    Examples:
        - "domestic cat" vs "cat" → True (specific instance)
        - "Moorhen" vs "bird" → True (specific instance)
        - "cat" vs "domestic cat" → False (too general)
        - "animal" vs "cat" → False (too general)

    """

    _cache: dict[tuple[str, str], ResultClassification] | None

    def __init__(
        self,
        llm: AbstractLlm,
        prompt: str | None = None,
        cache_enabled: bool = True,
    ) -> None:
        self.llm = llm
        self.prompt = prompt or (
            "You are comparing an AI classification result against a ground truth label. "
            "Determine if the result qualifies as a specific instance or type of the label. "
            "Rules:\n"
            "- If result is a specific type/instance of the label, answer True (e.g., 'domestic cat' IS A 'cat')\n"
            "- If result is the same as the label, answer True\n"
            "- If result is too general or unrelated, answer False (e.g., 'cat' is NOT specifically 'domestic cat')\n"
            "- If result is a different category, answer False\n"
            "- If both result and label is missing, empty, or contains only 'none', 'no', answer True\n"
            "\n"
            "result is '{result}'\n"
            "label is '{label}'"
        )
        self._cache = {} if cache_enabled else None

    def compare(self, result: RichResult, label: str) -> ResultClassification:
        result_lower = result.species_name.lower()
        label_lower = label.lower()

        if result.is_animal_unknown:
            return ResultClassification.UNKNOWN

        if CONFIDENCE_ORDER[result.confidence] < CONFIDENCE_ORDER[ConfidenceLevel.HIGH]:
            return ResultClassification.UNKNOWN

        if not result.is_animal_present:
            if label_lower in ABSENCE_MARKERS:
                return ResultClassification.CORRECT
            return ResultClassification.INCORRECT

        special_case = self._check_special_cases(result_lower, label_lower)
        if special_case is not None:
            return special_case

        if self._cache is not None:
            cache_key = (result_lower, label_lower)
            if cache_key in self._cache:
                logger.debug("Using cached comparison for (%s, %s)", result.species_name, label)
                return self._cache[cache_key]

        message = self.prompt.format(result=result.species_name, label=label)
        logger.debug("Comparing '%s' vs '%s' using LLM", result.species_name, label)

        response = self.llm.message_with_schema(
            message=message,
            images=(),
            response_class=BoolResponse,
        )

        classification = ResultClassification.CORRECT if response.answer else ResultClassification.INCORRECT

        if self._cache is not None:
            self._cache[cache_key] = classification
        logger.info("Comparison result: '%s' vs '%s' = %s", result.species_name, label, response.answer)
        return classification

    @property
    def method_name(self) -> str:
        return "llm"

    def clear_cache(self) -> None:
        """Clear the comparison cache."""
        if self._cache is not None:
            self._cache.clear()
