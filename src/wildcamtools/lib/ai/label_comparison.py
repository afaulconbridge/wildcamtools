import logging
from abc import ABC, abstractmethod

from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.types import BoolResponse, ResultClassification

logger = logging.getLogger(__name__)

ABSENCE_MARKERS = {"", "none", "no animal", "missing", "empty", "no", "no detection"}
UNKNOWN_MARKERS = {"unknown", "uncertain", "unsure"}


class AbstractLabelComparator(ABC):
    """Abstract base class for label comparison strategies."""

    @staticmethod
    def _check_special_cases(result: str, label: str) -> tuple[bool, ResultClassification] | None:
        """Check for special cases (unknown markers, absence markers).

        Args:
            result: The raw result string from the pipeline.
            label: The ground truth label.

        Returns:
            Tuple of (is_correct, classification) if a special case is detected,
            None otherwise.
        """
        result_lower = result.lower()
        label_lower = label.lower()

        if result_lower in UNKNOWN_MARKERS:
            return False, ResultClassification.UNKNOWN

        if result_lower in ABSENCE_MARKERS:
            if label_lower in ABSENCE_MARKERS:
                return True, ResultClassification.CORRECT
            else:
                return False, ResultClassification.INCORRECT

        if result_lower == label_lower:
            return True, ResultClassification.CORRECT

        return None

    @abstractmethod
    def compare(self, result: str, label: str) -> tuple[bool, ResultClassification]:
        """Compare a pipeline result against a ground truth label.

        Args:
            result: The raw result string from the pipeline.
            label: The ground truth label.

        Returns:
            Tuple of (is_correct, classification) where:
            - is_correct: True if the result matches the label
            - classification: The type of result (correct, incorrect, unknown)
        """
        ...

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of the comparison method for logging/output."""
        ...


class ExactLabelComparator(AbstractLabelComparator):
    """Exact string matching comparator (case-insensitive)."""

    def compare(self, result: str, label: str) -> tuple[bool, ResultClassification]:
        special_case = self._check_special_cases(result, label)
        if special_case is not None:
            return special_case

        return False, ResultClassification.INCORRECT

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

    _cache: dict[tuple[str, str], tuple[bool, ResultClassification]] | None

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
            "result is '{result}'"
            "label is '{label}'"
        )
        self._cache = {} if cache_enabled else None

    def compare(self, result: str, label: str) -> tuple[bool, ResultClassification]:
        special_case = self._check_special_cases(result, label)
        if special_case is not None:
            return special_case

        result_lower = result.lower()
        label_lower = label.lower()

        if self._cache is not None:
            cache_key = (result_lower, label_lower)
            if cache_key in self._cache:
                logger.debug("Using cached comparison for (%s, %s)", result, label)
                is_correct, classification = self._cache[cache_key]
                return is_correct, classification

        message = self.prompt.format(result=result, label=label)
        logger.debug("Comparing '%s' vs '%s' using LLM", result, label)

        response = self.llm.message_with_schema(
            message=message,
            images=(),
            response_class=BoolResponse,
        )

        classification = ResultClassification.CORRECT if response.answer else ResultClassification.INCORRECT

        if self._cache is not None:
            self._cache[cache_key] = (response.answer, classification)
        logger.info("Comparison result: '%s' vs '%s' = %s", result, label, response.answer)
        return response.answer, classification

    @property
    def method_name(self) -> str:
        return "llm"

    def clear_cache(self) -> None:
        """Clear the comparison cache."""
        if self._cache is not None:
            self._cache.clear()
