from unittest.mock import MagicMock

import pytest

from wildcamtools.lib.ai import BoolResponse
from wildcamtools.lib.ai.label_comparison import (
    ExactLabelComparator,
    LLMLabelComparator,
)
from wildcamtools.lib.ai.types import ResultClassification


class TestExactLabelComparator:
    def test_exact_match_case_insensitive(self) -> None:
        comparator = ExactLabelComparator()
        assert comparator.compare("cat", "cat") == (True, ResultClassification.CORRECT)
        assert comparator.compare("CAT", "cat") == (True, ResultClassification.CORRECT)
        assert comparator.compare("Cat", "CAT") == (True, ResultClassification.CORRECT)

    def test_exact_match_mismatch(self) -> None:
        comparator = ExactLabelComparator()
        assert comparator.compare("cat", "dog") == (False, ResultClassification.INCORRECT)
        assert comparator.compare("cat", "cats") == (False, ResultClassification.INCORRECT)
        assert comparator.compare("domestic cat", "cat") == (False, ResultClassification.INCORRECT)

    def test_unknown_result(self) -> None:
        comparator = ExactLabelComparator()
        assert comparator.compare("unknown", "cat") == (False, ResultClassification.UNKNOWN)
        assert comparator.compare("Uncertain", "dog") == (False, ResultClassification.UNKNOWN)
        assert comparator.compare("unsure", "bird") == (False, ResultClassification.UNKNOWN)

    def test_method_name(self) -> None:
        comparator = ExactLabelComparator()
        assert comparator.method_name == "exact"


class TestLLMLabelComparator:
    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        mock = MagicMock()
        mock.model = "test-model"
        mock.backend = "ollama"
        mock.url = "http://localhost:8080/v1"
        return mock

    def test_semantic_match_specific_to_general(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        assert comparator.compare("domestic cat", "cat") == (True, ResultClassification.CORRECT)
        assert comparator.compare("Moorhen", "bird") == (True, ResultClassification.CORRECT)
        assert comparator.compare("Roe deer", "deer") == (True, ResultClassification.CORRECT)

    def test_semantic_match_general_to_specific(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=False)
        mock_llm.message_with_schema.return_value = mock_response

        assert comparator.compare("cat", "domestic cat") == (False, ResultClassification.INCORRECT)
        assert comparator.compare("bird", "Moorhen") == (False, ResultClassification.INCORRECT)
        assert comparator.compare("deer", "Roe deer") == (False, ResultClassification.INCORRECT)

    def test_unknown_result(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)
        assert comparator.compare("unknown", "cat") == (False, ResultClassification.UNKNOWN)
        assert comparator.compare("uncertain", "dog") == (False, ResultClassification.UNKNOWN)

    def test_caching(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("domestic cat", "cat")
        comparator.compare("domestic cat", "cat")

        assert mock_llm.message_with_schema.call_count == 1

    def test_cache_clear(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("domestic cat", "cat")
        comparator.clear_cache()
        comparator.compare("domestic cat", "cat")

        assert mock_llm.message_with_schema.call_count == 2

    def test_cache_disabled(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm, cache_enabled=False)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("domestic cat", "cat")
        comparator.compare("domestic cat", "cat")

        assert mock_llm.message_with_schema.call_count == 2
        assert comparator._cache is None

    def test_llm_call_made_for_semantic_comparison(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("domestic cat", "cat")

        assert mock_llm.message_with_schema.call_count == 1

    def test_llm_call_with_false_response(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=False)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("domestic cat", "cat")

        assert mock_llm.message_with_schema.call_count == 1

    def test_custom_prompt(self, mock_llm: MagicMock) -> None:
        custom_prompt = "Custom prompt: {result} vs {label}"
        comparator = LLMLabelComparator(llm=mock_llm, prompt=custom_prompt)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("domestic cat", "cat")

        mock_llm.message_with_schema.assert_called_once()
        call_args = mock_llm.message_with_schema.call_args
        assert call_args.kwargs["message"] == "Custom prompt: domestic cat vs cat"

    def test_default_prompt(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("domestic cat", "cat")

        call_args = mock_llm.message_with_schema.call_args
        assert "You are comparing an AI classification result" in call_args.kwargs["message"]

    def test_method_name(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)
        assert comparator.method_name == "llm"

    def test_case_insensitive_caching(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("Domestic Cat", "CAT")
        comparator.compare("domestic cat", "cat")

        assert mock_llm.message_with_schema.call_count == 1
