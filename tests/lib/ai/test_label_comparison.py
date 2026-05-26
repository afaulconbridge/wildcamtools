from unittest.mock import MagicMock

import pytest

from wildcamtools.lib.ai import BoolResponse
from wildcamtools.lib.ai.label_comparison import (
    ExactLabelComparator,
    LLMLabelComparator,
)


class TestExactLabelComparator:
    def test_exact_match_case_insensitive(self) -> None:
        comparator = ExactLabelComparator()
        assert comparator.compare("cat", "cat") is True
        assert comparator.compare("CAT", "cat") is True
        assert comparator.compare("Cat", "CAT") is True

    def test_exact_match_mismatch(self) -> None:
        comparator = ExactLabelComparator()
        assert comparator.compare("cat", "dog") is False
        assert comparator.compare("cat", "cats") is False
        assert comparator.compare("domestic cat", "cat") is False

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

        assert comparator.compare("domestic cat", "cat") is True
        assert comparator.compare("Moorhen", "bird") is True
        assert comparator.compare("Roe deer", "deer") is True

    def test_semantic_match_general_to_specific(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=False)
        mock_llm.message_with_schema.return_value = mock_response

        assert comparator.compare("cat", "domestic cat") is False
        assert comparator.compare("bird", "Moorhen") is False
        assert comparator.compare("deer", "Roe deer") is False

    def test_caching(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("cat", "cat")
        comparator.compare("cat", "cat")

        assert mock_llm.message_with_schema.call_count == 1

    def test_cache_clear(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("cat", "cat")
        comparator.clear_cache()
        comparator.compare("cat", "cat")

        assert mock_llm.message_with_schema.call_count == 2

    def test_cache_disabled(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm, cache_enabled=False)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("cat", "cat")
        comparator.compare("cat", "cat")

        assert mock_llm.message_with_schema.call_count == 2
        assert comparator._cache is None

    def test_invalid_json_response(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("cat", "cat")

        assert mock_llm.message_with_schema.call_count == 1

    def test_missing_match_field_raises_error(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=False)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("cat", "cat")

        assert mock_llm.message_with_schema.call_count == 1

    def test_custom_prompt(self, mock_llm: MagicMock) -> None:
        custom_prompt = "Custom prompt: {result} vs {label}"
        comparator = LLMLabelComparator(llm=mock_llm, prompt=custom_prompt)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("cat", "cat")

        mock_llm.message_with_schema.assert_called_once()
        call_args = mock_llm.message_with_schema.call_args
        assert call_args.kwargs["message"] == "Custom prompt: cat vs cat"

    def test_default_prompt(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("cat", "cat")

        call_args = mock_llm.message_with_schema.call_args
        assert "You are comparing an AI classification result" in call_args.kwargs["message"]

    def test_method_name(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)
        assert comparator.method_name == "llm"

    def test_case_insensitive_caching(self, mock_llm: MagicMock) -> None:
        comparator = LLMLabelComparator(llm=mock_llm)

        mock_response = BoolResponse(answer=True)
        mock_llm.message_with_schema.return_value = mock_response

        comparator.compare("Cat", "CAT")
        comparator.compare("cat", "cat")

        assert mock_llm.message_with_schema.call_count == 1
