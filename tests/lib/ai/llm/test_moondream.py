from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from wildcamtools.lib.ai import SpeciesResult, StringResponse
from wildcamtools.lib.ai.llm.abstract import DEFAULT_SYSTEM_MESSAGE
from wildcamtools.lib.ai.llm.moondream import MoondreamLlm


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    image_file = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(image_file, format="JPEG")
    return image_file


class TestMoondreamLlmSystemMessage:
    def test_default_system_message(self) -> None:
        with patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained"):
            llm = MoondreamLlm(model="test-model")
            assert llm.system_message == DEFAULT_SYSTEM_MESSAGE

    def test_custom_system_message(self) -> None:
        custom_message = "Custom system message: {schema}"
        with patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained"):
            llm = MoondreamLlm(model="test-model", system_message=custom_message)
            assert llm.system_message == custom_message

    def test_empty_system_message_uses_default(self) -> None:
        with patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained"):
            llm = MoondreamLlm(model="test-model", system_message=None)
            assert llm.system_message == DEFAULT_SYSTEM_MESSAGE

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_message_with_schema_includes_system_message(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": '{"species_name": "test"}'}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        mock_model.query.assert_called_once()
        call_kwargs = mock_model.query.call_args.kwargs
        question = call_kwargs.get("question")

        assert question is not None
        assert DEFAULT_SYSTEM_MESSAGE.split("\n\n")[0] in question

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_message_with_schema_includes_json_schema(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": '{"species_name": "test"}'}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        call_kwargs = mock_model.query.call_args.kwargs
        question = call_kwargs.get("question")

        assert "species_name" in question
        expected_schema = SpeciesResult.model_json_schema()
        assert any(prop in question for prop in expected_schema.get("properties", {}))

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_message_with_schema_different_response_classes(
        self, mock_model_class: MagicMock, sample_image: Path
    ) -> None:
        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": '{"species_name": "test"}'}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")

        for response_class in [SpeciesResult, StringResponse]:
            mock_model.query.reset_mock()
            mock_model.query.return_value = {
                "answer": '{"species_name": "test"}' if response_class == SpeciesResult else '{"message": "test"}'
            }

            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=response_class,
            )

            call_kwargs = mock_model.query.call_args.kwargs
            question = call_kwargs.get("question")

            schema = response_class.model_json_schema()
            assert any(prop in question for prop in schema.get("properties", {}))

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_custom_system_message_receives_schema(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        custom_message = "Custom: {schema}"
        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": '{"species_name": "test"}'}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model", system_message=custom_message)
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        call_kwargs = mock_model.query.call_args.kwargs
        question = call_kwargs.get("question")

        assert question.startswith("Custom:")
        schema = SpeciesResult.model_json_schema()
        assert any(prop in question for prop in schema.get("properties", {}))

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_message_with_schema_includes_expected_schema_text(
        self, mock_model_class: MagicMock, sample_image: Path
    ) -> None:
        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": '{"species_name": "test"}'}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        mock_model.query.assert_called_once()
        call_kwargs = mock_model.query.call_args.kwargs
        question = call_kwargs.get("question")

        assert "Expected JSON schema" in question

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_system_message_logged_at_debug_level(
        self, mock_model_class: MagicMock, sample_image: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": '{"species_name": "test"}'}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")

        with caplog.at_level(logging.DEBUG):
            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=SpeciesResult,
            )

        assert "raw response:" in caplog.text.lower()

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_message_with_schema_strips_markdown_code_blocks(
        self, mock_model_class: MagicMock, sample_image: Path
    ) -> None:
        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": '```json\n{"species_name": "test"}\n```'}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        result = llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        assert result.species_name == "test"

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_message_with_schema_strips_plain_markdown_blocks(
        self, mock_model_class: MagicMock, sample_image: Path
    ) -> None:
        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": '```\n{"species_name": "test"}\n```'}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        result = llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        assert result.species_name == "test"

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_message_with_schema_single_image(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": '{"species_name": "test"}'}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        result = llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        assert result.species_name == "test"
        call_kwargs = mock_model.query.call_args.kwargs
        assert "image" in call_kwargs

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_message_with_schema_invalid_json_response(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        from pydantic import ValidationError

        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": "not valid json"}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        with pytest.raises(ValidationError):
            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=SpeciesResult,
            )

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_message_with_schema_empty_response(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": '{"message": ""}'}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        result = llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=StringResponse,
        )

        assert result.message == ""

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_message_with_schema_missing_answer_key(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        mock_model = MagicMock()
        mock_model.query.return_value = {"wrong_key": "value"}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        with pytest.raises(ValueError, match="Empty or invalid response"):
            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=SpeciesResult,
            )

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_message_with_schema_none_response(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        mock_model = MagicMock()
        mock_model.query.return_value = None
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        with pytest.raises(ValueError, match="Empty or invalid response"):
            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=SpeciesResult,
            )


class TestMoondreamRetryMechanism:
    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_success_on_first_attempt(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        """No retry needed when first response is valid."""
        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": '{"species_name": "lion"}'}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        result = llm.message_with_schema(
            message="What animal?",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        assert result.species_name == "lion"
        mock_model.query.assert_called_once()

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_success_on_second_attempt(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        """Retry succeeds after first failure."""
        mock_model = MagicMock()
        mock_model.query.side_effect = [
            {"answer": "invalid json"},
            {"answer": '{"species_name": "lion"}'},
        ]
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        result = llm.message_with_schema(
            message="What animal?",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        assert result.species_name == "lion"
        assert mock_model.query.call_count == 2

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_failure_after_max_retries(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        """Raises ValidationError after exhausting retries."""
        from pydantic import ValidationError

        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": "invalid json"}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")

        with pytest.raises(ValidationError):
            llm.message_with_schema(
                message="What animal?",
                images=[sample_image],
                response_class=SpeciesResult,
                max_retries=3,
            )

        assert mock_model.query.call_count == 3

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_retry_prompt_includes_error_details(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        """Retry prompt contains full validation error context."""
        mock_model = MagicMock()
        mock_model.query.side_effect = [
            {"answer": '{"wrong_field": "value"}'},
            {"answer": '{"species_name": "lion"}'},
        ]
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        llm.message_with_schema(
            message="What animal?",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        second_call_kwargs = mock_model.query.call_args_list[1].kwargs
        retry_prompt = second_call_kwargs["question"]

        assert "Original request:" in retry_prompt
        assert "Your invalid response:" in retry_prompt
        assert "Validation error:" in retry_prompt
        assert "Expected JSON schema:" in retry_prompt

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_custom_max_retries(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        """Custom max_retries parameter is respected."""
        from pydantic import ValidationError

        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": "invalid json"}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")

        with pytest.raises(ValidationError):
            llm.message_with_schema(
                message="What animal?",
                images=[sample_image],
                response_class=SpeciesResult,
                max_retries=5,
            )

        assert mock_model.query.call_count == 5

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_max_retries_one(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        """max_retries=1 means one attempt only."""
        from pydantic import ValidationError

        mock_model = MagicMock()
        mock_model.query.return_value = {"answer": "invalid json"}
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")

        with pytest.raises(ValidationError):
            llm.message_with_schema(
                message="What animal?",
                images=[sample_image],
                response_class=SpeciesResult,
                max_retries=1,
            )

        assert mock_model.query.call_count == 1

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_success_on_third_attempt(self, mock_model_class: MagicMock, sample_image: Path) -> None:
        """Retry succeeds on third attempt."""
        mock_model = MagicMock()
        mock_model.query.side_effect = [
            {"answer": "invalid json 1"},
            {"answer": "invalid json 2"},
            {"answer": '{"species_name": "lion"}'},
        ]
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")
        result = llm.message_with_schema(
            message="What animal?",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        assert result.species_name == "lion"
        assert mock_model.query.call_count == 3

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_retry_logging_warning_on_failure(
        self, mock_model_class: MagicMock, sample_image: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Validation failures are logged at WARNING level."""
        import logging

        mock_model = MagicMock()
        mock_model.query.side_effect = [
            {"answer": "invalid json"},
            {"answer": '{"species_name": "lion"}'},
        ]
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")

        with caplog.at_level(logging.WARNING):
            llm.message_with_schema(
                message="What animal?",
                images=[sample_image],
                response_class=SpeciesResult,
            )

        assert "Schema validation failed" in caplog.text
        assert "attempt 1" in caplog.text.lower()

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_retry_logging_info_on_success(
        self, mock_model_class: MagicMock, sample_image: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Successful retry is logged at INFO level."""
        import logging

        mock_model = MagicMock()
        mock_model.query.side_effect = [
            {"answer": "invalid json"},
            {"answer": '{"species_name": "lion"}'},
        ]
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")

        with caplog.at_level(logging.INFO):
            llm.message_with_schema(
                message="What animal?",
                images=[sample_image],
                response_class=SpeciesResult,
            )

        assert "succeeded on attempt" in caplog.text.lower()

    @patch("wildcamtools.lib.ai.llm.moondream.AutoModelForCausalLM.from_pretrained")
    def test_wrong_image_count_raises(self, mock_model_class: MagicMock, sample_image: Path, tmp_path: Path) -> None:
        """Moondream requires exactly 1 image."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        llm = MoondreamLlm(model="test-model")

        with pytest.raises(ValueError, match="Moondream only supports one image"):
            llm.message_with_schema(
                message="What animal?",
                images=[],
                response_class=SpeciesResult,
            )

        second_image = tmp_path / "test2.jpg"
        Image.new("RGB", (100, 100), color="blue").save(second_image, format="JPEG")

        with pytest.raises(ValueError, match="Moondream only supports one image"):
            llm.message_with_schema(
                message="What animal?",
                images=[sample_image, second_image],
                response_class=SpeciesResult,
            )
