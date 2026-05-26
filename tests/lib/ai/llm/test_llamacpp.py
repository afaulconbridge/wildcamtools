import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wildcamtools.lib.ai import SpeciesResult, StringResponse
from wildcamtools.lib.ai.llm.abstract import DEFAULT_SYSTEM_MESSAGE
from wildcamtools.lib.ai.llm.llamacpp import LlamaCppLlm


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b"fake jpeg data")
    return image_file


class TestLlamaCppLlmSystemMessage:
    def test_default_system_message(self) -> None:
        llm = LlamaCppLlm(model="test-model", base_url="http://localhost:8080/v1")
        assert llm.system_message == DEFAULT_SYSTEM_MESSAGE

    def test_custom_system_message(self) -> None:
        custom_message = "Custom system message: {schema}"
        llm = LlamaCppLlm(
            model="test-model",
            base_url="http://localhost:8080/v1",
            system_message=custom_message,
        )
        assert llm.system_message == custom_message

    def test_empty_system_message_uses_default(self) -> None:
        llm = LlamaCppLlm(model="test-model", base_url="http://localhost:8080/v1", system_message=None)
        assert llm.system_message == DEFAULT_SYSTEM_MESSAGE

    @patch("wildcamtools.lib.ai.llm.llamacpp.openai_lib.OpenAI")
    def test_message_with_schema_includes_system_message(
        self, mock_client_class: MagicMock, sample_image: Path
    ) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"species_name": "test"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LlamaCppLlm(model="test-model", base_url="http://localhost:8080/v1")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert DEFAULT_SYSTEM_MESSAGE.split("\n\n")[0] in messages[0]["content"]

    @patch("wildcamtools.lib.ai.llm.llamacpp.openai_lib.OpenAI")
    def test_message_with_schema_includes_json_schema(self, mock_client_class: MagicMock, sample_image: Path) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"species_name": "test"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LlamaCppLlm(model="test-model", base_url="http://localhost:8080/v1")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_content = messages[0]["content"]

        expected_schema = SpeciesResult.model_json_schema()
        assert "species_name" in system_content
        assert json.dumps(expected_schema) in system_content or expected_schema["properties"] is not None

    @patch("wildcamtools.lib.ai.llm.llamacpp.openai_lib.OpenAI")
    def test_message_with_schema_different_response_classes(
        self, mock_client_class: MagicMock, sample_image: Path
    ) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LlamaCppLlm(model="test-model", base_url="http://localhost:8080/v1")

        for response_class in [SpeciesResult, StringResponse]:
            mock_client.chat.completions.create.reset_mock()
            mock_response.choices[0].message.content = (
                '{"species_name": "test"}' if response_class == SpeciesResult else '{"message": "test"}'
            )

            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=response_class,
            )

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            system_content = messages[0]["content"]

            schema = response_class.model_json_schema()
            assert any(prop in system_content for prop in schema.get("properties", {}))

    @patch("wildcamtools.lib.ai.llm.llamacpp.openai_lib.OpenAI")
    def test_custom_system_message_receives_schema(self, mock_client_class: MagicMock, sample_image: Path) -> None:
        custom_message = "Custom: {schema}"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"species_name": "test"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LlamaCppLlm(
            model="test-model",
            base_url="http://localhost:8080/v1",
            system_message=custom_message,
        )
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_content = messages[0]["content"]

        assert system_content.startswith("Custom:")
        schema = SpeciesResult.model_json_schema()
        assert any(prop in system_content for prop in schema.get("properties", {}))

    @pytest.mark.parametrize("image_count", [0, 1, 3, 5])
    @patch("wildcamtools.lib.ai.llm.llamacpp.openai_lib.OpenAI")
    def test_message_with_schema_various_image_counts(
        self, mock_client_class: MagicMock, tmp_path: Path, image_count: int
    ) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"species_name": "test"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        images = []
        for i in range(image_count):
            img = tmp_path / f"test_{i}.jpg"
            img.write_bytes(b"fake jpeg data")
            images.append(img)

        llm = LlamaCppLlm(model="test-model", base_url="http://localhost:8080/v1")
        llm.message_with_schema(
            message="Test message",
            images=images,
            response_class=SpeciesResult,
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Expected JSON schema" in messages[0]["content"]

    @patch("wildcamtools.lib.ai.llm.llamacpp.openai_lib.OpenAI")
    def test_system_message_logged_at_debug_level(
        self, mock_client_class: MagicMock, sample_image: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"species_name": "test"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LlamaCppLlm(model="test-model", base_url="http://localhost:8080/v1")

        with caplog.at_level(logging.DEBUG):
            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=SpeciesResult,
            )

        assert "System message:" in caplog.text
        assert "Expected JSON schema" in caplog.text

    @patch("wildcamtools.lib.ai.llm.llamacpp.openai_lib.OpenAI")
    def test_message_with_schema_preserves_other_params(self, mock_client_class: MagicMock, sample_image: Path) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"species_name": "test"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LlamaCppLlm(model="custom-model", base_url="http://custom-host:8080/v1")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "custom-model"
        assert call_args.kwargs["temperature"] == 0

    @patch("wildcamtools.lib.ai.llm.llamacpp.openai_lib.OpenAI")
    def test_system_message_with_images_in_user_content(self, mock_client_class: MagicMock, sample_image: Path) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"species_name": "test"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LlamaCppLlm(model="test-model", base_url="http://localhost:8080/v1")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert isinstance(user_content, list)
        assert any(item.get("type") == "image_url" for item in user_content)
        assert any(item.get("type") == "text" for item in user_content)
