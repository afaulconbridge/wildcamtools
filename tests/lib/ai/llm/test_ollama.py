import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wildcamtools.lib.ai import RichResult
from wildcamtools.lib.ai.llm.abstract import DEFAULT_SYSTEM_MESSAGE
from wildcamtools.lib.ai.llm.ollama import OllamaLlm


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b"fake jpeg data")
    return image_file


class TestOllamaLlmSystemMessage:
    def test_default_system_message(self) -> None:
        llm = OllamaLlm(model="test-model")
        assert llm.system_message == DEFAULT_SYSTEM_MESSAGE

    def test_custom_system_message(self) -> None:
        custom_message = "Custom system message: {schema}"
        llm = OllamaLlm(model="test-model", system_message=custom_message)
        assert llm.system_message == custom_message

    def test_empty_system_message_uses_default(self) -> None:
        llm = OllamaLlm(model="test-model", system_message=None)
        assert llm.system_message == DEFAULT_SYSTEM_MESSAGE

    @patch("wildcamtools.lib.ai.llm.ollama.ollama_lib.Client")
    def test_message_with_schema_includes_system_message(
        self, mock_client_class: MagicMock, sample_image: Path
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()
        mock_client.chat.return_value.message.content = json.dumps({
            "is_animal_present": True,
            "is_animal_unknown": False,
            "defining_features": "test features",
            "species_name": "test",
            "confidence": "high",
        })
        mock_client_class.return_value = mock_client

        llm = OllamaLlm(model="test-model")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=RichResult,
        )

        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert DEFAULT_SYSTEM_MESSAGE.split("\n\n")[0] in messages[0]["content"]

    @patch("wildcamtools.lib.ai.llm.ollama.ollama_lib.Client")
    def test_message_with_schema_includes_json_schema(self, mock_client_class: MagicMock, sample_image: Path) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()
        mock_client.chat.return_value.message.content = json.dumps({
            "is_animal_present": True,
            "is_animal_unknown": False,
            "defining_features": "test features",
            "species_name": "test",
            "confidence": "high",
        })
        mock_client_class.return_value = mock_client

        llm = OllamaLlm(model="test-model")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=RichResult,
        )

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        system_content = messages[0]["content"]

        assert "species_name" in system_content
        assert "Expected JSON schema" in system_content

    @patch("wildcamtools.lib.ai.llm.ollama.ollama_lib.Client")
    def test_message_with_schema_rich_result(self, mock_client_class: MagicMock, sample_image: Path) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()
        mock_client.chat.return_value.message.content = json.dumps({
            "is_animal_present": True,
            "is_animal_unknown": False,
            "defining_features": "test features",
            "species_name": "test",
            "confidence": "high",
        })
        mock_client_class.return_value = mock_client

        llm = OllamaLlm(model="test-model")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=RichResult,
        )

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        system_content = messages[0]["content"]

        schema = RichResult.model_json_schema()
        assert any(prop in system_content for prop in schema.get("properties", {}))

    @patch("wildcamtools.lib.ai.llm.ollama.ollama_lib.Client")
    def test_custom_system_message_receives_schema(self, mock_client_class: MagicMock, sample_image: Path) -> None:
        custom_message = "Custom: {schema}"
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()
        mock_client.chat.return_value.message.content = json.dumps({
            "is_animal_present": True,
            "is_animal_unknown": False,
            "defining_features": "test features",
            "species_name": "test",
            "confidence": "high",
        })
        mock_client_class.return_value = mock_client

        llm = OllamaLlm(model="test-model", system_message=custom_message)
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=RichResult,
        )

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        system_content = messages[0]["content"]

        assert system_content.startswith("Custom:")
        schema = RichResult.model_json_schema()
        assert any(prop in system_content for prop in schema.get("properties", {}))

    @pytest.mark.parametrize("image_count", [0, 1, 3, 5])
    @patch("wildcamtools.lib.ai.llm.ollama.ollama_lib.Client")
    def test_message_with_schema_various_image_counts(
        self, mock_client_class: MagicMock, tmp_path: Path, image_count: int
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()
        mock_client.chat.return_value.message.content = json.dumps({
            "is_animal_present": True,
            "is_animal_unknown": False,
            "defining_features": "test features",
            "species_name": "test",
            "confidence": "high",
        })
        mock_client_class.return_value = mock_client

        images = []
        for i in range(image_count):
            img = tmp_path / f"test_{i}.jpg"
            img.write_bytes(b"fake jpeg data")
            images.append(img)

        llm = OllamaLlm(model="test-model")
        llm.message_with_schema(
            message="Test message",
            images=images,
            response_class=RichResult,
        )

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Expected JSON schema" in messages[0]["content"]

    @patch("wildcamtools.lib.ai.llm.ollama.ollama_lib.Client")
    def test_system_message_logged_at_debug_level(
        self, mock_client_class: MagicMock, sample_image: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()
        mock_client.chat.return_value.message.content = json.dumps({
            "is_animal_present": True,
            "is_animal_unknown": False,
            "defining_features": "test features",
            "species_name": "test",
            "confidence": "high",
        })
        mock_client_class.return_value = mock_client

        llm = OllamaLlm(model="test-model")

        with caplog.at_level(logging.DEBUG):
            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=RichResult,
            )

        assert "System message:" in caplog.text
        assert "Expected JSON schema" in caplog.text

    @patch("wildcamtools.lib.ai.llm.ollama.ollama_lib.Client")
    def test_message_with_schema_preserves_other_params(self, mock_client_class: MagicMock, sample_image: Path) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()
        mock_client.chat.return_value.message.content = json.dumps({
            "is_animal_present": True,
            "is_animal_unknown": False,
            "defining_features": "test features",
            "species_name": "test",
            "confidence": "high",
        })
        mock_client_class.return_value = mock_client

        llm = OllamaLlm(model="qwen3.5:cloud", host="http://custom-host:11434")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=RichResult,
        )

        call_args = mock_client.chat.call_args
        assert call_args.kwargs["model"] == "qwen3.5:cloud"
        assert call_args.kwargs["format"] == RichResult.model_json_schema()
