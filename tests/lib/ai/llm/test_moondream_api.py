import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from wildcamtools.lib.ai import SpeciesResult, StringResponse
from wildcamtools.lib.ai.llm.abstract import DEFAULT_SYSTEM_MESSAGE
from wildcamtools.lib.ai.llm.moondream_api import MoondreamApiLlm


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    image_file = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(image_file, format="JPEG")
    return image_file


@pytest.fixture
def mock_httpx_response() -> MagicMock:
    mock_response = MagicMock()
    mock_response.json.return_value = {"answer": '{"species_name": "Red Fox"}'}
    mock_response.raise_for_status = MagicMock()
    return mock_response


class TestMoondreamApiLlmSystemMessage:
    def test_default_system_message(self) -> None:
        llm = MoondreamApiLlm()
        assert llm.system_message == DEFAULT_SYSTEM_MESSAGE

    def test_custom_system_message(self) -> None:
        custom_message = "Custom system message: {schema}"
        llm = MoondreamApiLlm(system_message=custom_message)
        assert llm.system_message == custom_message

    def test_empty_system_message_uses_default(self) -> None:
        llm = MoondreamApiLlm(system_message=None)
        assert llm.system_message == DEFAULT_SYSTEM_MESSAGE

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_message_with_schema_includes_system_message(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post = MagicMock(return_value=mock_httpx_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm()
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args.kwargs
        payload = call_kwargs.get("json")

        assert payload is not None
        question = payload.get("question")
        assert question is not None
        assert DEFAULT_SYSTEM_MESSAGE.split("\n\n")[0] in question

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_message_with_schema_includes_json_schema(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post = MagicMock(return_value=mock_httpx_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm()
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        call_kwargs = mock_client.post.call_args.kwargs
        payload = call_kwargs.get("json")
        question = payload.get("question")

        assert "species_name" in question
        expected_schema = SpeciesResult.model_json_schema()
        assert any(prop in question for prop in expected_schema.get("properties", {}))

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_message_with_schema_different_response_classes(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post = MagicMock(return_value=mock_httpx_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm()

        for response_class in [SpeciesResult, StringResponse]:
            mock_client.post.reset_mock()
            mock_response = MagicMock()
            if response_class == SpeciesResult:
                mock_response.json.return_value = {"answer": '{"species_name": "test"}'}
            else:
                mock_response.json.return_value = {"answer": '{"message": "test"}'}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=response_class,
            )

            call_kwargs = mock_client.post.call_args.kwargs
            payload = call_kwargs.get("json")
            question = payload.get("question")

            schema = response_class.model_json_schema()
            assert any(prop in question for prop in schema.get("properties", {}))

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_custom_system_message_receives_schema(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        custom_message = "Custom: {schema}"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post = MagicMock(return_value=mock_httpx_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm(system_message=custom_message)
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        call_kwargs = mock_client.post.call_args.kwargs
        payload = call_kwargs.get("json")
        question = payload.get("question")

        assert question.startswith("Custom:")
        schema = SpeciesResult.model_json_schema()
        assert any(prop in question for prop in schema.get("properties", {}))

    @pytest.mark.parametrize("image_count", [0, 2, 3])
    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_message_with_schema_invalid_image_counts(
        self, mock_client_class: MagicMock, tmp_path: Path, image_count: int
    ) -> None:
        images = []
        for i in range(image_count):
            img_path = tmp_path / f"test_{i}.jpg"
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(img_path, format="JPEG")
            images.append(img_path)

        llm = MoondreamApiLlm()
        with pytest.raises(ValueError, match="Moondream API only supports one image at a time"):
            llm.message_with_schema(
                message="Test message",
                images=images,
                response_class=SpeciesResult,
            )

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_message_with_schema_strips_markdown_code_blocks(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        mock_response = MagicMock()
        mock_response.json.return_value = {"answer": '```json\n{"species_name": "test"}\n```'}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm()
        result = llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        assert result.species_name == "test"

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_message_with_schema_invalid_json_response(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        from pydantic import ValidationError

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        mock_response = MagicMock()
        mock_response.json.return_value = {"answer": "not valid json"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm()
        with pytest.raises(ValidationError):
            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=SpeciesResult,
            )

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_message_with_schema_empty_response(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        mock_response = MagicMock()
        mock_response.json.return_value = {"answer": '{"message": ""}'}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm()
        result = llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=StringResponse,
        )

        assert result.message == ""

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_message_with_schema_missing_answer_key(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        mock_response = MagicMock()
        mock_response.json.return_value = {"wrong_key": "value"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm()
        with pytest.raises(ValueError, match="Empty or invalid response"):
            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=SpeciesResult,
            )

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_message_with_schema_none_response(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        mock_response = MagicMock()
        mock_response.json.return_value = None
        mock_response.raise_for_status = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm()
        with pytest.raises(ValueError, match="Empty or invalid response"):
            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=SpeciesResult,
            )


class TestMoondreamApiLlmAuthentication:
    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_api_key_included_in_headers(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post = MagicMock(return_value=mock_httpx_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm(api_key="test_api_key_123")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        call_kwargs = mock_client.post.call_args.kwargs
        headers = call_kwargs.get("headers")
        assert headers is not None
        assert headers.get("X-Moondream-Auth") == "test_api_key_123"

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_no_api_key_when_not_provided(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post = MagicMock(return_value=mock_httpx_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm(api_key=None)
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        call_kwargs = mock_client.post.call_args.kwargs
        headers = call_kwargs.get("headers")
        assert headers is not None
        assert "X-Moondream-Auth" not in headers


class TestMoondreamApiLlmImageEncoding:
    def test_encode_image_to_base64(self, sample_image: Path) -> None:
        with sample_image.open("rb") as f:
            expected_base64 = base64.b64encode(f.read()).decode("utf-8")

        result = MoondreamApiLlm._encode_image_to_base64(sample_image)
        assert result == expected_base64

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_image_encoded_as_data_url(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post = MagicMock(return_value=mock_httpx_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm()
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        call_kwargs = mock_client.post.call_args.kwargs
        payload = call_kwargs.get("json")
        assert payload is not None
        image_url = payload.get("image_url")
        assert image_url is not None
        assert image_url.startswith("data:image/jpeg;base64,")


class TestMoondreamApiLlmHttpCall:
    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_correct_endpoint_called(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post = MagicMock(return_value=mock_httpx_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm(url="https://api.moondream.ai/v1/")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args.args
        assert call_args[0] == "https://api.moondream.ai/v1/query"

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_custom_url_used(
        self, mock_client_class: MagicMock, sample_image: Path, mock_httpx_response: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post = MagicMock(return_value=mock_httpx_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm(url="https://custom.api.example.com/v1/")
        llm.message_with_schema(
            message="Test message",
            images=[sample_image],
            response_class=SpeciesResult,
        )

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args.args
        assert call_args[0] == "https://custom.api.example.com/v1/query"

    @patch("wildcamtools.lib.ai.llm.moondream_api.httpx.Client")
    def test_http_error_raises_exception(self, mock_client_class: MagicMock, sample_image: Path) -> None:
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=MagicMock(status_code=400)
        )
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        llm = MoondreamApiLlm()
        with pytest.raises(httpx.HTTPStatusError):
            llm.message_with_schema(
                message="Test message",
                images=[sample_image],
                response_class=SpeciesResult,
            )
