import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wildcamtools.lib.ai import Backend
from wildcamtools.lib.ai.pipeline_config import (
    AiPipelineConfig,
    FrameExtractorConfig,
    FrameSelectorConfig,
    FrameSelectorType,
    LlmConfig,
    QueryConfig,
    ReconcilerConfig,
    ReconcilerType,
    ResponseSchemaType,
)


class TestFrameSelectorConfig:
    def test_default_values(self) -> None:
        config = FrameSelectorConfig()
        assert config.selector_type == FrameSelectorType.FPS_RESCALING
        assert config.fps == 1.0

    def test_custom_fps(self) -> None:
        config = FrameSelectorConfig(fps=0.5)
        assert config.fps == 0.5

    def test_invalid_fps_zero(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameSelectorConfig(fps=0.0)
        assert "gt=0.0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()

    def test_invalid_fps_negative(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameSelectorConfig(fps=-1.0)
        assert "gt=0.0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()

    def test_create_frame_selector(self) -> None:
        config = FrameSelectorConfig(fps=2.0)
        selector = config.create_frame_selector()
        assert selector.fps == 2.0

    @pytest.mark.parametrize("fps_value", [0.1, 0.5, 1.0, 5.0, 30.0])
    def test_various_fps_values(self, fps_value: float) -> None:
        config = FrameSelectorConfig(fps=fps_value)
        assert config.fps == fps_value
        selector = config.create_frame_selector()
        assert selector.fps == fps_value

    def test_strict_type_validation(self) -> None:
        with pytest.raises(ValidationError):
            FrameSelectorConfig(fps="1.0")

    def test_selector_type_serialization(self) -> None:
        config = FrameSelectorConfig()
        data = config.model_dump()
        assert data["selector_type"] == "fps_rescaling"

    def test_from_dict_with_enum_string(self) -> None:
        config = FrameSelectorConfig.model_validate({"selector_type": "fps_rescaling", "fps": 5.0})
        assert config.selector_type == FrameSelectorType.FPS_RESCALING
        assert config.fps == 5.0


class TestFrameExtractorConfig:
    def test_default_values(self) -> None:
        config = FrameExtractorConfig()
        assert config.resolution == (640, 360)

    def test_custom_resolution(self) -> None:
        config = FrameExtractorConfig(resolution=(1280, 720))
        assert config.resolution == (1280, 720)

    def test_invalid_resolution_zero_width(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameExtractorConfig(resolution=(0, 360))
        assert "gt=0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()

    def test_invalid_resolution_zero_height(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameExtractorConfig(resolution=(640, 0))
        assert "gt=0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()

    def test_invalid_resolution_negative(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameExtractorConfig(resolution=(-100, 200))
        assert "gt=0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()

    def test_create_frame_extractor(self) -> None:
        config = FrameExtractorConfig(resolution=(800, 600))
        extractor = config.create_frame_extractor()
        assert extractor.resolution == (800, 600)

    def test_strict_type_validation_resolution(self) -> None:
        with pytest.raises(ValidationError):
            FrameExtractorConfig(resolution=("640", "360"))


class TestLlmConfig:
    def test_default_values(self) -> None:
        config = LlmConfig(model="test-model")
        assert config.backend == Backend.OLLAMA
        assert config.model == "test-model"
        assert config.url == "http://localhost:8080/v1"
        assert config.api_key is None

    def test_custom_values(self) -> None:
        config = LlmConfig(
            backend=Backend.LLAMACPP,
            model="llama-3",
            url="http://localhost:8080/v1",
            api_key="test-key",
        )
        assert config.backend == Backend.LLAMACPP
        assert config.model == "llama-3"
        assert config.api_key == "test-key"

    def test_empty_model_name(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            LlmConfig(model="")
        assert "min_length=1" in str(exc_info.value) or "at least 1" in str(exc_info.value).lower()

    def test_invalid_url_no_protocol(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            LlmConfig(model="test", url="localhost:8080")
        assert "http" in str(exc_info.value).lower()

    def test_env_var_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_API_KEY", "secret-key-123")
        config = LlmConfig(model="test", api_key="${TEST_API_KEY}")
        assert config.api_key == "secret-key-123"

    def test_env_var_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_KEY", raising=False)
        config = LlmConfig(model="test", api_key="${MISSING_KEY}")
        assert config.api_key is None

    def test_env_var_not_resolved_plain_string(self) -> None:
        config = LlmConfig(model="test", api_key="plain-api-key")
        assert config.api_key == "plain-api-key"

    def test_env_var_partial_match(self) -> None:
        config = LlmConfig(model="test", api_key="${INCOMPLETE")
        assert config.api_key == "${INCOMPLETE"

    def test_create_llm(self) -> None:
        config = LlmConfig(model="test-model", url="http://localhost:8080/v1")
        llm = config.create_llm()
        assert llm.model == "test-model"
        assert llm.url == "http://localhost:8080/v1"

    @pytest.mark.parametrize("backend_value", ["ollama", "llamacpp"])
    def test_backend_serialization(self, backend_value: str) -> None:
        config = LlmConfig.model_validate({"model": "test", "backend": backend_value})
        assert config.backend == Backend(backend_value)


class TestQueryConfig:
    def test_default_values(self) -> None:
        config = QueryConfig(prompt="Test prompt")
        assert config.prompt == "Test prompt"
        assert config.response_schema == ResponseSchemaType.SPECIES_RESULT

    def test_custom_values(self) -> None:
        config = QueryConfig(
            prompt="Custom prompt",
            response_schema=ResponseSchemaType.STRING_RESPONSE,
        )
        assert config.prompt == "Custom prompt"
        assert config.response_schema == ResponseSchemaType.STRING_RESPONSE

    def test_empty_prompt(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            QueryConfig(prompt="")
        assert "min_length=1" in str(exc_info.value) or "at least 1" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "schema_value,expected_class",
        [
            ("SpeciesResult", "SpeciesResult"),
            ("Result", "Result"),
            ("StringResponse", "StringResponse"),
        ],
    )
    def test_response_schema_values(self, schema_value: str, expected_class: str) -> None:
        config = QueryConfig.model_validate({"prompt": "test", "response_schema": schema_value})
        assert config.response_schema.value == expected_class

    def test_invalid_response_schema(self) -> None:
        with pytest.raises(ValidationError):
            QueryConfig.model_validate({"prompt": "test", "response_schema": "InvalidSchema"})

    def test_get_response_class_species(self) -> None:
        from wildcamtools.lib.ai import SpeciesResult

        config = QueryConfig(prompt="test", response_schema=ResponseSchemaType.SPECIES_RESULT)
        assert config.get_response_class() == SpeciesResult

    def test_get_response_class_result(self) -> None:
        from wildcamtools.lib.ai import Result

        config = QueryConfig(prompt="test", response_schema=ResponseSchemaType.RESULT)
        assert config.get_response_class() == Result

    def test_get_response_class_string(self) -> None:
        from wildcamtools.lib.ai import StringResponse

        config = QueryConfig(prompt="test", response_schema=ResponseSchemaType.STRING_RESPONSE)
        assert config.get_response_class() == StringResponse

    def test_strict_type_validation(self) -> None:
        with pytest.raises(ValidationError):
            QueryConfig(prompt=123)


class TestReconcilerConfig:
    def test_default_values(self) -> None:
        config = ReconcilerConfig()
        assert config.reconciler_type == ReconcilerType.MAJORITY

    def test_custom_reconciler_type(self) -> None:
        config = ReconcilerConfig(reconciler_type=ReconcilerType.MAJORITY)
        assert config.reconciler_type == ReconcilerType.MAJORITY

    def test_create_reconciler(self) -> None:
        from wildcamtools.lib.ai.pipeline import MajorityResultReconciler

        config = ReconcilerConfig()
        reconciler = config.create_reconciler()
        assert isinstance(reconciler, MajorityResultReconciler)

    def test_reconciler_type_serialization(self) -> None:
        config = ReconcilerConfig()
        data = config.model_dump()
        assert data["reconciler_type"] == "majority"

    def test_from_dict_with_enum_string(self) -> None:
        config = ReconcilerConfig.model_validate({"reconciler_type": "majority"})
        assert config.reconciler_type == ReconcilerType.MAJORITY


class TestAiPipelineConfig:
    def test_minimal_config(self) -> None:
        config = AiPipelineConfig(llm=LlmConfig(model="test-model"), query=QueryConfig(prompt="test"))
        assert config.frame_selector.selector_type == FrameSelectorType.FPS_RESCALING
        assert config.frame_extractor.resolution == (640, 360)
        assert config.llm.model == "test-model"
        assert config.query.prompt == "test"
        assert config.reconciler.reconciler_type == ReconcilerType.MAJORITY

    def test_full_config(self) -> None:
        config = AiPipelineConfig(
            frame_selector=FrameSelectorConfig(fps=0.5),
            frame_extractor=FrameExtractorConfig(resolution=(1280, 720)),
            llm=LlmConfig(model="llama-3", backend=Backend.LLAMACPP, url="http://localhost:8080/v1"),
            query=QueryConfig(prompt="Custom prompt", response_schema=ResponseSchemaType.STRING_RESPONSE),
            reconciler=ReconcilerConfig(reconciler_type=ReconcilerType.MAJORITY),
        )
        assert config.frame_selector.fps == 0.5
        assert config.frame_extractor.resolution == (1280, 720)
        assert config.llm.backend == Backend.LLAMACPP
        assert config.query.response_schema == ResponseSchemaType.STRING_RESPONSE

    def test_from_json(self, tmp_path: Path) -> None:
        json_content = """
        {
            "llm": {
                "model": "test-model",
                "backend": "ollama",
                "url": "http://localhost:8080/v1"
            },
            "query": {
                "prompt": "Test prompt from JSON"
            }
        }
        """
        config_file = tmp_path / "config.json"
        config_file.write_text(json_content)

        config = AiPipelineConfig.from_json(config_file)
        assert config.llm.model == "test-model"
        assert config.query.prompt == "Test prompt from JSON"

    def test_to_json(self, tmp_path: Path) -> None:
        config = AiPipelineConfig(llm=LlmConfig(model="test-model"), query=QueryConfig(prompt="test"))
        output_file = tmp_path / "output_config.json"

        config.to_json(output_file)

        assert output_file.exists()
        loaded_data = json.loads(output_file.read_text())
        assert loaded_data["llm"]["model"] == "test-model"
        assert loaded_data["query"]["prompt"] == "test"

    def test_roundtrip_json(self, tmp_path: Path) -> None:
        original = AiPipelineConfig(
            frame_selector=FrameSelectorConfig(fps=2.0),
            frame_extractor=FrameExtractorConfig(resolution=(800, 600)),
            llm=LlmConfig(model="qwen3.5:cloud"),
            query=QueryConfig(prompt="Test prompt"),
            reconciler=ReconcilerConfig(),
        )

        output_file = tmp_path / "roundtrip.json"
        original.to_json(output_file)
        restored = AiPipelineConfig.from_json(output_file)

        assert restored.frame_selector.fps == original.frame_selector.fps
        assert restored.frame_extractor.resolution == original.frame_extractor.resolution
        assert restored.llm.model == original.llm.model
        assert restored.query.prompt == original.query.prompt

    def test_create_pipeline(self) -> None:
        from wildcamtools.lib.ai.pipeline import AiPipeline

        config = AiPipelineConfig(
            llm=LlmConfig(model="test-model", url="http://localhost:8080/v1"),
            query=QueryConfig(prompt="test"),
        )

        pipeline = config.create_pipeline()
        assert isinstance(pipeline, AiPipeline)
        assert pipeline.frame_selector.fps == 1.0
        assert pipeline.frame_image_extractor.resolution == (640, 360)

    def test_create_pipeline_with_custom_params(self) -> None:
        config = AiPipelineConfig(
            frame_selector=FrameSelectorConfig(fps=5.0),
            frame_extractor=FrameExtractorConfig(resolution=(1024, 768)),
            llm=LlmConfig(model="custom-model", url="http://example.com"),
            query=QueryConfig(prompt="custom prompt", response_schema=ResponseSchemaType.RESULT),
            reconciler=ReconcilerConfig(),
        )

        pipeline = config.create_pipeline()
        assert pipeline.frame_selector.fps == 5.0
        assert pipeline.frame_image_extractor.resolution == (1024, 768)

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            AiPipelineConfig.model_validate({})

    def test_missing_llm_config(self) -> None:
        with pytest.raises(ValidationError):
            AiPipelineConfig.model_validate({"query": {"prompt": "test"}})

    def test_missing_query_config(self) -> None:
        with pytest.raises(ValidationError):
            AiPipelineConfig.model_validate({"llm": {"model": "test"}})

    def test_example_schema(self) -> None:
        schema = AiPipelineConfig.model_json_schema()
        assert "example" in schema.get("jsonSchemaExtra", {}) or "example" in str(schema)


class TestIntegration:
    def test_env_var_in_full_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MY_API_KEY", "super-secret-key")

        json_content = """
        {
            "llm": {
                "model": "test-model",
                "backend": "ollama",
                "url": "http://localhost:8080/v1",
                "api_key": "${MY_API_KEY}"
            },
            "query": {
                "prompt": "Test with env var"
            }
        }
        """
        config_file = tmp_path / "config_with_env.json"
        config_file.write_text(json_content)

        config = AiPipelineConfig.from_json(config_file)
        assert config.llm.api_key == "super-secret-key"

    def test_create_and_run_pipeline_structure(self) -> None:
        config = AiPipelineConfig(
            llm=LlmConfig(model="test", url="http://localhost:8080/v1"),
            query=QueryConfig(prompt="test"),
        )

        pipeline = config.create_pipeline()

        assert hasattr(pipeline, "frame_selector")
        assert hasattr(pipeline, "frame_image_extractor")
        assert hasattr(pipeline, "image_batch_query")
        assert hasattr(pipeline, "result_reconciler")
        assert hasattr(pipeline, "run")
