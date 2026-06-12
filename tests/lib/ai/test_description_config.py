"""Tests for the description query and reconciler config options."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from wildcamtools.lib.ai import (
    DescriptionImageBatchQuery,
    LlmDescriptionReconciler,
    LlmImageBatchQuery,
)
from wildcamtools.lib.ai.pipeline_config import (
    AiPipelineConfig,
    FrameExtractorConfig,
    FrameSelectorConfig,
    ImageBatchQueryConfig,
    ImageBatchQueryType,
    LlmConfig,
    ReconcilerConfig,
    ReconcilerType,
)


class TestImageBatchQueryConfigDescription:
    def test_default_query_type_is_llm(self) -> None:
        config = ImageBatchQueryConfig(
            query_type=ImageBatchQueryType.LLM,
            prompt="test",
            llm=LlmConfig(backend="ollama", model="test"),
        )
        assert config.query_type == ImageBatchQueryType.LLM

    def test_description_query_type_does_not_require_prompt(self) -> None:
        # The 'prompt' field is no longer required when query_type is 'description'.
        config = ImageBatchQueryConfig(
            query_type=ImageBatchQueryType.DESCRIPTION,
            llm=LlmConfig(backend="ollama", model="test"),
        )
        assert config.prompt is None
        assert config.effective_description_prompt()  # falls back to default

    def test_llm_query_type_still_requires_prompt(self) -> None:
        with pytest.raises(ValidationError, match="'prompt' is required"):
            ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                llm=LlmConfig(backend="ollama", model="test"),
            )

    def test_verified_query_type_still_requires_prompt(self) -> None:
        with pytest.raises(ValidationError, match="'prompt' is required"):
            ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.VERIFIED,
                llm=LlmConfig(backend="ollama", model="test"),
            )

    def test_description_query_type_accepts_explicit_description_prompt(self) -> None:
        config = ImageBatchQueryConfig(
            query_type=ImageBatchQueryType.DESCRIPTION,
            description_prompt="Describe the scene",
            prompt="unused",
            llm=LlmConfig(backend="ollama", model="test"),
        )
        assert config.effective_description_prompt() == "Describe the scene"

    def test_effective_description_prompt_falls_back_to_default(self) -> None:
        config = ImageBatchQueryConfig(
            query_type=ImageBatchQueryType.DESCRIPTION,
            description_prompt="explicit",
            prompt="unused",
            llm=LlmConfig(backend="ollama", model="test"),
        )
        assert config.effective_description_prompt() == "explicit"

    def test_create_query_returns_description_query(self) -> None:
        config = ImageBatchQueryConfig(
            query_type=ImageBatchQueryType.DESCRIPTION,
            description_prompt="describe",
            prompt="unused",
            llm=LlmConfig(backend="ollama", model="test"),
        )
        query = config.create_image_batch_query()
        assert isinstance(query, DescriptionImageBatchQuery)

    def test_create_query_llm_for_backwards_compat(self) -> None:
        config = ImageBatchQueryConfig(
            query_type=ImageBatchQueryType.LLM,
            prompt="test",
            llm=LlmConfig(backend="ollama", model="test"),
        )
        query = config.create_image_batch_query()
        assert isinstance(query, LlmImageBatchQuery)


class TestReconcilerConfigDescription:
    def test_default_reconciler_type_is_majority(self) -> None:
        config = ReconcilerConfig()
        assert config.reconciler_type == ReconcilerType.MAJORITY

    def test_description_reconciler_requires_llm(self) -> None:
        config = ReconcilerConfig(reconciler_type=ReconcilerType.DESCRIPTION)
        with pytest.raises(ValueError, match="An LLM is required"):
            config.create_reconciler()

    def test_description_reconciler_uses_fallback_llm(self) -> None:
        config = ReconcilerConfig(reconciler_type=ReconcilerType.DESCRIPTION)
        reconciler = config.create_reconciler(fallback_llm=MagicMock())
        assert isinstance(reconciler, LlmDescriptionReconciler)

    def test_description_reconciler_uses_dedicated_llm(self) -> None:
        config = ReconcilerConfig(
            reconciler_type=ReconcilerType.DESCRIPTION,
            llm=LlmConfig(model="separate-model", url="http://localhost:8080/v1"),
        )
        reconciler = config.create_reconciler(fallback_llm=None)
        assert isinstance(reconciler, LlmDescriptionReconciler)
        # The reconciler should be using the dedicated model, not the fallback.
        assert reconciler.llm.model == "separate-model"

    def test_description_reconciler_uses_custom_combine_prompt(self) -> None:
        config = ReconcilerConfig(
            reconciler_type=ReconcilerType.DESCRIPTION,
            combine_prompt="Combine: {descriptions}",
        )
        reconciler = config.create_reconciler(fallback_llm=MagicMock())
        assert isinstance(reconciler, LlmDescriptionReconciler)
        assert reconciler.prompt == "Combine: {descriptions}"


class TestAiPipelineConfigDescription:
    def test_create_pipeline_with_description_query(self) -> None:
        config = AiPipelineConfig(
            frame_selector=FrameSelectorConfig(selector_type="fps_rescaling", fps=1.0),
            frame_extractor=FrameExtractorConfig(),
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.DESCRIPTION,
                description_prompt="describe",
                prompt="unused",
                llm=LlmConfig(model="test-model", url="http://localhost:8080/v1"),
            ),
            reconciler=ReconcilerConfig(reconciler_type=ReconcilerType.MAJORITY),
        )
        pipeline = config.create_pipeline()
        assert isinstance(pipeline.image_batch_query, DescriptionImageBatchQuery)

    def test_description_query_round_trip_json(self, tmp_path: Path) -> None:
        config = AiPipelineConfig(
            frame_selector=FrameSelectorConfig(selector_type="fps_rescaling", fps=1.0),
            frame_extractor=FrameExtractorConfig(),
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.DESCRIPTION,
                description_prompt="describe",
                prompt="unused",
                llm=LlmConfig(model="test-model", url="http://localhost:8080/v1"),
            ),
            reconciler=ReconcilerConfig(
                reconciler_type=ReconcilerType.MAJORITY,
            ),
        )
        path = tmp_path / "config.json"
        config.to_json(path)
        loaded = AiPipelineConfig.from_json(path)
        assert loaded.query.query_type == ImageBatchQueryType.DESCRIPTION
        assert loaded.query.description_prompt == "describe"

    def test_description_query_json_shape(self, tmp_path: Path) -> None:
        raw_config = {
            "query": {
                "query_type": "description",
                "description_prompt": "describe",
                "prompt": "unused",
                "llm": {"model": "test", "url": "http://localhost:8080/v1"},
            },
            "reconciler": {"reconciler_type": "majority"},
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(raw_config))
        config = AiPipelineConfig.from_json(path)
        assert config.query.query_type == ImageBatchQueryType.DESCRIPTION
        assert config.reconciler.reconciler_type == ReconcilerType.MAJORITY
