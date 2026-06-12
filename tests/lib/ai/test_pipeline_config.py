import json
from pathlib import Path

import pytest
from pydantic import AnyHttpUrl, ValidationError

from wildcamtools.lib.ai import Backend
from wildcamtools.lib.ai.llm import create_analyser
from wildcamtools.lib.ai.pipeline import (
    AICroppedFrameImageExtractor,
    AiPipeline,
    ContrastEnhancedFrameImageExtractor,
    LlmImageBatchQuery,
    MotionFrameSelector,
    RescaledFrameImageExtractor,
    RichResultMajorityReconciler,
    SSIMFrameSelector,
    VerifiedImageBatchQuery,
)
from wildcamtools.lib.ai.pipeline_config import (
    AiPipelineConfig,
    FrameExtractorConfig,
    FrameExtractorType,
    FrameSelectorConfig,
    FrameSelectorType,
    ImageBatchQueryConfig,
    ImageBatchQueryType,
    LlmConfig,
    ReconcilerConfig,
    ReconcilerType,
)
from wildcamtools.lib.ai.types import ConfidenceLevel


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
            FrameSelectorConfig(selector_type=FrameSelectorType.FPS_RESCALING, fps=0.0)
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

    def test_motion_selector_default_values(self) -> None:
        config = FrameSelectorConfig(selector_type=FrameSelectorType.MOTION)
        assert config.selector_type == FrameSelectorType.MOTION
        assert config.fps == 1.0
        assert config.motion_threshold == 0.01
        assert config.resolution is None
        assert config.history == 30

    def test_motion_selector_custom_values(self) -> None:
        config = FrameSelectorConfig(
            selector_type=FrameSelectorType.MOTION,
            fps=10.0,
            motion_threshold=0.05,
            resolution=(320, 240),
            history=50,
        )
        assert config.fps == 10.0
        assert config.motion_threshold == 0.05
        assert config.resolution == (320, 240)
        assert config.history == 50

    def test_motion_selector_invalid_threshold_negative(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameSelectorConfig(selector_type=FrameSelectorType.MOTION, motion_threshold=-0.1)
        assert "ge=0.0" in str(exc_info.value) or "greater than or equal to 0" in str(exc_info.value).lower()

    def test_motion_selector_invalid_threshold_above_one(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameSelectorConfig(selector_type=FrameSelectorType.MOTION, motion_threshold=1.5)
        assert "le=1.0" in str(exc_info.value) or "less than or equal to 1" in str(exc_info.value).lower()

    def test_motion_selector_invalid_history_negative(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameSelectorConfig(selector_type=FrameSelectorType.MOTION, history=-5)
        assert "ge=0" in str(exc_info.value) or "greater than or equal to 0" in str(exc_info.value).lower()

    def test_motion_selector_invalid_resolution_zero_width(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameSelectorConfig(selector_type=FrameSelectorType.MOTION, resolution=(0, 240))
        assert "gt=0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()

    def test_motion_selector_invalid_resolution_zero_height(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameSelectorConfig(selector_type=FrameSelectorType.MOTION, resolution=(320, 0))
        assert "gt=0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()

    def test_motion_selector_invalid_fps_negative(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameSelectorConfig(selector_type=FrameSelectorType.MOTION, fps=-1.0)
        assert "gt=0.0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()

    def test_motion_selector_create_frame_selector(self) -> None:
        config = FrameSelectorConfig(
            selector_type=FrameSelectorType.MOTION,
            fps=10.0,
            motion_threshold=0.02,
            resolution=(640, 480),
            history=40,
        )
        selector = config.create_frame_selector()
        assert isinstance(selector, MotionFrameSelector)
        assert selector.fps == 10.0
        assert selector.motion_threshold == 0.02
        assert selector.resolution == (640, 480)
        assert selector.history == 40

    def test_motion_selector_create_default(self) -> None:
        config = FrameSelectorConfig(selector_type=FrameSelectorType.MOTION)
        selector = config.create_frame_selector()
        assert isinstance(selector, MotionFrameSelector)
        assert selector.fps == 1.0
        assert selector.motion_threshold == 0.01
        assert selector.resolution is None
        assert selector.history == 30

    def test_motion_selector_serialization(self) -> None:
        config = FrameSelectorConfig(
            selector_type=FrameSelectorType.MOTION,
            fps=8.0,
            motion_threshold=0.03,
            resolution=(400, 300),
            history=25,
        )
        data = config.model_dump()
        assert data["selector_type"] == "motion"
        assert data["fps"] == 8.0
        assert data["motion_threshold"] == 0.03
        assert data["resolution"] == (400, 300)
        assert data["history"] == 25

    def test_motion_selector_from_dict(self) -> None:
        config = FrameSelectorConfig.model_validate({
            "selector_type": "motion",
            "fps": 7.0,
            "motion_threshold": 0.015,
            "resolution": [480, 360],
            "history": 35,
        })
        assert config.selector_type == FrameSelectorType.MOTION
        assert config.fps == 7.0
        assert config.motion_threshold == 0.015
        assert config.resolution == (480, 360)
        assert config.history == 35

    def test_ssim_selector_default_values(self) -> None:
        config = FrameSelectorConfig(selector_type=FrameSelectorType.SSIM)
        assert config.selector_type == FrameSelectorType.SSIM
        assert config.fps == 1.0
        assert config.similarity_minimum == 0.9
        assert config.resolution is None

    def test_ssim_selector_custom_values(self) -> None:
        config = FrameSelectorConfig(
            selector_type=FrameSelectorType.SSIM,
            fps=10.0,
            similarity_minimum=0.95,
            resolution=(320, 240),
            history=5,
        )
        assert config.fps == 10.0
        assert config.similarity_minimum == 0.95
        assert config.resolution == (320, 240)
        assert config.history == 5

    def test_ssim_selector_invalid_similarity_negative(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameSelectorConfig(selector_type=FrameSelectorType.SSIM, similarity_minimum=-0.1)
        assert "ge=0.0" in str(exc_info.value) or "greater than or equal to 0" in str(exc_info.value).lower()

    def test_ssim_selector_invalid_similarity_above_one(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameSelectorConfig(selector_type=FrameSelectorType.SSIM, similarity_minimum=1.5)
        assert "le=1.0" in str(exc_info.value) or "less than or equal to 1" in str(exc_info.value).lower()

    def test_ssim_selector_create_frame_selector(self) -> None:
        config = FrameSelectorConfig(
            selector_type=FrameSelectorType.SSIM,
            fps=10.0,
            similarity_minimum=0.85,
            resolution=(640, 480),
        )
        selector = config.create_frame_selector()
        assert isinstance(selector, SSIMFrameSelector)
        assert selector.fps == 10.0
        assert selector.similarity_minimum == 0.85
        assert selector.resolution == (640, 480)

    def test_ssim_selector_create_default(self) -> None:
        config = FrameSelectorConfig(selector_type=FrameSelectorType.SSIM)
        selector = config.create_frame_selector()
        assert isinstance(selector, SSIMFrameSelector)
        assert selector.fps == 1.0
        assert selector.similarity_minimum == 0.9
        assert selector.resolution is None

    def test_ssim_selector_serialization(self) -> None:
        config = FrameSelectorConfig(
            selector_type=FrameSelectorType.SSIM,
            fps=8.0,
            similarity_minimum=0.92,
            resolution=(400, 300),
            history=15,
        )
        data = config.model_dump()
        assert data["selector_type"] == "ssim"
        assert data["fps"] == 8.0
        assert data["similarity_minimum"] == 0.92
        assert data["resolution"] == (400, 300)
        assert data["history"] == 15

    def test_ssim_selector_from_dict(self) -> None:
        config = FrameSelectorConfig.model_validate({
            "selector_type": "ssim",
            "fps": 7.0,
            "similarity_minimum": 0.88,
            "resolution": [480, 360],
            "history": 20,
        })
        assert config.selector_type == FrameSelectorType.SSIM
        assert config.fps == 7.0
        assert config.similarity_minimum == 0.88
        assert config.resolution == (480, 360)
        assert config.history == 20


class TestFrameExtractorConfig:
    def test_default_values(self) -> None:
        config = FrameExtractorConfig()
        assert config.extractor_type == FrameExtractorType.RESCALED
        assert config.resolution == (640, 360)
        assert config.max_batch_size == 30
        assert config.crop_expansion == 0.25

    def test_custom_resolution(self) -> None:
        config = FrameExtractorConfig(resolution=(1280, 720))
        assert config.resolution == (1280, 720)

    def test_custom_max_batch_size(self) -> None:
        config = FrameExtractorConfig(max_batch_size=50)
        assert config.max_batch_size == 50

    def test_custom_resolution_and_max_batch_size(self) -> None:
        config = FrameExtractorConfig(resolution=(1280, 720), max_batch_size=25)
        assert config.resolution == (1280, 720)
        assert config.max_batch_size == 25

    def test_custom_crop_expansion(self) -> None:
        config = FrameExtractorConfig(crop_expansion=0.5)
        assert config.crop_expansion == 0.5

    def test_invalid_crop_expansion_negative(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameExtractorConfig(crop_expansion=-0.1)
        assert "ge=0.0" in str(exc_info.value) or "greater than or equal to 0" in str(exc_info.value).lower()

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

    def test_invalid_max_batch_size_zero(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameExtractorConfig(max_batch_size=0)
        assert "gt=0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()

    def test_invalid_max_batch_size_negative(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameExtractorConfig(max_batch_size=-10)
        assert "gt=0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()

    def test_create_frame_extractor_rescaled(self) -> None:
        config = FrameExtractorConfig(extractor_type=FrameExtractorType.RESCALED, resolution=(800, 600))
        extractor = config.create_frame_extractor()
        assert isinstance(extractor, RescaledFrameImageExtractor)
        assert extractor.resolution == (800, 600)
        assert extractor.max_batch_size == 30

    def test_create_frame_extractor_with_custom_max_batch_size(self) -> None:
        config = FrameExtractorConfig(resolution=(800, 600), max_batch_size=50)
        extractor = config.create_frame_extractor()
        assert extractor.resolution == (800, 600)
        assert extractor.max_batch_size == 50

    def test_create_frame_extractor_ai_cropped_requires_analyser(self) -> None:
        config = FrameExtractorConfig(extractor_type=FrameExtractorType.AI_CROPPED)
        with pytest.raises(ValueError, match="analyser_llm must be provided or configured in analyser"):
            config.create_frame_extractor()

    def test_create_frame_extractor_ai_cropped_with_analyser_llm(self) -> None:
        config = FrameExtractorConfig(extractor_type=FrameExtractorType.AI_CROPPED, resolution=(320, 240))
        analyser = create_analyser(backend=Backend.OLLAMA, model="test", url="http://test", api_key=None)
        extractor = config.create_frame_extractor(analyser_llm=analyser)

        assert isinstance(extractor, AICroppedFrameImageExtractor)
        assert extractor.resolution == (320, 240)
        assert extractor.aicropfinder.expansion == 0.25

    def test_create_frame_extractor_ai_cropped_with_configured_analyser(self) -> None:
        config = FrameExtractorConfig(
            extractor_type=FrameExtractorType.AI_CROPPED,
            resolution=(320, 240),
            crop_expansion=0.5,
            analyser=LlmConfig(model="test", url="http://test"),
        )
        extractor = config.create_frame_extractor()
        assert isinstance(extractor, AICroppedFrameImageExtractor)
        assert extractor.resolution == (320, 240)
        assert extractor.aicropfinder.expansion == 0.5

    def test_strict_type_validation_resolution(self) -> None:
        with pytest.raises(ValidationError):
            FrameExtractorConfig(resolution=("640", "360"))

    def test_strict_type_validation_max_batch_size(self) -> None:
        with pytest.raises(ValidationError):
            FrameExtractorConfig(max_batch_size="30")

    def test_extractor_type_serialization(self) -> None:
        config = FrameExtractorConfig()
        data = config.model_dump()
        assert data["extractor_type"] == "rescaled"

    def test_extractor_type_from_dict(self) -> None:
        config = FrameExtractorConfig.model_validate({"extractor_type": "ai_cropped"})
        assert config.extractor_type == FrameExtractorType.AI_CROPPED

    def test_extractor_type_contrast_enhanced(self) -> None:
        config = FrameExtractorConfig.model_validate({"extractor_type": "contrast_enhanced"})
        assert config.extractor_type == FrameExtractorType.CONTRAST_ENHANCED

    def test_create_frame_extractor_contrast_enhanced_default(self) -> None:
        config = FrameExtractorConfig(extractor_type=FrameExtractorType.CONTRAST_ENHANCED)
        extractor = config.create_frame_extractor()
        assert isinstance(extractor, ContrastEnhancedFrameImageExtractor)
        assert extractor.resolution == (640, 360)
        assert extractor.max_batch_size == 30
        assert extractor.contrast_enhancer.clip_limit == 2.0
        assert extractor.contrast_enhancer.tile_grid_size == (8, 8)

    def test_create_frame_extractor_contrast_enhanced_custom(self) -> None:
        config = FrameExtractorConfig(
            extractor_type=FrameExtractorType.CONTRAST_ENHANCED,
            resolution=(1280, 720),
            max_batch_size=20,
            clip_limit=3.0,
            tile_grid_size=(16, 16),
        )
        extractor = config.create_frame_extractor()
        assert isinstance(extractor, ContrastEnhancedFrameImageExtractor)
        assert extractor.resolution == (1280, 720)
        assert extractor.max_batch_size == 20
        assert extractor.contrast_enhancer.clip_limit == 3.0
        assert extractor.contrast_enhancer.tile_grid_size == (16, 16)

    def test_contrast_enhanced_clip_limit_validation(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameExtractorConfig(extractor_type=FrameExtractorType.CONTRAST_ENHANCED, clip_limit=0.0)
        assert "gt=0.0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()

    def test_contrast_enhanced_tile_grid_size_validation(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FrameExtractorConfig(
                extractor_type=FrameExtractorType.CONTRAST_ENHANCED,
                tile_grid_size=(0, 8),
            )
        assert "gt=0" in str(exc_info.value) or "greater than 0" in str(exc_info.value).lower()


class TestLlmConfig:
    def test_default_values(self) -> None:
        config = LlmConfig(model="test-model")
        assert config.backend == Backend.OLLAMA
        assert config.model == "test-model"
        assert config.url == AnyHttpUrl("http://localhost:8080/v1")
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
        assert config.api_key.get_secret_value() == "test-key"

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
        assert config.api_key.get_secret_value() == "secret-key-123"

    def test_env_var_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_KEY", raising=False)
        config = LlmConfig(model="test", api_key="${MISSING_KEY}")
        assert config.api_key is None

    def test_env_var_not_resolved_plain_string(self) -> None:
        config = LlmConfig(model="test", api_key="plain-api-key")
        assert config.api_key.get_secret_value() == "plain-api-key"

    def test_env_var_partial_match(self) -> None:
        config = LlmConfig(model="test", api_key="${INCOMPLETE")
        assert config.api_key.get_secret_value() == "${INCOMPLETE"

    def test_create_llm(self) -> None:
        config = LlmConfig(model="test-model", url="http://localhost:8080/v1")
        llm = config.create_llm()
        assert llm.model == "test-model"
        assert llm.url == "http://localhost:8080/v1"

    def test_url_type(self) -> None:
        config = LlmConfig(model="test-model")
        assert isinstance(config.url, AnyHttpUrl)
        assert str(config.url) == "http://localhost:8080/v1"

    @pytest.mark.parametrize("backend_value", ["ollama", "llamacpp"])
    def test_backend_serialization(self, backend_value: str) -> None:
        config = LlmConfig.model_validate({"model": "test", "backend": backend_value})
        assert config.backend == Backend(backend_value)


class TestImageBatchQueryConfig:
    def test_default_values(self) -> None:
        config = ImageBatchQueryConfig(
            query_type=ImageBatchQueryType.LLM,
            prompt="Test prompt",
            llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
        )
        assert config.query_type == ImageBatchQueryType.LLM
        assert config.prompt == "Test prompt"
        assert config.verification_prompt is None
        assert config.min_confidence == "medium"

    def test_custom_values(self) -> None:
        config = ImageBatchQueryConfig(
            query_type=ImageBatchQueryType.VERIFIED,
            prompt="Custom prompt",
            verification_prompt="Verify: {initial_species}",
            min_confidence="high",
            llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
        )
        assert config.query_type == ImageBatchQueryType.VERIFIED
        assert config.prompt == "Custom prompt"
        assert config.verification_prompt == "Verify: {initial_species}"
        assert config.min_confidence == "high"

    def test_empty_prompt(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt="",
                llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
            )
        assert "min_length=1" in str(exc_info.value) or "at least 1" in str(exc_info.value).lower()

    def test_strict_type_validation(self) -> None:
        with pytest.raises(ValidationError):
            ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt=123,  # type: ignore[arg-type]
                llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
            )

    def test_invalid_query_type(self) -> None:
        with pytest.raises(ValidationError):
            ImageBatchQueryConfig.model_validate({
                "prompt": "test",
                "query_type": "invalid",
                "llm": {"backend": "ollama", "model": "test"},
            })

    def test_create_llm_image_batch_query(self) -> None:

        config = ImageBatchQueryConfig(
            query_type=ImageBatchQueryType.LLM,
            prompt="test prompt",
            llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
        )
        query = config.create_image_batch_query()

        assert isinstance(query, LlmImageBatchQuery)
        assert query.prompt == "test prompt"

    def test_create_verified_image_batch_query(self) -> None:

        config = ImageBatchQueryConfig(
            query_type=ImageBatchQueryType.VERIFIED,
            prompt="test prompt",
            min_confidence="high",
            llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
        )
        query = config.create_image_batch_query()

        assert isinstance(query, VerifiedImageBatchQuery)
        assert query.prompt == "test prompt"
        assert query.min_confidence == "high"

    def test_create_verified_image_batch_query_with_custom_verification_prompt(self) -> None:

        config = ImageBatchQueryConfig(
            query_type=ImageBatchQueryType.VERIFIED,
            prompt="test prompt",
            verification_prompt="custom verify: {initial_species}",
            llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
        )
        query = config.create_image_batch_query()

        assert isinstance(query, VerifiedImageBatchQuery)
        assert query.verification_prompt == "custom verify: {initial_species}"


class TestReconcilerConfig:
    def test_default_values(self) -> None:
        config = ReconcilerConfig()
        assert config.reconciler_type == ReconcilerType.MAJORITY

    def test_custom_reconciler_type(self) -> None:
        config = ReconcilerConfig(reconciler_type=ReconcilerType.MAJORITY)
        assert config.reconciler_type == ReconcilerType.MAJORITY

    def test_create_reconciler(self) -> None:

        config = ReconcilerConfig()
        reconciler = config.create_reconciler()
        assert isinstance(reconciler, RichResultMajorityReconciler)

    def test_reconciler_type_serialization(self) -> None:
        config = ReconcilerConfig()
        data = config.model_dump()
        assert data["reconciler_type"] == "majority"

    def test_from_dict_with_enum_string(self) -> None:
        config = ReconcilerConfig.model_validate({"reconciler_type": "majority"})
        assert config.reconciler_type == ReconcilerType.MAJORITY


class TestAiPipelineConfig:
    def test_minimal_config(self) -> None:
        config = AiPipelineConfig(
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt="test",
                llm=LlmConfig(model="test-model"),
            ),
        )
        assert config.frame_selector.selector_type == FrameSelectorType.FPS_RESCALING
        assert config.frame_extractor.resolution == (640, 360)
        assert config.query.prompt == "test"
        assert config.reconciler.reconciler_type == ReconcilerType.MAJORITY

    def test_full_config(self) -> None:
        config = AiPipelineConfig(
            frame_selector=FrameSelectorConfig(fps=0.5),
            frame_extractor=FrameExtractorConfig(resolution=(1280, 720)),
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt="Custom prompt",
                llm=LlmConfig(model="llama-3", backend=Backend.LLAMACPP, url="http://localhost:8080/v1"),
            ),
            reconciler=ReconcilerConfig(reconciler_type=ReconcilerType.MAJORITY),
        )
        assert config.frame_selector.fps == 0.5
        assert config.frame_extractor.resolution == (1280, 720)
        assert config.query.llm.backend == Backend.LLAMACPP

    def test_from_json(self, tmp_path: Path) -> None:
        json_content = """
        {
            "query": {
                "query_type": "llm",
                "prompt": "Test prompt from JSON",
                "llm": {
                    "model": "test-model",
                    "backend": "ollama",
                    "url": "http://localhost:8080/v1"
                }
            }
        }
        """
        config_file = tmp_path / "config.json"
        config_file.write_text(json_content)

        config = AiPipelineConfig.from_json(config_file)
        assert config.query.llm.model == "test-model"
        assert config.query.prompt == "Test prompt from JSON"

    def test_to_json(self, tmp_path: Path) -> None:
        config = AiPipelineConfig(
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt="test",
                llm=LlmConfig(model="test-model"),
            ),
        )
        output_file = tmp_path / "output_config.json"

        config.to_json(output_file)

        assert output_file.exists()
        loaded_data = json.loads(output_file.read_text())
        assert loaded_data["query"]["llm"]["model"] == "test-model"
        assert loaded_data["query"]["prompt"] == "test"

    def test_roundtrip_json(self, tmp_path: Path) -> None:
        original = AiPipelineConfig(
            frame_selector=FrameSelectorConfig(fps=2.0),
            frame_extractor=FrameExtractorConfig(resolution=(800, 600)),
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt="Test prompt",
                llm=LlmConfig(model="qwen3.5:cloud"),
            ),
            reconciler=ReconcilerConfig(),
        )

        output_file = tmp_path / "roundtrip.json"
        original.to_json(output_file)
        restored = AiPipelineConfig.from_json(output_file)

        assert restored.frame_selector.fps == original.frame_selector.fps
        assert restored.frame_extractor.resolution == original.frame_extractor.resolution
        assert restored.query.llm.model == original.query.llm.model
        assert restored.query.prompt == original.query.prompt

    def test_create_pipeline(self) -> None:

        config = AiPipelineConfig(
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt="test",
                llm=LlmConfig(model="test-model", url="http://localhost:8080/v1"),
            ),
        )

        pipeline = config.create_pipeline()
        assert isinstance(pipeline, AiPipeline)
        assert pipeline.frame_selector.fps == 1.0
        assert pipeline.frame_image_extractor.resolution == (640, 360)
        assert isinstance(pipeline.image_batch_query, LlmImageBatchQuery)

    def test_create_pipeline_with_custom_params(self) -> None:
        config = AiPipelineConfig(
            frame_selector=FrameSelectorConfig(fps=5.0),
            frame_extractor=FrameExtractorConfig(resolution=(1024, 768)),
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt="custom prompt",
                llm=LlmConfig(model="custom-model", url="http://example.com"),
            ),
            reconciler=ReconcilerConfig(),
        )

        pipeline = config.create_pipeline()
        assert pipeline.frame_selector.fps == 5.0
        assert pipeline.frame_image_extractor.resolution == (1024, 768)

    def test_create_pipeline_with_ai_cropped_extractor(self) -> None:

        config = AiPipelineConfig(
            frame_extractor=FrameExtractorConfig(extractor_type=FrameExtractorType.AI_CROPPED),
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt="test",
                llm=LlmConfig(model="test-model", url="http://localhost:8080/v1"),
            ),
        )

        pipeline = config.create_pipeline()
        assert isinstance(pipeline.frame_image_extractor, AICroppedFrameImageExtractor)
        assert pipeline.frame_image_extractor.aicropfinder.expansion == 0.25

    def test_create_pipeline_with_verified_query(self) -> None:

        config = AiPipelineConfig(
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.VERIFIED,
                prompt="test",
                llm=LlmConfig(model="test-model", url="http://localhost:8080/v1"),
            ),
        )

        pipeline = config.create_pipeline()
        assert isinstance(pipeline.image_batch_query, VerifiedImageBatchQuery)
        assert pipeline.image_batch_query.min_confidence == "medium"

    def test_create_pipeline_with_verified_query_custom_params(self) -> None:

        config = AiPipelineConfig(
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.VERIFIED,
                prompt="test",
                verification_prompt="custom verify",
                min_confidence=ConfidenceLevel.HIGH,
                llm=LlmConfig(model="test-model", url="http://localhost:8080/v1"),
            ),
        )

        pipeline = config.create_pipeline()
        assert isinstance(pipeline.image_batch_query, VerifiedImageBatchQuery)
        assert pipeline.image_batch_query.verification_prompt == "custom verify"
        assert pipeline.image_batch_query.min_confidence == ConfidenceLevel.HIGH

    def test_create_pipeline_with_motion_selector(self) -> None:
        config = AiPipelineConfig(
            frame_selector=FrameSelectorConfig(
                selector_type=FrameSelectorType.MOTION,
                fps=10.0,
                motion_threshold=0.02,
                resolution=(320, 240),
                history=40,
            ),
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt="test",
                llm=LlmConfig(model="test-model", url="http://localhost:8080/v1"),
            ),
        )

        pipeline = config.create_pipeline()
        assert isinstance(pipeline.frame_selector, MotionFrameSelector)
        assert pipeline.frame_selector.fps == 10.0
        assert pipeline.frame_selector.motion_threshold == 0.02
        assert pipeline.frame_selector.resolution == (320, 240)
        assert pipeline.frame_selector.history == 40

    def test_create_pipeline_with_motion_selector_default(self) -> None:
        config = AiPipelineConfig(
            frame_selector=FrameSelectorConfig(selector_type=FrameSelectorType.MOTION),
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt="test",
                llm=LlmConfig(model="test-model", url="http://localhost:8080/v1"),
            ),
        )

        pipeline = config.create_pipeline()
        assert isinstance(pipeline.frame_selector, MotionFrameSelector)
        assert pipeline.frame_selector.fps == 1.0
        assert pipeline.frame_selector.motion_threshold == 0.01
        assert pipeline.frame_selector.resolution is None
        assert pipeline.frame_selector.history == 30


class TestIntegration:
    def test_env_var_in_full_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MY_API_KEY", "super-secret-key")

        json_content = """
        {
            "query": {
                "query_type": "llm",
                "prompt": "Test with env var",
                "llm": {
                    "model": "test-model",
                    "backend": "ollama",
                    "url": "http://localhost:8080/v1",
                    "api_key": "${MY_API_KEY}"
                }
            }
        }
        """
        config_file = tmp_path / "config_with_env.json"
        config_file.write_text(json_content)

        config = AiPipelineConfig.from_json(config_file)
        assert config.query.llm.api_key.get_secret_value() == "super-secret-key"

    def test_create_and_run_pipeline_structure(self) -> None:
        config = AiPipelineConfig(
            query=ImageBatchQueryConfig(
                query_type=ImageBatchQueryType.LLM,
                prompt="test",
                llm=LlmConfig(model="test", url="http://localhost:8080/v1"),
            ),
        )

        pipeline = config.create_pipeline()

        assert hasattr(pipeline, "frame_selector")
        assert hasattr(pipeline, "frame_image_extractor")
        assert hasattr(pipeline, "image_batch_query")
        assert hasattr(pipeline, "result_reconciler")
        assert hasattr(pipeline, "run")
