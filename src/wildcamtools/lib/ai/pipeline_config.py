import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr, field_validator

from wildcamtools.lib.ai.crop import AICropFinder
from wildcamtools.lib.ai.llm import create_analyser
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.pipeline import (
    AICroppedFrameImageExtractor,
    AiPipeline,
    ContrastEnhancedFrameImageExtractor,
    FpsRescalingFrameSelector,
    FrameImageExtractor,
    FrameSelector,
    ImageBatchQuery,
    LlmImageBatchQuery,
    MajorityResultReconciler,
    MotionFrameSelector,
    RescaledFrameImageExtractor,
    ResultReconciler,
    SSIMFrameSelector,
    VerifiedImageBatchQuery,
)
from wildcamtools.lib.ai.types import Backend, ConfidenceLevel


class FrameSelectorType(StrEnum):
    FPS_RESCALING = "fps_rescaling"
    MOTION = "motion"
    SSIM = "ssim"


class FrameExtractorType(StrEnum):
    RESCALED = "rescaled"
    AI_CROPPED = "ai_cropped"
    CONTRAST_ENHANCED = "contrast_enhanced"


class ReconcilerType(StrEnum):
    MAJORITY = "majority"


class ImageBatchQueryType(StrEnum):
    LLM = "llm"
    VERIFIED = "verified"


class FrameSelectorConfig(BaseModel):
    selector_type: FrameSelectorType = FrameSelectorType.FPS_RESCALING
    fps: Annotated[float, Field(strict=True, gt=0.0, description="Target frames per second")] = 1.0
    motion_threshold: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, description="Motion detection threshold"),
    ] = 0.01
    similarity_minimum: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, description="Minimum SSIM similarity threshold"),
    ] = 0.9
    resolution: Annotated[
        tuple[Annotated[int, Field(strict=True, gt=0)], Annotated[int, Field(strict=True, gt=0)]] | None,
        Field(description="Motion detection resolution as (width, height)"),
    ] = None
    history: Annotated[int, Field(strict=True, ge=0, description="Motion detector warm-up frames")] = 30

    def create_frame_selector(self) -> FrameSelector:
        """Create a FrameSelector instance based on the configuration.

        Returns:
            FrameSelector: The configured frame selector instance.

        Raises:
            NotImplementedError: If the selector_type is not supported.
        """
        match self.selector_type:
            case FrameSelectorType.FPS_RESCALING:
                return FpsRescalingFrameSelector(fps=self.fps)
            case FrameSelectorType.MOTION:
                return MotionFrameSelector(
                    fps=self.fps,
                    motion_threshold=self.motion_threshold,
                    resolution=self.resolution,
                    history=self.history,
                )
            case FrameSelectorType.SSIM:
                return SSIMFrameSelector(
                    fps=self.fps,
                    similarity_minimum=self.similarity_minimum,
                    resolution=self.resolution,
                )
            case _:
                raise NotImplementedError(f"Unsupported frame selector type: {self.selector_type}")


class ContrastEnhancedFrameExtractorConfig(BaseModel):
    resolution: Annotated[
        tuple[Annotated[int, Field(strict=True, gt=0)], Annotated[int, Field(strict=True, gt=0)]],
        Field(description="Target resolution as (width, height)"),
    ] = (640, 360)
    max_batch_size: Annotated[
        int,
        Field(strict=True, gt=0, description="Maximum number of images per batch"),
    ] = 30
    clip_limit: Annotated[
        float,
        Field(strict=True, gt=0.0, description="CLAHE clip limit parameter"),
    ] = 2.0
    tile_grid_size: Annotated[
        tuple[Annotated[int, Field(strict=True, gt=0)], Annotated[int, Field(strict=True, gt=0)]],
        Field(description="CLAHE tile grid size as (width, height)"),
    ] = (8, 8)


class FrameExtractorConfig(BaseModel):
    extractor_type: FrameExtractorType = FrameExtractorType.RESCALED
    resolution: Annotated[
        tuple[Annotated[int, Field(strict=True, gt=0)], Annotated[int, Field(strict=True, gt=0)]],
        Field(description="Target resolution as (width, height)"),
    ] = (640, 360)
    max_batch_size: Annotated[
        int,
        Field(strict=True, gt=0, description="Maximum number of images per batch"),
    ] = 30
    crop_max_resolution: Annotated[
        tuple[Annotated[int, Field(strict=True, gt=0)], Annotated[int, Field(strict=True, gt=0)]],
        Field(description="Maximum crop resolution as (width, height)"),
    ] = (640, 360)
    crop_expansion: Annotated[
        float,
        Field(
            strict=True,
            ge=0.0,
            description="Expansion factor for AI crop bounding box (0.0 = no expansion)",
        ),
    ] = 0.25
    clip_limit: Annotated[
        float,
        Field(strict=True, gt=0.0, description="CLAHE clip limit parameter (for contrast_enhanced extractor)"),
    ] = 2.0
    tile_grid_size: Annotated[
        tuple[Annotated[int, Field(strict=True, gt=0)], Annotated[int, Field(strict=True, gt=0)]],
        Field(description="CLAHE tile grid size as (width, height) (for contrast_enhanced extractor)"),
    ] = (8, 8)
    analyser: "LlmConfig | None" = None

    def create_frame_extractor(self, analyser_llm: AbstractLlm | None = None) -> FrameImageExtractor:
        """Create a FrameImageExtractor instance based on the configuration.

        Args:
            analyser_llm: The LLM instance to use for AI crop detection (required for AI_CROPPED extractor).

        Returns:
            FrameImageExtractor: The configured frame extractor instance.

        Raises:
            ValueError: If analyser_llm is not provided for AI_CROPPED extractor type.
        """
        match self.extractor_type:
            case FrameExtractorType.RESCALED:
                return RescaledFrameImageExtractor(resolution=self.resolution, max_batch_size=self.max_batch_size)
            case FrameExtractorType.AI_CROPPED:
                if analyser_llm is None:
                    if self.analyser is None:
                        raise ValueError(
                            "analyser_llm must be provided or configured in analyser for AI_CROPPED extractor"
                        )
                    analyser_llm = self.analyser.create_llm()
                aicropfinder = AICropFinder(analyser=analyser_llm, expansion=self.crop_expansion)
                return AICroppedFrameImageExtractor(
                    aicropfinder=aicropfinder,
                    resolution=self.resolution,
                    crop_max_resolution=self.crop_max_resolution,
                    max_batch_size=self.max_batch_size,
                )
            case FrameExtractorType.CONTRAST_ENHANCED:
                return ContrastEnhancedFrameImageExtractor(
                    resolution=self.resolution,
                    max_batch_size=self.max_batch_size,
                    clip_limit=self.clip_limit,
                    tile_grid_size=self.tile_grid_size,
                )
            case _:
                raise NotImplementedError(f"Unsupported frame extractor type: {self.extractor_type}")


class LlmConfig(BaseModel):
    backend: Backend = Backend.OLLAMA
    model: Annotated[str, Field(strict=True, min_length=1, description="Model name/identifier")]
    url: Annotated[AnyHttpUrl, Field(description="Base URL for the LLM service")] = AnyHttpUrl(
        "http://localhost:8080/v1"
    )
    api_key: SecretStr | None = None

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_env_var(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if isinstance(v, SecretStr):
            v = v.get_secret_value()
        if v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            return os.environ.get(env_var)
        return v

    def create_llm(self) -> AbstractLlm:
        """Create an AbstractLlm instance based on the configuration.

        Returns:
            AbstractLlm: The configured LLM instance.
        """
        return create_analyser(
            backend=self.backend,
            model=self.model,
            url=str(self.url),
            api_key=self.api_key.get_secret_value() if self.api_key else None,
        )


class ImageBatchQueryConfig(BaseModel):
    query_type: ImageBatchQueryType = ImageBatchQueryType.LLM
    prompt: Annotated[str, Field(strict=True, min_length=1, description="Prompt sent to LLM")]
    verification_prompt: Annotated[
        str | None,
        Field(
            description="Custom prompt for verification. Uses {initial_species} placeholder.",
            min_length=1,
        ),
    ] = None
    min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM

    def create_image_batch_query(self, llm: AbstractLlm) -> ImageBatchQuery:
        """Create an ImageBatchQuery instance based on the configuration.

        Args:
            llm: The LLM instance to use for queries.

        Returns:
            ImageBatchQuery: The configured image batch query instance (LlmImageBatchQuery or VerifiedImageBatchQuery).
            Both return RichResult.
        """
        match self.query_type:
            case ImageBatchQueryType.LLM:
                return LlmImageBatchQuery(llm=llm, prompt=self.prompt)
            case ImageBatchQueryType.VERIFIED:
                return VerifiedImageBatchQuery(
                    llm=llm,
                    prompt=self.prompt,
                    verification_prompt=self.verification_prompt,
                    min_confidence=self.min_confidence,
                )
            case _:
                raise NotImplementedError(f"Unsupported query type: {self.query_type}")


class ReconcilerConfig(BaseModel):
    reconciler_type: ReconcilerType = ReconcilerType.MAJORITY

    def create_reconciler(self) -> ResultReconciler:
        """Create a ResultReconciler instance based on the configuration.

        Returns:
            ResultReconciler: The configured reconciler instance (returns RichResult).

        Raises:
            NotImplementedError: If the reconciler_type is not supported.
        """
        match self.reconciler_type:
            case ReconcilerType.MAJORITY:
                return MajorityResultReconciler()
            case _:
                raise NotImplementedError(f"Unsupported reconciler type: {self.reconciler_type}")


class AiPipelineConfig(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "frame_selector": {"selector_type": "fps_rescaling", "fps": 1.0},
                "frame_extractor": {"extractor_type": "rescaled", "resolution": [640, 360]},
                "llm": {"backend": "ollama", "model": "qwen3.5:cloud"},
                "query": {"query_type": "llm", "prompt": "What species is in this image?"},
                "reconciler": {"reconciler_type": "majority"},
            }
        }
    )

    frame_selector: FrameSelectorConfig = Field(default_factory=FrameSelectorConfig)
    frame_extractor: FrameExtractorConfig = Field(default_factory=FrameExtractorConfig)
    llm: LlmConfig
    query: ImageBatchQueryConfig
    reconciler: ReconcilerConfig = Field(default_factory=ReconcilerConfig)

    @classmethod
    def from_json(cls, path: Path) -> Self:
        content = path.read_text()
        return cls.model_validate_json(content)

    def to_json(self, path: Path, indent: int = 2) -> None:
        json_str = self.model_dump_json(indent=indent)
        path.write_text(json_str)

    def create_pipeline(self) -> AiPipeline:
        frame_selector = self.frame_selector.create_frame_selector()
        llm = self.llm.create_llm()
        frame_extractor = self.frame_extractor.create_frame_extractor(analyser_llm=llm)
        image_batch_query = self.query.create_image_batch_query(llm)
        reconciler = self.reconciler.create_reconciler()

        return AiPipeline(
            frame_selector=frame_selector,
            frame_image_extractor=frame_extractor,
            image_batch_query=image_batch_query,
            result_reconciler=reconciler,
        )
