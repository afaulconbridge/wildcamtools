import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from wildcamtools.lib.ai.crop import AICropFinder
from wildcamtools.lib.ai.llm import create_analyser
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.pipeline import (
    AICroppedFrameImageExtractor,
    AiPipeline,
    ConcatenatingDescriptionReconciler,
    ContrastEnhancedFrameImageExtractor,
    DescriptionImageBatchQuery,
    FpsRescalingFrameSelector,
    FrameImageExtractor,
    FrameSelector,
    ImageBatchQuery,
    LlmDescriptionReconciler,
    LlmImageBatchQuery,
    MotionFrameSelector,
    RescaledFrameImageExtractor,
    ResultReconciler,
    RichResultMajorityReconciler,
    SSIMFrameSelector,
    VerifiedImageBatchQuery,
)
from wildcamtools.lib.ai.types import (
    DEFAULT_BATCH_DESCRIPTION_PROMPT,
    DEFAULT_COMBINE_DESCRIPTION_PROMPT,
    NO_ACTIVITY_DESCRIPTION,
    Backend,
    BatchDescription,
    ConfidenceLevel,
    RichResult,
)


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
    DESCRIPTION = "description"


class ImageBatchQueryType(StrEnum):
    LLM = "llm"
    VERIFIED = "verified"
    DESCRIPTION = "description"


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
    llm: LlmConfig
    prompt: Annotated[
        str | None,
        Field(
            strict=True,
            min_length=1,
            description=(
                "Prompt sent to the LLM. Required for 'llm' and 'verified' query types; "
                "ignored for 'description' (use 'description_prompt' instead)."
            ),
        ),
    ] = None
    description_prompt: Annotated[
        str | None,
        Field(
            min_length=1,
            description=(
                "Custom prompt used for the description query. If unset, the default description prompt is used."
            ),
        ),
    ] = None
    verification_prompt: Annotated[
        str | None,
        Field(
            description="Custom prompt for verification. Uses {initial_species} placeholder.",
            min_length=1,
        ),
    ] = None
    min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM

    @model_validator(mode="after")
    def _check_prompt_requirements(self) -> Self:
        if self.query_type in (ImageBatchQueryType.LLM, ImageBatchQueryType.VERIFIED) and not self.prompt:
            raise ValueError(
                f"'prompt' is required when query_type is '{self.query_type.value}'. "
                "Provide a non-empty 'prompt' in the query config."
            )
        return self

    def effective_description_prompt(self) -> str:
        """Return the description prompt to use, falling back to the default."""
        return self.description_prompt or DEFAULT_BATCH_DESCRIPTION_PROMPT

    def create_image_batch_query(self) -> ImageBatchQuery[Any]:
        """Create an ImageBatchQuery instance based on the configuration.

        Returns:
            The configured image batch query instance.
        """
        llm = self.llm.create_llm()
        match self.query_type:
            case ImageBatchQueryType.LLM:
                if not self.prompt:
                    raise ValueError("'prompt' is required for the 'llm' query type")
                return LlmImageBatchQuery(llm=llm, prompt=self.prompt)
            case ImageBatchQueryType.VERIFIED:
                if not self.prompt:
                    raise ValueError("'prompt' is required for the 'verified' query type")
                return VerifiedImageBatchQuery(
                    llm=llm,
                    prompt=self.prompt,
                    verification_prompt=self.verification_prompt,
                    min_confidence=self.min_confidence,
                )
            case ImageBatchQueryType.DESCRIPTION:
                return DescriptionImageBatchQuery(llm=llm, prompt=self.effective_description_prompt())
            case _:
                raise NotImplementedError(f"Unsupported query type: {self.query_type}")


class ReconcilerConfig(BaseModel):
    reconciler_type: ReconcilerType = ReconcilerType.MAJORITY
    combine_prompt: Annotated[
        str | None,
        Field(
            min_length=1,
            description=(
                "Custom prompt for combining batch descriptions into a final description. "
                "Uses {descriptions} placeholder. Used when reconciler_type is 'description'."
            ),
        ),
    ] = None
    llm: LlmConfig | None = Field(
        default=None,
        description=(
            "Optional separate LLM configuration for the description reconciler. "
            "If unset, the main pipeline LLM is used."
        ),
    )

    def create_reconciler(self, fallback_llm: AbstractLlm | None = None) -> ResultReconciler[Any]:
        """Create a ResultReconciler instance based on the configuration.

        Args:
            fallback_llm: LLM to use if the reconciler needs an LLM but no dedicated one is configured.

        Returns:
            The configured reconciler instance.

        Raises:
            NotImplementedError: If the reconciler_type is not supported.
        """
        match self.reconciler_type:
            case ReconcilerType.MAJORITY:
                return RichResultMajorityReconciler()
            case ReconcilerType.DESCRIPTION:
                if self.llm is not None:
                    reconciler_llm: AbstractLlm = self.llm.create_llm()
                elif fallback_llm is not None:
                    reconciler_llm = fallback_llm
                else:
                    raise ValueError(
                        "An LLM is required for the description reconciler. "
                        "Provide one via 'llm' in the reconciler config or as a fallback."
                    )
                return LlmDescriptionReconciler(
                    llm=reconciler_llm,
                    prompt=self.combine_prompt,
                    fallback=ConcatenatingDescriptionReconciler(),
                )
            case _:
                raise NotImplementedError(f"Unsupported reconciler type: {self.reconciler_type}")


class DescriptionConfig(BaseModel):
    llm: LlmConfig
    description_prompt: Annotated[
        str | None,
        Field(
            min_length=1,
            description="Custom prompt for per-batch descriptions. Uses default if not provided.",
        ),
    ] = None
    combine_prompt: Annotated[
        str | None,
        Field(
            min_length=1,
            description=(
                "Custom prompt for combining batch descriptions. Uses {descriptions} placeholder. "
                "Uses default if not provided."
            ),
        ),
    ] = None


class AiPipelineConfig(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "frame_selector": {"selector_type": "fps_rescaling", "fps": 1.0},
                "frame_extractor": {"extractor_type": "rescaled", "resolution": [640, 360]},
                "query": {
                    "query_type": "llm",
                    "prompt": "What species is in this image?",
                    "llm": {"backend": "ollama", "model": "qwen3.5:cloud"},
                },
                "reconciler": {"reconciler_type": "majority"},
                "description": {
                    "llm": {"backend": "ollama", "model": "gemma4:31b-cloud"},
                    "description_prompt": "Describe what is happening in these frames.",
                },
            }
        }
    )

    frame_selector: FrameSelectorConfig = Field(default_factory=FrameSelectorConfig)
    frame_extractor: FrameExtractorConfig = Field(default_factory=FrameExtractorConfig)
    query: ImageBatchQueryConfig
    reconciler: ReconcilerConfig = Field(default_factory=ReconcilerConfig)
    description: DescriptionConfig | None = None

    @classmethod
    def from_json(cls, path: Path) -> Self:
        content = path.read_text()
        return cls.model_validate_json(content)

    def to_json(self, path: Path, indent: int = 2) -> None:
        json_str = self.model_dump_json(indent=indent)
        path.write_text(json_str)

    def create_pipeline(self) -> AiPipeline[Any]:
        frame_selector = self.frame_selector.create_frame_selector()
        query_llm = self.query.llm.create_llm()

        # For AI crop extractor, use query LLM if no dedicated analyser configured
        analyser_llm = None
        if self.frame_extractor.analyser is not None:
            analyser_llm = self.frame_extractor.analyser.create_llm()
        else:
            analyser_llm = query_llm

        frame_extractor = self.frame_extractor.create_frame_extractor(analyser_llm=analyser_llm)
        image_batch_query = self.query.create_image_batch_query()

        # Use query LLM as fallback for reconciler
        reconciler = self.reconciler.create_reconciler(fallback_llm=query_llm)

        if self.query.query_type == ImageBatchQueryType.DESCRIPTION:
            empty_result: BatchDescription = BatchDescription(description=NO_ACTIVITY_DESCRIPTION)
            return AiPipeline[BatchDescription](
                frame_selector=frame_selector,
                frame_image_extractor=frame_extractor,
                image_batch_query=image_batch_query,
                result_reconciler=reconciler,
                empty_result=empty_result,
            )

        # Build description pipeline components if description config is provided
        description_query: DescriptionImageBatchQuery | None = None
        description_reconciler: ResultReconciler[BatchDescription] | None = None

        if self.description is not None:
            description_llm = self.description.llm.create_llm()
            description_prompt = self.description.description_prompt or DEFAULT_BATCH_DESCRIPTION_PROMPT
            description_query = DescriptionImageBatchQuery(llm=description_llm, prompt=description_prompt)

            combine_prompt = self.description.combine_prompt or DEFAULT_COMBINE_DESCRIPTION_PROMPT
            description_reconciler = LlmDescriptionReconciler(
                llm=description_llm,
                prompt=combine_prompt,
                fallback=ConcatenatingDescriptionReconciler(),
            )

        # For classification pipelines, use RichResult as the empty result
        empty_rich_result = RichResult(
            is_animal_present=False,
            is_animal_unknown=False,
            defining_features="",
            species_name="no animal",
            confidence=ConfidenceLevel.HIGH,
        )

        return AiPipeline(
            frame_selector=frame_selector,
            frame_image_extractor=frame_extractor,
            image_batch_query=image_batch_query,
            result_reconciler=reconciler,
            empty_result=empty_rich_result,
            description_query=description_query,
            description_reconciler=description_reconciler,
        )
