import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from wildcamtools.lib.ai.llm import create_analyser
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.pipeline import (
    AiPipeline,
    FpsRescalingFrameSelector,
    FrameSelector,
    LlmImageBatchQuery,
    MajorityResultReconciler,
    RescaledFrameImageExtractor,
    ResultReconciler,
)
from wildcamtools.lib.ai.types import Backend, Result, SpeciesResult, StringResponse


class FrameSelectorType(StrEnum):
    FPS_RESCALING = "fps_rescaling"


class ReconcilerType(StrEnum):
    MAJORITY = "majority"


class ResponseSchemaType(StrEnum):
    SPECIES_RESULT = "SpeciesResult"
    RESULT = "Result"
    STRING_RESPONSE = "StringResponse"


RESPONSE_SCHEMA_MAP: dict[ResponseSchemaType, type[BaseModel]] = {
    ResponseSchemaType.SPECIES_RESULT: SpeciesResult,
    ResponseSchemaType.RESULT: Result,
    ResponseSchemaType.STRING_RESPONSE: StringResponse,
}


class FrameSelectorConfig(BaseModel):
    selector_type: FrameSelectorType = FrameSelectorType.FPS_RESCALING
    fps: Annotated[float, Field(strict=True, gt=0.0, description="Target frames per second")] = 1.0

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
            case _:
                raise NotImplementedError(f"Unsupported frame selector type: {self.selector_type}")


class FrameExtractorConfig(BaseModel):
    resolution: Annotated[
        tuple[Annotated[int, Field(strict=True, gt=0)], Annotated[int, Field(strict=True, gt=0)]],
        Field(description="Target resolution as (width, height)"),
    ] = (640, 360)

    def create_frame_extractor(self) -> RescaledFrameImageExtractor:
        """Create a FrameImageExtractor instance based on the configuration.

        Returns:
            RescaledFrameImageExtractor: The configured frame extractor instance.
        """
        return RescaledFrameImageExtractor(resolution=self.resolution)


class LlmConfig(BaseModel):
    backend: Backend = Backend.OLLAMA
    model: Annotated[str, Field(strict=True, min_length=1, description="Model name/identifier")]
    url: Annotated[str, Field(description="Base URL for the LLM service")] = "http://localhost:8080/v1"
    api_key: str | None = None

    @field_validator("api_key")
    @classmethod
    def resolve_env_var(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            return os.environ.get(env_var)
        return v

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    def create_llm(self) -> AbstractLlm:
        """Create an AbstractLlm instance based on the configuration.

        Returns:
            AbstractLlm: The configured LLM instance.
        """
        return create_analyser(
            backend=self.backend,
            model=self.model,
            url=self.url,
            api_key=self.api_key,
        )


class QueryConfig(BaseModel):
    prompt: Annotated[str, Field(strict=True, min_length=1, description="Prompt sent to LLM")]
    response_schema: ResponseSchemaType = ResponseSchemaType.SPECIES_RESULT

    def get_response_class(self) -> type[BaseModel]:
        """Get the Pydantic model class for the response schema.

        Returns:
            type[BaseModel]: The Pydantic model class corresponding to response_schema.
        """
        return RESPONSE_SCHEMA_MAP[self.response_schema]


class ReconcilerConfig(BaseModel):
    reconciler_type: ReconcilerType = ReconcilerType.MAJORITY

    def create_reconciler(self) -> ResultReconciler:
        """Create a ResultReconciler instance based on the configuration.

        Returns:
            ResultReconciler: The configured reconciler instance.

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
                "frame_extractor": {"resolution": [640, 360]},
                "llm": {"backend": "ollama", "model": "qwen3.5:cloud"},
                "query": {"prompt": "What species is in this image?"},
                "reconciler": {"reconciler_type": "majority"},
            }
        }
    )

    frame_selector: FrameSelectorConfig = Field(default_factory=FrameSelectorConfig)
    frame_extractor: FrameExtractorConfig = Field(default_factory=FrameExtractorConfig)
    llm: LlmConfig
    query: QueryConfig
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
        frame_extractor = self.frame_extractor.create_frame_extractor()
        llm = self.llm.create_llm()
        response_class = self.query.get_response_class()
        image_batch_query = LlmImageBatchQuery(
            llm=llm,
            prompt=self.query.prompt,
            response_class=response_class,
        )
        reconciler = self.reconciler.create_reconciler()

        return AiPipeline(
            frame_selector=frame_selector,
            frame_image_extractor=frame_extractor,
            image_batch_query=image_batch_query,
            result_reconciler=reconciler,
        )
