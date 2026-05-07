from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from wildcamtools.lib import FrameHandler
from wildcamtools.lib.ai import AbstractAnalyser, Backend
from wildcamtools.lib.ai.llamacpp import LlamaCppAnalyser
from wildcamtools.lib.ai.ollama import OllamaAnalyser
from wildcamtools.lib.frames import CropPanHandler, FilterSSIM, FrameImageWriter, Rescaler
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.vidio import VideoReader

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AIFrameCreation:
    filename: Path
    video_directory: Path
    tmpdir: Path
    history: int = 30
    threshold: float = 16.0
    kernel_size: float = 0.02
    x: int | None = None
    y: int | None = None
    fps: float | None = None
    crop_expansion: float = 0.75
    crop_inertia: float = 10.0
    similarity_minimum: float | None = None
    do_croppan: bool = False


@dataclass(kw_only=True)
class AIFrameCreationResult(AIFrameCreation):
    frame_count: int

    @classmethod
    def from_creation(cls, creation: AIFrameCreation, *, frame_count: int) -> AIFrameCreationResult:
        return cls(
            filename=creation.filename,
            video_directory=creation.video_directory,
            tmpdir=creation.tmpdir,
            history=creation.history,
            threshold=creation.threshold,
            kernel_size=creation.kernel_size,
            x=creation.x,
            y=creation.y,
            fps=creation.fps,
            crop_expansion=creation.crop_expansion,
            crop_inertia=creation.crop_inertia,
            similarity_minimum=creation.similarity_minimum,
            do_croppan=creation.do_croppan,
            frame_count=frame_count,
        )


@dataclass(kw_only=True)
class AIEvaluation:
    frame_directory: Path
    label: str
    backend: Backend
    url: str
    model: str
    api_key: str = "API_KEY"
    prompt: str | None = None


@dataclass(kw_only=True)
class AIEvaluationResult(AIEvaluation):
    correct: bool
    raw_result: str

    @classmethod
    def from_evaluation(cls, evaluation: AIEvaluation, *, correct: bool, raw_result: str) -> AIEvaluationResult:
        return cls(
            frame_directory=evaluation.frame_directory,
            label=evaluation.label,
            backend=evaluation.backend,
            url=evaluation.url,
            model=evaluation.model,
            api_key=evaluation.api_key,
            prompt=evaluation.prompt,
            correct=correct,
            raw_result=raw_result,
        )


def create_frames(
    frame_creation: AIFrameCreation,
) -> AIFrameCreationResult:
    stats = get_video_stats(frame_creation.video_directory / frame_creation.filename)
    handlers: list[FrameHandler] = []

    if frame_creation.do_croppan:
        handlers.append(
            CropPanHandler(
                motion_handler=MogMotion(
                    history=frame_creation.history,
                    threshold=int(frame_creation.threshold),
                    kernel_size=frame_creation.kernel_size,
                ),
                expansion=frame_creation.crop_expansion,
                inertia=frame_creation.crop_inertia,
            )
        )
    # TODO rescale after similarity vs rescale before similarity
    if frame_creation.x or frame_creation.y or frame_creation.fps:
        handlers.append(
            Rescaler(
                stats=stats,
                x=frame_creation.x,
                y=frame_creation.y,
                fps=frame_creation.fps,
            )
        )

    if frame_creation.similarity_minimum is not None:
        handlers.append(FilterSSIM(similarity_minimum=frame_creation.similarity_minimum))

    handlers.append(FrameImageWriter(frame_creation.tmpdir))

    frame_count = 0
    with VideoReader(frame_creation.video_directory / frame_creation.filename) as video_input:
        for _frame in video_input:
            frame_count += 1
            for handler in handlers:
                _frame = handler.handle(_frame)

    return AIFrameCreationResult.from_creation(frame_creation, frame_count=frame_count)


def evaluate_frames(
    evaluation: AIEvaluation,
) -> AIEvaluationResult:
    analyser: AbstractAnalyser
    match evaluation.backend:
        case Backend.LLAMACPP:
            analyser = LlamaCppAnalyser(model=evaluation.model, base_url=evaluation.url, message=evaluation.prompt)
        case Backend.OLLAMA:
            analyser = OllamaAnalyser(
                model=evaluation.model, host=evaluation.url, api_key=evaluation.api_key, message=evaluation.prompt
            )
        case _:
            raise ValueError(f"Unsupported backend: {evaluation.backend}")

    images = list(evaluation.frame_directory.iterdir())
    raw_result = analyser.analyze_video(images) if images else "no"
    logger.info("Label: %s Result: %s", evaluation.label, raw_result)
    correct = evaluation.label.lower() == raw_result.lower()

    return AIEvaluationResult.from_evaluation(evaluation, correct=correct, raw_result=raw_result)
