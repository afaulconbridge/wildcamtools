from __future__ import annotations

import logging

from wildcamtools.lib.ai import AbstractAnalyser, Backend
from wildcamtools.lib.ai.llamacpp import LlamaCppAnalyser
from wildcamtools.lib.ai.ollama import OllamaAnalyser
from wildcamtools.lib.frames import (
    AIEvaluation,
    AIEvaluationResult,
)

logger = logging.getLogger(__name__)


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
