from __future__ import annotations

import json
import logging
import multiprocessing
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.pool import AsyncResult
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from wildcamtools.lib.ai.llm import create_analyser
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.types import (
    AIEvaluation,
    AIEvaluationResult,
    Backend,
    SpeciesResult,
)

logger = logging.getLogger(__name__)


def evaluate_frames(
    evaluation: AIEvaluation,
) -> AIEvaluationResult:
    analyser = create_analyser(
        backend=evaluation.backend,
        model=evaluation.model,
        url=evaluation.url,
        api_key=evaluation.api_key,
    )

    images = list(evaluation.frame_directory.iterdir())
    if images:
        result = analyser.message_with_schema(
            message=evaluation.prompt,
            images=images,
            response_class=SpeciesResult,
        )
        raw_result = result.species_name
    else:
        raw_result = "no"
    logger.info("Label: %s Result: %s", evaluation.label, raw_result)
    correct = evaluation.label.lower() == raw_result.lower()

    return AIEvaluationResult.from_evaluation(evaluation, correct=correct, raw_result=raw_result)


@dataclass(kw_only=True)
class PipelineEvaluationResult:
    filename: str
    correct: bool
    raw_result: str
    crop_count: int
    label: str
    model: str
    backend: str
    crop_expansion: float


@dataclass
class PipelineEvaluationSummary:
    results: list[PipelineEvaluationResult]
    counters: dict[tuple[Any, ...], list[int]]

    def print_summary(self) -> None:
        for counter_key in self.counters:
            successes = self.counters[counter_key][0]
            attempts = self.counters[counter_key][1]
            ratio = successes / attempts
            logger.info("%s: %d / %d = %.4f", counter_key, successes, attempts, ratio)


def _process_pipeline_evaluation_result(
    frame_result: tuple[str, Sequence[Path], Sequence[Path]],
    labelled_data: dict[str, Any],
    crops_output_dir: Path,
    analyser: AbstractLlm,
    crop_expansion: float,
    result_jsonl_path: Path,
) -> PipelineEvaluationResult:
    from wildcamtools.lib.pipeline import process_pipeline_result

    filename, _, _ = frame_result
    label = labelled_data[filename]

    correct, raw_result, crop_count = process_pipeline_result(
        frame_result=frame_result,
        labelled_data=labelled_data,
        crops_output_dir=crops_output_dir,
        analyser=analyser,
        crop_expansion=crop_expansion,
        filename=filename,
    )

    with open(result_jsonl_path, "a") as result_file:
        output_parameter_dict = {
            "filename": filename,
            "model": analyser.model,
            "backend": analyser.backend.value,
            "crop_expansion": crop_expansion,
        }
        result_dict = {
            "result": correct,
            "raw_result": raw_result,
            "label": label,
            "crop_count": crop_count,
        }
        output_dict = (output_parameter_dict, result_dict)
        result_file.write(json.dumps(output_dict))
        result_file.write("\n")

    logger.info("Video %s: correct=%s", filename, correct)

    return PipelineEvaluationResult(
        filename=filename,
        correct=correct,
        raw_result=raw_result,
        crop_count=crop_count,
        label=label,
        model=analyser.model,
        backend=analyser.backend.value,
        crop_expansion=crop_expansion,
    )


def _aggregate_counters(results: list[PipelineEvaluationResult]) -> dict[tuple[Any, ...], list[int]]:
    counters: dict[tuple[Any, ...], list[int]] = {}
    for result in results:
        counter_key = ("crop_expansion", result.crop_expansion)
        if counter_key not in counters:
            counters[counter_key] = [0, 0]
        counters[counter_key][0] += 1 if result.correct else 0
        counters[counter_key][1] += 1
    return counters


def evaluate_pipeline(
    labels_path: Path,
    model: str,
    backend: Backend = Backend.OLLAMA,
    url: str = "http://localhost:8080/v1",
    api_key: str | None = None,
    fps: float | None = None,
    low_res_size: tuple[int, int] = (640, 360),
    crop_expansion: float = 0.25,
    crops_output_dir: Path | None = None,
    result_jsonl_path: Path | None = None,
) -> PipelineEvaluationSummary:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not labels_path.is_file():
        raise ValueError(f"Labels is not a file: {labels_path}")

    from wildcamtools.lib.pipeline import generate_frames_for_video
    from wildcamtools.lib.web.label import load_labels

    labelled_data = load_labels(labels_path)
    video_directory = labels_path.resolve().parent

    if low_res_size[0] <= 0 or low_res_size[1] <= 0:
        raise ValueError("low_res_size must be positive")

    if crops_output_dir is None:
        crops_output_dir = Path("crops_output/")
    crops_output_dir.mkdir(parents=True, exist_ok=True)

    if result_jsonl_path is None:
        result_jsonl_path = Path("result.jsonl")
    result_jsonl_path.unlink(missing_ok=True)

    analyser = create_analyser(backend=backend, model=model, url=url, api_key=api_key)

    ctx = multiprocessing.get_context("spawn")
    results: list[PipelineEvaluationResult] = []

    with ctx.Pool() as pool:
        frame_tmpdirs: list[TemporaryDirectory] = []
        futures: list[tuple[AsyncResult, str, TemporaryDirectory]] = []
        for filename in labelled_data:
            video_path = video_directory / filename
            if not video_path.exists():
                logger.warning("Video not found: %s", filename)
                continue
            frame_tmpdir = TemporaryDirectory()
            frame_tmpdirs.append(frame_tmpdir)
            future = pool.apply_async(
                generate_frames_for_video,
                args=(
                    filename,
                    video_directory,
                    fps,
                    low_res_size,
                    Path(frame_tmpdir.name),
                ),
            )
            futures.append((future, filename, frame_tmpdir))

        for future, _, frame_tmpdir in futures:
            try:
                frame_result = future.get()
                result = _process_pipeline_evaluation_result(
                    frame_result=frame_result,
                    labelled_data=labelled_data,
                    crops_output_dir=crops_output_dir,
                    analyser=analyser,
                    crop_expansion=crop_expansion,
                    result_jsonl_path=result_jsonl_path,
                )
                results.append(result)
            except Exception:
                logger.exception("Worker failed")
                raise
            finally:
                frame_tmpdir.cleanup()

    counters = _aggregate_counters(results)
    return PipelineEvaluationSummary(results=results, counters=counters)
