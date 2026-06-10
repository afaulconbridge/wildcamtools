import logging
from pathlib import Path

from pydantic import BaseModel, Field

from wildcamtools.lib.ai.label_comparison_config import LabelComparisonConfig
from wildcamtools.lib.ai.parallel_processing import (
    run_parallel_worker_pool_with_labels,
    time_pipeline_execution,
    validate_evaluation_paths,
)
from wildcamtools.lib.ai.pipeline import PipelineOutcome
from wildcamtools.lib.ai.pipeline_config import AiPipelineConfig
from wildcamtools.lib.ai.types import ResultClassification, RichResult
from wildcamtools.lib.labels import load_labels

logger = logging.getLogger(__name__)


class _WorkerResult(BaseModel):
    """Result from worker process before label comparison."""

    filename: str
    outcome: PipelineOutcome
    processing_time_seconds: float = 0.0
    frame_ids: list[int] = Field(default_factory=list)
    error: str | None = None


class PipelineEvaluationResult(BaseModel):
    filename: str
    classification: ResultClassification
    result: RichResult
    label: str
    error: str | None = None
    comparison_method: str = "exact"
    processing_time_seconds: float = 0.0
    frame_ids: list[int] = Field(default_factory=list)
    stats: dict | None = None


class PipelineEvaluationSummary(BaseModel):
    results: list[PipelineEvaluationResult] = Field(default_factory=list)
    correct_count: int = 0
    incorrect_count: int = 0
    unknown_count: int = 0
    total_count: int = 0
    error_count: int = 0
    average_processing_time_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total_count == 0:
            return 0.0
        valid_count = self.total_count - self.error_count - self.unknown_count
        if valid_count == 0:
            return 0.0
        return self.correct_count / valid_count

    @property
    def success_rate(self) -> float:
        return self.accuracy

    @property
    def failure_count(self) -> int:
        return self.total_count - self.correct_count

    @property
    def detection_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        confident_count = self.total_count - self.unknown_count - self.error_count
        return confident_count / self.total_count

    @property
    def precision_when_confident(self) -> float:
        confident_total = self.correct_count + self.incorrect_count
        if confident_total == 0:
            return 0.0
        return self.correct_count / confident_total

    def print_summary(self) -> None:
        logger.info("Evaluation Summary:")
        logger.info("  Total: %d", self.total_count)
        logger.info("  Correct: %d", self.correct_count)
        logger.info("  Incorrect: %d", self.incorrect_count)
        logger.info("  Unknown: %d", self.unknown_count)
        logger.info("  Errors: %d", self.error_count)
        logger.info("  Accuracy: %.4f", self.accuracy)
        logger.info("  Detection Rate: %.4f", self.detection_rate)
        logger.info("  Precision (when confident): %.4f", self.precision_when_confident)
        logger.info("  Average processing time: %.2f seconds", self.average_processing_time_seconds)


def _evaluate_video_worker(
    video_path_str: str,
    pipeline_config: AiPipelineConfig,
) -> _WorkerResult:
    video_path = Path(video_path_str)
    outcome, processing_time = time_pipeline_execution(video_path, pipeline_config)
    # Extract frame numbers from batch_results
    frame_nos = [pair.frame_no for batch in outcome.batches for pair in batch.selected_frames]

    logger.info(
        "Video %s: result=%s, frames=%s",
        video_path.name,
        outcome.result.species_name,
        frame_nos,
    )

    return _WorkerResult(
        filename=video_path.name,
        outcome=outcome,
        processing_time_seconds=processing_time,
        frame_ids=frame_nos,
        error=None,
    )


def _run_worker_pool(
    labelled_data: dict[str, str],
    video_dir: Path,
    pipeline_config: AiPipelineConfig,
    max_workers: int | None,
    comparison_config: LabelComparisonConfig,
) -> list[PipelineEvaluationResult]:
    results: list[PipelineEvaluationResult] = []
    comparator = comparison_config.create_comparator()

    tasks_with_labels = []
    for filename, label in labelled_data.items():
        video_path = video_dir / filename
        if not video_path.exists():
            logger.warning("Video not found: %s", filename)
            continue
        tasks_with_labels.append((label, str(video_path), pipeline_config))

    worker_results = run_parallel_worker_pool_with_labels(
        tasks_with_labels=tasks_with_labels,
        worker_fn=_evaluate_video_worker,
        max_workers=max_workers,
        task_description="videos",
    )

    for label, worker_result in worker_results:
        if not isinstance(worker_result, _WorkerResult):
            logger.error("Result of unexpected class")
            continue
        classification = comparator.compare(worker_result.outcome.result, label)
        results.append(
            PipelineEvaluationResult(
                filename=worker_result.filename,
                result=worker_result.outcome.result,
                classification=classification,
                label=label,
                error=worker_result.error,
                comparison_method=comparator.method_name,
                processing_time_seconds=worker_result.processing_time_seconds,
                frame_ids=worker_result.frame_ids,
                stats=worker_result.outcome.stats.model_dump(),
            )
        )

    return results


def evaluate_ai_pipeline(
    config_path: Path,
    labels_path: Path,
    video_dir: Path | None = None,
    max_workers: int | None = None,
    comparison_config_path: Path | None = None,
) -> PipelineEvaluationSummary:
    """Evaluate AiPipeline against labeled videos.

    Args:
        config_path: Path to JSON configuration file for AiPipeline.
        labels_path: Path to JSONL file with video labels.
        video_dir: Directory containing video files. Defaults to parent of labels_path.
        max_workers: Maximum number of worker processes. Defaults to CPU count.
        comparison_config_path: Path to JSON config for label comparison. Defaults to None for exact matching.

    Returns:
        PipelineEvaluationSummary with results and statistics.

    Raises:
        FileNotFoundError: If config_path, labels_path, or comparison_config_path doesn't exist.
        ValueError: If config_path, labels_path, or comparison_config_path is not a file.
    """
    validate_evaluation_paths(config_path, labels_path, video_dir, comparison_config_path)

    labelled_data = load_labels(labels_path)
    if video_dir is None:
        video_dir = labels_path.parent

    pipeline_config = AiPipelineConfig.from_json(config_path)
    comparison_config = (
        LabelComparisonConfig.from_json(comparison_config_path)
        if comparison_config_path is not None
        else LabelComparisonConfig()
    )

    results = _run_worker_pool(labelled_data, video_dir, pipeline_config, max_workers, comparison_config)

    correct_count = sum(1 for r in results if r.classification == ResultClassification.CORRECT)
    incorrect_count = sum(1 for r in results if r.classification == ResultClassification.INCORRECT)
    unknown_count = sum(1 for r in results if r.classification == ResultClassification.UNKNOWN)
    error_count = sum(1 for r in results if r.error is not None)
    total_count = len(results)
    total_processing_time = sum(r.processing_time_seconds for r in results)
    average_processing_time = total_processing_time / total_count if total_count > 0 else 0.0

    return PipelineEvaluationSummary(
        results=results,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        unknown_count=unknown_count,
        total_count=total_count,
        error_count=error_count,
        average_processing_time_seconds=average_processing_time,
    )
