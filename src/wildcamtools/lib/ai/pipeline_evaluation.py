import json
import logging
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

from wildcamtools.lib.ai.label_comparison_config import LabelComparisonConfig
from wildcamtools.lib.ai.pipeline_config import AiPipelineConfig
from wildcamtools.lib.ai.types import SpeciesResult
from wildcamtools.lib.labels import load_labels

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class PipelineEvaluationResult:
    filename: str
    correct: bool
    raw_result: str
    label: str
    error: str | None = None
    comparison_method: str = "exact"


@dataclass(kw_only=True)
class PipelineEvaluationSummary:
    results: list[PipelineEvaluationResult]
    correct_count: int
    total_count: int
    error_count: int

    @property
    def accuracy(self) -> float:
        if self.total_count == 0:
            return 0.0
        valid_count = self.total_count - self.error_count
        if valid_count == 0:
            return 0.0
        return self.correct_count / valid_count

    def print_summary(self) -> None:
        logger.info("Evaluation Summary:")
        logger.info("  Total: %d", self.total_count)
        logger.info("  Correct: %d", self.correct_count)
        logger.info("  Errors: %d", self.error_count)
        logger.info("  Accuracy: %.4f", self.accuracy)


def _evaluate_video_worker(
    video_path_str: str,
    ground_truth_label: str,
    pipeline_config: AiPipelineConfig,
    comparator_config: LabelComparisonConfig,
) -> PipelineEvaluationResult:
    video_path = Path(video_path_str)
    try:
        pipeline = pipeline_config.create_pipeline()
        result = pipeline.run(video_path)

        if isinstance(result, SpeciesResult) or hasattr(result, "species_name"):
            raw_result = result.species_name
        elif hasattr(result, "message"):
            raw_result = result.message
        else:
            raw_result = str(result)

        comparator = comparator_config.create_comparator()
        correct = comparator.compare(raw_result, ground_truth_label)
        logger.info(
            "Video %s: label=%s, result=%s, correct=%s (method=%s)",
            video_path.name,
            ground_truth_label,
            raw_result,
            correct,
            comparator.method_name,
        )

        return PipelineEvaluationResult(
            filename=video_path.name,
            correct=correct,
            raw_result=raw_result,
            label=ground_truth_label,
            error=None,
            comparison_method=comparator.method_name,
        )
    except Exception:
        logger.exception("Worker failed for video %s", video_path.name)
        raise


def _validate_paths(config_path: Path, labels_path: Path, video_dir: Path | None) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not labels_path.is_file():
        raise ValueError(f"Labels path is not a file: {labels_path}")
    if video_dir is not None and not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if video_dir is not None and not video_dir.is_dir():
        raise ValueError(f"Video path is not a directory: {video_dir}")


def _run_worker_pool(
    labelled_data: dict[str, str],
    video_dir: Path,
    pipeline_config: AiPipelineConfig,
    max_workers: int | None,
    comparison_config: LabelComparisonConfig,
) -> list[PipelineEvaluationResult]:
    ctx = multiprocessing.get_context("spawn")
    results: list[PipelineEvaluationResult] = []

    with ctx.Pool(processes=max_workers) as pool:
        futures = []
        for filename, label in labelled_data.items():
            video_path = video_dir / filename
            if not video_path.exists():
                logger.warning("Video not found: %s", filename)
                continue
            # TODO use a Pydantic model to a dictionary as kwargs
            future = pool.apply_async(
                _evaluate_video_worker,
                args=(str(video_path), label, pipeline_config, comparison_config),
            )
            futures.append(future)

        for future in futures:
            try:
                result = future.get()
                results.append(result)
            except Exception as e:
                logger.exception("Worker task failed")
                results.append(
                    PipelineEvaluationResult(
                        filename="unknown",
                        correct=False,
                        raw_result="",
                        label="unknown",
                        error=str(e),
                        comparison_method="exact",
                    )
                )

    return results


def _write_results(results: list[PipelineEvaluationResult], result_jsonl_path: Path) -> None:
    result_jsonl_path.unlink(missing_ok=True)
    with open(result_jsonl_path, "w") as result_file:
        for result in results:
            result_dict = {
                "filename": result.filename,
                "correct": result.correct,
                "raw_result": result.raw_result,
                "label": result.label,
                "error": result.error,
                "comparison_method": result.comparison_method,
            }
            result_file.write(json.dumps(result_dict) + "\n")


def evaluate_ai_pipeline(
    config_path: Path,
    labels_path: Path,
    video_dir: Path | None = None,
    max_workers: int | None = None,
    result_jsonl_path: Path | None = None,
    comparison_config_path: Path | None = None,
) -> PipelineEvaluationSummary:
    """Evaluate AiPipeline against labeled videos.

    Args:
        config_path: Path to JSON configuration file for AiPipeline.
        labels_path: Path to JSONL file with video labels.
        video_dir: Directory containing video files. Defaults to parent of labels_path.
        max_workers: Maximum number of worker processes. Defaults to CPU count.
        result_jsonl_path: Path to write results. Defaults to current directory.
        comparison_config_path: Path to JSON config for label comparison. Defaults to None for exact matching.

    Returns:
        PipelineEvaluationSummary with results and statistics.

    Raises:
        FileNotFoundError: If config_path, labels_path, or comparison_config_path doesn't exist.
        ValueError: If config_path, labels_path, or comparison_config_path is not a file.
    """
    # TODO refactor these into more modular and reusable convenience functions (e.g validate_file, validate_dir)
    _validate_paths(config_path, labels_path, video_dir)
    if comparison_config_path is not None:
        if not comparison_config_path.exists():
            raise FileNotFoundError(f"Label comparison config file not found: {comparison_config_path}")
        if not comparison_config_path.is_file():
            raise ValueError(f"Label comparison config path is not a file: {comparison_config_path}")

    labelled_data = load_labels(labels_path)
    if video_dir is None:
        video_dir = labels_path.parent

    if result_jsonl_path is None:
        result_jsonl_path = Path.cwd() / "pipeline_evaluation_result.jsonl"

    pipeline_config = AiPipelineConfig.from_json(config_path)
    comparison_config = (
        LabelComparisonConfig.from_json(comparison_config_path)
        if comparison_config_path is not None
        else LabelComparisonConfig()
    )

    results = _run_worker_pool(labelled_data, video_dir, pipeline_config, max_workers, comparison_config)
    _write_results(results, result_jsonl_path)

    correct_count = sum(1 for r in results if r.correct and r.error is None)
    error_count = sum(1 for r in results if r.error is not None)
    total_count = len(results)

    return PipelineEvaluationSummary(
        results=results,
        correct_count=correct_count,
        total_count=total_count,
        error_count=error_count,
    )
