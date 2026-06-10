import logging
import multiprocessing
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wildcamtools.lib.ai.pipeline import PipelineOutcome
from wildcamtools.lib.ai.pipeline_config import AiPipelineConfig

logger = logging.getLogger(__name__)


def time_pipeline_execution(
    video_path: Path,
    pipeline_config: AiPipelineConfig,
) -> tuple[PipelineOutcome, float]:
    """Execute a pipeline and return the outcome with timing information.

    Args:
        video_path: Path to the video file to process
        pipeline_config: AI pipeline configuration

    Returns:
        Tuple of (PipelineOutcome, processing_time_seconds)

    Raises:
        Exception: Any exception from pipeline execution is propagated
    """
    start_time = time.time()
    pipeline = pipeline_config.create_pipeline()
    outcome = pipeline.run(video_path)
    end_time = time.time()
    processing_time = end_time - start_time
    return outcome, processing_time


def run_parallel_worker_pool(
    tasks: list[tuple[Any, ...]],
    worker_fn: Callable[..., Any],
    max_workers: int | None = None,
    task_description: str = "tasks",
) -> list[Any]:
    """Run tasks in parallel using a multiprocessing pool.

    Args:
        tasks: List of task argument tuples to pass to worker_fn
        worker_fn: Worker function to execute for each task
        max_workers: Maximum number of worker processes (default: CPU count)
        task_description: Description of tasks for logging

    Returns:
        List of results from worker_fn (excludes failed tasks)
    """
    ctx = multiprocessing.get_context("spawn")
    results: list[Any] = []

    worker_count = max_workers or multiprocessing.cpu_count()
    logger.info(
        "Processing %d %s using %d workers",
        len(tasks),
        task_description,
        worker_count,
    )

    with ctx.Pool(processes=max_workers) as pool:
        futures = []
        for task_args in tasks:
            future = pool.apply_async(worker_fn, args=task_args)
            futures.append(future)

        for future in futures:
            try:
                result = future.get()
                results.append(result)
            except Exception:
                logger.exception("Worker task failed")

    return results


def run_parallel_worker_pool_with_labels(
    tasks_with_labels: list[tuple[Any, ...]],
    worker_fn: Callable[..., Any],
    max_workers: int | None = None,
    task_description: str = "tasks",
) -> list[tuple[Any, Any]]:
    """Run tasks in parallel and return results paired with their labels.

    Similar to run_parallel_worker_pool but preserves label information for each task.

    Args:
        tasks_with_labels: List of tuples where first element is label, rest are task args
        worker_fn: Worker function to execute for each task
        max_workers: Maximum number of worker processes (default: CPU count)
        task_description: Description of tasks for logging

    Returns:
        List of (label, result) tuples (excludes failed tasks)
    """
    ctx = multiprocessing.get_context("spawn")
    results: list[tuple[Any, Any]] = []

    worker_count = max_workers or multiprocessing.cpu_count()
    logger.info(
        "Processing %d %s using %d workers",
        len(tasks_with_labels),
        task_description,
        worker_count,
    )

    with ctx.Pool(processes=max_workers) as pool:
        futures = []
        for item in tasks_with_labels:
            label = item[0]
            task_args = item[1:]
            future = pool.apply_async(worker_fn, args=task_args)
            futures.append((label, future))

        for label, future in futures:
            try:
                result = future.get()
                results.append((label, result))
            except Exception:
                logger.exception("Worker task failed")

    return results


def validate_evaluation_paths(
    config_path: Path,
    labels_path: Path,
    video_dir: Path | None = None,
    comparison_config_path: Path | None = None,
) -> None:
    """Validate paths for pipeline evaluation.

    Args:
        config_path: Path to config file
        labels_path: Path to labels file
        video_dir: Path to video directory (optional)
        comparison_config_path: Path to comparison config file (optional)

    Raises:
        FileNotFoundError: If a required path doesn't exist
        ValueError: If a path is not the correct type (file vs directory)
    """
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
    if comparison_config_path is not None:
        if not comparison_config_path.exists():
            raise FileNotFoundError(f"Label comparison config file not found: {comparison_config_path}")
        if not comparison_config_path.is_file():
            raise ValueError(f"Label comparison config path is not a file: {comparison_config_path}")
