import logging
import time
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from wildcamtools.lib.ai.parallel_processing import (
    run_parallel_worker_pool,
)
from wildcamtools.lib.ai.pipeline import (
    CombinedPipelineOutcome,
    PipelineOutcome,
)
from wildcamtools.lib.ai.pipeline_config import AiPipelineConfig
from wildcamtools.lib.ai.types import RichResult

logger = logging.getLogger(__name__)

R = TypeVar("R", bound=BaseModel)


class BatchPipelineOutput[R: BaseModel](BaseModel):
    """Output format for batch pipeline processing."""

    config: AiPipelineConfig
    outcome: PipelineOutcome[R] | CombinedPipelineOutcome[R]


class RichResultBatchPipelineOutput(BatchPipelineOutput[RichResult]):
    """Concrete BatchPipelineOutput parameterised on RichResult."""


class BatchWorkerResult(BaseModel):
    """Result from batch worker process."""

    video_path: str
    output_path: str
    processing_time_seconds: float = 0.0
    error: str | None = None


def _build_combined_output[R: BaseModel](
    outcome: PipelineOutcome[R] | CombinedPipelineOutcome[R], config: AiPipelineConfig
) -> BatchPipelineOutput[R]:
    return BatchPipelineOutput[R](config=config, outcome=outcome)


def _run_pipeline_worker(
    video_path_str: str,
    output_path_str: str,
    pipeline_config: AiPipelineConfig,
) -> BatchWorkerResult:
    """Worker function that runs the AI pipeline on a single video and writes the result.

    Args:
        video_path_str: Path to the video file
        output_path_str: Path to write the JSON output
        pipeline_config_json: JSON string of AI pipeline configuration

    Returns:
        BatchWorkerResult with processing details
    """
    video_path = Path(video_path_str)
    output_path = Path(output_path_str)

    try:
        start_time = time.time()
        pipeline = pipeline_config.create_pipeline()
        outcome = pipeline.run(video_path)
        processing_time = time.time() - start_time

        output_data = _build_combined_output(outcome, pipeline_config)
        json_output = output_data.model_dump_json(indent=2)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_output)

        logger.info(
            "Processed %s -> %s (%.2f seconds)",
            video_path.name,
            output_path.name,
            processing_time,
        )

        return BatchWorkerResult(
            video_path=str(video_path),
            output_path=str(output_path),
            processing_time_seconds=processing_time,
            error=None,
        )
    except Exception as e:
        logger.exception("Worker failed for video %s", video_path.name)
        return BatchWorkerResult(
            video_path=str(video_path),
            output_path=str(output_path),
            processing_time_seconds=0.0,
            error=f"Processing failed: {video_path.name}: {type(e).__name__}: {e}",
        )


def discover_video_files(video_dir: Path, recursive: bool = True) -> list[Path]:
    """Discover video files in a directory.

    Args:
        video_dir: Directory to search
        recursive: Whether to search recursively

    Returns:
        List of video file paths sorted by path
    """
    patterns = ["*.mp4", "*.MP4"]
    video_files: list[Path] = []

    if recursive:
        for pattern in patterns:
            video_files.extend(video_dir.rglob(pattern))
    else:
        for pattern in patterns:
            video_files.extend(video_dir.glob(pattern))

    return sorted(video_files)


def compute_output_path(video_path: Path, video_dir: Path, output_dir: Path) -> Path:
    """Compute the output JSON path for a video file, mirroring the directory structure.

    Args:
        video_path: Path to the video file
        video_dir: Base video directory
        output_dir: Base output directory

    Returns:
        Path to the output JSON file
    """
    try:
        relative_path = video_path.relative_to(video_dir)
    except ValueError:
        # video_path is not under video_dir, use the filename only
        relative_path = Path(video_path.name)

    # Replace .mp4 extension with .json
    output_path = output_dir / relative_path.with_suffix(".json")
    return output_path


def run_batch_pipeline(
    video_paths: list[Path],
    video_dir: Path,
    output_dir: Path,
    pipeline_config: AiPipelineConfig,
    max_workers: int | None = None,
    skip_existing: bool = True,
) -> list[BatchWorkerResult]:
    """Run the AI pipeline on multiple videos in parallel.

    Args:
        video_paths: List of video file paths to process
        video_dir: Base video directory (used for computing output paths)
        output_dir: Base output directory for JSON results
        pipeline_config: AI pipeline configuration
        max_workers: Maximum number of worker processes (default: CPU count)
        skip_existing: Whether to skip videos that already have output files

    Returns:
        List of BatchWorkerResult objects
    """
    # Filter out videos that already have output files
    pending_videos: list[tuple[Path, Path]] = []
    skipped_count = 0

    for video_path in video_paths:
        output_path = compute_output_path(video_path, video_dir, output_dir)
        if skip_existing and output_path.exists():
            logger.info("Skipping %s (output already exists)", video_path.name)
            skipped_count += 1
        else:
            pending_videos.append((video_path, output_path))

    if not pending_videos:
        logger.info("All %d videos already processed, skipping", len(video_paths))
        return []

    tasks = [(str(video_path), str(output_path), pipeline_config) for video_path, output_path in pending_videos]

    results = run_parallel_worker_pool(
        tasks=tasks,
        worker_fn=_run_pipeline_worker,
        max_workers=max_workers,
        task_description="videos",
    )

    # Log summary
    success_count = sum(1 for r in results if r.error is None)
    error_count = sum(1 for r in results if r.error is not None)
    logger.info(
        "Batch processing completed: %d successful, %d errors, %d skipped",
        success_count,
        error_count,
        skipped_count,
    )

    return results
