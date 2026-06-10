import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlmodel import Session

from wildcamtools.lib.ai.pipeline import PipelineOutcome
from wildcamtools.lib.ai.pipeline_config import AiPipelineConfig
from wildcamtools.lib.ai.types import RichResult
from wildcamtools.lib.persistence.models import (
    ClassificationResult,
    PipelineBatch,
    PipelineRun,
    Video,
    VideoStat,
)
from wildcamtools.lib.stats import VideoStats

logger = logging.getLogger(__name__)

# mypy: disable-error-code="arg-type,call-overload,no-any-return"
# SQLModel's Session.exec() has type issues with select() that are well-known
# These ignores are safe as the queries are valid SQLAlchemy/SQLModel patterns


def _rich_result_to_classification(result: RichResult) -> dict[str, Any]:
    """Convert a RichResult to a ClassificationResult dict."""
    return {
        "species_name": result.species_name,
        "confidence": result.confidence.value,
        "is_animal_present": result.is_animal_present,
        "is_animal_unknown": result.is_animal_unknown,
    }


def _get_or_create_classification_result(session: Session, result_data: dict[str, Any]) -> ClassificationResult:
    """Get or create a ClassificationResult based on unique constraint fields."""
    stmt = select(ClassificationResult).where(
        ClassificationResult.species_name == result_data["species_name"],
        ClassificationResult.confidence == result_data["confidence"],
        ClassificationResult.is_animal_present == result_data["is_animal_present"],
        ClassificationResult.is_animal_unknown == result_data["is_animal_unknown"],
    )
    results = session.exec(stmt)
    existing = results.scalars().first()
    if existing:
        logger.debug("Found existing ClassificationResult: id=%d", existing.id)
        return existing

    new_result = ClassificationResult(**result_data)
    session.add(new_result)
    session.flush()
    logger.debug("Created new ClassificationResult: id=%d", new_result.id)
    return new_result


def _get_or_create_video_stat(session: Session, stats: VideoStats) -> VideoStat:
    """Get or create a VideoStat based on unique constraint fields."""
    stat_data = {
        "width": stats.x,
        "height": stats.y,
        "fps": stats.fps,
        "total_frames": stats.frame_count,
    }
    stmt = select(VideoStat).where(
        VideoStat.width == stat_data["width"],
        VideoStat.height == stat_data["height"],
        VideoStat.fps == stat_data["fps"],
        VideoStat.total_frames == stat_data["total_frames"],
    )
    results = session.exec(stmt)
    existing = results.scalars().first()
    if existing:
        logger.debug("Found existing VideoStat: id=%d", existing.id)
        return existing

    new_stat = VideoStat(**stat_data)
    session.add(new_stat)
    session.flush()
    logger.debug("Created new VideoStat: id=%d", new_stat.id)
    return new_stat


def _get_or_create_video(session: Session, filename: str, stat: VideoStat) -> Video:
    """Get or create a Video entity."""
    stmt = select(Video).where(Video.filename == filename)
    results = session.exec(stmt)
    existing = results.scalars().first()
    if existing:
        logger.debug("Found existing Video: filename=%s", filename)
        if existing.stat_id != stat.id:
            existing.stat_id = stat.id
        return existing

    new_video = Video(filename=filename, stat=stat)
    session.add(new_video)
    session.flush()
    logger.debug("Created new Video: filename=%s", filename)
    return new_video


def save_pipeline_run(
    session: Session,
    video_path: Path,
    config: AiPipelineConfig,
    outcome: PipelineOutcome,
) -> PipelineRun:
    """Save a pipeline run to the database.

    This function normalizes data by:
    - Reusing VideoStat entries for identical video metadata
    - Reusing ClassificationResult entries for identical AI results
    - Creating new PipelineRun and PipelineBatch entries for each execution

    Note:
        The caller is responsible for committing the session. This function only
        adds objects and flushes to generate IDs.

    Args:
        session: SQLModel session
        video_path: Path to the video file
        config: Pipeline configuration
        outcome: Pipeline execution outcome

    Returns:
        The created PipelineRun entity
    """

    video_stat = _get_or_create_video_stat(session, outcome.stats)
    _get_or_create_video(session, str(video_path), video_stat)

    final_result_data = _rich_result_to_classification(outcome.result)
    final_result = _get_or_create_classification_result(session, final_result_data)

    config_json = config.model_dump_json()

    run = PipelineRun(
        video_filename=str(video_path),
        config_json=config_json,
        final_result=final_result,
    )
    session.add(run)
    session.flush()

    for batch in outcome.batches:
        if batch.result is None:
            continue
        batch_result_data = _rich_result_to_classification(batch.result)
        batch_result = _get_or_create_classification_result(session, batch_result_data)

        frame_numbers = json.dumps([pair.frame_no for pair in batch.selected_frames])

        pipeline_batch = PipelineBatch(
            run_id=run.id,
            frame_numbers=frame_numbers,
            result=batch_result,
        )
        session.add(pipeline_batch)

    session.flush()
    logger.info("Saved PipelineRun: id=%d, video=%s, batches=%d", run.id, str(video_path), len(outcome.batches))
    return run


def get_pipeline_run(session: Session, run_id: int) -> PipelineRun | None:
    """Get a pipeline run by ID with all relationships loaded."""
    stmt = select(PipelineRun).where(PipelineRun.id == run_id)
    return session.exec(stmt).scalars().first()


def get_pipeline_runs_by_video(session: Session, filename: str) -> list[PipelineRun]:
    """Get all pipeline runs for a specific video file."""
    stmt = select(PipelineRun).where(PipelineRun.video_filename == filename)
    return list(session.exec(stmt).scalars().all())


def get_classification_result(session: Session, result_id: int) -> ClassificationResult | None:
    """Get a classification result by ID."""
    stmt = select(ClassificationResult).where(ClassificationResult.id == result_id)
    return session.exec(stmt).scalars().first()
