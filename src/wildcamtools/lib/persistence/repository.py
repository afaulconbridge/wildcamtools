import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import selectinload
from sqlmodel import Session

from wildcamtools.lib.ai.pipeline import (
    CombinedBatchResult,
    CombinedPipelineOutcome,
    PipelineOutcome,
)
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


def _get_or_create_video(
    session: Session,
    filename: str,
    stat: VideoStat,
    recorded_at: datetime | None = None,
) -> Video:
    """Get or create a Video entity."""
    stmt = select(Video).where(Video.filename == filename)
    results = session.exec(stmt)
    existing = results.scalars().first()
    if existing:
        logger.debug("Found existing Video: filename=%s", filename)
        if existing.stat_id != stat.id:
            existing.stat_id = stat.id
        if recorded_at is not None and existing.recorded_at is None:
            existing.recorded_at = recorded_at
        return existing

    new_video = Video(filename=filename, stat=stat, recorded_at=recorded_at)
    session.add(new_video)
    session.flush()
    logger.debug("Created new Video: filename=%s, recorded_at=%s", filename, recorded_at)
    return new_video


def save_pipeline_run(
    session: Session,
    video_path: Path,
    config: AiPipelineConfig,
    outcome: PipelineOutcome | CombinedPipelineOutcome,
    recorded_at: datetime | None = None,
) -> PipelineRun:
    """Save a pipeline run to the database.

    This function normalizes data by:
    - Reusing VideoStat entries for identical video metadata
    - Reusing ClassificationResult entries for identical AI results
    - Creating new PipelineRun and PipelineBatch entries for each execution

    Note:
        The caller is responsible for committing the session. This function only
        adds objects and flushes to generate IDs.
        Description data from CombinedPipelineOutcome batches is persisted to the database.

    Args:
        session: SQLModel session
        video_path: Path to the video file
        config: Pipeline configuration
        outcome: Pipeline execution outcome
        recorded_at: Optional timestamp inferred from the filename. Only used
            when creating a new ``Video`` row, or to backfill an existing row
            whose ``recorded_at`` is ``None``.

    Returns:
        The created PipelineRun entity

    """
    video_stat = _get_or_create_video_stat(session, outcome.stats)
    _get_or_create_video(session, str(video_path), video_stat, recorded_at=recorded_at)

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

    # Handle both PipelineOutcome and CombinedPipelineOutcome batch structures
    for batch in outcome.batches:
        if isinstance(batch, CombinedBatchResult):
            classification_result = batch.classification
            if classification_result is None:
                continue
            batch_result_data = _rich_result_to_classification(classification_result)
            # Extract description text from BatchDescription object
            description_text = batch.description.description if batch.description else None
        else:
            if batch.result is None:
                continue
            batch_result_data = _rich_result_to_classification(batch.result)
            description_text = None

        batch_result = _get_or_create_classification_result(session, batch_result_data)

        frame_numbers = json.dumps([pair.frame_no for pair in batch.selected_frames])

        pipeline_batch = PipelineBatch(
            run_id=run.id,
            frame_numbers=frame_numbers,
            result=batch_result,
            description=description_text,
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


def list_pipeline_runs_filtered(
    session: Session,
    *,
    confidences: list[str] | None = None,
    species: list[str] | None = None,
    animal_present_only: bool = False,
    limit: int = 25,
    offset: int = 0,
) -> list[PipelineRun]:
    """Return pipeline runs whose final classification matches the filters.

    Results are ordered by ``timestamp`` descending and paginated via
    ``limit``/``offset``. ``confidences`` and ``species`` of ``None`` mean
    'no filter'. An empty list means 'filter to nothing'.
    """
    if confidences is not None and len(confidences) == 0:
        return []
    if species is not None and len(species) == 0:
        return []

    stmt = (
        _filtered_classification_subquery(
            confidences=confidences or [],
            species=species,
            animal_present_only=animal_present_only,
        )
        .order_by(desc(PipelineRun.timestamp))
        .offset(offset)
        .limit(limit)
        .options(
            selectinload(PipelineRun.video),
            selectinload(PipelineRun.final_result),
        )
    )

    rows = session.exec(stmt).all()
    return [run for run, _ in rows]


def list_runs_for_video(session: Session, filename: str) -> list[PipelineRun]:
    """Return pipeline runs for ``filename`` ordered by timestamp desc.

    Eagerly loads the final classification, the related video, and per-batch
    classification so callers can render results without triggering N+1
    queries.
    """
    stmt = (
        select(PipelineRun)
        .where(PipelineRun.video_filename == filename)
        .order_by(desc(PipelineRun.timestamp))
        .options(
            selectinload(PipelineRun.final_result),
            selectinload(PipelineRun.video),
            selectinload(PipelineRun.batches).selectinload(PipelineBatch.result),
        )
    )
    return list(session.exec(stmt).scalars().all())


def list_all_video_filenames(session: Session) -> list[str]:
    """Return all video filenames ordered alphabetically."""
    stmt = select(Video.filename).order_by(Video.filename)
    return list(session.exec(stmt).scalars().all())


def list_recent_pipeline_runs(session: Session, limit: int = 10) -> list[PipelineRun]:
    """Return the most recent pipeline runs ordered by timestamp desc.

    Eagerly loads the related video and final classification so callers can
    render captions without triggering N+1 queries.
    """
    stmt = (
        select(PipelineRun)
        .order_by(desc(PipelineRun.timestamp))
        .limit(limit)
        .options(
            selectinload(PipelineRun.video),
            selectinload(PipelineRun.final_result),
        )
    )
    return list(session.exec(stmt).scalars().all())


def count_animal_status(
    session: Session,
    *,
    confidences: list[str] | None = None,
    species: list[str] | None = None,
) -> tuple[int, int, int]:
    """Count runs by animal status (present, absent, unknown) under filters.

    Unlike :func:`aggregate_statistics`, this does not apply
    ``animal_present_only`` so the caller can compute ``absent`` as
    ``unfiltered - present`` without that filter masking the absent count.
    """
    if confidences is not None and len(confidences) == 0:
        return (0, 0, 0)
    if species is not None and len(species) == 0:
        return (0, 0, 0)

    base = _filtered_classification_subquery(
        confidences=confidences or [],
        species=species,
        animal_present_only=False,
    ).subquery()

    present = int(
        session.exec(
            select(func.count(base.c.video_filename)).select_from(base).where(base.c.is_animal_present.is_(True)),
        ).scalar()
        or 0,
    )
    absent = int(
        session.exec(
            select(func.count(base.c.video_filename)).select_from(base).where(base.c.is_animal_present.is_(False)),
        ).scalar()
        or 0,
    )
    unknown = int(
        session.exec(
            select(func.count(base.c.video_filename)).select_from(base).where(base.c.is_animal_unknown.is_(True)),
        ).scalar()
        or 0,
    )
    return (present, absent, unknown)


@dataclass
class StatisticsSummary:
    """Pre-aggregated statistics for the database browser stats tab."""

    total_runs: int
    total_videos: int
    distinct_species: int
    animal_present_count: int
    animal_absent_count: int
    animal_unknown_count: int
    species_counts: dict[str, int]
    confidence_counts: dict[str, int]


def _filtered_classification_subquery(
    *,
    confidences: list[str],
    species: list[str] | None,
    animal_present_only: bool,
) -> Any:
    """Build a where-clause shared by run filtering and statistics.

    Joins ``PipelineRun.final_result`` so the result columns are usable in the
    outer query.
    """
    stmt: Any = select(PipelineRun, ClassificationResult).join(
        ClassificationResult,
        PipelineRun.final_result_id == ClassificationResult.id,
    )
    if confidences:
        stmt = stmt.where(ClassificationResult.confidence.in_(confidences))  # type: ignore[attr-defined]
    if species:
        stmt = stmt.where(ClassificationResult.species_name.in_(species))  # type: ignore[attr-defined]
    if animal_present_only:
        stmt = stmt.where(ClassificationResult.is_animal_present.is_(True))  # type: ignore[attr-defined]
    return stmt


def count_pipeline_runs_filtered(
    session: Session,
    *,
    confidences: list[str] | None = None,
    species: list[str] | None = None,
    animal_present_only: bool = False,
) -> int:
    """Count pipeline runs whose final classification matches the filters."""
    if confidences is not None and len(confidences) == 0:
        return 0
    if species is not None and len(species) == 0:
        return 0

    base = _filtered_classification_subquery(
        confidences=confidences or [],
        species=species,
        animal_present_only=animal_present_only,
    ).subquery()
    count_stmt = select(func.count()).select_from(base)
    return int(session.exec(count_stmt).scalar() or 0)


def list_species_with_counts(
    session: Session,
    *,
    confidences: list[str] | None = None,
) -> list[tuple[str, int]]:
    """Return distinct species and their run counts, most frequent first.

    ``confidences`` is an optional list of confidence values to pre-filter by.
    """
    stmt: Any = (
        select(ClassificationResult.species_name, func.count(PipelineRun.id))
        .join(PipelineRun, PipelineRun.final_result_id == ClassificationResult.id)
        .group_by(ClassificationResult.species_name)
        .order_by(desc(func.count(PipelineRun.id)))
    )
    if confidences:
        stmt = stmt.where(ClassificationResult.confidence.in_(confidences))  # type: ignore[attr-defined]
    return [(name, int(count)) for name, count in session.exec(stmt).all()]


def get_classification_result(session: Session, result_id: int) -> ClassificationResult | None:
    """Get a classification result by ID."""
    stmt = select(ClassificationResult).where(ClassificationResult.id == result_id)
    return session.exec(stmt).scalars().first()


def aggregate_statistics(
    session: Session,
    *,
    confidences: list[str] | None = None,
    species: list[str] | None = None,
    animal_present_only: bool = False,
) -> StatisticsSummary:
    """Aggregate final-run classification data for the statistics tab."""
    if confidences is not None and len(confidences) == 0:
        return StatisticsSummary(0, 0, 0, 0, 0, 0, {}, {})
    if species is not None and len(species) == 0:
        return StatisticsSummary(0, 0, 0, 0, 0, 0, {}, {})

    base = _filtered_classification_subquery(
        confidences=confidences or [],
        species=species,
        animal_present_only=animal_present_only,
    ).subquery()

    total_runs = int(session.exec(select(func.count(base.c.video_filename)).select_from(base)).scalar() or 0)

    total_videos = int(session.exec(select(func.count(func.distinct(base.c.video_filename)))).scalar() or 0)

    distinct_species = int(session.exec(select(func.count(func.distinct(base.c.species_name)))).scalar() or 0)

    present, absent, unknown = count_animal_status(
        session,
        confidences=confidences,
        species=species,
    )

    species_rows = session.exec(
        select(base.c.species_name, func.count(base.c.video_filename))
        .group_by(base.c.species_name)
        .order_by(desc(func.count(base.c.video_filename))),
    ).all()
    species_counts = {name: int(count) for name, count in species_rows}

    confidence_rows = session.exec(
        select(base.c.confidence, func.count(base.c.video_filename))
        .group_by(base.c.confidence)
        .order_by(desc(func.count(base.c.video_filename))),
    ).all()
    confidence_counts = {level: int(count) for level, count in confidence_rows}

    return StatisticsSummary(
        total_runs=total_runs,
        total_videos=total_videos,
        distinct_species=distinct_species,
        animal_present_count=present,
        animal_absent_count=absent,
        animal_unknown_count=unknown,
        species_counts=species_counts,
        confidence_counts=confidence_counts,
    )
