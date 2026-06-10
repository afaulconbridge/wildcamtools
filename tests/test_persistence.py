"""Tests for the persistence layer."""

import tempfile
from pathlib import Path

import pytest

from wildcamtools.lib.ai.pipeline import (
    BatchResult,
    ExtractedFrame,
    PipelineOutcome,
)
from wildcamtools.lib.ai.pipeline_config import (
    AiPipelineConfig,
    FrameExtractorConfig,
    FrameSelectorConfig,
    ImageBatchQueryConfig,
    LlmConfig,
    ReconcilerConfig,
)
from wildcamtools.lib.ai.types import Backend, ConfidenceLevel, RichResult
from wildcamtools.lib.persistence.database import create_engine_and_tables, get_session
from wildcamtools.lib.persistence.models import (
    ClassificationResult,
    Video,
)
from wildcamtools.lib.persistence.repository import (
    get_classification_result,
    get_pipeline_run,
    get_pipeline_runs_by_video,
    save_pipeline_run,
)
from wildcamtools.lib.stats import Colourspace, VideoStats


@pytest.fixture
def db_engine():
    """Create an in-memory SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        connection_string = f"sqlite:///{db_path}"
        engine = create_engine_and_tables(connection_string)
        yield engine


@pytest.fixture
def sample_video_stats() -> VideoStats:
    """Sample video statistics for testing."""
    return VideoStats(
        fps=30.0,
        frame_count=100,
        x=1920,
        y=1080,
        colourspace=Colourspace.RGB,
    )


@pytest.fixture
def sample_rich_result() -> RichResult:
    """Sample AI classification result for testing."""
    return RichResult(
        is_animal_present=True,
        is_animal_unknown=False,
        defining_features="short fur, pointy ears",
        species_name="Red Fox",
        confidence=ConfidenceLevel.HIGH,
    )


@pytest.fixture
def sample_pipeline_config() -> AiPipelineConfig:
    """Sample pipeline configuration for testing."""
    return AiPipelineConfig(
        frame_selector=FrameSelectorConfig(selector_type="fps_rescaling", fps=5.0),
        frame_extractor=FrameExtractorConfig(extractor_type="rescaled", resolution=(640, 360)),
        llm=LlmConfig(backend=Backend.OLLAMA, model="test-model"),
        query=ImageBatchQueryConfig(query_type="llm", prompt="Test prompt"),
        reconciler=ReconcilerConfig(reconciler_type="majority"),
    )


def test_save_pipeline_run(db_engine, sample_video_stats, sample_rich_result, sample_pipeline_config):
    """Test saving a pipeline run to the database."""
    outcome = PipelineOutcome(
        result=sample_rich_result,
        stats=sample_video_stats,
        batches=[],
    )

    video_path = Path("/test/videos/fox_video.mp4")

    with get_session(db_engine) as session:
        run = save_pipeline_run(session, video_path, sample_pipeline_config, outcome)
        session.commit()

        assert run.id is not None
        assert run.video_filename == "/test/videos/fox_video.mp4"
        assert run.final_result is not None
        assert run.final_result.species_name == "Red Fox"


def test_videostat_deduplication(db_engine, sample_video_stats, sample_rich_result, sample_pipeline_config):
    """Test that identical video stats are deduplicated."""
    outcome = PipelineOutcome(
        result=sample_rich_result,
        stats=sample_video_stats,
        batches=[],
    )

    video_path_1 = Path("/test/videos/video1.mp4")
    video_path_2 = Path("/test/videos/video2.mp4")

    with get_session(db_engine) as session:
        run1 = save_pipeline_run(session, video_path_1, sample_pipeline_config, outcome)
        run2 = save_pipeline_run(session, video_path_2, sample_pipeline_config, outcome)
        session.commit()

        assert run1.video_filename == "/test/videos/video1.mp4"
        assert run2.video_filename == "/test/videos/video2.mp4"

        video1_stat = session.get(Video, "/test/videos/video1.mp4").stat
        video2_stat = session.get(Video, "/test/videos/video2.mp4").stat

        assert video1_stat.id == video2_stat.id


def test_classification_result_deduplication(db_engine, sample_video_stats, sample_rich_result, sample_pipeline_config):
    """Test that identical classification results are deduplicated."""
    outcome = PipelineOutcome(
        result=sample_rich_result,
        stats=sample_video_stats,
        batches=[],
    )

    video_path_1 = Path("/test/videos/video1.mp4")
    video_path_2 = Path("/test/videos/video2.mp4")

    with get_session(db_engine) as session:
        run1 = save_pipeline_run(session, video_path_1, sample_pipeline_config, outcome)
        run2 = save_pipeline_run(session, video_path_2, sample_pipeline_config, outcome)
        session.commit()

        assert run1.final_result_id == run2.final_result_id


def test_save_pipeline_run_with_batches(db_engine, sample_video_stats, sample_rich_result, sample_pipeline_config):
    """Test saving a pipeline run with batch results."""
    batch_result = BatchResult(
        selected_frames=[
            ExtractedFrame(path=Path("/test/frames/frame_00001.jpg"), frame_no=1),
            ExtractedFrame(path=Path("/test/frames/frame_00005.jpg"), frame_no=5),
        ],
        result=sample_rich_result,
    )

    outcome = PipelineOutcome(
        result=sample_rich_result,
        stats=sample_video_stats,
        batches=[batch_result],
    )

    video_path = Path("/test/videos/batch_video.mp4")

    with get_session(db_engine) as session:
        run = save_pipeline_run(session, video_path, sample_pipeline_config, outcome)
        session.commit()

        assert len(run.batches) == 1
        batch = run.batches[0]
        assert batch.result is not None
        assert batch.result.species_name == "Red Fox"


def test_get_pipeline_run(db_engine, sample_video_stats, sample_rich_result, sample_pipeline_config):
    """Test retrieving a pipeline run by ID."""
    outcome = PipelineOutcome(
        result=sample_rich_result,
        stats=sample_video_stats,
        batches=[],
    )

    video_path = Path("/test/videos/retrieval_test.mp4")

    with get_session(db_engine) as session:
        run = save_pipeline_run(session, video_path, sample_pipeline_config, outcome)
        session.commit()

        retrieved = get_pipeline_run(session, run.id)
        assert retrieved is not None
        assert retrieved.id == run.id
        assert retrieved.video_filename == "/test/videos/retrieval_test.mp4"


def test_get_pipeline_runs_by_video(db_engine, sample_video_stats, sample_rich_result, sample_pipeline_config):
    """Test retrieving all pipeline runs for a specific video."""
    outcome = PipelineOutcome(
        result=sample_rich_result,
        stats=sample_video_stats,
        batches=[],
    )

    video_path = Path("/test/videos/multi_run_video.mp4")

    with get_session(db_engine) as session:
        run1 = save_pipeline_run(session, video_path, sample_pipeline_config, outcome)
        run2 = save_pipeline_run(session, video_path, sample_pipeline_config, outcome)
        session.commit()

        runs = get_pipeline_runs_by_video(session, "/test/videos/multi_run_video.mp4")
        assert len(runs) == 2
        run_ids = {r.id for r in runs}
        assert run1.id in run_ids
        assert run2.id in run_ids


def test_classification_result_unique_constraint(
    db_engine, sample_video_stats, sample_rich_result, sample_pipeline_config
):
    """Test that the unique constraint on ClassificationResult works correctly."""
    outcome = PipelineOutcome(
        result=sample_rich_result,
        stats=sample_video_stats,
        batches=[],
    )

    video_path = Path("/test/videos/constraint_test.mp4")

    with get_session(db_engine) as session:
        run1 = save_pipeline_run(session, video_path, sample_pipeline_config, outcome)
        session.commit()

        result1 = get_classification_result(session, run1.final_result_id)
        assert result1 is not None

        stmt = (
            pytest
            .importorskip("sqlalchemy")
            .select(pytest.importorskip("sqlalchemy").func.count(ClassificationResult.id))
            .where(
                ClassificationResult.species_name == "Red Fox",
                ClassificationResult.confidence == "high",
                ClassificationResult.is_animal_present.is_(True),
                ClassificationResult.is_animal_unknown.is_(False),
            )
        )
        count_result = session.exec(stmt)
        count = count_result.scalar()
        assert count == 1
