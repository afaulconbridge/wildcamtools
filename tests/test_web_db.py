"""Tests for the web/db Streamlit UI and its repository helpers."""

import tempfile
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
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
    PipelineRun,
)
from wildcamtools.lib.persistence.repository import (
    aggregate_statistics,
    count_pipeline_runs_filtered,
    list_pipeline_runs_filtered,
    list_recent_pipeline_runs,
    list_runs_for_video,
    list_species_with_counts,
    save_pipeline_run,
)
from wildcamtools.lib.stats import Colourspace, VideoStats
from wildcamtools.web.lib.thumbnails import (
    PLACEHOLDER_HEIGHT,
    PLACEHOLDER_WIDTH,
    extract_thumbnail,
    thumbnail_or_placeholder,
)


def _make_synthetic_mp4(path: Path, width: int = 320, height: int = 240, frame_count: int = 30, fps: int = 30) -> None:
    """Write a tiny synthetic H.264 MP4 with solid frames."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    rng = np.random.default_rng(0)
    for _i in range(frame_count):
        frame_array = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture
def db_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine = create_engine_and_tables(f"sqlite:///{db_path}")
        yield engine


@pytest.fixture
def sample_video_stats() -> VideoStats:
    return VideoStats(
        fps=30.0,
        frame_count=100,
        x=1920,
        y=1080,
        colourspace=Colourspace.RGB,
    )


@pytest.fixture
def sample_pipeline_config() -> AiPipelineConfig:
    return AiPipelineConfig(
        frame_selector=FrameSelectorConfig(selector_type="fps_rescaling", fps=5.0),
        frame_extractor=FrameExtractorConfig(extractor_type="rescaled", resolution=(640, 360)),
        llm=LlmConfig(backend=Backend.OLLAMA, model="test-model"),
        query=ImageBatchQueryConfig(
            query_type="llm",
            prompt="Test prompt",
            llm=LlmConfig(backend=Backend.OLLAMA, model="test-model"),
        ),
        reconciler=ReconcilerConfig(reconciler_type="majority"),
    )


def _save_run(
    session,
    video_path: str,
    config: AiPipelineConfig,
    result: RichResult,
    timestamp: datetime | None = None,
    recorded_at: datetime | None = None,
) -> PipelineRun:
    """Save a run via the production repository, which dedupes by design."""
    stats = VideoStats(fps=30.0, frame_count=100, x=1920, y=1080, colourspace=Colourspace.RGB)
    outcome = PipelineOutcome(result=result, stats=stats, batches=[])
    run = save_pipeline_run(session, Path(video_path), config, outcome, recorded_at=recorded_at)
    if timestamp is not None:
        run.timestamp = timestamp
        session.add(run)
        session.flush()
    return run


def test_extract_thumbnail_returns_jpeg(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _make_synthetic_mp4(video_path)

    thumbnail = extract_thumbnail(video_path, max_width=160)

    assert thumbnail is not None
    assert len(thumbnail) > 0
    assert thumbnail[:2] == b"\xff\xd8"
    np_arr = np.frombuffer(thumbnail, dtype=np.uint8)
    decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    assert decoded.shape[1] <= 160


def test_extract_thumbnail_missing_file_returns_none(tmp_path: Path) -> None:
    assert extract_thumbnail(tmp_path / "missing.mp4") is None


def test_thumbnail_or_placeholder_returns_jpeg(tmp_path: Path) -> None:
    video_path = tmp_path / "real.mp4"
    _make_synthetic_mp4(video_path)
    data = thumbnail_or_placeholder(video_path, max_width=200)
    assert data[:2] == b"\xff\xd8"

    placeholder = thumbnail_or_placeholder(tmp_path / "missing.mp4")
    assert placeholder[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(placeholder, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[0] == PLACEHOLDER_HEIGHT
    assert decoded.shape[1] == PLACEHOLDER_WIDTH


def test_placeholder_dimensions_constants() -> None:
    assert PLACEHOLDER_WIDTH == 320
    assert PLACEHOLDER_HEIGHT == 180


def test_list_species_with_counts_orders_by_frequency(db_engine, sample_pipeline_config) -> None:
    species_repeats = [
        ("Red Fox", ConfidenceLevel.HIGH, True, False),
        ("Red Fox", ConfidenceLevel.HIGH, True, False),
        ("Red Fox", ConfidenceLevel.HIGH, True, False),
        ("Owl", ConfidenceLevel.MEDIUM, True, False),
        ("Owl", ConfidenceLevel.MEDIUM, True, False),
        ("Deer", ConfidenceLevel.LOW, True, False),
    ]
    with get_session(db_engine) as session:
        for i, (species, confidence, present, unknown) in enumerate(species_repeats):
            _save_run(
                session,
                f"/videos/video_{i}.mp4",
                sample_pipeline_config,
                RichResult(
                    is_animal_present=present,
                    is_animal_unknown=unknown,
                    defining_features="x",
                    species_name=species,
                    confidence=confidence,
                ),
            )
        session.commit()
        with get_session(db_engine) as read_session:
            counts = list_species_with_counts(read_session)

    assert [name for name, _ in counts] == ["Red Fox", "Owl", "Deer"]
    assert dict(counts) == {"Red Fox": 3, "Owl": 2, "Deer": 1}


def test_list_pipeline_runs_filtered_by_confidence(db_engine, sample_pipeline_config) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, 0)
    with get_session(db_engine) as session:
        for i, confidence in enumerate([ConfidenceLevel.HIGH, ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]):
            _save_run(
                session,
                f"/videos/v_{i}.mp4",
                sample_pipeline_config,
                RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="x",
                    species_name="Red Fox",
                    confidence=confidence,
                ),
                timestamp=timestamp.replace(hour=i),
            )
        session.commit()

        with get_session(db_engine) as read_session:
            high_only = list_pipeline_runs_filtered(read_session, confidences=["high"])
            assert [r.final_result.confidence for r in high_only] == ["high"]

            empty_filter = list_pipeline_runs_filtered(read_session, confidences=[])
            assert empty_filter == []

            all_runs = list_pipeline_runs_filtered(read_session)
            assert len(all_runs) == 3

            count_all = count_pipeline_runs_filtered(read_session)
            assert count_all == 3


def test_list_pipeline_runs_filtered_by_species_and_animal_present(db_engine, sample_pipeline_config) -> None:
    cases = [
        ("Red Fox", ConfidenceLevel.HIGH, True, False),
        ("Red Fox", ConfidenceLevel.HIGH, False, False),
        ("Owl", ConfidenceLevel.HIGH, True, False),
    ]
    with get_session(db_engine) as session:
        for i, (species, conf, present, unknown) in enumerate(cases):
            _save_run(
                session,
                f"/videos/s_{i}.mp4",
                sample_pipeline_config,
                RichResult(
                    is_animal_present=present,
                    is_animal_unknown=unknown,
                    defining_features="x",
                    species_name=species,
                    confidence=conf,
                ),
            )
        session.commit()

        with get_session(db_engine) as read_session:
            only_fox = list_pipeline_runs_filtered(read_session, species=["Red Fox"])
            assert {r.final_result.species_name for r in only_fox} == {"Red Fox"}

            present_only = list_pipeline_runs_filtered(read_session, animal_present_only=True)
            assert len(present_only) == 2
            assert all(r.final_result.is_animal_present for r in present_only)

            combined = list_pipeline_runs_filtered(
                read_session,
                species=["Red Fox"],
                animal_present_only=True,
            )
            assert len(combined) == 1
            assert combined[0].final_result.species_name == "Red Fox"
            assert combined[0].final_result.is_animal_present is True


def test_list_pipeline_runs_pagination(db_engine, sample_pipeline_config) -> None:
    with get_session(db_engine) as session:
        for i in range(7):
            _save_run(
                session,
                f"/videos/p_{i}.mp4",
                sample_pipeline_config,
                RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="x",
                    species_name="Red Fox",
                    confidence=ConfidenceLevel.HIGH,
                ),
                timestamp=datetime(2024, 1, 1, 0, 0, i),
            )
        session.commit()

        with get_session(db_engine) as read_session:
            page0 = list_pipeline_runs_filtered(read_session, limit=3, offset=0)
            page1 = list_pipeline_runs_filtered(read_session, limit=3, offset=3)
            page2 = list_pipeline_runs_filtered(read_session, limit=3, offset=6)
            assert len(page0) == 3
            assert len(page1) == 3
            assert len(page2) == 1
            total = count_pipeline_runs_filtered(read_session)
            assert total == 7


def test_aggregate_statistics(db_engine, sample_pipeline_config) -> None:
    cases = [
        ("Red Fox", ConfidenceLevel.HIGH, True, False),
        ("Red Fox", ConfidenceLevel.HIGH, True, False),
        ("Owl", ConfidenceLevel.LOW, False, True),
        ("Owl", ConfidenceLevel.MEDIUM, True, False),
        ("Deer", ConfidenceLevel.LOW, False, False),
    ]
    with get_session(db_engine) as session:
        for i, (species, conf, present, unknown) in enumerate(cases):
            _save_run(
                session,
                f"/videos/a_{i}.mp4",
                sample_pipeline_config,
                RichResult(
                    is_animal_present=present,
                    is_animal_unknown=unknown,
                    defining_features="x",
                    species_name=species,
                    confidence=conf,
                ),
            )
        session.commit()

        with get_session(db_engine) as read_session:
            stats = aggregate_statistics(read_session)

    assert stats.total_runs == 5
    assert stats.total_videos == 5
    assert stats.distinct_species == 3
    assert stats.animal_present_count == 3
    assert stats.animal_absent_count == 2
    assert stats.animal_unknown_count == 1
    assert stats.species_counts == {"Red Fox": 2, "Owl": 2, "Deer": 1}
    assert stats.confidence_counts == {"high": 2, "low": 2, "medium": 1}


def test_aggregate_statistics_with_filters(db_engine, sample_pipeline_config) -> None:
    cases = [
        ("Red Fox", ConfidenceLevel.HIGH, True, False),
        ("Owl", ConfidenceLevel.LOW, False, True),
    ]
    with get_session(db_engine) as session:
        for i, (species, conf, present, unknown) in enumerate(cases):
            _save_run(
                session,
                f"/videos/af_{i}.mp4",
                sample_pipeline_config,
                RichResult(
                    is_animal_present=present,
                    is_animal_unknown=unknown,
                    defining_features="x",
                    species_name=species,
                    confidence=conf,
                ),
            )
        session.commit()

        with get_session(db_engine) as read_session:
            present_only = aggregate_statistics(
                read_session,
                confidences=["high", "low"],
                animal_present_only=True,
            )
            assert present_only.total_runs == 1
            assert present_only.species_counts == {"Red Fox": 1}

            empty_filter = aggregate_statistics(read_session, confidences=[])
            assert empty_filter.total_runs == 0
            assert empty_filter.species_counts == {}


def test_aggregate_statistics_empty_db(db_engine) -> None:
    with get_session(db_engine) as session:
        stats = aggregate_statistics(session)
    assert stats.total_runs == 0
    assert stats.species_counts == {}
    assert stats.confidence_counts == {}


def test_list_runs_for_video_eager_loads_batches(db_engine, sample_pipeline_config) -> None:
    """Accessing ``run.batches[0].result`` after the query must not issue a SELECT."""
    outcome_batches = [
        BatchResult(
            selected_frames=[ExtractedFrame(path=Path("/f/1.jpg"), frame_no=1)],
            result=RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="x",
                species_name="Red Fox",
                confidence=ConfidenceLevel.HIGH,
            ),
        )
    ]
    stats = VideoStats(fps=30.0, frame_count=10, x=1280, y=720, colourspace=Colourspace.RGB)
    outcome = PipelineOutcome(
        result=RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="x",
            species_name="Red Fox",
            confidence=ConfidenceLevel.HIGH,
        ),
        stats=stats,
        batches=outcome_batches,
    )

    with get_session(db_engine) as session:
        save_pipeline_run(session, Path("/videos/eager.mp4"), sample_pipeline_config, outcome)
        session.commit()

        with get_session(db_engine) as read_session:
            runs = list_runs_for_video(read_session, "/videos/eager.mp4")
            assert len(runs) == 1
            assert runs[0].batches[0].result is not None
            assert runs[0].batches[0].result.species_name == "Red Fox"
            assert runs[0].final_result is not None


def test_list_recent_pipeline_runs_orders_by_timestamp(db_engine, sample_pipeline_config) -> None:
    with get_session(db_engine) as session:
        for i, hour in enumerate([1, 5, 3]):
            _save_run(
                session,
                f"/videos/r_{i}.mp4",
                sample_pipeline_config,
                RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="x",
                    species_name="Red Fox",
                    confidence=ConfidenceLevel.HIGH,
                ),
                timestamp=datetime(2024, 1, 1, hour, 0, 0),
            )
        session.commit()

        with get_session(db_engine) as read_session:
            recent = list_recent_pipeline_runs(read_session, limit=10)
            assert [r.video_filename for r in recent] == [
                "/videos/r_1.mp4",
                "/videos/r_2.mp4",
                "/videos/r_0.mp4",
            ]


def test_animal_status_count_independent_of_animal_present_filter(db_engine, sample_pipeline_config) -> None:
    """``count_animal_status`` must not apply ``animal_present_only`` so the
    absent count isn't masked when the user toggles that filter on.
    """
    cases = [
        ("Red Fox", True, False),
        ("Red Fox", False, False),
        ("Owl", True, False),
    ]
    with get_session(db_engine) as session:
        for i, (species, present, unknown) in enumerate(cases):
            _save_run(
                session,
                f"/videos/as_{i}.mp4",
                sample_pipeline_config,
                RichResult(
                    is_animal_present=present,
                    is_animal_unknown=unknown,
                    defining_features="x",
                    species_name=species,
                    confidence=ConfidenceLevel.HIGH,
                ),
            )
        session.commit()

        with get_session(db_engine) as read_session:
            summary = aggregate_statistics(read_session, animal_present_only=True)
            assert summary.total_runs == 2
            assert summary.animal_present_count == 2
            assert summary.animal_absent_count == 1
            assert summary.animal_unknown_count == 0


def test_format_run_caption_includes_recorded_at(db_engine, sample_pipeline_config) -> None:
    """The Browse caption should mention the recording time when known."""
    from wildcamtools.web.db import _format_run_caption

    recorded = datetime(2023, 8, 16, 20, 21, 16)
    with get_session(db_engine) as session:
        run = _save_run(
            session,
            "/videos/cap.mp4",
            sample_pipeline_config,
            RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="x",
                species_name="Red Fox",
                confidence=ConfidenceLevel.HIGH,
            ),
            recorded_at=recorded,
        )
        session.commit()
        target_id = run.id

    with get_session(db_engine) as read_session:
        runs = list_recent_pipeline_runs(read_session, limit=10)
        target = next(r for r in runs if r.id == target_id)
        caption = _format_run_caption(target)

    assert "recorded 2023-08-16 20:21:16" in caption
    assert "Red Fox" in caption


def test_format_run_caption_omits_recorded_at_when_unknown(db_engine, sample_pipeline_config) -> None:
    """The caption should not contain 'recorded' when no timestamp is set."""
    from wildcamtools.web.db import _format_run_caption

    with get_session(db_engine) as session:
        run = _save_run(
            session,
            "/videos/cap2.mp4",
            sample_pipeline_config,
            RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="x",
                species_name="Red Fox",
                confidence=ConfidenceLevel.HIGH,
            ),
        )
        session.commit()
        target_id = run.id

    with get_session(db_engine) as read_session:
        runs = list_recent_pipeline_runs(read_session, limit=10)
        target = next(r for r in runs if r.id == target_id)
        caption = _format_run_caption(target)

    assert "recorded " not in caption
