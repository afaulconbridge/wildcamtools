"""Tests for the db CLI commands."""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wildcamtools.cli import app
from wildcamtools.cli.db import PipelineRunOutput
from wildcamtools.lib.ai.pipeline import RichResultPipelineOutcome
from wildcamtools.lib.ai.pipeline_config import (
    AiPipelineConfig,
    FrameExtractorConfig,
    FrameSelectorConfig,
    ImageBatchQueryConfig,
    LlmConfig,
    ReconcilerConfig,
)
from wildcamtools.lib.ai.types import Backend, ConfidenceLevel, RichResult
from wildcamtools.lib.stats import Colourspace, VideoStats

runner = CliRunner()


@pytest.fixture
def sample_result_json() -> str:
    """Sample result JSON string matching the PipelineRunOutput format."""
    config = AiPipelineConfig(
        frame_selector=FrameSelectorConfig(selector_type="fps_rescaling", fps=5.0),
        frame_extractor=FrameExtractorConfig(extractor_type="rescaled", resolution=(640, 360)),
        query=ImageBatchQueryConfig(
            query_type="llm",
            prompt="Test prompt",
            llm=LlmConfig(backend=Backend.OLLAMA, model="test-model"),
        ),
        reconciler=ReconcilerConfig(reconciler_type="majority"),
    )
    outcome = RichResultPipelineOutcome(
        result=RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="short fur, pointy ears",
            species_name="Red Fox",
            confidence=ConfidenceLevel.HIGH,
        ),
        stats=VideoStats(
            fps=30.0,
            frame_count=100,
            x=1920,
            y=1080,
            colourspace=Colourspace.RGB,
        ),
        batches=[],
    )
    output = PipelineRunOutput(config=config, outcome=outcome)
    return output.model_dump_json()


@pytest.fixture
def temp_result_file(sample_result_json: str) -> Path:
    """Create a temporary JSON result file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(sample_result_json)
        return Path(f.name)


@pytest.fixture
def temp_video_file() -> Path:
    """Create a temporary video file placeholder."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mp4", delete=False) as f:
        f.write("fake video content")
        return Path(f.name)


def test_db_import_result(temp_result_file: Path, temp_video_file: Path):
    """Test importing a result file into the database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        result = runner.invoke(
            app,
            ["db", "import", str(temp_result_file), str(temp_video_file), "-d", str(db_path)],
        )

        assert result.exit_code == 0
        assert "Successfully imported pipeline run" in result.stdout
        assert db_path.exists()


def test_db_import_result_missing_file(temp_video_file: Path):
    """Test error handling for missing input file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        missing_file = Path(tmpdir) / "nonexistent.json"

        result = runner.invoke(
            app,
            ["db", "import", str(missing_file), str(temp_video_file), "-d", str(db_path)],
        )

        assert result.exit_code == 1
        assert "Error: Input path not found" in result.stderr


def test_db_import_result_missing_video(temp_result_file: Path):
    """Test error handling for missing video file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        missing_video = Path(tmpdir) / "nonexistent.mp4"

        result = runner.invoke(
            app,
            ["db", "import", str(temp_result_file), str(missing_video), "-d", str(db_path)],
        )

        assert result.exit_code == 1
        assert "Error: Video path not found" in result.stderr


def test_db_import_result_invalid_json(temp_video_file: Path):
    """Test error handling for invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        invalid_json = Path(tmpdir) / "invalid.json"
        invalid_json.write_text("not valid json")

        result = runner.invoke(
            app,
            ["db", "import", str(invalid_json), str(temp_video_file), "-d", str(db_path)],
        )

        assert result.exit_code == 1
        assert "Error: Failed to parse JSON" in result.stderr


def test_db_import_result_directory_mode(temp_result_file: Path, temp_video_file: Path):
    """Test importing multiple files using directory mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_dir = tmpdir / "videos"
        result_dir = tmpdir / "results"
        db_path = tmpdir / "test.db"

        video_dir.mkdir()
        result_dir.mkdir()

        video1 = video_dir / "video1.mp4"
        video2 = video_dir / "video2.mp4"
        result1 = result_dir / "video1.json"
        result2 = result_dir / "video2.json"

        video1.write_text("fake video 1")
        video2.write_text("fake video 2")

        result_data = temp_result_file.read_text()
        result1.write_text(result_data)
        result2.write_text(result_data)

        result = runner.invoke(
            app,
            ["db", "import", str(result_dir), str(video_dir), "-d", str(db_path)],
        )

        assert result.exit_code == 0
        assert "Import completed: 2 successful, 0 failed" in result.stdout
        assert db_path.exists()


def test_db_import_result_directory_mode_mismatch(temp_result_file: Path, temp_video_file: Path):
    """Test directory mode with mismatched files (should warn and skip)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_dir = tmpdir / "videos"
        result_dir = tmpdir / "results"
        db_path = tmpdir / "test.db"

        video_dir.mkdir()
        result_dir.mkdir()

        video1 = video_dir / "video1.mp4"
        result1 = result_dir / "video1.json"
        result2 = result_dir / "video2.json"

        video1.write_text("fake video 1")
        result_data = temp_result_file.read_text()
        result1.write_text(result_data)
        result2.write_text(result_data)

        result = runner.invoke(
            app,
            ["db", "import", str(result_dir), str(video_dir), "-d", str(db_path)],
        )

        assert result.exit_code == 0
        assert "Warning: No matching video file for result" in result.stdout
        assert "Import completed: 1 successful" in result.stdout


def test_db_import_result_directory_mode_mixed_types(temp_video_file: Path):
    """Test error when one path is directory and other is file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_dir = tmpdir / "videos"
        db_path = tmpdir / "test.db"

        video_dir.mkdir()

        result = runner.invoke(
            app,
            ["db", "import", str(temp_video_file), str(video_dir), "-d", str(db_path)],
        )

        assert result.exit_code == 1
        assert "Both paths must be files or both must be directories" in result.stderr


def test_db_import_result_directory_mode_no_matches(temp_video_file: Path):
    """Test directory mode when no matching pairs are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_dir = tmpdir / "videos"
        result_dir = tmpdir / "results"
        db_path = tmpdir / "test.db"

        video_dir.mkdir()
        result_dir.mkdir()

        (video_dir / "video1.mp4").write_text("fake video")
        (result_dir / "other.json").write_text('{"config":{}, "outcome":{}}')

        result = runner.invoke(
            app,
            ["db", "import", str(result_dir), str(video_dir), "-d", str(db_path)],
        )

        assert result.exit_code == 1
        assert "No matching video-result pairs found" in result.stderr


def test_db_import_with_filename_date_format(temp_result_file: Path, tmp_path: Path) -> None:
    """Importing with --filename-date-format should populate recorded_at."""
    db_path = tmp_path / "test.db"
    video_path = tmp_path / "20230816202116_VD_00001.mp4"
    video_path.write_text("fake video")

    result = runner.invoke(
        app,
        [
            "db",
            "import",
            str(temp_result_file),
            str(video_path),
            "-d",
            str(db_path),
            "--filename-date-format",
            "%Y%m%d%H%M%S",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Successfully imported pipeline run" in result.stdout

    from sqlmodel import Session, create_engine

    from wildcamtools.lib.persistence.models import Video

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        video = session.get(Video, str(video_path.absolute()))
        assert video is not None
        assert video.recorded_at is not None
        assert video.recorded_at.year == 2023
        assert video.recorded_at.month == 8
        assert video.recorded_at.day == 16
        assert video.recorded_at.hour == 20
        assert video.recorded_at.minute == 21
        assert video.recorded_at.second == 16


def test_db_import_without_format_leaves_recorded_at_null(temp_result_file: Path, tmp_path: Path) -> None:
    """Importing without --filename-date-format should leave recorded_at as None."""
    db_path = tmp_path / "test.db"
    video_path = tmp_path / "no_timestamp.mp4"
    video_path.write_text("fake video")

    result = runner.invoke(
        app,
        [
            "db",
            "import",
            str(temp_result_file),
            str(video_path),
            "-d",
            str(db_path),
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr

    from sqlmodel import Session, create_engine

    from wildcamtools.lib.persistence.models import Video

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        video = session.get(Video, str(video_path.absolute()))
        assert video is not None
        assert video.recorded_at is None


def test_db_import_with_non_matching_format(temp_result_file: Path, tmp_path: Path) -> None:
    """A format that does not match the filename should leave recorded_at as None."""
    db_path = tmp_path / "test.db"
    video_path = tmp_path / "no_timestamp_here.mp4"
    video_path.write_text("fake video")

    result = runner.invoke(
        app,
        [
            "db",
            "import",
            str(temp_result_file),
            str(video_path),
            "-d",
            str(db_path),
            "--filename-date-format",
            "%Y%m%d%H%M%S",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr

    from sqlmodel import Session, create_engine

    from wildcamtools.lib.persistence.models import Video

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        video = session.get(Video, str(video_path.absolute()))
        assert video is not None
        assert video.recorded_at is None
