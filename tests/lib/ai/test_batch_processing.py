"""Tests for the batch processing functionality."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from wildcamtools.cli import app
from wildcamtools.lib.ai.batch_processing import (
    BatchPipelineOutput,
    compute_output_path,
    discover_video_files,
)
from wildcamtools.lib.ai.pipeline import (
    CombinedBatchResult,
    CombinedPipelineOutcome,
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
from wildcamtools.lib.ai.types import Backend, BatchDescription, ConfidenceLevel, RichResult
from wildcamtools.lib.stats import Colourspace, VideoStats

runner = CliRunner()


class TestComputeOutputPath:
    """Tests for compute_output_path function."""

    def test_nested_directory(self, tmp_path: Path) -> None:
        """Test path computation for nested directory structure."""
        video_dir = tmp_path / "videos"
        output_dir = tmp_path / "results"
        video_path = video_dir / "2023" / "08" / "video.mp4"

        result = compute_output_path(video_path, video_dir, output_dir)

        assert result == output_dir / "2023" / "08" / "video.json"

    def test_root_directory(self, tmp_path: Path) -> None:
        """Test path computation for video in root directory."""
        video_dir = tmp_path / "videos"
        output_dir = tmp_path / "results"
        video_path = video_dir / "video.mp4"

        result = compute_output_path(video_path, video_dir, output_dir)

        assert result == output_dir / "video.json"

    def test_different_extension(self, tmp_path: Path) -> None:
        """Test path computation with different video extension."""
        video_dir = tmp_path / "videos"
        output_dir = tmp_path / "results"
        video_path = video_dir / "video.MP4"

        result = compute_output_path(video_path, video_dir, output_dir)

        assert result == output_dir / "video.json"


class TestDiscoverVideoFiles:
    """Tests for discover_video_files function."""

    def test_recursive_discovery(self, tmp_path: Path) -> None:
        """Test recursive video file discovery."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "2023").mkdir()
        (video_dir / "2023" / "08").mkdir()

        video1 = video_dir / "video1.mp4"
        video2 = video_dir / "2023" / "video2.mp4"
        video3 = video_dir / "2023" / "08" / "video3.MP4"

        video1.write_text("fake")
        video2.write_text("fake")
        video3.write_text("fake")

        result = discover_video_files(video_dir, recursive=True)

        assert len(result) == 3
        assert video1 in result
        assert video2 in result
        assert video3 in result

    def test_non_recursive_discovery(self, tmp_path: Path) -> None:
        """Test non-recursive video file discovery."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "2023").mkdir()

        video1 = video_dir / "video1.mp4"
        video2 = video_dir / "2023" / "video2.mp4"

        video1.write_text("fake")
        video2.write_text("fake")

        result = discover_video_files(video_dir, recursive=False)

        assert len(result) == 1
        assert result[0] == video1

    def test_sorted_output(self, tmp_path: Path) -> None:
        """Test that results are sorted by path."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()

        video_c = video_dir / "c.mp4"
        video_a = video_dir / "a.mp4"
        video_b = video_dir / "b.mp4"

        video_c.write_text("fake")
        video_a.write_text("fake")
        video_b.write_text("fake")

        result = discover_video_files(video_dir, recursive=True)

        assert result == [video_a, video_b, video_c]

    def test_no_videos(self, tmp_path: Path) -> None:
        """Test discovery with no video files."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()

        result = discover_video_files(video_dir, recursive=True)

        assert result == []


class TestRunBatchCommand:
    """Tests for the run_batch CLI command."""

    def test_run_batch_missing_config(self, tmp_path: Path) -> None:
        """Test error handling for missing config file."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        config = tmp_path / "config.json"

        result = runner.invoke(
            app,
            ["ai", "run-batch", str(config), str(video_dir), str(output_dir)],
        )

        assert result.exit_code == 1
        assert "Error: Config file not found" in result.stderr

    def test_run_batch_missing_video_dir(self, tmp_path: Path) -> None:
        """Test error handling for missing video directory."""
        config = tmp_path / "config.json"
        config.write_text(
            AiPipelineConfig(
                frame_selector=FrameSelectorConfig(),
                frame_extractor=FrameExtractorConfig(),
                query=ImageBatchQueryConfig(
                    query_type="llm",
                    prompt="test",
                    llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
                ),
                reconciler=ReconcilerConfig(),
            ).model_dump_json(),
        )
        video_dir = tmp_path / "videos"
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            ["ai", "run-batch", str(config), str(video_dir), str(output_dir)],
        )

        assert result.exit_code == 1
        assert "Error: Video directory not found" in result.stderr

    def test_run_batch_no_videos(self, tmp_path: Path) -> None:
        """Test handling when no video files found."""
        config = tmp_path / "config.json"
        config.write_text(
            AiPipelineConfig(
                frame_selector=FrameSelectorConfig(),
                frame_extractor=FrameExtractorConfig(),
                query=ImageBatchQueryConfig(
                    query_type="llm",
                    prompt="test",
                    llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
                ),
                reconciler=ReconcilerConfig(),
            ).model_dump_json(),
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            ["ai", "run-batch", str(config), str(video_dir), str(output_dir)],
        )

        assert result.exit_code == 1
        assert "No video files found" in result.stderr

    def test_run_batch_creates_output_dir(self, tmp_path: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        config = tmp_path / "config.json"
        config.write_text(
            AiPipelineConfig(
                frame_selector=FrameSelectorConfig(),
                frame_extractor=FrameExtractorConfig(),
                query=ImageBatchQueryConfig(
                    query_type="llm",
                    prompt="test",
                    llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
                ),
                reconciler=ReconcilerConfig(),
            ).model_dump_json(),
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        output_dir = tmp_path / "results"

        result = runner.invoke(
            app,
            ["ai", "run-batch", str(config), str(video_dir), str(output_dir)],
        )

        # Should fail because no videos, but output dir should be created
        assert result.exit_code == 1
        assert output_dir.exists()

    def test_run_batch_invalid_max_workers(self, tmp_path: Path) -> None:
        """Test error handling for invalid max_workers."""
        config = tmp_path / "config.json"
        config.write_text(
            AiPipelineConfig(
                frame_selector=FrameSelectorConfig(),
                frame_extractor=FrameExtractorConfig(),
                query=ImageBatchQueryConfig(
                    query_type="llm",
                    prompt="test",
                    llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
                ),
                reconciler=ReconcilerConfig(),
            ).model_dump_json(),
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "ai",
                "run-batch",
                str(config),
                str(video_dir),
                str(output_dir),
                "-w",
                "0",
            ],
        )

        assert result.exit_code == 1
        assert "--max-workers must be at least 1" in result.stderr

    def test_run_batch_short_options(self, tmp_path: Path) -> None:
        """Test that short options work."""
        config = tmp_path / "config.json"
        config.write_text(
            AiPipelineConfig(
                frame_selector=FrameSelectorConfig(),
                frame_extractor=FrameExtractorConfig(),
                query=ImageBatchQueryConfig(
                    query_type="llm",
                    prompt="test",
                    llm=LlmConfig(backend=Backend.OLLAMA, model="test"),
                ),
                reconciler=ReconcilerConfig(),
            ).model_dump_json(),
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "ai",
                "run-batch",
                str(config),
                str(video_dir),
                str(output_dir),
                "-w",
                "2",
                "-r",
            ],
        )

        # Should fail because no videos, but options should be parsed
        assert result.exit_code == 1


class TestBatchPipelineOutput:
    """Tests for BatchPipelineOutput model."""

    def test_serialization(self) -> None:
        """Test that BatchPipelineOutput can be serialized to JSON."""
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
        outcome = PipelineOutcome(
            result=RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test",
                species_name="Test Species",
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

        output = BatchPipelineOutput(config=config, outcome=outcome)
        json_str = output.model_dump_json(indent=2)

        # Should be valid JSON
        data = json.loads(json_str)
        assert "config" in data
        assert "outcome" in data


class TestRunPipelineWorkerDispatch:
    """Tests for the worker writing pipeline JSON output."""

    @staticmethod
    def _make_config(has_description: bool = False) -> AiPipelineConfig:
        if has_description:
            return AiPipelineConfig(
                frame_selector=FrameSelectorConfig(selector_type="fps_rescaling", fps=1.0),
                frame_extractor=FrameExtractorConfig(),
                query=ImageBatchQueryConfig(
                    query_type="llm",
                    prompt="what species?",
                    llm=LlmConfig(backend=Backend.OLLAMA, model="test-model"),
                ),
                reconciler=ReconcilerConfig(reconciler_type="majority"),
                description={
                    "llm": {"backend": "ollama", "model": "test-model"},
                    "description_prompt": "describe the scene",
                },
            )
        return AiPipelineConfig(
            frame_selector=FrameSelectorConfig(selector_type="fps_rescaling", fps=1.0),
            frame_extractor=FrameExtractorConfig(),
            query=ImageBatchQueryConfig(
                query_type="llm",
                prompt="what species?",
                llm=LlmConfig(backend=Backend.OLLAMA, model="test-model"),
            ),
            reconciler=ReconcilerConfig(reconciler_type="majority"),
        )

    @staticmethod
    def _with_pipeline(config: AiPipelineConfig, pipeline_mock: MagicMock) -> AiPipelineConfig:
        """Return a copy of the config whose create_pipeline returns the given mock."""

        class _TestConfig(AiPipelineConfig):
            def create_pipeline(self) -> MagicMock:  # type: ignore[override]
                return pipeline_mock

        return _TestConfig.model_validate(config.model_dump())

    def test_worker_writes_rich_result_json(self, tmp_path: Path) -> None:
        from wildcamtools.lib.ai.batch_processing import _run_pipeline_worker

        config = self._make_config(has_description=False)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"x")
        output = tmp_path / "out.json"

        outcome = PipelineOutcome[RichResult](
            result=RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test",
                species_name="Red Fox",
                confidence=ConfidenceLevel.HIGH,
            ),
            stats=VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB),
        )
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = outcome
        config = self._with_pipeline(config, mock_pipeline)

        result = _run_pipeline_worker(str(video), str(output), config)

        assert result.error is None
        data = json.loads(output.read_text())
        assert "outcome" in data
        assert data["outcome"]["result"]["species_name"] == "Red Fox"

    def test_worker_writes_combined_json(self, tmp_path: Path) -> None:
        from wildcamtools.lib.ai.batch_processing import _run_pipeline_worker

        config = self._make_config(has_description=True)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"x")
        output = tmp_path / "out.json"

        outcome = CombinedPipelineOutcome(
            result=RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test",
                species_name="Red Fox",
                confidence=ConfidenceLevel.HIGH,
            ),
            description=BatchDescription(description="A badger forages in the undergrowth."),
            stats=VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB),
            batches=[
                CombinedBatchResult(
                    selected_frames=[ExtractedFrame(frame_no=0)],
                    classification=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test",
                        species_name="Red Fox",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    description=BatchDescription(description="A badger forages."),
                ),
            ],
        )
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = outcome
        config = self._with_pipeline(config, mock_pipeline)

        result = _run_pipeline_worker(str(video), str(output), config)

        assert result.error is None
        data = json.loads(output.read_text())
        assert "outcome" in data
        assert "description" in data["outcome"]
        assert data["outcome"]["description"]["description"] == "A badger forages in the undergrowth."
        assert len(data["outcome"]["batches"]) == 1
