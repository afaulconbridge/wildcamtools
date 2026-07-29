import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from wildcamtools.cli.ai import app as ai_app
from wildcamtools.lib.ai import PipelineEvaluationResult, PipelineEvaluationSummary
from wildcamtools.lib.ai.types import ConfidenceLevel, ResultClassification, RichResult

runner = CliRunner()


@pytest.fixture
def sample_config_file(tmp_path: Path) -> Path:
    config = {
        "llm": {
            "model": "test-model",
            "backend": "ollama",
            "url": "http://localhost:8080/v1",
        },
        "query": {
            "prompt": "Test prompt",
        },
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    return config_file


@pytest.fixture
def sample_comparison_config_file(tmp_path: Path) -> Path:
    comparison_config = {
        "comparator_type": "exact",
    }
    config_file = tmp_path / "comparison_config.json"
    config_file.write_text(json.dumps(comparison_config))
    return config_file


@pytest.fixture
def sample_video_file(tmp_path: Path) -> Path:
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video content")
    return video_file


class TestRunCommand:
    def test_run_missing_config(self, sample_video_file: Path) -> None:
        result = runner.invoke(ai_app, ["run", "nonexistent.json", str(sample_video_file)])
        assert result.exit_code == 1
        assert "Error: Config file not found" in result.stderr or "Error: Config file not found" in result.stdout

    def test_run_missing_video(self, sample_config_file: Path) -> None:
        result = runner.invoke(ai_app, ["run", str(sample_config_file), "nonexistent.mp4"])
        assert result.exit_code == 1
        assert "Error: Video file not found" in result.stderr or "Error: Video file not found" in result.stdout

    def test_run_config_not_file(self, tmp_path: Path, sample_video_file: Path) -> None:
        result = runner.invoke(ai_app, ["run", str(tmp_path), str(sample_video_file)])
        assert result.exit_code == 1
        assert (
            "Error: Config path is not a file" in result.stderr or "Error: Config path is not a file" in result.stdout
        )

    def test_run_video_not_file(self, tmp_path: Path, sample_config_file: Path) -> None:
        result = runner.invoke(ai_app, ["run", str(sample_config_file), str(tmp_path)])
        assert result.exit_code == 1
        assert "Error: Video path is not a file" in result.stderr or "Error: Video path is not a file" in result.stdout

    def test_run_invalid_json_config(self, tmp_path: Path, sample_video_file: Path) -> None:
        invalid_config = tmp_path / "invalid.json"
        invalid_config.write_text("not valid json {{{")
        result = runner.invoke(ai_app, ["run", "-c", str(invalid_config), str(sample_video_file)])
        assert result.exit_code != 0

    def test_run_missing_required_fields(self, tmp_path: Path, sample_video_file: Path) -> None:
        incomplete_config = tmp_path / "incomplete.json"
        incomplete_config.write_text("{}")
        result = runner.invoke(ai_app, ["run", "-c", str(incomplete_config), str(sample_video_file)])
        assert result.exit_code != 0

    def test_run_with_output_file(self, sample_config_file: Path, sample_video_file: Path, tmp_path: Path) -> None:
        output_file = tmp_path / "output.json"
        result = runner.invoke(
            ai_app,
            ["run", str(sample_config_file), "--output", str(output_file), str(sample_video_file)],
        )
        assert result.exit_code != 0

    def test_run_without_output_file(self, sample_config_file: Path, sample_video_file: Path) -> None:
        result = runner.invoke(ai_app, ["run", str(sample_config_file), str(sample_video_file)])
        assert result.exit_code != 0

    def test_run_short_options(self, sample_config_file: Path, sample_video_file: Path, tmp_path: Path) -> None:
        output_file = tmp_path / "out.json"
        result = runner.invoke(ai_app, ["run", str(sample_config_file), "-o", str(output_file), str(sample_video_file)])
        # Short options still work
        assert result.exit_code != 0

    def test_run_success_with_mocked_pipeline(
        self,
        sample_config_file: Path,
        sample_video_file: Path,
        tmp_path: Path,
    ) -> None:
        from wildcamtools.lib.ai import AiPipelineConfig, PipelineOutcome
        from wildcamtools.lib.ai.types import ConfidenceLevel, RichResult
        from wildcamtools.lib.stats import Colourspace, VideoStats

        mock_result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test",
            species_name="test",
            confidence=ConfidenceLevel.HIGH,
        )
        mock_stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
        mock_outcome = PipelineOutcome(result=mock_result, stats=mock_stats, frame_ids=[[]])

        mock_config = MagicMock(spec=AiPipelineConfig)
        mock_config.llm = MagicMock()
        mock_config.query = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_outcome
        mock_config.create_pipeline.return_value = mock_pipeline

        with patch("wildcamtools.cli.ai.AiPipelineConfig.from_json", return_value=mock_config):
            output_file = tmp_path / "output.json"
            result = runner.invoke(
                ai_app,
                ["run", str(sample_config_file), "--output", str(output_file), str(sample_video_file)],
            )
            assert result.exit_code == 0
            assert output_file.exists()
            output_json = json.loads(output_file.read_text())
            assert "config" in output_json
            assert "outcome" in output_json
            assert output_json["outcome"]["result"]["species_name"] == "test"
            mock_pipeline.run.assert_called_once_with(sample_video_file)

    def test_run_success_prints_to_console(self, sample_config_file: Path, sample_video_file: Path) -> None:
        from wildcamtools.lib.ai import AiPipelineConfig, PipelineOutcome
        from wildcamtools.lib.ai.types import ConfidenceLevel, RichResult
        from wildcamtools.lib.stats import Colourspace, VideoStats

        mock_result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test",
            species_name="badger",
            confidence=ConfidenceLevel.HIGH,
        )
        mock_stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
        mock_outcome = PipelineOutcome(result=mock_result, stats=mock_stats, frame_ids=[[]])

        mock_config = MagicMock(spec=AiPipelineConfig)
        mock_config.llm = MagicMock()
        mock_config.query = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_outcome
        mock_config.create_pipeline.return_value = mock_pipeline

        with patch("wildcamtools.cli.ai.AiPipelineConfig.from_json", return_value=mock_config):
            result = runner.invoke(ai_app, ["run", str(sample_config_file), str(sample_video_file)])
            assert result.exit_code == 0
            output_json = json.loads(result.stdout)
            assert "config" in output_json
            assert "outcome" in output_json
            assert output_json["outcome"]["result"]["species_name"] == "badger"


class TestRunCommandIntegration:
    def test_run_config_with_env_var(
        self,
        tmp_path: Path,
        sample_video_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY", "secret-key")

        config = {
            "llm": {
                "model": "test-model",
                "backend": "ollama",
                "url": "http://localhost:8080/v1",
                "api_key": "${TEST_API_KEY}",
            },
            "query": {
                "prompt": "Test prompt",
            },
        }
        config_file = tmp_path / "config_with_env.json"
        config_file.write_text(json.dumps(config))

        result = runner.invoke(ai_app, ["run", "-c", str(config_file), str(sample_video_file)])
        assert result.exit_code != 0

    @pytest.mark.parametrize("fps_value", [0.5, 1.0, 5.0, 10.0])
    def test_run_with_different_fps_configs(self, fps_value: float, tmp_path: Path, sample_video_file: Path) -> None:
        config = {
            "frame_selector": {
                "selector_type": "fps_rescaling",
                "fps": fps_value,
            },
            "llm": {
                "model": "test-model",
                "backend": "ollama",
                "url": "http://localhost:8080/v1",
            },
            "query": {
                "prompt": "Test prompt",
            },
        }
        config_file = tmp_path / f"config_fps_{fps_value}.json"
        config_file.write_text(json.dumps(config))

        result = runner.invoke(ai_app, ["run", "-c", str(config_file), str(sample_video_file)])
        assert result.exit_code != 0

    @pytest.mark.parametrize("resolution", [[640, 360], [1280, 720], [320, 240]])
    def test_run_with_different_resolutions(
        self,
        resolution: list[int],
        tmp_path: Path,
        sample_video_file: Path,
    ) -> None:
        config = {
            "frame_extractor": {
                "resolution": resolution,
            },
            "llm": {
                "model": "test-model",
                "backend": "ollama",
                "url": "http://localhost:8080/v1",
            },
            "query": {
                "prompt": "Test prompt",
            },
        }
        config_file = tmp_path / f"config_res_{resolution[0]}x{resolution[1]}.json"
        config_file.write_text(json.dumps(config))

        result = runner.invoke(ai_app, ["run", "-c", str(config_file), str(sample_video_file)])
        assert result.exit_code != 0

    def test_run_with_all_config_options(self, tmp_path: Path, sample_video_file: Path) -> None:
        config = {
            "frame_selector": {
                "selector_type": "fps_rescaling",
                "fps": 2.0,
            },
            "frame_extractor": {
                "resolution": [800, 600],
            },
            "llm": {
                "backend": "llamacpp",
                "model": "custom-model",
                "url": "http://example.com:8080/v1",
            },
            "query": {
                "prompt": "Custom prompt for testing",
            },
            "reconciler": {
                "reconciler_type": "majority",
            },
        }
        config_file = tmp_path / "full_config.json"
        config_file.write_text(json.dumps(config))

        result = runner.invoke(ai_app, ["run", "-c", str(config_file), str(sample_video_file)])
        assert result.exit_code != 0

    def test_run_output_file_created(self, sample_config_file: Path, sample_video_file: Path, tmp_path: Path) -> None:
        output_file = tmp_path / "result.json"
        result = runner.invoke(
            ai_app,
            ["run", "-c", str(sample_config_file), "-o", str(output_file), str(sample_video_file)],
        )
        assert result.exit_code != 0

    def test_run_error_bubbles_up(self, tmp_path: Path, sample_video_file: Path) -> None:
        config = {
            "llm": {
                "model": "invalid-model",
                "backend": "ollama",
                "url": "http://invalid-url:9999",
            },
            "query": {
                "prompt": "Test prompt",
            },
        }
        config_file = tmp_path / "bad_config.json"
        config_file.write_text(json.dumps(config))

        result = runner.invoke(ai_app, ["run", "-c", str(config_file), str(sample_video_file)])
        assert result.exit_code != 0


class TestRunEvaluateCommand:
    @pytest.fixture
    def sample_labels_file(self, tmp_path: Path) -> Path:
        labels = [
            {"video": "test.mp4", "label": "otter"},
            {"video": "short.mp4", "label": "cat"},
        ]
        labels_file = tmp_path / "labels.jsonl"
        with open(labels_file, "w") as f:
            f.writelines(json.dumps(label) + "\n" for label in labels)
        return labels_file

    def test_run_evaluate_missing_config(self, sample_comparison_config_file: Path, sample_labels_file: Path) -> None:
        result = runner.invoke(
            ai_app,
            ["run-evaluate", "nonexistent.json", str(sample_comparison_config_file), str(sample_labels_file)],
        )
        assert result.exit_code == 1
        assert "Error: Config file not found" in result.stderr or "Error: Config file not found" in result.stdout

    def test_run_evaluate_missing_labels(self, sample_config_file: Path, sample_comparison_config_file: Path) -> None:
        result = runner.invoke(
            ai_app,
            ["run-evaluate", str(sample_config_file), str(sample_comparison_config_file), "nonexistent.jsonl"],
        )
        assert result.exit_code == 1
        assert "Error: Labels file not found" in result.stderr or "Error: Labels file not found" in result.stdout

    def test_run_evaluate_config_not_file(
        self,
        tmp_path: Path,
        sample_comparison_config_file: Path,
        sample_labels_file: Path,
    ) -> None:
        result = runner.invoke(
            ai_app,
            ["run-evaluate", str(tmp_path), str(sample_comparison_config_file), str(sample_labels_file)],
        )
        assert result.exit_code == 1
        assert (
            "Error: Config path is not a file" in result.stderr or "Error: Config path is not a file" in result.stdout
        )

    def test_run_evaluate_labels_not_file(
        self,
        tmp_path: Path,
        sample_config_file: Path,
        sample_comparison_config_file: Path,
    ) -> None:
        result = runner.invoke(
            ai_app,
            ["run-evaluate", str(sample_config_file), str(sample_comparison_config_file), str(tmp_path)],
        )
        assert result.exit_code == 1
        assert (
            "Error: Labels path is not a file" in result.stderr or "Error: Labels path is not a file" in result.stdout
        )

    def test_run_evaluate_video_dir_not_found(
        self,
        sample_config_file: Path,
        sample_comparison_config_file: Path,
        sample_labels_file: Path,
    ) -> None:
        result = runner.invoke(
            ai_app,
            [
                "run-evaluate",
                str(sample_config_file),
                str(sample_comparison_config_file),
                str(sample_labels_file),
                "-v",
                "/nonexistent/dir",
            ],
        )
        assert result.exit_code == 1
        assert (
            "Error: Video directory not found" in result.stderr or "Error: Video directory not found" in result.stdout
        )

    def test_run_evaluate_video_dir_not_directory(
        self,
        sample_config_file: Path,
        sample_comparison_config_file: Path,
        sample_labels_file: Path,
    ) -> None:
        result = runner.invoke(
            ai_app,
            [
                "run-evaluate",
                str(sample_config_file),
                str(sample_comparison_config_file),
                str(sample_labels_file),
                "-v",
                str(sample_config_file),
            ],
        )
        assert result.exit_code == 1
        assert (
            "Error: Video path is not a directory" in result.stderr
            or "Error: Video path is not a directory" in result.stdout
        )

    def test_run_evaluate_invalid_max_workers(
        self,
        sample_config_file: Path,
        sample_comparison_config_file: Path,
        sample_labels_file: Path,
    ) -> None:
        result = runner.invoke(
            ai_app,
            [
                "run-evaluate",
                str(sample_config_file),
                str(sample_comparison_config_file),
                str(sample_labels_file),
                "-w",
                "0",
            ],
        )
        assert result.exit_code == 1
        assert (
            "Error: --max-workers must be at least 1" in result.stderr
            or "Error: --max-workers must be at least 1" in result.stdout
        )

    def test_run_evaluate_success_with_mocked(
        self,
        sample_config_file: Path,
        sample_comparison_config_file: Path,
        sample_labels_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_summary = PipelineEvaluationSummary(
            results=[
                PipelineEvaluationResult(
                    filename="test.mp4",
                    classification=ResultClassification.CORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test features",
                        species_name="otter",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="otter",
                    comparison_method="exact",
                    processing_time_seconds=1.0,
                ),
            ],
            correct_count=1,
            total_count=1,
            error_count=0,
            average_processing_time_seconds=1.0,
        )

        with patch("wildcamtools.cli.ai.evaluate_ai_pipeline") as mock_eval:
            mock_eval.return_value = mock_summary

            output_file = tmp_path / "results.json"
            result = runner.invoke(
                ai_app,
                [
                    "run-evaluate",
                    str(sample_config_file),
                    str(sample_comparison_config_file),
                    str(sample_labels_file),
                    "-o",
                    str(output_file),
                    "-w",
                    "2",
                ],
            )
            assert result.exit_code == 0
            mock_eval.assert_called_once_with(
                config_path=sample_config_file,
                labels_path=sample_labels_file,
                video_dir=None,
                max_workers=2,
                comparison_config_path=sample_comparison_config_file,
            )
            assert output_file.exists()
            json_content = json.loads(output_file.read_text())
            assert "results" in json_content
            assert "correct_count" in json_content
            assert "average_processing_time_seconds" in json_content

    def test_run_evaluate_with_label_comparison_config(
        self,
        sample_config_file: Path,
        sample_comparison_config_file: Path,
        sample_labels_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_summary = PipelineEvaluationSummary(
            results=[
                PipelineEvaluationResult(
                    filename="test.mp4",
                    classification=ResultClassification.CORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test features",
                        species_name="domestic cat",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="cat",
                    comparison_method="llm",
                    processing_time_seconds=1.0,
                ),
            ],
            correct_count=1,
            total_count=1,
            error_count=0,
            average_processing_time_seconds=1.0,
        )

        with patch("wildcamtools.cli.ai.evaluate_ai_pipeline") as mock_eval:
            mock_eval.return_value = mock_summary

            result = runner.invoke(
                ai_app,
                [
                    "run-evaluate",
                    str(sample_config_file),
                    str(sample_comparison_config_file),
                    str(sample_labels_file),
                ],
            )
            assert result.exit_code == 0
            mock_eval.assert_called_once()
            call_kwargs = mock_eval.call_args.kwargs
            assert call_kwargs["comparison_config_path"] == sample_comparison_config_file

    def test_run_evaluate_json_output_to_stdout(
        self,
        sample_config_file: Path,
        sample_comparison_config_file: Path,
        sample_labels_file: Path,
    ) -> None:
        mock_summary = PipelineEvaluationSummary(
            results=[
                PipelineEvaluationResult(
                    filename="test.mp4",
                    classification=ResultClassification.CORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test features",
                        species_name="otter",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="otter",
                    comparison_method="exact",
                    processing_time_seconds=1.5,
                ),
            ],
            correct_count=1,
            total_count=1,
            error_count=0,
            average_processing_time_seconds=1.5,
        )

        with patch("wildcamtools.cli.ai.evaluate_ai_pipeline") as mock_eval:
            mock_eval.return_value = mock_summary

            result = runner.invoke(
                ai_app,
                [
                    "run-evaluate",
                    str(sample_config_file),
                    str(sample_comparison_config_file),
                    str(sample_labels_file),
                ],
            )
            assert result.exit_code == 0
            json_output = json.loads(result.stdout)
            assert "results" in json_output
            assert json_output["correct_count"] == 1
            assert json_output["average_processing_time_seconds"] == 1.5

    def test_run_evaluate_json_output_to_file(
        self,
        sample_config_file: Path,
        sample_comparison_config_file: Path,
        sample_labels_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_summary = PipelineEvaluationSummary(
            results=[
                PipelineEvaluationResult(
                    filename="test.mp4",
                    classification=ResultClassification.CORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test features",
                        species_name="otter",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="otter",
                    comparison_method="exact",
                    processing_time_seconds=2.0,
                ),
            ],
            correct_count=1,
            total_count=1,
            error_count=0,
            average_processing_time_seconds=2.0,
        )

        with patch("wildcamtools.cli.ai.evaluate_ai_pipeline") as mock_eval:
            mock_eval.return_value = mock_summary

            output_file = tmp_path / "evaluation_result.json"
            result = runner.invoke(
                ai_app,
                [
                    "run-evaluate",
                    str(sample_config_file),
                    str(sample_comparison_config_file),
                    str(sample_labels_file),
                    "-o",
                    str(output_file),
                ],
            )
            assert result.exit_code == 0
            assert output_file.exists()
            json_content = json.loads(output_file.read_text())
            assert json_content["correct_count"] == 1
            assert json_content["average_processing_time_seconds"] == 2.0
            assert len(json_content["results"]) == 1
            assert json_content["results"][0]["processing_time_seconds"] == 2.0
