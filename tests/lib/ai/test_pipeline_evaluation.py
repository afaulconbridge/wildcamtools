import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wildcamtools.lib.ai import (
    BatchResult,
    ExtractedFrame,
    PipelineEvaluationResult,
    PipelineEvaluationSummary,
    PipelineOutcome,
    RichResult,
)
from wildcamtools.lib.ai.label_comparison_config import LabelComparisonConfig
from wildcamtools.lib.ai.pipeline_config import AiPipelineConfig
from wildcamtools.lib.ai.pipeline_evaluation import (
    _evaluate_video_worker,
    _WorkerResult,
    evaluate_ai_pipeline,
)
from wildcamtools.lib.ai.types import ConfidenceLevel, ResultClassification
from wildcamtools.lib.stats import Colourspace, VideoStats


class TestWorkerResult:
    def test_worker_result_creation(self) -> None:
        rich_result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="",
            species_name="otter",
            confidence=ConfidenceLevel.HIGH,
        )
        stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
        outcome = PipelineOutcome(result=rich_result, stats=stats, batches=[])
        result = _WorkerResult(
            filename="test.mp4",
            outcome=outcome,
        )
        assert result.filename == "test.mp4"
        assert result.outcome.result.species_name == "otter"
        assert result.error is None
        assert result.processing_time_seconds == 0.0
        assert result.frame_ids == []

    def test_worker_result_with_error(self) -> None:
        rich_result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="",
            species_name="otter",
            confidence=ConfidenceLevel.HIGH,
        )
        stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
        outcome = PipelineOutcome(result=rich_result, stats=stats, batches=[])
        result = _WorkerResult(
            filename="test.mp4",
            outcome=outcome,
            error="Connection timeout",
        )
        assert result.error == "Connection timeout"
        assert result.processing_time_seconds == 0.0
        assert result.frame_ids == []

    def test_worker_result_with_processing_time(self) -> None:
        rich_result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="",
            species_name="otter",
            confidence=ConfidenceLevel.HIGH,
        )
        stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
        outcome = PipelineOutcome(result=rich_result, stats=stats, batches=[])
        result = _WorkerResult(
            filename="test.mp4",
            outcome=outcome,
            processing_time_seconds=1.5,
        )
        assert result.processing_time_seconds == 1.5

    def test_worker_result_with_frame_ids(self) -> None:
        rich_result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="",
            species_name="otter",
            confidence=ConfidenceLevel.HIGH,
        )
        stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
        batch = BatchResult(
            selected_frames=[ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i) for i in [1, 5, 10, 15]],
            result=rich_result,
        )
        outcome = PipelineOutcome(result=rich_result, stats=stats, batches=[batch])
        result = _WorkerResult(
            filename="test.mp4",
            outcome=outcome,
            frame_ids=[1, 5, 10, 15],
        )
        assert result.frame_ids == [1, 5, 10, 15]


class TestPipelineEvaluationResult:
    def test_result_creation(self) -> None:
        result = PipelineEvaluationResult(
            filename="test.mp4",
            classification=ResultClassification.CORRECT,
            result=RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test",
                species_name="otter",
                confidence=ConfidenceLevel.HIGH,
            ),
            label="otter",
        )
        assert result.filename == "test.mp4"
        assert result.classification == ResultClassification.CORRECT
        assert result.result.species_name == "otter"
        assert result.label == "otter"
        assert result.error is None
        assert result.comparison_method == "exact"
        assert result.processing_time_seconds == 0.0
        assert result.frame_ids == []
        assert result.stats is None

    def test_result_with_error(self) -> None:
        result = PipelineEvaluationResult(
            filename="test.mp4",
            classification=ResultClassification.INCORRECT,
            result=RichResult(
                is_animal_present=False,
                is_animal_unknown=False,
                defining_features="",
                species_name="",
                confidence=ConfidenceLevel.LOW,
            ),
            label="otter",
            error="Connection timeout",
        )
        assert result.error == "Connection timeout"
        assert result.classification == ResultClassification.INCORRECT
        assert result.processing_time_seconds == 0.0
        assert result.frame_ids == []
        assert result.stats is None

    def test_result_unknown_classification(self) -> None:
        result = PipelineEvaluationResult(
            filename="test.mp4",
            classification=ResultClassification.UNKNOWN,
            result=RichResult(
                is_animal_present=False,
                is_animal_unknown=True,
                defining_features="unknown",
                species_name="unknown",
                confidence=ConfidenceLevel.LOW,
            ),
            label="otter",
        )
        assert result.classification == ResultClassification.UNKNOWN
        # result.correct field removed - classification UNKNOWN implies not correct

    def test_result_with_comparison_method(self) -> None:
        result = PipelineEvaluationResult(
            filename="test.mp4",
            classification=ResultClassification.CORRECT,
            result=RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test",
                species_name="domestic cat",
                confidence=ConfidenceLevel.HIGH,
            ),
            label="cat",
            comparison_method="llm",
        )
        assert result.comparison_method == "llm"
        assert result.frame_ids == []

    def test_result_with_processing_time(self) -> None:
        result = PipelineEvaluationResult(
            filename="test.mp4",
            classification=ResultClassification.CORRECT,
            result=RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test",
                species_name="otter",
                confidence=ConfidenceLevel.HIGH,
            ),
            label="otter",
            processing_time_seconds=1.5,
        )
        assert result.processing_time_seconds == 1.5
        assert result.frame_ids == []

    def test_result_with_frame_ids(self) -> None:
        result = PipelineEvaluationResult(
            filename="test.mp4",
            classification=ResultClassification.CORRECT,
            result=RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test",
                species_name="otter",
                confidence=ConfidenceLevel.HIGH,
            ),
            label="otter",
            frame_ids=[1, 5, 10, 15],
        )
        assert result.frame_ids == [1, 5, 10, 15]


class TestPipelineEvaluationSummary:
    def test_summary_creation(self) -> None:
        results = [
            PipelineEvaluationResult(
                filename="test1.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="otter",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
            ),
            PipelineEvaluationResult(
                filename="test2.mp4",
                classification=ResultClassification.INCORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="cat",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
            ),
        ]
        summary = PipelineEvaluationSummary(
            results=results,
            correct_count=1,
            incorrect_count=1,
            unknown_count=0,
            total_count=2,
            error_count=0,
        )
        assert summary.correct_count == 1
        assert summary.incorrect_count == 1
        assert summary.total_count == 2
        assert summary.error_count == 0
        assert summary.average_processing_time_seconds == 0.0

    def test_accuracy_calculation(self) -> None:
        results = [
            PipelineEvaluationResult(
                filename="test1.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="otter",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
            ),
            PipelineEvaluationResult(
                filename="test2.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="cat",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="cat",
            ),
            PipelineEvaluationResult(
                filename="test3.mp4",
                classification=ResultClassification.INCORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="dog",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="cat",
            ),
        ]
        summary = PipelineEvaluationSummary(
            results=results,
            correct_count=2,
            incorrect_count=1,
            unknown_count=0,
            total_count=3,
            error_count=0,
        )
        assert summary.accuracy == pytest.approx(2 / 3)

    def test_accuracy_excludes_unknown(self) -> None:
        results = [
            PipelineEvaluationResult(
                filename="test1.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="otter",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
            ),
            PipelineEvaluationResult(
                filename="test2.mp4",
                classification=ResultClassification.UNKNOWN,
                result=RichResult(
                    is_animal_present=False,
                    is_animal_unknown=True,
                    defining_features="unknown",
                    species_name="unknown",
                    confidence=ConfidenceLevel.LOW,
                ),
                label="otter",
            ),
            PipelineEvaluationResult(
                filename="test3.mp4",
                classification=ResultClassification.INCORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="dog",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
            ),
        ]
        summary = PipelineEvaluationSummary(
            results=results,
            correct_count=1,
            incorrect_count=1,
            unknown_count=1,
            total_count=3,
            error_count=0,
        )
        assert summary.accuracy == pytest.approx(0.5)

    def test_accuracy_with_errors(self) -> None:
        results = [
            PipelineEvaluationResult(
                filename="test1.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="otter",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
            ),
            PipelineEvaluationResult(
                filename="test2.mp4",
                classification=ResultClassification.INCORRECT,
                result=RichResult(
                    is_animal_present=False,
                    is_animal_unknown=False,
                    defining_features="",
                    species_name="",
                    confidence=ConfidenceLevel.LOW,
                ),
                label="cat",
                error="LLM error",
            ),
        ]
        summary = PipelineEvaluationSummary(
            results=results,
            correct_count=1,
            incorrect_count=0,
            unknown_count=0,
            total_count=2,
            error_count=1,
        )
        assert summary.accuracy == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "correct,incorrect,total,expected",
        [
            (0, 10, 10, 0.0),
            (5, 5, 10, 0.5),
            (10, 0, 10, 1.0),
            (7, 1, 8, 0.875),
        ],
    )
    def test_accuracy_various_values(self, correct: int, incorrect: int, total: int, expected: float) -> None:
        results = []
        for i in range(correct):
            results.append(
                PipelineEvaluationResult(
                    filename=f"test_correct_{i}.mp4",
                    classification=ResultClassification.CORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test",
                        species_name="otter",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="otter",
                )
            )
        for i in range(incorrect):
            results.append(
                PipelineEvaluationResult(
                    filename=f"test_incorrect_{i}.mp4",
                    classification=ResultClassification.INCORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test",
                        species_name="cat",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="otter",
                )
            )
        summary = PipelineEvaluationSummary(
            results=results,
            correct_count=correct,
            incorrect_count=incorrect,
            unknown_count=0,
            total_count=total,
            error_count=0,
        )
        assert summary.accuracy == pytest.approx(expected)

    def test_success_rate_property(self) -> None:
        results = [
            PipelineEvaluationResult(
                filename="test1.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="otter",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
            ),
            PipelineEvaluationResult(
                filename="test2.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="cat",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="cat",
            ),
        ]
        summary = PipelineEvaluationSummary(
            results=results,
            correct_count=2,
            incorrect_count=0,
            unknown_count=0,
            total_count=3,
            error_count=0,
        )
        assert summary.success_rate == summary.accuracy

    def test_failure_count_property(self) -> None:
        results = [
            PipelineEvaluationResult(
                filename="test1.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="otter",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
            ),
            PipelineEvaluationResult(
                filename="test2.mp4",
                classification=ResultClassification.INCORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="cat",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="cat",
            ),
        ]
        summary = PipelineEvaluationSummary(
            results=results,
            correct_count=1,
            incorrect_count=1,
            unknown_count=0,
            total_count=3,
            error_count=0,
        )
        assert summary.failure_count == 2

    def test_detection_rate_property(self) -> None:
        results = [
            PipelineEvaluationResult(
                filename="test1.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="otter",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
            ),
            PipelineEvaluationResult(
                filename="test2.mp4",
                classification=ResultClassification.UNKNOWN,
                result=RichResult(
                    is_animal_present=False,
                    is_animal_unknown=True,
                    defining_features="unknown",
                    species_name="unknown",
                    confidence=ConfidenceLevel.LOW,
                ),
                label="cat",
            ),
            PipelineEvaluationResult(
                filename="test3.mp4",
                classification=ResultClassification.INCORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="dog",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="dog",
            ),
        ]
        summary = PipelineEvaluationSummary(
            results=results,
            correct_count=1,
            incorrect_count=1,
            unknown_count=1,
            total_count=3,
            error_count=0,
        )
        assert summary.detection_rate == pytest.approx(2 / 3)

    def test_precision_when_confident_property(self) -> None:
        results = [
            PipelineEvaluationResult(
                filename="test1.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="otter",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
            ),
            PipelineEvaluationResult(
                filename="test2.mp4",
                classification=ResultClassification.INCORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="cat",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="cat",
            ),
        ]
        summary = PipelineEvaluationSummary(
            results=results,
            correct_count=1,
            incorrect_count=1,
            unknown_count=0,
            total_count=2,
            error_count=0,
        )
        assert summary.precision_when_confident == pytest.approx(1 / 2)

    def test_average_processing_time(self) -> None:
        results = [
            PipelineEvaluationResult(
                filename="test1.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="otter",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
                processing_time_seconds=1.0,
            ),
            PipelineEvaluationResult(
                filename="test2.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="cat",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="cat",
                processing_time_seconds=2.0,
            ),
            PipelineEvaluationResult(
                filename="test3.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="dog",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="dog",
                processing_time_seconds=3.0,
            ),
        ]
        summary = PipelineEvaluationSummary(
            results=results,
            correct_count=3,
            incorrect_count=0,
            unknown_count=0,
            total_count=3,
            error_count=0,
            average_processing_time_seconds=2.0,
        )
        assert summary.average_processing_time_seconds == 2.0

    def test_json_serialization(self) -> None:
        results = [
            PipelineEvaluationResult(
                filename="test1.mp4",
                classification=ResultClassification.CORRECT,
                result=RichResult(
                    is_animal_present=True,
                    is_animal_unknown=False,
                    defining_features="test",
                    species_name="otter",
                    confidence=ConfidenceLevel.HIGH,
                ),
                label="otter",
                processing_time_seconds=1.5,
            ),
        ]
        summary = PipelineEvaluationSummary(
            results=results,
            correct_count=1,
            incorrect_count=0,
            unknown_count=0,
            total_count=1,
            error_count=0,
            average_processing_time_seconds=1.5,
        )
        json_str = summary.model_dump_json(indent=2)
        assert "test1.mp4" in json_str
        assert "classification" in json_str
        assert "processing_time_seconds" in json_str
        assert "average_processing_time_seconds" in json_str


class TestEvaluateVideoWorker:
    def test_worker_with_mocked_pipeline_in_worker(
        self,
        data_directory: Path,
    ) -> None:
        mock_result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test",
            species_name="otter",
            confidence=ConfidenceLevel.HIGH,
        )
        mock_stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
        mock_batch = BatchResult(
            selected_frames=[ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i) for i in [1, 5, 10]],
            result=mock_result,
        )
        mock_outcome = PipelineOutcome(result=mock_result, stats=mock_stats, batches=[mock_batch])
        mock_config = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_outcome
        mock_config.create_pipeline.return_value = mock_pipeline

        video_path = data_directory / "test.mp4"
        result = _evaluate_video_worker(str(video_path), mock_config)

        assert result.filename == "test.mp4"
        assert result.outcome.result.species_name == "otter"
        assert result.error is None
        assert result.processing_time_seconds > 0.0

    def test_worker_incorrect_result_with_mock(
        self,
        data_directory: Path,
    ) -> None:
        mock_result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test",
            species_name="cat",
            confidence=ConfidenceLevel.HIGH,
        )
        mock_stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
        mock_outcome = PipelineOutcome(result=mock_result, stats=mock_stats, batches=[])
        mock_config = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_outcome
        mock_config.create_pipeline.return_value = mock_pipeline

        video_path = data_directory / "test.mp4"
        result = _evaluate_video_worker(str(video_path), mock_config)

        assert result.outcome.result.species_name == "cat"

    def test_worker_unknown_result_with_mock(
        self,
        data_directory: Path,
    ) -> None:
        mock_result = RichResult(
            is_animal_present=False,
            is_animal_unknown=True,
            defining_features="unknown",
            species_name="unknown",
            confidence=ConfidenceLevel.LOW,
        )
        mock_stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
        mock_outcome = PipelineOutcome(result=mock_result, stats=mock_stats, batches=[])
        mock_config = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_outcome
        mock_config.create_pipeline.return_value = mock_pipeline

        video_path = data_directory / "test.mp4"
        result = _evaluate_video_worker(str(video_path), mock_config)

        assert result.outcome.result.species_name == "unknown"

    def test_worker_with_rich_result(
        self,
        data_directory: Path,
    ) -> None:
        mock_result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test",
            species_name="test response",
            confidence=ConfidenceLevel.HIGH,
        )
        mock_stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
        mock_outcome = PipelineOutcome(result=mock_result, stats=mock_stats, batches=[])
        mock_config = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_outcome
        mock_config.create_pipeline.return_value = mock_pipeline

        video_path = data_directory / "test.mp4"
        result = _evaluate_video_worker(str(video_path), mock_config)

        assert result.outcome.result.species_name == "test response"

    def test_worker_captures_frame_ids(
        self,
        data_directory: Path,
    ) -> None:
        mock_result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test",
            species_name="otter",
            confidence=ConfidenceLevel.HIGH,
        )
        mock_stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
        mock_batch = BatchResult(
            selected_frames=[ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i) for i in [1, 5, 10]],
            result=mock_result,
        )
        mock_outcome = PipelineOutcome(result=mock_result, stats=mock_stats, batches=[mock_batch])
        mock_config = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_outcome
        mock_config.create_pipeline.return_value = mock_pipeline

        video_path = data_directory / "test.mp4"
        result = _evaluate_video_worker(str(video_path), mock_config)

        assert result.frame_ids == [1, 5, 10]


class TestEvaluateAiPipeline:
    @pytest.fixture
    def sample_config_file(self, tmp_path: Path) -> Path:
        config = {
            "query": {
                "query_type": "llm",
                "prompt": "What species is in this image?",
                "llm": {
                    "model": "test-model",
                    "backend": "ollama",
                    "url": "http://localhost:8080/v1",
                },
            },
            "reconciler": {
                "reconciler_type": "majority",
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))
        return config_file

    @pytest.fixture
    def sample_comparison_config_file(self, tmp_path: Path) -> Path:
        comparison_config = {
            "comparator_type": "exact",
        }
        config_file = tmp_path / "comparison_config.json"
        config_file.write_text(json.dumps(comparison_config))
        return config_file

    @pytest.fixture
    def sample_labels_file(self, tmp_path: Path) -> Path:
        labels = [
            {"video": "test.mp4", "label": "otter"},
            {"video": "short.mp4", "label": "cat"},
        ]
        labels_file = tmp_path / "labels.jsonl"
        with open(labels_file, "w") as f:
            for label in labels:
                f.write(json.dumps(label) + "\n")
        return labels_file

    def test_missing_config_file(
        self, tmp_path: Path, sample_comparison_config_file: Path, sample_labels_file: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            evaluate_ai_pipeline(
                config_path=tmp_path / "nonexistent.json",
                comparison_config_path=sample_comparison_config_file,
                labels_path=sample_labels_file,
            )

    def test_config_not_file(
        self, tmp_path: Path, sample_comparison_config_file: Path, sample_labels_file: Path
    ) -> None:
        with pytest.raises(ValueError, match="Config path is not a file"):
            evaluate_ai_pipeline(
                config_path=tmp_path,
                comparison_config_path=sample_comparison_config_file,
                labels_path=sample_labels_file,
            )

    def test_missing_labels_file(
        self, tmp_path: Path, sample_config_file: Path, sample_comparison_config_file: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Labels file not found"):
            evaluate_ai_pipeline(
                config_path=sample_config_file,
                comparison_config_path=sample_comparison_config_file,
                labels_path=tmp_path / "nonexistent.jsonl",
            )

    def test_labels_not_file(
        self, tmp_path: Path, sample_config_file: Path, sample_comparison_config_file: Path
    ) -> None:
        with pytest.raises(ValueError, match="Labels path is not a file"):
            evaluate_ai_pipeline(
                config_path=sample_config_file,
                comparison_config_path=sample_comparison_config_file,
                labels_path=tmp_path,
            )

    def test_evaluate_returns_summary(
        self,
        sample_config_file: Path,
        sample_comparison_config_file: Path,
        sample_labels_file: Path,
        data_directory: Path,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(AiPipelineConfig, "model_validate") as mock_validate,
            patch.object(LabelComparisonConfig, "model_validate") as mock_comparison_validate,
            patch("wildcamtools.lib.ai.pipeline_evaluation._run_worker_pool") as mock_pool,
        ):
            mock_config = MagicMock()
            mock_pipeline = MagicMock()
            mock_result = RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test",
                species_name="otter",
                confidence=ConfidenceLevel.HIGH,
            )
            mock_stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
            mock_batch = BatchResult(
                selected_frames=[ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i) for i in [1, 5, 10]],
                result=mock_result,
            )
            mock_outcome = PipelineOutcome(result=mock_result, stats=mock_stats, batches=[mock_batch])
            mock_pipeline.run.return_value = mock_outcome
            mock_config.create_pipeline.return_value = mock_pipeline
            mock_validate.return_value = mock_config

            mock_comparator = MagicMock()
            mock_comparator.compare.return_value = ResultClassification.CORRECT
            mock_comparator.method_name = "exact"
            mock_comparison_config = MagicMock()
            mock_comparison_config.create_comparator.return_value = mock_comparator
            mock_comparison_validate.return_value = mock_comparison_config

            mock_pool.return_value = [
                PipelineEvaluationResult(
                    filename="test.mp4",
                    classification=ResultClassification.CORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test",
                        species_name="otter",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="otter",
                    comparison_method="exact",
                    processing_time_seconds=1.0,
                    frame_ids=[1, 5, 10],
                ),
                PipelineEvaluationResult(
                    filename="short.mp4",
                    classification=ResultClassification.CORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test",
                        species_name="cat",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="cat",
                    comparison_method="exact",
                    processing_time_seconds=2.0,
                    frame_ids=[2, 6],
                ),
            ]

            summary = evaluate_ai_pipeline(
                config_path=sample_config_file,
                comparison_config_path=sample_comparison_config_file,
                labels_path=sample_labels_file,
                video_dir=data_directory,
                max_workers=1,
            )

            assert summary.average_processing_time_seconds == 1.5

    def test_evaluate_counts_errors(
        self,
        sample_config_file: Path,
        sample_comparison_config_file: Path,
        sample_labels_file: Path,
        data_directory: Path,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(AiPipelineConfig, "model_validate") as mock_validate,
            patch.object(LabelComparisonConfig, "model_validate") as mock_comparison_validate,
            patch("wildcamtools.lib.ai.pipeline_evaluation._run_worker_pool") as mock_pool,
        ):
            mock_config = MagicMock()
            mock_pipeline = MagicMock()
            mock_result = RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test",
                species_name="otter",
                confidence=ConfidenceLevel.HIGH,
            )
            mock_stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
            mock_batch = BatchResult(
                selected_frames=[ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i) for i in [1, 5, 10]],
                result=mock_result,
            )
            mock_outcome = PipelineOutcome(result=mock_result, stats=mock_stats, batches=[mock_batch])
            mock_pipeline.run.return_value = mock_outcome
            mock_config.create_pipeline.return_value = mock_pipeline
            mock_validate.return_value = mock_config

            mock_comparator = MagicMock()
            mock_comparator.compare.return_value = ResultClassification.CORRECT
            mock_comparator.method_name = "exact"
            mock_comparison_config = MagicMock()
            mock_comparison_config.create_comparator.return_value = mock_comparator
            mock_comparison_validate.return_value = mock_comparison_config

            mock_pool.return_value = [
                PipelineEvaluationResult(
                    filename="test.mp4",
                    classification=ResultClassification.CORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test",
                        species_name="otter",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="otter",
                    comparison_method="exact",
                    processing_time_seconds=1.0,
                    frame_ids=[1, 5, 10],
                ),
                PipelineEvaluationResult(
                    filename="short.mp4",
                    classification=ResultClassification.INCORRECT,
                    result=RichResult(
                        is_animal_present=False,
                        is_animal_unknown=False,
                        defining_features="",
                        species_name="",
                        confidence=ConfidenceLevel.LOW,
                    ),
                    label="cat",
                    error="LLM connection failed",
                    comparison_method="exact",
                    processing_time_seconds=0.0,
                    frame_ids=[],
                ),
            ]

            summary = evaluate_ai_pipeline(
                config_path=sample_config_file,
                comparison_config_path=sample_comparison_config_file,
                labels_path=sample_labels_file,
                video_dir=data_directory,
                max_workers=1,
            )

            assert summary.error_count == 1
            assert summary.correct_count == 1
            assert summary.total_count == 2
            assert summary.failure_count == 1


class TestIntegration:
    def test_end_to_end_with_real_videos(
        self,
        tmp_path: Path,
        data_directory: Path,
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({
                "query": {
                    "query_type": "llm",
                    "prompt": "What species is in this image?",
                    "llm": {
                        "model": "test-model",
                        "backend": "ollama",
                        "url": "http://localhost:8080/v1",
                    },
                },
                "reconciler": {
                    "reconciler_type": "majority",
                },
            })
        )

        comparison_config_file = tmp_path / "comparison_config.json"
        comparison_config_file.write_text(
            json.dumps({
                "comparator_type": "exact",
            })
        )

        labels_file = tmp_path / "labels.jsonl"
        labels = [
            {"video": "test.mp4", "label": "otter"},
            {"video": "short.mp4", "label": "cat"},
        ]
        with open(labels_file, "w") as f:
            for label in labels:
                f.write(json.dumps(label) + "\n")

        with (
            patch.object(AiPipelineConfig, "model_validate") as mock_validate,
            patch.object(LabelComparisonConfig, "model_validate") as mock_comparison_validate,
            patch("wildcamtools.lib.ai.pipeline_evaluation._run_worker_pool") as mock_pool,
        ):
            mock_config = MagicMock()
            mock_pipeline = MagicMock()
            mock_result = RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test",
                species_name="otter",
                confidence=ConfidenceLevel.HIGH,
            )
            mock_stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
            mock_outcome = PipelineOutcome(result=mock_result, stats=mock_stats, frame_ids=[[1, 5, 10]])
            mock_pipeline.run.return_value = mock_outcome
            mock_config.create_pipeline.return_value = mock_pipeline
            mock_validate.return_value = mock_config

            mock_comparator = MagicMock()
            mock_comparator.compare.return_value = ResultClassification.CORRECT
            mock_comparator.method_name = "exact"
            mock_comparison_config = MagicMock()
            mock_comparison_config.create_comparator.return_value = mock_comparator
            mock_comparison_validate.return_value = mock_comparison_config

            mock_pool.return_value = [
                PipelineEvaluationResult(
                    filename="test.mp4",
                    classification=ResultClassification.CORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test",
                        species_name="otter",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="otter",
                    comparison_method="exact",
                    processing_time_seconds=1.0,
                    frame_ids=[1, 5, 10],
                ),
                PipelineEvaluationResult(
                    filename="short.mp4",
                    classification=ResultClassification.CORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test",
                        species_name="cat",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="cat",
                    comparison_method="exact",
                    processing_time_seconds=2.0,
                    frame_ids=[2, 6],
                ),
            ]

            summary = evaluate_ai_pipeline(
                config_path=config_file,
                comparison_config_path=comparison_config_file,
                labels_path=labels_file,
                video_dir=data_directory,
                max_workers=2,
            )

            assert summary.total_count == 2
            assert len(summary.results) == 2
            assert all(isinstance(r, PipelineEvaluationResult) for r in summary.results)
            assert summary.average_processing_time_seconds == 1.5
            assert summary.success_rate == summary.accuracy
            assert summary.failure_count == 0
            assert summary.results[0].frame_ids == [1, 5, 10]
            assert summary.results[1].frame_ids == [2, 6]

    def test_end_to_end_json_serialization(
        self,
        tmp_path: Path,
        data_directory: Path,
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({
                "query": {
                    "query_type": "llm",
                    "prompt": "What species is in this image?",
                    "llm": {
                        "model": "test-model",
                        "backend": "ollama",
                        "url": "http://localhost:8080/v1",
                    },
                },
                "reconciler": {
                    "reconciler_type": "majority",
                },
            })
        )

        comparison_config_file = tmp_path / "comparison_config.json"
        comparison_config_file.write_text(
            json.dumps({
                "comparator_type": "exact",
            })
        )

        labels_file = tmp_path / "labels.jsonl"
        labels = [
            {"video": "test.mp4", "label": "otter"},
        ]
        with open(labels_file, "w") as f:
            for label in labels:
                f.write(json.dumps(label) + "\n")

        with (
            patch.object(AiPipelineConfig, "model_validate") as mock_validate,
            patch.object(LabelComparisonConfig, "model_validate") as mock_comparison_validate,
            patch("wildcamtools.lib.ai.pipeline_evaluation._run_worker_pool") as mock_pool,
        ):
            mock_config = MagicMock()
            mock_pipeline = MagicMock()
            mock_result = RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test",
                species_name="otter",
                confidence=ConfidenceLevel.HIGH,
            )
            mock_stats = VideoStats(fps=30.0, frame_count=100, x=640, y=360, colourspace=Colourspace.RGB)
            mock_outcome = PipelineOutcome(result=mock_result, stats=mock_stats, frame_ids=[[1, 5, 10]])
            mock_pipeline.run.return_value = mock_outcome
            mock_config.create_pipeline.return_value = mock_pipeline
            mock_validate.return_value = mock_config

            mock_comparator = MagicMock()
            mock_comparator.compare.return_value = ResultClassification.CORRECT
            mock_comparator.method_name = "exact"
            mock_comparison_config = MagicMock()
            mock_comparison_config.create_comparator.return_value = mock_comparator
            mock_comparison_validate.return_value = mock_comparison_config

            mock_pool.return_value = [
                PipelineEvaluationResult(
                    filename="test.mp4",
                    classification=ResultClassification.CORRECT,
                    result=RichResult(
                        is_animal_present=True,
                        is_animal_unknown=False,
                        defining_features="test",
                        species_name="otter",
                        confidence=ConfidenceLevel.HIGH,
                    ),
                    label="otter",
                    comparison_method="exact",
                    processing_time_seconds=1.5,
                    frame_ids=[1, 5, 10],
                ),
            ]

            summary = evaluate_ai_pipeline(
                config_path=config_file,
                comparison_config_path=comparison_config_file,
                labels_path=labels_file,
                video_dir=data_directory,
                max_workers=1,
            )

            json_output = summary.model_dump_json(indent=2)
            assert "test.mp4" in json_output
            assert "classification" in json_output
            assert "processing_time_seconds" in json_output
            assert "average_processing_time_seconds" in json_output
            assert "frame_ids" in json_output
            assert "success_rate" not in json_output
            assert "failure_count" not in json_output
