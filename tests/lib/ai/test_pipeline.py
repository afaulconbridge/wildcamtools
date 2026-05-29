"""Tests for AI pipeline components: FrameSelector, FrameImageExtractor, and ImageBatchQuery."""

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np
import pytest
from pydantic import BaseModel

from wildcamtools.lib import Frame
from wildcamtools.lib.ai import Backend
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.pipeline import (
    FpsRescalingFrameSelector,
    LlmImageBatchQuery,
    MajorityResultReconciler,
    RescaledFrameImageExtractor,
    VerifiedImageBatchQuery,
)
from wildcamtools.lib.ai.types import ConfidenceLevel, VerificationResult

T = TypeVar("T", bound=BaseModel)


@pytest.fixture(name="sample_frames")
def sample_frames() -> list[Frame]:
    """Create synthetic Frame objects for testing."""
    frames = []
    for i in range(10):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        raw[:, :] = [(i * 10) % 256, (i * 20) % 256, (i * 30) % 256]
        frame = Frame(raw=raw, frame_no=i, filter_keep=True)
        frames.append(frame)
    return frames


@pytest.fixture(name="sample_frames_with_filtering")
def sample_frames_with_filtering() -> list[Frame]:
    """Create synthetic Frame objects with some filtered out."""
    frames = []
    for i in range(10):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        raw[:, :] = [(i * 10) % 256, (i * 20) % 256, (i * 30) % 256]
        filter_keep = i % 2 == 0
        frame = Frame(raw=raw, frame_no=i, filter_keep=filter_keep)
        frames.append(frame)
    return frames


@pytest.fixture(name="sample_frames_with_rescale")
def sample_frames_with_rescale() -> list[Frame]:
    """Create synthetic Frame objects with rescaled images."""
    frames = []
    for i in range(5):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        rescale = np.zeros((50, 100, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=i, rescale=rescale, filter_keep=True)
        frames.append(frame)
    return frames


@pytest.fixture(name="sample_frames_all_filtered")
def sample_frames_all_filtered() -> list[Frame]:
    """Create synthetic Frame objects with all frames filtered out."""
    frames = []
    for i in range(5):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=i, filter_keep=False)
        frames.append(frame)
    return frames


@pytest.fixture(name="sample_image_paths")
def sample_image_paths(tmp_path: Path) -> list[Path]:
    """Create sample image files for testing."""
    image_paths = []
    for i in range(5):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, :] = [(i * 10) % 256, (i * 20) % 256, (i * 30) % 256]
        img_path = tmp_path / f"image_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        image_paths.append(img_path)
    return image_paths


class MockSpeciesResult(BaseModel):
    """Mock response model for testing."""

    species_name: str


class MockAbstractLlm(AbstractLlm):
    """Mock LLM for testing ImageBatchQuery implementations."""

    model: str
    backend: Backend
    url: str
    api_key: str | None = None

    def __init__(
        self,
        model: str = "test-model",
        backend: Backend = Backend.OLLAMA,
        url: str = "http://test",
        response_to_return: str = "test_species",
    ) -> None:
        self.model = model
        self.backend = backend
        self.url = url
        self.response_to_return = response_to_return
        self.call_count = 0
        self.last_images: list[Path] = []
        self.last_prompt: str = ""

    def message_with_schema[T](
        self,
        message: str,
        images: Sequence[Path] = (),
        response_class: type[T] = MockSpeciesResult,  # type: ignore[assignment]
    ) -> T:
        self.call_count += 1
        self.last_images = list(images)
        self.last_prompt = message
        result = MockSpeciesResult(species_name=self.response_to_return)
        return result  # type: ignore[return-value]


class MockVerifiedLlm(AbstractLlm):
    """Mock LLM for testing VerifiedImageBatchQuery with two-stage responses."""

    model: str
    backend: Backend
    url: str
    api_key: str | None = None

    def __init__(
        self,
        model: str = "test-model",
        backend: Backend = Backend.OLLAMA,
        url: str = "http://test",
        initial_species: str = "fox",
        verification_confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
        verification_verified: bool = True,
        verification_corrected_species: str | None = None,
    ) -> None:
        self.model = model
        self.backend = backend
        self.url = url
        self.initial_species = initial_species
        self.verification_confidence = verification_confidence
        self.verification_verified = verification_verified
        self.verification_corrected_species = verification_corrected_species
        self.call_count = 0
        self.last_images: list[Path] = []
        self.last_prompt: str = ""
        self.verification_prompt: str = ""

    def message_with_schema[T](
        self,
        message: str,
        images: Sequence[Path] = (),
        response_class: type[T] = MockSpeciesResult,  # type: ignore[assignment]
    ) -> T:
        self.call_count += 1
        self.last_images = list(images)

        if response_class is VerificationResult:
            self.verification_prompt = message
            result = VerificationResult(
                species_name=self.verification_corrected_species or self.initial_species,
                confidence=self.verification_confidence,
                verified=self.verification_verified,
            )
            return result  # type: ignore[return-value]
        else:
            self.last_prompt = message
            result = MockSpeciesResult(species_name=self.initial_species)
            return result  # type: ignore[return-value]


class TestFpsRescalingFrameSelector:
    """Tests for FpsRescalingFrameSelector specific functionality."""

    def test_fps_selector_initializes_with_fps(self) -> None:
        """Test constructor stores fps parameter."""
        selector = FpsRescalingFrameSelector(fps=5.0)
        assert selector.fps == 5.0

    def test_fps_selector_with_default_fps(self) -> None:
        """Test constructor with default fps value."""
        selector = FpsRescalingFrameSelector()
        assert selector.fps == 1.0

    def test_select_frames_returns_generator(self, video_path: Path) -> None:
        """Verify select_frames returns a Generator."""
        selector = FpsRescalingFrameSelector(fps=5.0)
        result = selector.select_frames(video_path)
        assert isinstance(result, Generator)

    def test_select_frames_yields_frames(self, video_path: Path) -> None:
        """Verify yielded items are Frame instances."""
        selector = FpsRescalingFrameSelector(fps=5.0)
        frames = list(selector.select_frames(video_path))
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, Frame)
            assert hasattr(frame, "raw")
            assert hasattr(frame, "frame_no")

    def test_fps_selector_reduces_frame_count(self, video_path: Path) -> None:
        """Test that higher FPS reduction results in fewer frames."""
        selector_10fps = FpsRescalingFrameSelector(fps=10.0)
        selector_2fps = FpsRescalingFrameSelector(fps=2.0)

        frames_10 = list(selector_10fps.select_frames(video_path))
        frames_2 = list(selector_2fps.select_frames(video_path))

        assert len(frames_2) < len(frames_10)

    def test_fps_selector_preserves_frame_order(self, video_path: Path) -> None:
        """Test that frames maintain sequential ordering."""
        selector = FpsRescalingFrameSelector(fps=5.0)
        frames = list(selector.select_frames(video_path))

        frame_numbers = [frame.frame_no for frame in frames]
        assert frame_numbers == sorted(frame_numbers)

    @pytest.mark.parametrize("fps_value", [1.0, 5.0, 10.0])
    def test_fps_selector_parametrized(self, video_path: Path, fps_value: float) -> None:
        """Test various FPS values."""
        selector = FpsRescalingFrameSelector(fps=fps_value)
        frames = list(selector.select_frames(video_path))
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, Frame)

    def test_fps_selector_zero_fps_raises_error(self, video_path: Path) -> None:
        """Test that fps=0 raises ZeroDivisionError."""
        selector = FpsRescalingFrameSelector(fps=0.0)
        with pytest.raises(ZeroDivisionError):
            list(selector.select_frames(video_path))


class TestRescaledFrameImageExtractor:
    """Tests for RescaledFrameImageExtractor specific functionality."""

    def test_rescaled_extractor_initializes_with_resolution(self) -> None:
        """Test constructor stores resolution parameter."""
        resolution = (320, 240)
        extractor = RescaledFrameImageExtractor(resolution=resolution)
        assert extractor.resolution == resolution

    def test_rescaled_extractor_with_default_resolution(self) -> None:
        """Test constructor with default 640x360 resolution."""
        extractor = RescaledFrameImageExtractor()
        assert extractor.resolution == (640, 360)

    def test_extract_images_returns_sequence(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Verify extract_images returns a Sequence."""
        extractor = RescaledFrameImageExtractor()
        result = extractor.extract_images(sample_frames, tmp_path)
        assert isinstance(result, Sequence)
        if result:
            assert isinstance(result[0], Sequence)
            assert all(isinstance(p, Path) for p in result[0])

    def test_extract_images_creates_files(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Verify images are written to output directory."""
        extractor = RescaledFrameImageExtractor()
        extractor.extract_images(sample_frames, tmp_path)

        image_files = list(tmp_path.glob("*.jpg"))
        assert len(image_files) == len(sample_frames)

    def test_extract_images_downscales_images(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test output images match target resolution."""
        resolution = (320, 240)
        extractor = RescaledFrameImageExtractor(resolution=resolution)
        extractor.extract_images(sample_frames, tmp_path)

        for image_file in tmp_path.glob("*.jpg"):
            img = cv2.imread(str(image_file))
            assert img is not None
            h, w = img.shape[:2]
            assert w <= resolution[0]
            assert h <= resolution[1]

    def test_rescaled_extractor_preserves_aspect_ratio(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test aspect ratio is maintained in output images."""
        resolution = (320, 240)
        extractor = RescaledFrameImageExtractor(resolution=resolution)
        extractor.extract_images(sample_frames, tmp_path)

        original_ratio = sample_frames[0].width_raw / sample_frames[0].height_raw

        for image_file in tmp_path.glob("*.jpg"):
            img = cv2.imread(str(image_file))
            assert img is not None
            h, w = img.shape[:2]
            output_ratio = w / h
            assert abs(output_ratio - original_ratio) < 0.05

    @pytest.mark.parametrize("resolution", [(128, 72), (640, 360), (800, 600), (1024, 768)])
    def test_rescaled_extractor_with_custom_resolution(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
        resolution: tuple[int, int],
    ) -> None:
        """Test custom resolutions work correctly."""
        extractor = RescaledFrameImageExtractor(resolution=resolution)
        extractor.extract_images(sample_frames, tmp_path)

        for image_file in tmp_path.glob("*.jpg"):
            img = cv2.imread(str(image_file))
            assert img is not None
            h, w = img.shape[:2]
            assert w <= resolution[0]
            assert h <= resolution[1]

    def test_rescaled_extractor_skips_filtered_frames(
        self,
        sample_frames_with_filtering: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test only filter_keep=True frames are processed."""
        extractor = RescaledFrameImageExtractor()
        extractor.extract_images(sample_frames_with_filtering, tmp_path)

        image_files = list(tmp_path.glob("*.jpg"))
        expected_count = sum(1 for f in sample_frames_with_filtering if f.filter_keep)
        assert len(image_files) == expected_count

    def test_rescaled_extractor_batch_structure(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test returns list of lists (batches)."""
        extractor = RescaledFrameImageExtractor()
        result = extractor.extract_images(sample_frames, tmp_path)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == len(sample_frames)
        assert all(isinstance(p, Path) for p in result[0])

    def test_extract_images_with_empty_frames(self, tmp_path: Path) -> None:
        """Test edge case with empty frame list."""
        extractor = RescaledFrameImageExtractor()
        result = extractor.extract_images([], tmp_path)

        assert result == []

    def test_rescaled_extractor_with_all_filtered_frames(
        self,
        sample_frames_all_filtered: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test when all frames have filter_keep=False."""
        extractor = RescaledFrameImageExtractor()
        result = extractor.extract_images(sample_frames_all_filtered, tmp_path)

        assert result == []
        image_files = list(tmp_path.glob("*.jpg"))
        assert len(image_files) == 0

    def test_extract_images_uses_outdir(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test files are created in correct directory."""
        subdir = tmp_path / "subdir" / "nested"
        extractor = RescaledFrameImageExtractor()
        extractor.extract_images(sample_frames, subdir)

        assert subdir.exists()
        image_files = list(subdir.glob("*.jpg"))
        assert len(image_files) == len(sample_frames)

    def test_rescaled_extractor_with_rescaled_frames(
        self,
        sample_frames_with_rescale: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test extractor handles frames with existing rescale attribute."""
        resolution = (160, 120)
        extractor = RescaledFrameImageExtractor(resolution=resolution)
        extractor.extract_images(sample_frames_with_rescale, tmp_path)

        image_files = list(tmp_path.glob("*.jpg"))
        assert len(image_files) == len(sample_frames_with_rescale)

        for image_file in image_files:
            img = cv2.imread(str(image_file))
            assert img is not None
            h, w = img.shape[:2]
            assert w <= resolution[0]
            assert h <= resolution[1]

    @pytest.mark.parametrize(
        "resolution",
        [(128, 72), (320, 240), (640, 360), (800, 600), (1024, 768)],
    )
    def test_rescaled_extractor_parametrized(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
        resolution: tuple[int, int],
    ) -> None:
        """Test various resolution values."""
        extractor = RescaledFrameImageExtractor(resolution=resolution)
        result = extractor.extract_images(sample_frames, tmp_path)

        assert len(result) == 1
        assert len(result[0]) == len(sample_frames)


class TestLlmImageBatchQuery:
    """Tests for LlmImageBatchQuery specific functionality."""

    def test_llm_query_initializes_with_params(self) -> None:
        """Test constructor stores parameters."""
        mock_llm = MockAbstractLlm()
        prompt = "Test prompt"
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt=prompt,
            response_class=MockSpeciesResult,
        )
        assert query.llm is mock_llm
        assert query.prompt == prompt
        assert query.response_class is MockSpeciesResult

    def test_query_images_returns_model_instance(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test query_images returns a model instance."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        result = query.query_images(sample_image_paths)
        assert isinstance(result, MockSpeciesResult)
        assert result.species_name == "test_species"

    def test_query_images_calls_analyser(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test query_images calls the analyser."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        query.query_images(sample_image_paths)
        assert mock_llm.call_count == 1

    def test_query_images_sorts_images(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test images are sorted before sending to analyser."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        query.query_images(list(reversed(sample_image_paths)))
        assert mock_llm.last_images == sorted(sample_image_paths)

    def test_query_images_passes_prompt(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test prompt is passed to analyser."""
        mock_llm = MockAbstractLlm()
        custom_prompt = "Custom test prompt"
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt=custom_prompt,
            response_class=MockSpeciesResult,
        )
        query.query_images(sample_image_paths)
        assert mock_llm.last_prompt == custom_prompt

    def test_query_images_with_custom_response(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test custom response value from analyser."""
        mock_llm = MockAbstractLlm(response_to_return="custom_species")
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        result = query.query_images(sample_image_paths)
        assert result.species_name == "custom_species"

    def test_query_images_empty_batch_raises_error(self) -> None:
        """Test empty batch raises ValueError."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        with pytest.raises(ValueError, match="Empty image batch"):
            query.query_images([])

    def test_query_image_batches_yields_results(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test query_image_batches yields results for each batch."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        batches = [sample_image_paths, sample_image_paths]
        results = list(query.query_image_batches(batches))
        assert len(results) == 2
        assert all(isinstance(r, MockSpeciesResult) for r in results)

    def test_query_image_batches_multiple_batches(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test query_image_batches processes multiple batches."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        batches = [sample_image_paths, sample_image_paths, sample_image_paths]
        list(query.query_image_batches(batches))
        assert mock_llm.call_count == 3

    @pytest.mark.parametrize("batch_count", [1, 3, 5])
    def test_query_image_batches_parametrized(
        self,
        sample_image_paths: list[Path],
        batch_count: int,
    ) -> None:
        """Test various batch counts."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        batches = [sample_image_paths] * batch_count
        results = list(query.query_image_batches(batches))
        assert len(results) == batch_count
        assert mock_llm.call_count == batch_count

    def test_query_images_return_type_matches_response_class(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test returned instance matches response_class."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        result = query.query_images(sample_image_paths)
        assert isinstance(result, MockSpeciesResult)
        assert type(result) is MockSpeciesResult

    def test_query_images_single_image(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test single image batch edge case."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        result = query.query_images([sample_image_paths[0]])
        assert isinstance(result, MockSpeciesResult)
        assert mock_llm.call_count == 1


class TestMajorityResultReconciler:
    """Tests for MajorityResultReconciler specific functionality."""

    def test_reconciler_with_single_result(self) -> None:
        """Test reconciler returns single result unchanged."""
        reconciler: MajorityResultReconciler[MockSpeciesResult] = MajorityResultReconciler()
        result = reconciler.reconcile_results([MockSpeciesResult(species_name="test")])
        assert result.species_name == "test"

    def test_reconciler_with_identical_results(self) -> None:
        """Test reconciler returns result when all are identical."""
        reconciler: MajorityResultReconciler[MockSpeciesResult] = MajorityResultReconciler()
        results = [MockSpeciesResult(species_name="same") for _ in range(5)]
        result = reconciler.reconcile_results(results)
        assert result.species_name == "same"

    def test_reconciler_with_clear_majority(self) -> None:
        """Test reconciler returns majority result."""
        reconciler: MajorityResultReconciler[MockSpeciesResult] = MajorityResultReconciler()
        results = [
            MockSpeciesResult(species_name="A"),
            MockSpeciesResult(species_name="A"),
            MockSpeciesResult(species_name="A"),
            MockSpeciesResult(species_name="B"),
            MockSpeciesResult(species_name="B"),
        ]
        result = reconciler.reconcile_results(results)
        assert result.species_name == "A"

    def test_reconciler_with_tie_takes_first(self) -> None:
        """Test reconciler returns first-seen result in case of tie."""
        reconciler: MajorityResultReconciler[MockSpeciesResult] = MajorityResultReconciler()
        results = [
            MockSpeciesResult(species_name="A"),
            MockSpeciesResult(species_name="B"),
            MockSpeciesResult(species_name="B"),
            MockSpeciesResult(species_name="A"),
        ]
        result = reconciler.reconcile_results(results)
        assert result.species_name == "A"

    def test_reconciler_with_empty_results_raises_error(self) -> None:
        """Test reconciler raises error with empty results."""
        reconciler: MajorityResultReconciler[MockSpeciesResult] = MajorityResultReconciler()
        with pytest.raises(ValueError, match="No results to reconcile"):
            reconciler.reconcile_results([])

    def test_reconciler_preserves_object_identity(self) -> None:
        """Test reconciler returns one of the input objects, not a copy."""
        reconciler: MajorityResultReconciler[MockSpeciesResult] = MajorityResultReconciler()
        result_obj = MockSpeciesResult(species_name="test")
        results = [result_obj, result_obj, result_obj]
        result = reconciler.reconcile_results(results)
        assert result is result_obj

    @pytest.mark.parametrize("count", [1, 2, 3, 5, 10])
    def test_reconciler_with_varying_result_counts(
        self,
        count: int,
    ) -> None:
        """Test reconciler with varying numbers of results."""
        reconciler: MajorityResultReconciler[MockSpeciesResult] = MajorityResultReconciler()
        results = [MockSpeciesResult(species_name="majority")] * count
        result = reconciler.reconcile_results(results)
        assert result.species_name == "majority"

    def test_reconciler_with_complex_tie_scenario(self) -> None:
        """Test reconciler with multiple results tied for majority."""
        reconciler: MajorityResultReconciler[MockSpeciesResult] = MajorityResultReconciler()
        results = [
            MockSpeciesResult(species_name="A"),
            MockSpeciesResult(species_name="B"),
            MockSpeciesResult(species_name="C"),
            MockSpeciesResult(species_name="A"),
            MockSpeciesResult(species_name="B"),
            MockSpeciesResult(species_name="C"),
        ]
        result = reconciler.reconcile_results(results)
        assert result.species_name == "A"

    def test_reconciler_with_generator_input(self) -> None:
        """Test reconciler accepts generator/iterator input."""
        reconciler: MajorityResultReconciler[MockSpeciesResult] = MajorityResultReconciler()

        def result_generator() -> Generator[MockSpeciesResult]:
            for _ in range(3):
                yield MockSpeciesResult(species_name="A")

        result = reconciler.reconcile_results(result_generator())
        assert result.species_name == "A"

    def test_reconciler_with_string_results(self) -> None:
        """Test reconciler works with non-BaseModel types."""
        reconciler: MajorityResultReconciler[str] = MajorityResultReconciler()
        results = ["A", "A", "B", "A", "B"]
        result = reconciler.reconcile_results(results)
        assert result == "A"


class TestVerifiedImageBatchQuery:
    """Tests for VerifiedImageBatchQuery two-stage verification."""

    def test_verified_query_initializes_with_params(self) -> None:
        """Test constructor stores parameters."""
        mock_llm = MockVerifiedLlm()
        prompt = "Test prompt"
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt=prompt,
            response_class=MockSpeciesResult,
        )
        assert query.llm is mock_llm
        assert query.prompt == prompt
        assert query.response_class is MockSpeciesResult
        assert query.min_confidence == ConfidenceLevel.MEDIUM

    def test_verified_query_with_custom_min_confidence(self) -> None:
        """Test constructor accepts custom min_confidence."""
        mock_llm = MockVerifiedLlm()
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
            min_confidence=ConfidenceLevel.HIGH,
        )
        assert query.min_confidence == ConfidenceLevel.HIGH

    def test_verified_query_with_custom_verification_prompt(self) -> None:
        """Test constructor accepts custom verification prompt."""
        mock_llm = MockVerifiedLlm()
        custom_prompt = "Custom verification: {initial_species}"
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
            verification_prompt=custom_prompt,
        )
        assert query.verification_prompt == custom_prompt

    def test_verified_query_images_makes_two_calls(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test query_images makes two LLM calls (initial + verification)."""
        mock_llm = MockVerifiedLlm()
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        query.query_images(sample_image_paths)
        assert mock_llm.call_count == 2

    def test_verified_query_returns_verification_result(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test query_images returns VerificationResult."""
        mock_llm = MockVerifiedLlm()
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        result = query.query_images(sample_image_paths)
        assert isinstance(result, VerificationResult)
        assert result.species_name == "fox"
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.verified is True

    def test_verified_query_passes_initial_species_to_verification(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test initial species is passed to verification prompt."""
        mock_llm = MockVerifiedLlm(initial_species="badger")
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        query.query_images(sample_image_paths)
        assert "badger" in mock_llm.verification_prompt

    def test_verified_query_low_confidence_becomes_unknown(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test low confidence result is marked as unknown."""
        mock_llm = MockVerifiedLlm(
            initial_species="fox",
            verification_confidence=ConfidenceLevel.LOW,
        )
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
            min_confidence=ConfidenceLevel.MEDIUM,
        )
        result = query.query_images(sample_image_paths)
        assert result.species_name == "unknown"
        assert result.verified is False
        assert result.confidence == ConfidenceLevel.LOW

    def test_verified_query_medium_confidence_passes_with_medium_threshold(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test medium confidence passes when threshold is medium."""
        mock_llm = MockVerifiedLlm(
            verification_confidence=ConfidenceLevel.MEDIUM,
        )
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
            min_confidence=ConfidenceLevel.MEDIUM,
        )
        result = query.query_images(sample_image_paths)
        assert result.species_name == "fox"
        assert result.verified is True

    def test_verified_query_medium_confidence_fails_with_high_threshold(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test medium confidence fails when threshold is high."""
        mock_llm = MockVerifiedLlm(
            verification_confidence=ConfidenceLevel.MEDIUM,
        )
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
            min_confidence=ConfidenceLevel.HIGH,
        )
        result = query.query_images(sample_image_paths)
        assert result.species_name == "unknown"
        assert result.verified is False

    def test_verified_query_high_confidence_passes_all_thresholds(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test high confidence passes all threshold levels."""
        for threshold in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]:
            mock_llm = MockVerifiedLlm(
                verification_confidence=ConfidenceLevel.HIGH,
            )
            query = VerifiedImageBatchQuery(
                llm=mock_llm,
                prompt="Test prompt",
                response_class=MockSpeciesResult,
                min_confidence=threshold,
            )
            result = query.query_images(sample_image_paths)
            assert result.species_name == "fox"
            assert result.verified is True

    def test_verified_query_empty_batch_raises_error(self) -> None:
        """Test empty batch raises ValueError."""
        mock_llm = MockVerifiedLlm()
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        with pytest.raises(ValueError, match="Empty image batch"):
            query.query_images([])

    def test_verified_query_sorts_images(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test images are sorted before sending to LLM."""
        mock_llm = MockVerifiedLlm()
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        query.query_images(list(reversed(sample_image_paths)))
        assert mock_llm.last_images == sorted(sample_image_paths)

    def test_verified_query_uses_corrected_species_when_provided(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test corrected species is used when verification provides one."""
        mock_llm = MockVerifiedLlm(
            initial_species="fox",
            verification_corrected_species="badger",
            verification_confidence=ConfidenceLevel.HIGH,
        )
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        result = query.query_images(sample_image_paths)
        assert result.species_name == "badger"
        assert result.verified is True

    def test_verified_query_verified_false_becomes_unknown(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test verified=false results in unknown species."""
        mock_llm = MockVerifiedLlm(
            verification_verified=False,
            verification_confidence=ConfidenceLevel.HIGH,
        )
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        result = query.query_images(sample_image_paths)
        assert result.species_name == "unknown"
        assert result.verified is False

    def test_query_image_batches_yields_verification_results(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test query_image_batches yields VerificationResult for each batch."""
        mock_llm = MockVerifiedLlm()
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        batches = [sample_image_paths, sample_image_paths]
        results = list(query.query_image_batches(batches))
        assert len(results) == 2
        assert all(isinstance(r, VerificationResult) for r in results)

    def test_query_image_batches_multiple_batches(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test query_image_batches processes multiple batches."""
        mock_llm = MockVerifiedLlm()
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        batches = [sample_image_paths, sample_image_paths, sample_image_paths]
        list(query.query_image_batches(batches))
        assert mock_llm.call_count == 6

    @pytest.mark.parametrize("batch_count", [1, 3, 5])
    def test_query_image_batches_parametrized(
        self,
        sample_image_paths: list[Path],
        batch_count: int,
    ) -> None:
        """Test various batch counts."""
        mock_llm = MockVerifiedLlm()
        query = VerifiedImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
            response_class=MockSpeciesResult,
        )
        batches = [sample_image_paths] * batch_count
        results = list(query.query_image_batches(batches))
        assert len(results) == batch_count
        assert mock_llm.call_count == batch_count * 2

    def test_verified_query_with_low_threshold_accepts_all(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test LOW threshold accepts all confidence levels."""
        for confidence in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]:
            mock_llm = MockVerifiedLlm(verification_confidence=confidence)
            query = VerifiedImageBatchQuery(
                llm=mock_llm,
                prompt="Test prompt",
                response_class=MockSpeciesResult,
                min_confidence=ConfidenceLevel.LOW,
            )
            result = query.query_images(sample_image_paths)
            assert result.species_name == "fox"
            assert result.verified is True
