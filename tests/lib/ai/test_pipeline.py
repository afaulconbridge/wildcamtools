"""Tests for AI pipeline components: FrameSelector, FrameImageExtractor, and ImageBatchQuery."""

from collections.abc import Generator, Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import BaseModel

from wildcamtools.lib import Frame
from wildcamtools.lib.ai import Backend
from wildcamtools.lib.ai.crop import AICropFinder
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.pipeline import (
    AICroppedFrameImageExtractor,
    BatchResult,
    ContrastEnhancedFrameImageExtractor,
    ExtractedBatch,
    ExtractedFrame,
    ExtractedFrames,
    ExtractedFramesWithResults,
    FpsRescalingFrameSelector,
    LlmImageBatchQuery,
    MotionFrameSelector,
    PipelineOutcome,
    RescaledFrameImageExtractor,
    RichResultBatchResult,
    RichResultPipelineOutcome,
    SSIMFrameSelector,
    VerifiedImageBatchQuery,
)
from wildcamtools.lib.ai.types import ConfidenceLevel, FrameResult, Result, ResultList, RichResult, VerificationResult
from wildcamtools.lib.stats import Colourspace, VideoStats


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
        result_list_to_return: ResultList | None = None,
        return_rich_result: bool = False,
    ) -> None:
        self.model = model
        self.backend = backend
        self.url = url
        self.response_to_return = response_to_return
        self.result_list_to_return = result_list_to_return
        self.return_rich_result = return_rich_result
        self.call_count = 0
        self.last_images: list[Path] = []
        self.last_prompt: str = ""

    def message_with_schema(
        self,
        message: str,
        images: Sequence[Path] = (),
        response_class: type = RichResult,
    ):
        self.call_count += 1
        self.last_images = list(images)
        self.last_prompt = message
        if response_class is ResultList and self.result_list_to_return is not None:
            return self.result_list_to_return
        if response_class is RichResult:
            result = RichResult(
                is_animal_present=True,
                is_animal_unknown=False,
                defining_features="test features",
                species_name=self.response_to_return,
                confidence=ConfidenceLevel.HIGH,
            )
        else:
            result = MockSpeciesResult(species_name=self.response_to_return)
        return result


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

    def message_with_schema(
        self,
        message: str,
        images: Sequence[Path] = (),
        response_class: type = RichResult,
    ):
        self.call_count += 1
        self.last_images = list(images)

        if response_class is VerificationResult:
            self.verification_prompt = message
            result = VerificationResult(
                species_name=self.verification_corrected_species or self.initial_species,
                confidence=self.verification_confidence,
                verified=self.verification_verified,
            )
            return result
        self.last_prompt = message
        result = RichResult(
            is_animal_present=self.initial_species not in ("unknown", "no animal"),
            is_animal_unknown=self.initial_species == "unknown",
            defining_features="test features",
            species_name=self.initial_species,
            confidence=ConfidenceLevel.HIGH,
        )
        return result


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


class TestMotionFrameSelector:
    """Tests for MotionFrameSelector specific functionality."""

    def test_motion_selector_initializes_with_defaults(self) -> None:
        """Test constructor with default values."""
        selector = MotionFrameSelector()
        assert selector.fps == 5.0
        assert selector.motion_threshold == 0.01
        assert selector.resolution is None
        assert selector.history == 30

    def test_motion_selector_initializes_with_custom_values(self) -> None:
        """Test constructor stores custom parameters."""
        selector = MotionFrameSelector(
            fps=10.0,
            motion_threshold=0.05,
            resolution=(640, 360),
            history=50,
        )
        assert selector.fps == 10.0
        assert selector.motion_threshold == 0.05
        assert selector.resolution == (640, 360)
        assert selector.history == 50

    def test_select_frames_returns_generator(self, video_path: Path) -> None:
        """Verify select_frames returns a Generator."""
        selector = MotionFrameSelector()
        result = selector.select_frames(video_path)
        assert isinstance(result, Generator)

    def test_select_frames_yields_frames(self, video_path: Path) -> None:
        """Verify yielded items are Frame instances."""
        selector = MotionFrameSelector()
        frames = list(selector.select_frames(video_path))
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, Frame)
            assert hasattr(frame, "raw")
            assert hasattr(frame, "frame_no")

    def test_motion_selector_respects_max_fps(self, video_path: Path) -> None:
        """Test that fps parameter reduces frame count."""
        selector_10fps = MotionFrameSelector(fps=10.0)
        selector_2fps = MotionFrameSelector(fps=2.0)

        frames_10 = list(selector_10fps.select_frames(video_path))
        frames_2 = list(selector_2fps.select_frames(video_path))

        assert len(frames_2) < len(frames_10)

    def test_motion_selector_preserves_frame_order(self, video_path: Path) -> None:
        """Test that frames maintain sequential ordering."""
        selector = MotionFrameSelector()
        frames = list(selector.select_frames(video_path))

        frame_numbers = [frame.frame_no for frame in frames]
        assert frame_numbers == sorted(frame_numbers)

    def test_motion_selector_with_resolution(self, video_path: Path) -> None:
        """Test that resolution parameter controls motion detection resolution."""
        selector = MotionFrameSelector(resolution=(320, 240))
        frames = list(selector.select_frames(video_path))

        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, Frame)
            assert hasattr(frame, "motion_proportion")

    def test_motion_selector_with_high_threshold(self, video_path: Path) -> None:
        """Test that high motion threshold reduces frame count."""
        selector_low = MotionFrameSelector(motion_threshold=0.001)
        selector_high = MotionFrameSelector(motion_threshold=0.5)

        frames_low = list(selector_low.select_frames(video_path))
        frames_high = list(selector_high.select_frames(video_path))

        assert len(frames_high) <= len(frames_low)

    @pytest.mark.parametrize("fps_value", [1.0, 5.0, 10.0])
    def test_motion_selector_parametrized_fps(self, video_path: Path, fps_value: float) -> None:
        """Test various FPS values."""
        selector = MotionFrameSelector(fps=fps_value)
        frames = list(selector.select_frames(video_path))
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, Frame)

    def test_motion_selector_zero_fps(self, video_path: Path) -> None:
        """Test that fps=0 does not apply FPS filtering."""
        selector = MotionFrameSelector(fps=0.0)
        frames = list(selector.select_frames(video_path))
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, Frame)

    def test_motion_selector_resolution_affects_motion_detection(
        self,
        video_path: Path,
    ) -> None:
        """Test that resolution parameter is passed to motion handler."""
        selector_low_res = MotionFrameSelector(resolution=(160, 120), motion_threshold=0.001)
        selector_full_res = MotionFrameSelector(resolution=None, motion_threshold=0.001)

        frames_low = list(selector_low_res.select_frames(video_path))
        frames_full = list(selector_full_res.select_frames(video_path))

        assert len(frames_low) > 0
        assert len(frames_full) > 0

        for frame in frames_low:
            assert isinstance(frame, Frame)
            assert frame.motion_proportion >= -1.0


class TestSSIMFrameSelector:
    """Tests for SSIMFrameSelector specific functionality."""

    def test_ssim_selector_initializes_with_defaults(self) -> None:
        """Test constructor with default values."""
        selector = SSIMFrameSelector()
        assert selector.fps == 5.0
        assert selector.similarity_minimum == 0.9
        assert selector.resolution is None

    def test_ssim_selector_initializes_with_custom_values(self) -> None:
        """Test constructor stores custom parameters."""
        selector = SSIMFrameSelector(
            fps=10.0,
            similarity_minimum=0.95,
            resolution=(640, 360),
        )
        assert selector.fps == 10.0
        assert selector.similarity_minimum == 0.95
        assert selector.resolution == (640, 360)

    def test_select_frames_returns_generator(self, video_path: Path) -> None:
        """Verify select_frames returns a Generator."""
        selector = SSIMFrameSelector()
        result = selector.select_frames(video_path)
        assert isinstance(result, Generator)

    def test_select_frames_yields_frames(self, data_directory: Path) -> None:
        """Verify yielded items are Frame instances."""
        video_path = data_directory / "short.mp4"
        selector = SSIMFrameSelector(fps=1.0, similarity_minimum=0.0, resolution=(40, 30))
        frames = list(selector.select_frames(video_path))
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, Frame)
            assert hasattr(frame, "raw")
            assert hasattr(frame, "frame_no")

    def test_ssim_selector_respects_max_fps(self, data_directory: Path) -> None:
        """Test that fps parameter reduces frame count."""
        video_path = data_directory / "short.mp4"
        selector_high = SSIMFrameSelector(fps=5.0, similarity_minimum=0.0, resolution=(40, 30))
        selector_low = SSIMFrameSelector(fps=0.5, similarity_minimum=0.0, resolution=(40, 30))

        frames_high = list(selector_high.select_frames(video_path))
        frames_low = list(selector_low.select_frames(video_path))

        assert len(frames_low) <= len(frames_high)

    def test_ssim_selector_preserves_frame_order(self, data_directory: Path) -> None:
        """Test that frames maintain sequential ordering."""
        video_path = data_directory / "short.mp4"
        selector = SSIMFrameSelector(fps=1.0, similarity_minimum=0.0, resolution=(40, 30))
        frames = list(selector.select_frames(video_path))

        frame_numbers = [frame.frame_no for frame in frames]
        assert frame_numbers == sorted(frame_numbers)

    def test_ssim_selector_with_resolution(self, data_directory: Path) -> None:
        """Test that resolution parameter controls output resolution."""
        video_path = data_directory / "short.mp4"
        selector = SSIMFrameSelector(resolution=(40, 30), fps=1.0, similarity_minimum=0.0)
        frames = list(selector.select_frames(video_path))

        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, Frame)

    def test_ssim_selector_with_high_similarity_minimum(self, data_directory: Path) -> None:
        """Test that high similarity_minimum reduces frame count."""
        video_path = data_directory / "short.mp4"
        selector_low = SSIMFrameSelector(similarity_minimum=0.0, fps=1.0, resolution=(40, 30))
        selector_high = SSIMFrameSelector(similarity_minimum=0.99, fps=1.0, resolution=(40, 30))

        frames_low = list(selector_low.select_frames(video_path))
        frames_high = list(selector_high.select_frames(video_path))

        assert len(frames_high) <= len(frames_low)

    @pytest.mark.parametrize("fps_value", [0.5, 1.0, 2.0])
    def test_ssim_selector_parametrized_fps(self, data_directory: Path, fps_value: float) -> None:
        """Test various FPS values."""
        video_path = data_directory / "short.mp4"
        selector = SSIMFrameSelector(fps=fps_value, similarity_minimum=0.0, resolution=(40, 30))
        frames = list(selector.select_frames(video_path))
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, Frame)

    def test_ssim_selector_zero_fps(self, data_directory: Path) -> None:
        """Test that fps=0 does not apply FPS filtering."""
        video_path = data_directory / "short.mp4"
        selector = SSIMFrameSelector(fps=0.0, similarity_minimum=0.0, resolution=None)
        frames = list(selector.select_frames(video_path))
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, Frame)


class TestFrameImagePair:
    """Tests for FrameImagePair class."""

    def test_frame_image_pair_creation(self) -> None:
        """Test FrameImagePair can be created with path and frame_no."""
        path = Path("/test/image.jpg")
        pair = ExtractedFrame(path=path, frame_no=42)
        assert pair.path == path
        assert pair.frame_no == 42

    def test_frame_image_pair_path_excluded_from_json(self) -> None:
        """Test path field is excluded from JSON serialization."""
        path = Path("/test/image.jpg")
        pair = ExtractedFrame(path=path, frame_no=42)
        json_data = pair.model_dump()
        assert "frame_no" in json_data
        assert "path" not in json_data

    def test_frame_image_pair_json_content(self) -> None:
        """Test JSON contains only frame_no."""
        pair = ExtractedFrame(path=Path("/test.jpg"), frame_no=123)
        json_data = pair.model_dump()
        assert json_data == {"frame_no": 123}

    def test_frame_image_pair_round_trip_serialization(self) -> None:
        """Test ExtractedFrame can be serialized and deserialized (round-trip)."""
        original = ExtractedFrame(path=Path("/test/image.jpg"), frame_no=42)
        json_str = original.model_dump_json()
        deserialized = ExtractedFrame.model_validate_json(json_str)
        assert deserialized.frame_no == 42
        assert deserialized.path is None

    def test_frame_image_pair_require_path_with_path_set(self) -> None:
        """Test require_path() returns path when it's set."""
        path = Path("/test/image.jpg")
        pair = ExtractedFrame(path=path, frame_no=42)
        result = pair.require_path()
        assert result == path
        assert isinstance(result, Path)

    def test_frame_image_pair_require_path_with_path_none(self) -> None:
        """Test require_path() raises ValueError when path is None."""
        pair = ExtractedFrame(frame_no=42)
        with pytest.raises(ValueError, match=r"ExtractedFrame\.path is required"):
            pair.require_path()

    def test_frame_image_pair_require_path_after_deserialization(self) -> None:
        """Test require_path() raises ValueError after JSON deserialization."""
        original = ExtractedFrame(path=Path("/test/image.jpg"), frame_no=42)
        json_str = original.model_dump_json()
        deserialized = ExtractedFrame.model_validate_json(json_str)
        with pytest.raises(ValueError, match=r"ExtractedFrame\.path is required"):
            deserialized.require_path()


class TestExtractedBatch:
    """Tests for ExtractedBatch class."""

    def test_extracted_batch_creation(self) -> None:
        """Test ExtractedBatch can be created with frame_image_pairs."""
        pairs = [ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i) for i in range(3)]
        batch = ExtractedBatch(selected_frames=pairs)
        assert len(batch.selected_frames) == 3
        assert batch.selected_frames[0].frame_no == 0

    def test_extracted_batch_json_excludes_paths(self) -> None:
        """Test JSON serialization excludes paths."""
        pairs = [ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i) for i in range(2)]
        batch = ExtractedBatch(selected_frames=pairs)
        json_data = batch.model_dump()
        assert "selected_frames" in json_data
        assert all("path" not in pair for pair in json_data["selected_frames"])
        assert all("frame_no" in pair for pair in json_data["selected_frames"])


class TestBatchResult:
    """Tests for BatchResult class."""

    def test_batch_result_creation_without_result(self) -> None:
        """Test BatchResult can be created without result."""
        pairs = [ExtractedFrame(path=Path("frame.jpg"), frame_no=1)]
        batch = BatchResult(selected_frames=pairs)
        assert batch.result is None

    def test_batch_result_creation_with_result(self) -> None:
        """Test BatchResult can be created with result."""
        pairs = [ExtractedFrame(path=Path("frame.jpg"), frame_no=1)]
        result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test",
            species_name="fox",
            confidence=ConfidenceLevel.HIGH,
        )
        batch = BatchResult(selected_frames=pairs, result=result)
        assert batch.result is not None
        assert batch.result.species_name == "fox"

    def test_batch_result_inherits_from_extracted_batch(self) -> None:
        """Test BatchResult inherits ExtractedBatch functionality."""
        pairs = [ExtractedFrame(path=Path("frame.jpg"), frame_no=1)]
        batch = BatchResult(selected_frames=pairs)
        assert isinstance(batch, ExtractedBatch)
        assert len(batch.selected_frames) == 1

    def test_batch_result_round_trip_serialization(self) -> None:
        """Test BatchResult can be serialized and deserialized (round-trip)."""
        pairs = [ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i) for i in range(3)]
        result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test features",
            species_name="Red Fox",
            confidence=ConfidenceLevel.HIGH,
        )
        original = RichResultBatchResult(selected_frames=pairs, result=result)
        json_str = original.model_dump_json()
        deserialized = RichResultBatchResult.model_validate_json(json_str)
        assert len(deserialized.selected_frames) == 3
        assert all(frame.path is None for frame in deserialized.selected_frames)
        assert [f.frame_no for f in deserialized.selected_frames] == [0, 1, 2]
        assert deserialized.result is not None
        assert deserialized.result.species_name == "Red Fox"

    def test_batch_result_round_trip_without_result(self) -> None:
        """Test BatchResult without result can be serialized and deserialized."""
        pairs = [ExtractedFrame(path=Path("frame.jpg"), frame_no=1)]
        original = BatchResult(selected_frames=pairs)
        json_str = original.model_dump_json()
        deserialized = BatchResult.model_validate_json(json_str)
        assert len(deserialized.selected_frames) == 1
        assert deserialized.selected_frames[0].frame_no == 1
        assert deserialized.result is None


class TestExtractedFrames:
    """Tests for ExtractedFrames class."""

    def test_extracted_frames_creation(self) -> None:
        """Test ExtractedFrames can be created with batches."""
        pairs = [ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i) for i in range(5)]
        batch = ExtractedBatch(selected_frames=pairs)
        frames = ExtractedFrames(batches=[batch])
        assert len(frames.batches) == 1
        assert len(frames.batches[0].selected_frames) == 5

    def test_extracted_frames_frame_ids_property(self) -> None:
        """Test frame_ids property extracts frame numbers correctly."""
        pairs = [ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i * 10) for i in range(5)]
        batch = ExtractedBatch(selected_frames=pairs)
        frames = ExtractedFrames(batches=[batch])
        assert frames.frame_ids == [[0, 10, 20, 30, 40]]

    def test_extracted_frames_frame_ids_multiple_batches(self) -> None:
        """Test frame_ids with multiple batches."""
        batch1 = ExtractedBatch(selected_frames=[ExtractedFrame(path=Path(f"f{i}.jpg"), frame_no=i) for i in range(3)])
        batch2 = ExtractedBatch(
            selected_frames=[ExtractedFrame(path=Path(f"f{i}.jpg"), frame_no=i + 10) for i in range(2)],
        )
        frames = ExtractedFrames(batches=[batch1, batch2])
        assert frames.frame_ids == [[0, 1, 2], [10, 11]]

    def test_extracted_frames_get_batches_yields_paths(self) -> None:
        """Test get_batches yields lists of paths."""
        pairs = [ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i) for i in range(5)]
        batch = ExtractedBatch(selected_frames=pairs)
        frames = ExtractedFrames(batches=[batch])
        batches = list(frames.get_batches())
        assert len(batches) == 1
        assert len(batches[0]) == 5
        assert all(isinstance(p, Path) for p in batches[0])
        assert batches[0][0] == Path("frame_0.jpg")

    def test_extracted_frames_len_returns_batch_count(self) -> None:
        """Test __len__ returns number of batches."""
        batch1 = ExtractedBatch(selected_frames=[ExtractedFrame(path=Path("f1.jpg"), frame_no=1)])
        batch2 = ExtractedBatch(selected_frames=[ExtractedFrame(path=Path("f2.jpg"), frame_no=2)])
        frames = ExtractedFrames(batches=[batch1, batch2])
        assert len(frames) == 2

    def test_extracted_frames_empty_batches(self) -> None:
        """Test ExtractedFrames with no batches."""
        frames = ExtractedFrames(batches=[])
        assert len(frames) == 0
        assert frames.frame_ids == []
        assert list(frames.get_batches()) == []


class TestExtractedFramesWithResults:
    """Tests for ExtractedFramesWithResults class."""

    def test_extracted_frames_with_results_creation(self) -> None:
        """Test ExtractedFramesWithResults can be created."""
        pairs = [ExtractedFrame(path=Path("frame.jpg"), frame_no=1)]
        result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test",
            species_name="fox",
            confidence=ConfidenceLevel.HIGH,
        )
        batch = BatchResult(selected_frames=pairs, result=result)
        frames = ExtractedFramesWithResults(batches=[batch])
        assert len(frames.batches) == 1
        assert frames.batches[0].result is not None

    def test_get_batch_results_extracts_results(self) -> None:
        """Test get_batch_results extracts non-None results."""
        pairs = [ExtractedFrame(path=Path("frame.jpg"), frame_no=1)]
        result1 = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test1",
            species_name="fox",
            confidence=ConfidenceLevel.HIGH,
        )
        result2 = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test2",
            species_name="badger",
            confidence=ConfidenceLevel.HIGH,
        )
        batch1 = BatchResult(selected_frames=pairs, result=result1)
        batch2 = BatchResult(selected_frames=pairs, result=result2)
        batch3 = BatchResult(selected_frames=pairs, result=None)
        frames = ExtractedFramesWithResults(batches=[batch1, batch2, batch3])
        results = frames.get_batch_results()
        assert len(results) == 2
        assert results[0].species_name == "fox"
        assert results[1].species_name == "badger"

    def test_inherits_from_extracted_frames(self) -> None:
        """Test ExtractedFramesWithResults inherits ExtractedFrames functionality."""
        pairs = [ExtractedFrame(path=Path("frame.jpg"), frame_no=1)]
        result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test",
            species_name="fox",
            confidence=ConfidenceLevel.HIGH,
        )
        batch = BatchResult(selected_frames=pairs, result=result)
        frames = ExtractedFramesWithResults(batches=[batch])
        assert isinstance(frames, ExtractedFrames)
        assert frames.frame_ids == [[1]]
        assert len(frames) == 1


class TestPipelineOutcome:
    """Tests for PipelineOutcome serialization and round-trip."""

    def test_pipeline_outcome_creation(self) -> None:
        """Test PipelineOutcome can be created with result, stats, and batches."""
        result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test",
            species_name="fox",
            confidence=ConfidenceLevel.HIGH,
        )
        stats = VideoStats(fps=30.0, frame_count=100, x=1920, y=1080, colourspace=Colourspace.RGB)
        pairs = [ExtractedFrame(path=Path("frame.jpg"), frame_no=1)]
        batch = BatchResult(selected_frames=pairs, result=result)
        outcome = PipelineOutcome(result=result, stats=stats, batches=[batch])
        assert outcome.result.species_name == "fox"
        assert outcome.stats.fps == 30.0
        assert len(outcome.batches) == 1

    def test_pipeline_outcome_round_trip_serialization(self) -> None:
        """Test PipelineOutcome can be serialized and deserialized (round-trip)."""
        result = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="test features",
            species_name="Red Fox",
            confidence=ConfidenceLevel.HIGH,
        )
        stats = VideoStats(fps=30.0, frame_count=100, x=1920, y=1080, colourspace=Colourspace.RGB)
        pairs = [ExtractedFrame(path=Path(f"frame_{i}.jpg"), frame_no=i) for i in range(3)]
        batch = RichResultBatchResult(selected_frames=pairs, result=result)
        original = RichResultPipelineOutcome(result=result, stats=stats, batches=[batch])

        json_str = original.model_dump_json()
        deserialized = RichResultPipelineOutcome.model_validate_json(json_str)

        assert deserialized.result.species_name == "Red Fox"
        assert deserialized.stats.fps == 30.0
        assert deserialized.stats.frame_count == 100
        assert len(deserialized.batches) == 1
        assert len(deserialized.batches[0].selected_frames) == 3
        assert all(frame.path is None for frame in deserialized.batches[0].selected_frames)
        assert [f.frame_no for f in deserialized.batches[0].selected_frames] == [0, 1, 2]

    def test_pipeline_outcome_round_trip_empty_batches(self) -> None:
        """Test PipelineOutcome with empty batches can be serialized and deserialized."""
        result = RichResult(
            is_animal_present=False,
            is_animal_unknown=False,
            defining_features="",
            species_name="no animal",
            confidence=ConfidenceLevel.HIGH,
        )
        stats = VideoStats(fps=30.0, frame_count=100, x=1920, y=1080, colourspace=Colourspace.RGB)
        original = RichResultPipelineOutcome(result=result, stats=stats, batches=[])

        json_str = original.model_dump_json()
        deserialized = RichResultPipelineOutcome.model_validate_json(json_str)

        assert deserialized.result.is_animal_present is False
        assert deserialized.stats.fps == 30.0
        assert len(deserialized.batches) == 0

    def test_pipeline_outcome_round_trip_multiple_batches(self) -> None:
        """Test PipelineOutcome with multiple batches can be serialized and deserialized."""
        result1 = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="features 1",
            species_name="Fox",
            confidence=ConfidenceLevel.HIGH,
        )
        result2 = RichResult(
            is_animal_present=True,
            is_animal_unknown=False,
            defining_features="features 2",
            species_name="Badger",
            confidence=ConfidenceLevel.MEDIUM,
        )
        stats = VideoStats(fps=30.0, frame_count=100, x=1920, y=1080, colourspace=Colourspace.RGB)
        batch1 = RichResultBatchResult(
            selected_frames=[ExtractedFrame(path=Path("f1.jpg"), frame_no=1)],
            result=result1,
        )
        batch2 = RichResultBatchResult(
            selected_frames=[ExtractedFrame(path=Path("f2.jpg"), frame_no=2)],
            result=result2,
        )
        original = RichResultPipelineOutcome(result=result1, stats=stats, batches=[batch1, batch2])

        json_str = original.model_dump_json()
        deserialized = RichResultPipelineOutcome.model_validate_json(json_str)

        assert len(deserialized.batches) == 2
        assert deserialized.batches[0].result is not None
        assert deserialized.batches[0].result.species_name == "Fox"
        assert deserialized.batches[1].result is not None
        assert deserialized.batches[1].result.species_name == "Badger"


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
        """Verify extract_images returns ExtractedFrames."""
        extractor = RescaledFrameImageExtractor()
        result = extractor.extract_images(sample_frames, tmp_path)
        assert isinstance(result, ExtractedFrames)
        if len(result) > 0:
            assert isinstance(result.batches[0], ExtractedBatch)
            assert all(isinstance(pair, ExtractedFrame) for pair in result.batches[0].selected_frames)
            assert all(isinstance(pair.path, Path) for pair in result.batches[0].selected_frames)

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
        """Test returns ExtractedFrames with proper batch structure."""
        extractor = RescaledFrameImageExtractor()
        result = extractor.extract_images(sample_frames, tmp_path)

        assert isinstance(result, ExtractedFrames)
        assert len(result.batches) == 1
        assert len(result.batches[0].selected_frames) == len(sample_frames)
        assert all(isinstance(pair, ExtractedFrame) for pair in result.batches[0].selected_frames)
        assert all(isinstance(pair.path, Path) for pair in result.batches[0].selected_frames)

    def test_rescaled_extractor_max_batch_size(
        self,
        tmp_path: Path,
    ) -> None:
        """Test max_batch_size splits batches correctly."""
        frames = []
        for i in range(75):
            raw = np.zeros((100, 200, 3), dtype=np.uint8)
            raw[:, :] = [(i * 10) % 256, (i * 20) % 256, (i * 30) % 256]
            frame = Frame(raw=raw, frame_no=i, filter_keep=True)
            frames.append(frame)
        extractor = RescaledFrameImageExtractor(max_batch_size=30)
        result = extractor.extract_images(frames, tmp_path)

        assert len(result.batches) == 3
        assert len(result.batches[0].selected_frames) == 25
        assert len(result.batches[1].selected_frames) == 25
        assert len(result.batches[2].selected_frames) == 25

    def test_rescaled_extractor_max_batch_size_default(
        self,
        tmp_path: Path,
    ) -> None:
        """Test default max_batch_size is 30."""
        extractor = RescaledFrameImageExtractor()
        assert extractor.max_batch_size == 30

    def test_rescaled_extractor_max_batch_size_custom(
        self,
        tmp_path: Path,
    ) -> None:
        """Test custom max_batch_size is stored."""
        extractor = RescaledFrameImageExtractor(max_batch_size=50)
        assert extractor.max_batch_size == 50

    def test_rescaled_extractor_max_batch_size_exact_multiple(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test when frames exactly match batch size."""
        extractor = RescaledFrameImageExtractor(max_batch_size=len(sample_frames))
        result = extractor.extract_images(sample_frames, tmp_path)

        assert len(result.batches) == 1
        assert len(result.batches[0].selected_frames) == len(sample_frames)

    def test_rescaled_extractor_max_batch_size_one(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test max_batch_size=1 creates one batch per frame."""
        extractor = RescaledFrameImageExtractor(max_batch_size=1)
        result = extractor.extract_images(sample_frames, tmp_path)

        assert len(result.batches) == len(sample_frames)
        for batch in result.batches:
            assert len(batch.selected_frames) == 1

    def test_extract_images_with_empty_frames(self, tmp_path: Path) -> None:
        """Test edge case with empty frame list."""
        extractor = RescaledFrameImageExtractor()
        result = extractor.extract_images([], tmp_path)

        assert len(result.batches) == 0

    def test_rescaled_extractor_with_all_filtered_frames(
        self,
        sample_frames_all_filtered: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test when all frames have filter_keep=False."""
        extractor = RescaledFrameImageExtractor()
        result = extractor.extract_images(sample_frames_all_filtered, tmp_path)

        assert len(result.batches) == 0
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
        assert len(result.batches[0].selected_frames) == len(sample_frames)


class TestAICroppedFrameImageExtractor:
    """Tests for AICroppedFrameImageExtractor specific functionality."""

    def test_ai_cropped_extractor_initializes_with_params(self) -> None:
        """Test constructor stores parameters."""
        mock_llm = MockAbstractLlm()
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.25)
        extractor = AICroppedFrameImageExtractor(
            aicropfinder=aicropfinder,
            resolution=(320, 240),
            max_batch_size=20,
        )
        assert extractor.aicropfinder is aicropfinder
        assert extractor.resolution == (320, 240)
        assert extractor.max_batch_size == 20

    def test_ai_cropped_extractor_with_default_resolution(self) -> None:
        """Test constructor with default 640x360 resolution."""
        mock_llm = MockAbstractLlm()
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.25)
        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder)
        assert extractor.resolution == (640, 360)
        assert extractor.max_batch_size == 30

    def test_extract_images_returns_sequence(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Verify extract_images returns ExtractedFrames."""
        mock_llm = MockAbstractLlm(result_list_to_return=ResultList(results=[]))
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.25)
        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder)
        result = extractor.extract_images(sample_frames, tmp_path)
        assert isinstance(result, ExtractedFrames)

    def test_extract_images_drops_frames_without_detections(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test frames are dropped when no animals detected."""
        mock_llm = MockAbstractLlm(result_list_to_return=ResultList(results=[]))
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.25)
        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder)
        result = extractor.extract_images(sample_frames, tmp_path)

        assert len(result.batches) == 0
        image_files = list(tmp_path.glob("frame_crop_*.jpg"))
        assert len(image_files) == 0

    def test_extract_images_drops_frames_with_none_crop(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test frames are dropped when crop is None after processing."""
        mock_llm = MockAbstractLlm(
            result_list_to_return=ResultList(
                results=[
                    Result(
                        species_name="fox",
                        frames=[
                            FrameResult(
                                frame_no=999,
                                left=0.2,
                                right=0.8,
                                top=0.2,
                                bottom=0.8,
                            ),
                        ],
                    ),
                ],
            ),
        )
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.25)
        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder)
        result = extractor.extract_images(sample_frames, tmp_path)

        assert len(result.batches) == 0
        image_files = list(tmp_path.glob("frame_crop_*.jpg"))
        assert len(image_files) == 0

    def test_extract_images_extracts_cropped_frames(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test cropped images are extracted when animals detected."""
        mock_llm = MockAbstractLlm(
            result_list_to_return=ResultList(
                results=[
                    Result(
                        species_name="fox",
                        frames=[
                            FrameResult(
                                frame_no=0,
                                left=0.2,
                                right=0.8,
                                top=0.2,
                                bottom=0.8,
                            ),
                        ],
                    ),
                ],
            ),
        )
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.0)
        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder, resolution=(320, 240))
        result = extractor.extract_images(sample_frames, tmp_path)

        assert len(result.batches) == 1
        assert len(result.batches[0].selected_frames) == 1
        image_files = list(tmp_path.glob("frame_crop_*.jpg"))
        assert len(image_files) == 1

    def test_extract_images_downscales_cropped_images(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test output images match target resolution."""
        mock_llm = MockAbstractLlm(
            result_list_to_return=ResultList(
                results=[
                    Result(
                        species_name="fox",
                        frames=[
                            FrameResult(
                                frame_no=i,
                                left=0.2,
                                right=0.8,
                                top=0.2,
                                bottom=0.8,
                            )
                            for i in range(len(sample_frames))
                        ],
                    ),
                ],
            ),
        )
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.0)
        resolution = (320, 240)
        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder, crop_max_resolution=resolution)
        extractor.extract_images(sample_frames, tmp_path)

        for image_file in tmp_path.glob("frame_crop_*.jpg"):
            img = cv2.imread(str(image_file))
            assert img is not None
            h, w = img.shape[:2]
            assert w <= resolution[0]
            assert h <= resolution[1]

    def test_ai_cropped_extractor_skips_filtered_frames(
        self,
        sample_frames_with_filtering: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test only filter_keep=True frames are processed."""
        mock_llm = MockAbstractLlm(
            result_list_to_return=ResultList(
                results=[
                    Result(
                        species_name="fox",
                        frames=[
                            FrameResult(
                                frame_no=i,
                                left=0.2,
                                right=0.8,
                                top=0.2,
                                bottom=0.8,
                            )
                            for i in range(10)
                        ],
                    ),
                ],
            ),
        )
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.0)
        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder)
        extractor.extract_images(sample_frames_with_filtering, tmp_path)

        image_files = list(tmp_path.glob("frame_crop_*.jpg"))
        expected_count = sum(1 for f in sample_frames_with_filtering if f.filter_keep)
        assert len(image_files) == expected_count

    def test_ai_cropped_extractor_batch_structure(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test returns ExtractedFrames with proper batch structure."""
        mock_llm = MockAbstractLlm(
            result_list_to_return=ResultList(
                results=[
                    Result(
                        species_name="fox",
                        frames=[
                            FrameResult(
                                frame_no=i,
                                left=0.2,
                                right=0.8,
                                top=0.2,
                                bottom=0.8,
                            )
                            for i in range(len(sample_frames))
                        ],
                    ),
                ],
            ),
        )
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.0)
        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder)
        result = extractor.extract_images(sample_frames, tmp_path)

        assert isinstance(result, ExtractedFrames)
        assert len(result.batches) == 1
        assert len(result.batches[0].selected_frames) == len(sample_frames)
        assert all(isinstance(pair, ExtractedFrame) for pair in result.batches[0].selected_frames)
        assert all(isinstance(pair.path, Path) for pair in result.batches[0].selected_frames)

    def test_ai_cropped_extractor_max_batch_size(
        self,
        tmp_path: Path,
    ) -> None:
        """Test max_batch_size splits batches correctly."""
        mock_llm = MockAbstractLlm(
            result_list_to_return=ResultList(
                results=[
                    Result(
                        species_name="fox",
                        frames=[
                            FrameResult(
                                frame_no=i,
                                left=0.2,
                                right=0.8,
                                top=0.2,
                                bottom=0.8,
                            )
                            for i in range(75)
                        ],
                    ),
                ],
            ),
        )
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.0)
        frames = []
        for i in range(75):
            raw = np.zeros((100, 200, 3), dtype=np.uint8)
            raw[:, :] = [(i * 10) % 256, (i * 20) % 256, (i * 30) % 256]
            frame = Frame(raw=raw, frame_no=i, filter_keep=True)
            frames.append(frame)

        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder, max_batch_size=30)
        result = extractor.extract_images(frames, tmp_path)

        assert len(result.batches) == 3
        assert len(result.batches[0].selected_frames) == 30
        assert len(result.batches[1].selected_frames) == 30
        assert len(result.batches[2].selected_frames) == 15

    def test_ai_cropped_extractor_preserves_aspect_ratio(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test aspect ratio is maintained in output images."""
        mock_llm = MockAbstractLlm(
            result_list_to_return=ResultList(
                results=[
                    Result(
                        species_name="fox",
                        frames=[
                            FrameResult(
                                frame_no=i,
                                left=0.2,
                                right=0.8,
                                top=0.2,
                                bottom=0.8,
                            )
                            for i in range(len(sample_frames))
                        ],
                    ),
                ],
            ),
        )
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.0)
        resolution = (320, 240)
        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder, resolution=resolution)
        extractor.extract_images(sample_frames, tmp_path)

        for image_file in tmp_path.glob("*.jpg"):
            img = cv2.imread(str(image_file))
            assert img is not None
            h, w = img.shape[:2]
            crop_w = int(sample_frames[0].width_raw * 0.6)
            crop_h = int(sample_frames[0].height_raw * 0.6)
            original_ratio = crop_w / crop_h
            output_ratio = w / h
            assert abs(output_ratio - original_ratio) < 0.05

    def test_extract_images_with_empty_frames(self, tmp_path: Path) -> None:
        """Test edge case with empty frame list."""
        mock_llm = MockAbstractLlm(result_list_to_return=ResultList(results=[]))
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.25)
        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder)
        result = extractor.extract_images([], tmp_path)

        assert len(result.batches) == 0

    def test_extract_images_uses_outdir(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test files are created in correct directory."""
        mock_llm = MockAbstractLlm(
            result_list_to_return=ResultList(
                results=[
                    Result(
                        species_name="fox",
                        frames=[
                            FrameResult(
                                frame_no=i,
                                left=0.2,
                                right=0.8,
                                top=0.2,
                                bottom=0.8,
                            )
                            for i in range(len(sample_frames))
                        ],
                    ),
                ],
            ),
        )
        aicropfinder = AICropFinder(analyser=mock_llm, expansion=0.0)
        extractor = AICroppedFrameImageExtractor(aicropfinder=aicropfinder)
        subdir = tmp_path / "subdir" / "nested"
        extractor.extract_images(sample_frames, subdir)

        assert subdir.exists()
        image_files = list(subdir.glob("*.jpg"))
        # should be only cropped images
        assert len(image_files) == len(sample_frames)


class TestContrastEnhancedFrameImageExtractor:
    """Tests for ContrastEnhancedFrameImageExtractor specific functionality."""

    def test_contrast_enhanced_extractor_initializes_with_params(self) -> None:
        """Test constructor stores parameters."""
        extractor = ContrastEnhancedFrameImageExtractor(
            resolution=(320, 240),
            max_batch_size=20,
            clip_limit=2.5,
            tile_grid_size=(16, 16),
        )
        assert extractor.resolution == (320, 240)
        assert extractor.max_batch_size == 20
        assert extractor.contrast_enhancer.clip_limit == 2.5
        assert extractor.contrast_enhancer.tile_grid_size == (16, 16)

    def test_contrast_enhanced_extractor_with_default_values(self) -> None:
        """Test constructor with default values."""
        extractor = ContrastEnhancedFrameImageExtractor()
        assert extractor.resolution == (640, 360)
        assert extractor.max_batch_size == 30
        assert extractor.contrast_enhancer.clip_limit == 2.0
        assert extractor.contrast_enhancer.tile_grid_size == (8, 8)

    def test_extract_images_returns_sequence(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Verify extract_images returns a Sequence."""
        extractor = ContrastEnhancedFrameImageExtractor()
        result = extractor.extract_images(sample_frames, tmp_path)
        assert isinstance(result, ExtractedFrames)

    def test_extract_images_creates_files(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Verify images are written to output directory."""
        extractor = ContrastEnhancedFrameImageExtractor()
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
        extractor = ContrastEnhancedFrameImageExtractor(resolution=resolution)
        extractor.extract_images(sample_frames, tmp_path)

        for image_file in tmp_path.glob("*.jpg"):
            img = cv2.imread(str(image_file))
            assert img is not None
            h, w = img.shape[:2]
            assert w <= resolution[0]
            assert h <= resolution[1]

    def test_contrast_enhanced_extractor_preserves_aspect_ratio(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test aspect ratio is maintained in output images."""
        resolution = (320, 240)
        extractor = ContrastEnhancedFrameImageExtractor(resolution=resolution)
        extractor.extract_images(sample_frames, tmp_path)

        original_ratio = sample_frames[0].width_raw / sample_frames[0].height_raw

        for image_file in tmp_path.glob("*.jpg"):
            img = cv2.imread(str(image_file))
            assert img is not None
            h, w = img.shape[:2]
            output_ratio = w / h
            assert abs(output_ratio - original_ratio) < 0.05

    @pytest.mark.parametrize("resolution", [(128, 72), (640, 360), (800, 600), (1024, 768)])
    def test_contrast_enhanced_extractor_with_custom_resolution(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
        resolution: tuple[int, int],
    ) -> None:
        """Test custom resolutions work correctly."""
        extractor = ContrastEnhancedFrameImageExtractor(resolution=resolution)
        extractor.extract_images(sample_frames, tmp_path)

        for image_file in tmp_path.glob("*.jpg"):
            img = cv2.imread(str(image_file))
            assert img is not None
            h, w = img.shape[:2]
            assert w <= resolution[0]
            assert h <= resolution[1]

    def test_contrast_enhanced_extractor_skips_filtered_frames(
        self,
        sample_frames_with_filtering: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test only filter_keep=True frames are processed."""
        extractor = ContrastEnhancedFrameImageExtractor()
        extractor.extract_images(sample_frames_with_filtering, tmp_path)

        image_files = list(tmp_path.glob("*.jpg"))
        expected_count = sum(1 for f in sample_frames_with_filtering if f.filter_keep)
        assert len(image_files) == expected_count

    def test_contrast_enhanced_extractor_batch_structure(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test returns list of lists (batches)."""
        extractor = ContrastEnhancedFrameImageExtractor()
        result = extractor.extract_images(sample_frames, tmp_path)

        assert isinstance(result, ExtractedFrames)
        assert len(result.batches) == 1
        assert len(result.batches[0].selected_frames) == len(sample_frames)

    def test_contrast_enhanced_extractor_max_batch_size(
        self,
        tmp_path: Path,
    ) -> None:
        """Test max_batch_size splits batches correctly."""
        frames = []
        for i in range(75):
            raw = np.zeros((100, 200, 3), dtype=np.uint8)
            raw[:, :] = [(i * 10) % 256, (i * 20) % 256, (i * 30) % 256]
            frame = Frame(raw=raw, frame_no=i, filter_keep=True)
            frames.append(frame)
        extractor = ContrastEnhancedFrameImageExtractor(max_batch_size=30)
        result = extractor.extract_images(frames, tmp_path)

        assert len(result.batches) == 3
        assert len(result.batches[0].selected_frames) == 25
        assert len(result.batches[1].selected_frames) == 25
        assert len(result.batches[2].selected_frames) == 25

    def test_extract_images_with_empty_frames(self, tmp_path: Path) -> None:
        """Test edge case with empty frame list."""
        extractor = ContrastEnhancedFrameImageExtractor()
        result = extractor.extract_images([], tmp_path)

        assert result == ExtractedFrames(batches=[])

    def test_contrast_enhanced_extractor_with_all_filtered_frames(
        self,
        sample_frames_all_filtered: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test when all frames have filter_keep=False."""
        extractor = ContrastEnhancedFrameImageExtractor()
        result = extractor.extract_images(sample_frames_all_filtered, tmp_path)

        assert result == ExtractedFrames(batches=[])
        image_files = list(tmp_path.glob("*.jpg"))
        assert len(image_files) == 0

    def test_extract_images_uses_outdir(
        self,
        sample_frames: list[Frame],
        tmp_path: Path,
    ) -> None:
        """Test files are created in correct directory."""
        subdir = tmp_path / "subdir" / "nested"
        extractor = ContrastEnhancedFrameImageExtractor()
        extractor.extract_images(sample_frames, subdir)

        assert subdir.exists()
        image_files = list(subdir.glob("*.jpg"))
        assert len(image_files) == len(sample_frames)


class TestLlmImageBatchQuery:
    """Tests for LlmImageBatchQuery specific functionality."""

    def test_llm_query_initializes_with_params(self) -> None:
        """Test constructor stores parameters."""
        mock_llm = MockAbstractLlm()
        prompt = "Test prompt"
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt=prompt,
        )
        assert query.llm is mock_llm
        assert query.prompt == prompt

    def test_query_images_returns_rich_result(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test query_images returns a RichResult instance."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
        )
        result = query.query_images(sample_image_paths)
        assert isinstance(result, RichResult)
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
        )
        result = query.query_images(sample_image_paths)
        assert result.species_name == "custom_species"

    def test_query_images_empty_batch_raises_error(self) -> None:
        """Test empty batch raises ValueError."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
        )
        with pytest.raises(ValueError, match="Empty image batch"):
            query.query_images([])

    def test_query_image_batches_returns_enriched_frames(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test query_image_batches returns ExtractedFramesWithResults."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
        )
        batch1 = ExtractedBatch(
            selected_frames=[ExtractedFrame(path=p, frame_no=i) for i, p in enumerate(sample_image_paths)],
        )
        batch2 = ExtractedBatch(
            selected_frames=[ExtractedFrame(path=p, frame_no=i) for i, p in enumerate(sample_image_paths)],
        )
        extracted_frames = ExtractedFrames(batches=[batch1, batch2])
        enriched_frames = query.query_image_batches(extracted_frames)
        assert isinstance(enriched_frames, ExtractedFramesWithResults)
        assert len(enriched_frames.batches) == 2
        assert all(isinstance(batch, BatchResult) for batch in enriched_frames.batches)
        assert all(batch.result is not None for batch in enriched_frames.batches)

    def test_query_image_batches_multiple_batches(
        self,
        sample_image_paths: list[Path],
    ) -> None:
        """Test query_image_batches processes multiple batches."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
        )
        batches = []
        for _ in range(3):
            batch = ExtractedBatch(
                selected_frames=[ExtractedFrame(path=p, frame_no=i) for i, p in enumerate(sample_image_paths)],
            )
            batches.append(batch)
        extracted_frames = ExtractedFrames(batches=batches)
        enriched_frames = query.query_image_batches(extracted_frames)
        assert mock_llm.call_count == 3
        assert len(enriched_frames.batches) == 3

    @pytest.mark.parametrize("batch_count", [1, 3, 5])
    def test_query_image_batches_parametrized(
        self,
        sample_image_paths: list[Path],
        batch_count: int,
    ) -> None:
        """Test query_image_batches with various batch counts."""
        mock_llm = MockAbstractLlm()
        query = LlmImageBatchQuery(
            llm=mock_llm,
            prompt="Test prompt",
        )
        batches = []
        for _ in range(batch_count):
            batch = ExtractedBatch(
                selected_frames=[ExtractedFrame(path=p, frame_no=i) for i, p in enumerate(sample_image_paths)],
            )
            batches.append(batch)
        extracted_frames = ExtractedFrames(batches=batches)
        enriched_frames = query.query_image_batches(extracted_frames)
        assert len(enriched_frames.batches) == batch_count
        assert mock_llm.call_count == batch_count

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
                min_confidence=ConfidenceLevel.LOW,
            )
            result = query.query_images(sample_image_paths)
            assert result.species_name == "fox"
            assert result.is_animal_present is True
