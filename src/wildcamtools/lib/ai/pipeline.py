import collections
import itertools
import logging
import math
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar

import cv2
from pydantic import BaseModel, Field

from wildcamtools.lib import Frame
from wildcamtools.lib.ai.crop import AICropFinder
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.types import (
    DEFAULT_BATCH_DESCRIPTION_PROMPT,
    DEFAULT_COMBINE_DESCRIPTION_PROMPT,
    NO_ACTIVITY_DESCRIPTION,
    BatchDescription,
    CombinedDescription,
    ConfidenceLevel,
    ResultList,
    RichResult,
    VerificationResult,
)
from wildcamtools.lib.frames import ContrastEnhancer, FilterSSIM, Rescaler, resize_with_aspect_ratio
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.stats import VideoStats, get_video_stats
from wildcamtools.lib.vidio import VideoReader

logger = logging.getLogger(__name__)

R = TypeVar("R", bound=BaseModel)


class ExtractedFrame(BaseModel):
    """Pairs an image path with its frame number.

    Attributes:
        path: Path to the image file (excluded from JSON serialization, optional for deserialization)
        frame_no: Frame number (included in JSON serialization)
    """

    path: Path | None = Field(exclude=True, default=None)
    frame_no: int

    def require_path(self) -> Path:
        """Require that path is set, raising ValueError if not.

        This should be called before accessing path during pipeline execution
        to ensure the field was properly populated.

        Returns:
            The path to the image file

        Raises:
            ValueError: If path is None
        """
        if self.path is None:
            raise ValueError(
                f"ExtractedFrame.path is required but is None for frame_no={self.frame_no}. "
                "This indicates a bug in the pipeline execution."
            )
        return self.path


class ExtractedBatch(BaseModel):
    """Base class for batch of extracted frames.

    Attributes:
        frame_image_pairs: List of FrameImagePair objects (atomic pairing)
    """

    selected_frames: list[ExtractedFrame]


class BatchResult[R: BaseModel](ExtractedBatch):
    """Batch with AI result attached.

    Attributes:
        frame_image_pairs: List of FrameImagePair (inherited)
        result: RichResult from AI analysis (None before processing)
    """

    result: R | None = None


class RichResultBatchResult(BatchResult[RichResult]):
    """Concrete BatchResult parameterised on RichResult.

    This concrete subclass exists so that JSON round-trip serialisation works
    (Pydantic cannot reconstruct a parameterised generic without a concrete
    type). Behaviour is identical to BatchResult[RichResult].
    """


class BatchDescriptionBatchResult(BatchResult[BatchDescription]):
    """Concrete BatchResult parameterised on BatchDescription.

    Used for the description pipeline so that JSON round-trip serialisation
    works (Pydantic cannot reconstruct a parameterised generic without a
    concrete type).
    """


class ExtractedFrames(BaseModel):
    """Container for extracted frame batches (before AI processing).

    Attributes:
        batches: List of ExtractedBatch objects
    """

    batches: list[ExtractedBatch]

    @property
    def frame_ids(self) -> list[list[int]]:
        """Get frame numbers organized by batch (for JSON serialization)."""
        return [[pair.frame_no for pair in batch.selected_frames] for batch in self.batches]

    def get_batches(self) -> Iterator[list[Path]]:
        """Iterate over batches of image paths."""
        for batch in self.batches:
            yield [pair.require_path() for pair in batch.selected_frames]

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batches)


class ExtractedFramesWithResults[R: BaseModel](ExtractedFrames):
    """Container for extracted frame batches with AI results.

    Attributes:
        batches: List of BatchResult objects (contains results after AI processing)
    """

    batches: Sequence[BatchResult[R]]  # type: ignore[assignment]

    def get_batch_results(self) -> list[R]:
        """Extract non-None results from batches."""
        return [batch.result for batch in self.batches if batch.result is not None]


class RichResultExtractedFramesWithResults(ExtractedFramesWithResults[RichResult]):
    """Concrete ExtractedFramesWithResults parameterised on RichResult."""


class PipelineOutcome[R: BaseModel](BaseModel):
    """Container for pipeline execution results with intermediate stage data.

    Attributes:
        result: The final result from the AI pipeline
        stats: Video statistics captured at the start of processing
        batches: List of BatchResult objects with frames and per-batch AI results
    """

    result: R
    stats: VideoStats
    batches: list[BatchResult[R]] = Field(default_factory=list)


class RichResultPipelineOutcome(PipelineOutcome[RichResult]):
    """Concrete PipelineOutcome parameterised on RichResult.

    This concrete subclass exists so that JSON round-trip serialisation works
    (Pydantic cannot reconstruct a parameterised generic without a concrete
    type). Behaviour is identical to PipelineOutcome[RichResult].
    """


class BatchDescriptionPipelineOutcome(PipelineOutcome[BatchDescription]):
    """Concrete PipelineOutcome parameterised on BatchDescription."""


class CombinedBatchResult(BaseModel):
    """Batch result containing both classification and description.

    Attributes:
        selected_frames: List of extracted frames
        classification: RichResult from classification (None if classification not run)
        description: BatchDescription from description (None if description not run)
    """

    selected_frames: list[ExtractedFrame]
    classification: RichResult | None = None
    description: BatchDescription | None = None


class CombinedPipelineOutcome[R: BaseModel](BaseModel):
    """Pipeline outcome containing both classification and description results.

    Attributes:
        result: The final classification result
        description: Combined description result (None if description not enabled)
        stats: Video statistics
        batches: List of CombinedBatchResult with both classification and description per batch
    """

    result: R
    description: BatchDescription | None = None
    stats: VideoStats
    batches: list[CombinedBatchResult] = Field(default_factory=list)


CONFIDENCE_ORDER: dict[ConfidenceLevel, int] = {
    ConfidenceLevel.LOW: 0,
    ConfidenceLevel.MEDIUM: 1,
    ConfidenceLevel.HIGH: 2,
}

DEFAULT_VERIFICATION_PROMPT = (
    "You are verifying an AI classification of wildlife from camera trap images.\n\n"
    'Initial classification: "{initial_species}"\n\n'
    "Your task:\n"
    "1. Review the images carefully\n"
    "2. Determine if the initial classification is correct\n"
    '3. Rate your confidence level as exactly one of: "high", "medium", or "low"\n\n'
    "Use these guidelines:\n"
    '- "high": You are highly confident (>80%) the classification is correct\n'
    '- "medium": You are moderately confident (50-80%) but see some uncertainty\n'
    '- "low": You have significant doubts (<50%) or cannot verify\n\n'
)


class FrameSelector(ABC):
    @abstractmethod
    def select_frames(self, video: Path) -> Iterator[Frame]: ...


class FpsRescalingFrameSelector(FrameSelector):
    fps: float

    def __init__(self, fps: float = 1.0) -> None:
        self.fps = fps

    def select_frames(self, video: Path) -> Iterator[Frame]:
        stats = get_video_stats(video)
        rescaler = Rescaler(stats, fps=self.fps)
        with VideoReader(video) as video_reader:
            for frame in video_reader:
                frame = rescaler.handle(frame)
                if frame.filter_keep:
                    yield frame


class MotionFrameSelector(FrameSelector):
    fps: float
    motion_threshold: float
    resolution: tuple[int, int] | None
    history: int

    def __init__(
        self,
        fps: float = 5.0,
        motion_threshold: float = 0.01,
        resolution: tuple[int, int] | None = None,
        history: int = 30,
    ) -> None:
        self.fps = fps
        self.motion_threshold = motion_threshold
        self.resolution = resolution
        self.history = history

    def select_frames(self, video: Path) -> Iterator[Frame]:
        stats = get_video_stats(video)
        motion_handler = MogMotion(history=self.history, resolution=self.resolution)
        rescaler = Rescaler(stats, fps=self.fps) if self.fps > 0 else None

        with VideoReader(video) as video_reader:
            for frame in video_reader:
                frame = motion_handler.handle(frame)

                has_motion = frame.frame_no <= self.history or frame.motion_proportion >= self.motion_threshold

                if not has_motion:
                    frame.filter_keep = False
                    continue

                if rescaler is not None:
                    frame = rescaler.handle(frame)

                if not frame.filter_keep:
                    continue

                yield frame


class SSIMFrameSelector(FrameSelector):
    fps: float
    similarity_minimum: float
    resolution: tuple[int, int] | None

    def __init__(
        self,
        fps: float = 5.0,
        similarity_minimum: float = 0.9,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        self.fps = fps
        self.similarity_minimum = similarity_minimum
        self.resolution = resolution

    def select_frames(self, video: Path) -> Iterator[Frame]:
        stats = get_video_stats(video)
        ssim_filter = FilterSSIM(similarity_minimum=self.similarity_minimum)
        rescaler = None
        if self.fps > 0:
            if self.resolution is not None:
                rescaler = Rescaler(stats, fps=self.fps, x=self.resolution[0], y=self.resolution[1])
            else:
                rescaler = Rescaler(stats, fps=self.fps)

        with VideoReader(video) as video_reader:
            for frame in video_reader:
                if rescaler is not None:
                    frame = rescaler.handle(frame)
                    if not frame.filter_keep:
                        continue

                frame = ssim_filter.handle(frame)
                if frame.filter_keep:
                    yield frame


class FrameImageExtractor(ABC):
    @abstractmethod
    def extract_images(self, frames: Iterable[Frame], outdir: Path) -> ExtractedFrames: ...


FrameExtractor = FrameImageExtractor


class RescaledFrameImageExtractor(FrameImageExtractor):
    def __init__(self, resolution: tuple[int, int] = (640, 360), max_batch_size: int = 30) -> None:
        self.resolution = resolution
        self.max_batch_size = max_batch_size

    def extract_images(self, frames: Iterable[Frame], outdir: Path) -> ExtractedFrames:
        outdir.mkdir(parents=True, exist_ok=True)

        current_batch: list[ExtractedFrame] = []
        all_batches: list[list[ExtractedFrame]] = []

        for frame in frames:
            if not frame.filter_keep:
                continue

            rescaled_image = resize_with_aspect_ratio(frame.output, self.resolution)
            image_path = outdir / f"frame_{frame.frame_no:05d}.jpg"
            cv2.imwrite(str(image_path), rescaled_image)
            current_batch.append(ExtractedFrame(path=image_path, frame_no=frame.frame_no))

            if len(current_batch) >= self.max_batch_size:
                all_batches.append(current_batch)
                current_batch = []

        if current_batch:
            all_batches.append(current_batch)

        # adjust batch boundaries to make them as equal sized as possible
        total_frame_count = sum(len(b) for b in all_batches)
        equalised_batch_count = math.ceil(total_frame_count / self.max_batch_size) if self.max_batch_size > 0 else 1
        equalised_batch_size = math.ceil(total_frame_count / equalised_batch_count) if equalised_batch_count > 0 else 1

        # Flatten and re-batch
        flat_pairs = list(itertools.chain(*all_batches))
        equalised_batches = list(itertools.batched(flat_pairs, equalised_batch_size, strict=False))

        return ExtractedFrames(batches=[ExtractedBatch(selected_frames=list(b)) for b in equalised_batches])


class ContrastEnhancedFrameImageExtractor(FrameImageExtractor):
    """Extracts images with CLAHE contrast enhancement applied."""

    resolution: tuple[int, int]
    max_batch_size: int
    contrast_enhancer: ContrastEnhancer

    def __init__(
        self,
        resolution: tuple[int, int] = (640, 360),
        max_batch_size: int = 30,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        self.resolution = resolution
        self.max_batch_size = max_batch_size
        self.contrast_enhancer = ContrastEnhancer(
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )

    def extract_images(self, frames: Iterable[Frame], outdir: Path) -> ExtractedFrames:
        outdir.mkdir(parents=True, exist_ok=True)

        current_batch: list[ExtractedFrame] = []
        all_batches: list[list[ExtractedFrame]] = []

        for frame in frames:
            if not frame.filter_keep:
                continue

            frame = self.contrast_enhancer.handle(frame)

            rescaled_image = resize_with_aspect_ratio(frame.output, self.resolution)
            image_path = outdir / f"frame_{frame.frame_no:05d}.jpg"
            cv2.imwrite(str(image_path), rescaled_image)
            current_batch.append(ExtractedFrame(path=image_path, frame_no=frame.frame_no))

            if len(current_batch) >= self.max_batch_size:
                all_batches.append(current_batch)
                current_batch = []

        if current_batch:
            all_batches.append(current_batch)

        # adjust batch boundaries to make them as equal sized as possible
        total_frame_count = sum(len(b) for b in all_batches)
        equalised_batch_count = math.ceil(total_frame_count / self.max_batch_size) if self.max_batch_size > 0 else 1
        equalised_batch_size = math.ceil(total_frame_count / equalised_batch_count) if equalised_batch_count > 0 else 1

        # Flatten and re-batch
        flat_pairs = list(itertools.chain(*all_batches))
        equalised_batches = list(itertools.batched(flat_pairs, equalised_batch_size, strict=False))

        return ExtractedFrames(batches=[ExtractedBatch(selected_frames=list(b)) for b in equalised_batches])


class AICroppedFrameImageExtractor(FrameImageExtractor):
    DETECTION_PROMPT = """Analyze these wildlife camera trap frames and identify all animals present.

For each species detected, provide:
1. The species name
2. For each frame where the species appears, provide normalized bounding box coordinates:
   - left, right, top, bottom (values between 0.0 and 1.0)
   - frame_no (the frame number)

Return results as a ResultList with species_name and frames array."""

    def __init__(
        self,
        aicropfinder: AICropFinder,
        resolution: tuple[int, int] = (640, 360),
        crop_max_resolution: tuple[int, int] = (640, 360),
        max_batch_size: int = 30,
    ) -> None:
        self.aicropfinder = aicropfinder
        self.resolution = resolution
        self.crop_max_resolution = crop_max_resolution
        self.max_batch_size = max_batch_size

    def extract_images(self, frames: Iterable[Frame], outdir: Path) -> ExtractedFrames:
        outdir.mkdir(parents=True, exist_ok=True)

        all_batches = itertools.batched(frames, self.max_batch_size, strict=False)
        output_batches: list[list[ExtractedFrame]] = []

        for batch in all_batches:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                # for each image in this batch, produce a downscaled file
                batch_pairs: list[ExtractedFrame] = []
                for frame in batch:
                    rescaled_image = resize_with_aspect_ratio(frame.output, self.resolution)
                    image_path = tmpdir_path / f"frame_{frame.frame_no:05d}.jpg"
                    cv2.imwrite(str(image_path), rescaled_image)
                    batch_pairs.append(ExtractedFrame(path=image_path, frame_no=frame.frame_no))

                self.aicropfinder.detections = self.aicropfinder.analyser.message_with_schema(
                    message=self.DETECTION_PROMPT,
                    images=[pair.require_path() for pair in batch_pairs],
                    response_class=ResultList,
                )

                # align bbox with original and crop

                batch_crop_pairs: list[ExtractedFrame] = []
                for frame in batch:
                    frame = self.aicropfinder.handle(frame)
                    if not frame.filter_keep or frame.crop is None:
                        continue
                    rescaled_image = resize_with_aspect_ratio(frame.crop, self.crop_max_resolution)
                    image_path = outdir / f"frame_crop_{frame.frame_no:05d}.jpg"
                    cv2.imwrite(str(image_path), rescaled_image)
                    batch_crop_pairs.append(ExtractedFrame(path=image_path, frame_no=frame.frame_no))

                if batch_crop_pairs:
                    output_batches.append(batch_crop_pairs)
        return ExtractedFrames(batches=[ExtractedBatch(selected_frames=b) for b in output_batches])


class ImageBatchQuery[R: BaseModel](ABC):
    def query_image_batches(self, image_batches: ExtractedFrames) -> ExtractedFramesWithResults[R]:
        batch_results: list[BatchResult[R]] = []
        for batch in image_batches.batches:
            result = self.query_images([pair.require_path() for pair in batch.selected_frames])
            batch_results.append(BatchResult[R](selected_frames=batch.selected_frames, result=result))
        return ExtractedFramesWithResults[R](batches=batch_results)

    @abstractmethod
    def query_images(self, images: Sequence[Path]) -> R: ...


class LlmImageBatchQuery[R: BaseModel](ImageBatchQuery[R]):
    llm: AbstractLlm
    prompt: str
    response_class: type[R]

    def __init__(
        self,
        llm: AbstractLlm,
        prompt: str,
        response_class: type[R] = RichResult,  # type: ignore[assignment]
    ) -> None:
        self.llm = llm
        self.prompt = prompt
        self.response_class = response_class

    def query_images(self, images: Sequence[Path]) -> R:
        images_list = sorted(images)
        if not images_list:
            logger.warning("Empty image batch received")
            raise ValueError("Empty image batch")
        logger.info("Sending %d images to analyser", len(images_list))
        result: R = self.llm.message_with_schema(
            message=self.prompt,
            images=images_list,
            response_class=self.response_class,
        )
        logger.info("Received response from analyser")
        return result


class DescriptionImageBatchQuery(LlmImageBatchQuery[BatchDescription]):
    """Convenience subclass of LlmImageBatchQuery that returns BatchDescription.

    If no prompt is provided, the default batch description prompt is used.
    """

    def __init__(
        self,
        llm: AbstractLlm,
        prompt: str = DEFAULT_BATCH_DESCRIPTION_PROMPT,
    ) -> None:
        super().__init__(llm=llm, prompt=prompt, response_class=BatchDescription)


class VerifiedImageBatchQuery[R: BaseModel](ImageBatchQuery[R]):
    llm: AbstractLlm
    prompt: str
    response_class: type[R]
    verification_prompt: str
    min_confidence: ConfidenceLevel

    def __init__(
        self,
        llm: AbstractLlm,
        prompt: str,
        response_class: type[R] = RichResult,  # type: ignore[assignment]
        verification_prompt: str | None = None,
        min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ) -> None:
        self.llm = llm
        self.prompt = prompt
        self.response_class = response_class
        self.verification_prompt = verification_prompt or DEFAULT_VERIFICATION_PROMPT
        self.min_confidence = min_confidence

    def query_images(self, images: Sequence[Path]) -> R:
        images_list = sorted(images)
        if not images_list:
            logger.warning("Empty image batch received")
            raise ValueError("Empty image batch")

        logger.debug("Sending %d images to analyser for initial classification", len(images_list))
        initial_result: R = self.llm.message_with_schema(
            message=self.prompt,
            images=images_list,
            response_class=self.response_class,
        )
        # Note: VerifiedImageBatchQuery expects R to have species_name, confidence, is_animal_unknown fields
        # This is enforced by the typical usage with RichResult as the default response_class
        logger.debug("Initial classification: %s", getattr(initial_result, "species_name", initial_result))

        verification_message = self.verification_prompt.format(
            initial_species=getattr(initial_result, "species_name", "")
        )
        verification_result: VerificationResult = self.llm.message_with_schema(
            message=verification_message,
            images=images_list,
            response_class=VerificationResult,
        )

        # update confidence and species name from verification
        if hasattr(initial_result, "confidence"):
            initial_result.confidence = verification_result.confidence
        if hasattr(initial_result, "species_name"):
            initial_result.species_name = verification_result.species_name

        if not verification_result.verified or not self._meets_confidence_threshold(verification_result.confidence):
            logger.debug(
                "Verification failed or confidence %s below threshold %s, marking as unknown",
                verification_result.confidence,
                self.min_confidence,
            )
            if hasattr(initial_result, "is_animal_unknown"):
                initial_result.is_animal_unknown = True

        return initial_result

    def _meets_confidence_threshold(self, confidence: ConfidenceLevel) -> bool:
        return CONFIDENCE_ORDER[confidence] >= CONFIDENCE_ORDER[self.min_confidence]


class ResultReconciler[R: BaseModel](ABC):
    @abstractmethod
    def reconcile_results(self, results: Iterable[R]) -> R: ...


class RichResultMajorityReconciler(ResultReconciler[RichResult]):
    """Reconciles multiple RichResult classifications by selecting the most common species.

    Uses equality comparison on species_name to determine uniqueness.
    In case of ties, returns the first-seen result among those tied.

    Raises:
        ValueError: If results iterable is empty.
    """

    def reconcile_results(self, results: Iterable[RichResult]) -> RichResult:
        results_list = list(results)
        if not results_list:
            raise ValueError("No results to reconcile")
        if len(results_list) == 1:
            return results_list[0]

        logger.debug("Reconciling %d results", len(results_list))

        # return first no animal if all no animal
        results_ex_no_animal = [r for r in results_list if r.is_animal_present]
        if not results_ex_no_animal:
            return results_list[0]
        # return first unknown if all no animal or unknown
        results_ex_unknown_no_animal = [r for r in results_ex_no_animal if not r.is_animal_unknown]
        if not results_ex_unknown_no_animal:
            return results_ex_no_animal[0]

        # group by species name and return the first result from the highest frequency name
        # these results might be different in confidence and description!
        species_name_results: dict[str, list[RichResult]] = collections.defaultdict(list)
        for result in results_ex_unknown_no_animal:
            species_name_results[result.species_name].append(result)

        species_results = sorted(species_name_results.items(), key=lambda i: len(i[1]), reverse=True)

        logger.debug("Choosing from: %s", [(i[0], len(i[1])) for i in species_name_results.items()])

        return species_results[0][1][0]


class ConcatenatingDescriptionReconciler(ResultReconciler[BatchDescription]):
    """Concatenates batch descriptions into a single description.

    Joins descriptions with double newlines and wraps the result in a new
    BatchDescription. The output reflects the input order. An empty input
    yields the no-activity placeholder.
    """

    def reconcile_results(self, results: Iterable[BatchDescription]) -> BatchDescription:
        results_list = list(results)
        if not results_list:
            return BatchDescription(description=NO_ACTIVITY_DESCRIPTION)
        joined = "\n\n".join(r.description for r in results_list)
        return BatchDescription(description=joined)

    @property
    def method_name(self) -> str:
        return "concatenate"


class LlmDescriptionReconciler(ResultReconciler[BatchDescription]):
    """Reconciles multiple batch descriptions into a single final description using an LLM.

    When the input contains zero or one batch descriptions, no LLM call is made
    and the description is returned unchanged. With two or more batch descriptions,
    the LLM is invoked with the configured `combine_prompt` to produce a merged
    `CombinedDescription`. If the LLM call fails, the configured `fallback`
    reconciler is used (default: ConcatenatingDescriptionReconciler).

    The `last_method_name` property reflects the method actually used for the
    most recent `reconcile_results` call, so callers can record whether the
    LLM-combine path or the fallback path produced the final description.
    """

    def __init__(
        self,
        llm: AbstractLlm,
        prompt: str | None = None,
        fallback: ConcatenatingDescriptionReconciler | None = None,
    ) -> None:
        self.llm = llm
        self.prompt = prompt or DEFAULT_COMBINE_DESCRIPTION_PROMPT
        self.fallback = fallback or ConcatenatingDescriptionReconciler()
        self.last_method_name: str = self.method_name

    def reconcile_results(self, results: Iterable[BatchDescription]) -> BatchDescription:
        results_list = list(results)
        if not results_list:
            self.last_method_name = self.method_name
            return BatchDescription(description=NO_ACTIVITY_DESCRIPTION)
        if len(results_list) == 1:
            self.last_method_name = self.method_name
            return results_list[0]

        formatted_descriptions = "\n\n".join(f"[Segment {i + 1}]\n{r.description}" for i, r in enumerate(results_list))
        message = self.prompt.format(descriptions=formatted_descriptions)
        try:
            combined: CombinedDescription = self.llm.message_with_schema(
                message=message,
                images=(),
                response_class=CombinedDescription,
            )
        except Exception:
            logger.exception("LLM combine failed, falling back to concatenation")
            self.last_method_name = self.fallback.method_name
            return self.fallback.reconcile_results(results_list)

        self.last_method_name = self.method_name
        return BatchDescription(description=combined.description)

    @property
    def method_name(self) -> str:
        return "llm_combine"


class AiPipeline[R: BaseModel]:
    empty_result: R
    description_query: DescriptionImageBatchQuery | None = None
    description_reconciler: ResultReconciler[BatchDescription] | None = None

    def __init__(
        self,
        frame_selector: FrameSelector,
        frame_image_extractor: FrameImageExtractor,
        image_batch_query: ImageBatchQuery[R],
        result_reconciler: ResultReconciler[R],
        empty_result: R,
        description_query: DescriptionImageBatchQuery | None = None,
        description_reconciler: ResultReconciler[BatchDescription] | None = None,
    ) -> None:
        self.frame_selector = frame_selector
        self.frame_image_extractor = frame_image_extractor
        self.image_batch_query = image_batch_query
        self.result_reconciler = result_reconciler
        self.empty_result = empty_result
        self.description_query = description_query
        self.description_reconciler = description_reconciler

    def run(self, video: Path) -> PipelineOutcome[R] | CombinedPipelineOutcome[R]:
        stats = get_video_stats(video)
        frames = self.frame_selector.select_frames(video)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            extracted_frames = self.frame_image_extractor.extract_images(frames, tmpdir_path)
            if len(extracted_frames) == 0:
                if self.description_query is not None:
                    return CombinedPipelineOutcome[R](
                        result=self.empty_result,
                        description=BatchDescription(description=NO_ACTIVITY_DESCRIPTION),
                        stats=stats,
                        batches=[],
                    )
                return PipelineOutcome[R](result=self.empty_result, stats=stats, batches=[])

            enriched_frames = self.image_batch_query.query_image_batches(extracted_frames)
            query_results = enriched_frames.get_batch_results()
            consolidated_result = self.result_reconciler.reconcile_results(query_results)

            if self.description_query is not None and self.description_reconciler is not None:
                description_enriched = self.description_query.query_image_batches(extracted_frames)
                description_results = description_enriched.get_batch_results()
                combined_description = self.description_reconciler.reconcile_results(description_results)

                combined_batches = []
                for classification_batch, description_batch in zip(
                    enriched_frames.batches, description_enriched.batches, strict=True
                ):
                    combined_batches.append(
                        CombinedBatchResult(
                            selected_frames=classification_batch.selected_frames,
                            classification=classification_batch.result,  # type: ignore[arg-type] # CombinedBatchResult.classification is RichResult|None; classification_batch.result is R|None where R=RichResult in this context
                            description=description_batch.result,
                        )
                    )

                return CombinedPipelineOutcome[R](
                    result=consolidated_result,
                    description=BatchDescription(
                        description=combined_description.description,
                    ),
                    stats=stats,
                    batches=combined_batches,
                )

        return PipelineOutcome[R](result=consolidated_result, stats=stats, batches=list(enriched_frames.batches))
