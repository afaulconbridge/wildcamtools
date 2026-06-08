import collections
import itertools
import logging
import math
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import cv2
from pydantic import BaseModel, Field

from wildcamtools.lib import Frame
from wildcamtools.lib.ai.crop import AICropFinder
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.types import (
    ConfidenceLevel,
    ResultList,
    RichResult,
    VerificationResult,
)
from wildcamtools.lib.frames import FilterSSIM, Rescaler, resize_with_aspect_ratio
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.stats import VideoStats, get_video_stats
from wildcamtools.lib.vidio import VideoReader

logger = logging.getLogger(__name__)


class ExtractedFrame(BaseModel):
    """Pairs an image path with its frame number.

    Attributes:
        path: Path to the image file (excluded from JSON serialization)
        frame_no: Frame number (included in JSON serialization)
    """

    path: Path = Field(exclude=True)
    frame_no: int


class ExtractedBatch(BaseModel):
    """Base class for batch of extracted frames.

    Attributes:
        frame_image_pairs: List of FrameImagePair objects (atomic pairing)
    """

    selected_frames: list[ExtractedFrame]


class BatchResult(ExtractedBatch):
    """Batch with AI result attached.

    Attributes:
        frame_image_pairs: List of FrameImagePair (inherited)
        result: RichResult from AI analysis (None before processing)
    """

    result: RichResult | None = None


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
            yield [pair.path for pair in batch.selected_frames]

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batches)


class ExtractedFramesWithResults(ExtractedFrames):
    """Container for extracted frame batches with AI results.

    Attributes:
        batches: List of BatchResult objects (contains results after AI processing)
    """

    batches: Sequence[BatchResult]  # type: ignore[assignment]

    def get_batch_results(self) -> list[RichResult]:
        """Extract non-None results from batches."""
        return [batch.result for batch in self.batches if batch.result is not None]


class PipelineOutcome(BaseModel):
    """Container for pipeline execution results with intermediate stage data.

    Attributes:
        result: The final RichResult from the AI pipeline
        stats: Video statistics captured at the start of processing
        batch_results: List of BatchResult objects with frames and per-batch AI results
    """

    result: RichResult
    stats: VideoStats
    batches: list[BatchResult] = Field(default_factory=list)


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
            # for each image in this batch, produce a downscaled file
            batch_pairs: list[ExtractedFrame] = []
            for frame in batch:
                rescaled_image = resize_with_aspect_ratio(frame.output, self.resolution)
                image_path = outdir / f"frame_{frame.frame_no:05d}.jpg"
                cv2.imwrite(str(image_path), rescaled_image)
                batch_pairs.append(ExtractedFrame(path=image_path, frame_no=frame.frame_no))

            self.aicropfinder.detections = self.aicropfinder.analyser.message_with_schema(
                message=self.DETECTION_PROMPT,
                images=[pair.path for pair in batch_pairs],
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


class ImageBatchQuery(ABC):
    def query_image_batches(self, image_batches: ExtractedFrames) -> ExtractedFramesWithResults:
        batch_results: list[BatchResult] = []
        for batch in image_batches.batches:
            result = self.query_images([pair.path for pair in batch.selected_frames])
            batch_results.append(BatchResult(selected_frames=batch.selected_frames, result=result))
        return ExtractedFramesWithResults(batches=batch_results)

    @abstractmethod
    def query_images(self, images: Sequence[Path]) -> RichResult: ...


class LlmImageBatchQuery(ImageBatchQuery):
    llm: AbstractLlm
    prompt: str

    def __init__(
        self,
        llm: AbstractLlm,
        prompt: str,
    ) -> None:
        self.llm = llm
        self.prompt = prompt

    def query_images(self, images: Sequence[Path]) -> RichResult:
        images_list = sorted(images)
        if not images_list:
            logger.warning("Empty image batch received")
            raise ValueError("Empty image batch")
        logger.info("Sending %d images to analyser", len(images_list))
        result: RichResult = self.llm.message_with_schema(
            message=self.prompt,
            images=images_list,
            response_class=RichResult,
        )
        logger.info("Received response from analyser")
        return result


class VerifiedImageBatchQuery(ImageBatchQuery):
    llm: AbstractLlm
    prompt: str
    verification_prompt: str
    min_confidence: ConfidenceLevel

    def __init__(
        self,
        llm: AbstractLlm,
        prompt: str,
        verification_prompt: str | None = None,
        min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ) -> None:
        self.llm = llm
        self.prompt = prompt
        self.verification_prompt = verification_prompt or DEFAULT_VERIFICATION_PROMPT
        self.min_confidence = min_confidence

    def query_images(self, images: Sequence[Path]) -> RichResult:
        images_list = sorted(images)
        if not images_list:
            logger.warning("Empty image batch received")
            raise ValueError("Empty image batch")

        logger.debug("Sending %d images to analyser for initial classification", len(images_list))
        initial_result: RichResult = self.llm.message_with_schema(
            message=self.prompt,
            images=images_list,
            response_class=RichResult,
        )
        logger.debug("Initial classification: %s", initial_result.species_name)

        verification_message = self.verification_prompt.format(initial_species=initial_result.species_name)
        verification_result: VerificationResult = self.llm.message_with_schema(
            message=verification_message,
            images=images_list,
            response_class=VerificationResult,
        )

        # update confidence and species name from verification
        initial_result.confidence = verification_result.confidence
        initial_result.species_name = verification_result.species_name

        if not verification_result.verified or not self._meets_confidence_threshold(verification_result.confidence):
            logger.debug(
                "Verification failed or confidence %s below threshold %s, marking as unknown",
                verification_result.confidence,
                self.min_confidence,
            )
            initial_result.is_animal_unknown = True

        return initial_result

    def _meets_confidence_threshold(self, confidence: ConfidenceLevel) -> bool:
        return CONFIDENCE_ORDER[confidence] >= CONFIDENCE_ORDER[self.min_confidence]


class ResultReconciler(ABC):
    @abstractmethod
    def reconcile_results(self, results: Iterable[RichResult]) -> RichResult: ...


class MajorityResultReconciler(ResultReconciler):
    """Reconciles multiple results by selecting the most common value.

    Uses equality comparison (__eq__) to determine uniqueness.
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


class AiPipeline:
    def __init__(
        self,
        frame_selector: FrameSelector,
        frame_image_extractor: FrameImageExtractor,
        image_batch_query: ImageBatchQuery,
        result_reconciler: ResultReconciler,
    ) -> None:
        self.frame_selector = frame_selector
        self.frame_image_extractor = frame_image_extractor
        self.image_batch_query = image_batch_query
        self.result_reconciler = result_reconciler

    def run(self, video: Path) -> PipelineOutcome:
        stats = get_video_stats(video)
        # select frames from the video
        # e.g. fps, similarity
        frames = self.frame_selector.select_frames(video)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # extract images from frames into files
            # e.g. downscale, tile, crop
            extracted_frames = self.frame_image_extractor.extract_images(frames, tmpdir_path)
            if len(extracted_frames) == 0:
                # no images extracted
                result = RichResult(
                    is_animal_present=False,
                    is_animal_unknown=False,
                    defining_features="",
                    species_name="no animal",
                    confidence=ConfidenceLevel.HIGH,
                )
                return PipelineOutcome(result=result, stats=stats, batches=[])
            # send each batch to the AI for identification
            enriched_frames = self.image_batch_query.query_image_batches(extracted_frames)
            # reconcile multiple identifications (if applicable)
            query_results = enriched_frames.get_batch_results()
            consolidated_result = self.result_reconciler.reconcile_results(query_results)

        return PipelineOutcome(result=consolidated_result, stats=stats, batches=list(enriched_frames.batches))
