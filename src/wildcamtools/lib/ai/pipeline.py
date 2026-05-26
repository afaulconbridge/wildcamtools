import logging
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar

import cv2
from pydantic import BaseModel

from wildcamtools.lib import Frame
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.frames import Rescaler, resize_with_aspect_ratio
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.vidio import VideoReader

logger = logging.getLogger(__name__)


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


class FrameImageExtractor(ABC):
    @abstractmethod
    def extract_images(self, frames: Iterable[Frame], outdir: Path) -> Sequence[Sequence[Path]]: ...


class RescaledFrameImageExtractor(FrameImageExtractor):
    def __init__(self, resolution: tuple[int, int] = (640, 360)) -> None:
        self.resolution = resolution

    def extract_images(self, frames: Iterable[Frame], outdir: Path) -> Sequence[Sequence[Path]]:
        outdir.mkdir(parents=True, exist_ok=True)

        current_batch: list[Path] = []

        for frame in frames:
            if not frame.filter_keep:
                continue

            rescaled_image = resize_with_aspect_ratio(frame.output, self.resolution)
            image_path = outdir / f"frame_{frame.frame_no:05d}.jpg"
            cv2.imwrite(str(image_path), rescaled_image)
            current_batch.append(image_path)

        return [current_batch] if current_batch else []


T = TypeVar("T", bound=BaseModel)


class ImageBatchQuery[T](ABC):
    def query_image_batches(self, image_batches: Iterable[Iterable[Path]]) -> Iterator[T]:
        for batch in image_batches:
            yield self.query_images(list(batch))

    @abstractmethod
    def query_images(self, images: Sequence[Path]) -> T: ...


class LlmImageBatchQuery[T](ImageBatchQuery[T]):
    llm: AbstractLlm
    prompt: str
    response_class: type[T]

    def __init__(
        self,
        llm: AbstractLlm,
        prompt: str,
        response_class: type[T],
    ) -> None:
        self.llm = llm
        self.prompt = prompt
        self.response_class = response_class

    def query_images(self, images: Sequence[Path]) -> T:
        images_list = sorted(images)
        if not images_list:
            logger.warning("Empty image batch received")
            raise ValueError("Empty image batch")
        logger.info("Sending %d images to analyser", len(images_list))
        result: T = self.llm.message_with_schema(
            message=self.prompt,
            images=images_list,
            response_class=self.response_class,
        )  # type: ignore[type-var]
        logger.info("Received response from analyser")
        return result


class ResultReconciler[T](ABC):
    @abstractmethod
    def reconcile_results(self, results: Iterable[T]) -> T: ...


class MajorityResultReconciler[T](ResultReconciler[T]):
    """Reconciles multiple results by selecting the most common value.

    Uses equality comparison (__eq__) to determine uniqueness.
    In case of ties, returns the first-seen result among those tied.

    Raises:
        ValueError: If results iterable is empty.
    """

    def reconcile_results(self, results: Iterable[T]) -> T:
        results_list = list(results)
        if not results_list:
            raise ValueError("No results to reconcile")
        if len(results_list) == 1:
            return results_list[0]

        logger.debug("Reconciling %d results", len(results_list))
        seen_order: list[T] = []
        counts: list[int] = []

        for result in results_list:
            found_index = -1
            for i, seen in enumerate(seen_order):
                if seen == result:
                    found_index = i
                    break

            if found_index >= 0:
                counts[found_index] += 1
            else:
                seen_order.append(result)
                counts.append(1)

        max_count = max(counts)
        for i, count in enumerate(counts):
            if count == max_count:
                logger.debug("Selected result with count %d (first-seen wins ties)", max_count)
                return seen_order[i]

        raise RuntimeError("Unable to reconcile results")  # pragma: no cover


class AiPipeline[T](ABC):
    frame_selector: FrameSelector
    frame_image_extractor: FrameImageExtractor
    image_batch_query: ImageBatchQuery[T]
    result_reconciler: ResultReconciler[T]

    def __init__(
        self,
        frame_selector: FrameSelector,
        frame_image_extractor: FrameImageExtractor,
        image_batch_query: ImageBatchQuery[T],
        result_reconciler: ResultReconciler[T],
    ) -> None:
        self.frame_selector = frame_selector
        self.frame_image_extractor = frame_image_extractor
        self.image_batch_query = image_batch_query
        self.result_reconciler = result_reconciler

    def run(self, video: Path) -> T:
        # select frames from the video
        # e.g. fps, similarity
        frames = self.frame_selector.select_frames(video)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # extract images from frames into files
            # e.g. downscale, tile, crop
            image_batches = self.frame_image_extractor.extract_images(frames, tmpdir_path)
            # send each batch to the AI for identification
            query_results = self.image_batch_query.query_image_batches(image_batches)
            # reconcile multiple identifications (if applicable)
            consolidated_result = self.result_reconciler.reconcile_results(query_results)

        return consolidated_result
