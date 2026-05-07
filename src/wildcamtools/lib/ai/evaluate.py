import logging
from pathlib import Path

from wildcamtools.lib import FrameHandler
from wildcamtools.lib.ai import AbstractAnalyser, Backend
from wildcamtools.lib.ai.llamacpp import LlamaCppAnalyser
from wildcamtools.lib.ai.ollama import OllamaAnalyser
from wildcamtools.lib.frames import CropPanHandler, FilterSSIM, FrameImageWriter, Rescaler
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.vidio import VideoReader

logger = logging.getLogger(__name__)


def create_frames(
    filename: Path,
    video_directory: Path,
    tmpdir: Path,
    history: int = 30,
    threshold: float = 16.0,
    kernel_size: float = 0.02,
    x: int | None = None,
    y: int | None = None,
    fps: float | None = None,
    crop_expansion: float = 0.75,
    crop_inertia: float = 10.0,
    similarity_minimum: float | None = None,
    do_croppan: bool = False,
) -> None:
    stats = get_video_stats(video_directory / filename)
    handlers: list[FrameHandler] = []

    if do_croppan:
        handlers.append(
            CropPanHandler(
                motion_handler=MogMotion(
                    history=history,
                    threshold=int(threshold),
                    kernel_size=kernel_size,
                ),
                expansion=crop_expansion,
                inertia=crop_inertia,
            )
        )
    # TODO rescale after similarity vs rescale before similarity
    if x or y or fps:
        handlers.append(
            Rescaler(
                stats=stats,
                x=x,
                y=y,
                fps=fps,
            )
        )

    if similarity_minimum is not None:
        handlers.append(FilterSSIM(similarity_minimum=similarity_minimum))

    handlers.append(FrameImageWriter(Path(tmpdir)))

    with VideoReader(video_directory / filename) as video_input:
        for frame in video_input:
            for handler in handlers:
                frame = handler.handle(frame)


def evaluate_frames(
    frame_directory: Path,
    label: str,
    backend: Backend,
    url: str,
    model: str,
    api_key: str = "API_KEY",
    prompt: str | None = None,
) -> bool:
    analyser: AbstractAnalyser
    match backend:
        case Backend.LLAMACPP:
            analyser = LlamaCppAnalyser(model=model, base_url=url, message=prompt)
        case Backend.OLLAMA:
            analyser = OllamaAnalyser(model=model, host=url, api_key=api_key, message=prompt)
        case _:
            raise ValueError(f"Unsupported backend: {backend}")

    images = list(frame_directory.iterdir())
    result = analyser.analyze_video(images) if images else "no"
    logger.info("Label: %s Result: %s", label, result)
    return label.lower() == result.lower()
