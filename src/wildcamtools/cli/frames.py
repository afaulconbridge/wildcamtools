import logging
from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib import Frame, FrameHandler
from wildcamtools.lib.errors.cli import OutputNotDirectoryError
from wildcamtools.lib.frames import CropPanHandler, FilterSSIM, FrameImageWriter, Rescaler
from wildcamtools.lib.motion import FlowMotion
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.vidio import FrameSourceFFMPEG

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def ssim(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path | None, typer.Argument(metavar="OUTPUT")] = None,
    threshold: float | None = None,
    x: int | None = None,
    y: int | None = None,
    fps: float | None = None,
) -> None:

    if output is None:
        output = input_.with_suffix("")

    if output.exists() and not output.is_dir():
        raise OutputNotDirectoryError()

    stats = get_video_stats(input_)

    handlers: list[FrameHandler] = []
    if x or y or fps:
        handlers.append(Rescaler(stats=stats, x=x, y=y, fps=fps))
    if threshold:
        handlers.append(FilterSSIM(threshold))
    handlers.append(FrameImageWriter(output))

    timer = Timer()

    with FrameSourceFFMPEG(input_) as video_input:
        frame: Frame
        for frame in video_input:
            with timer:
                for handler in handlers:
                    frame = handler.handle(frame)

    typer.secho(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


@app.command()
def flow(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path | None, typer.Argument(metavar="OUTPUT")] = None,
    history: int = 30,
    threshold: float = 0.1,
    kernel_size: float = 0.02,
    x: int | None = None,
    y: int | None = None,
    fps: float | None = None,
    crop_expansion: float = 0.75,
    crop_inertia: float = 10.0,
    similarity_minimum: float = 0.5,
) -> None:

    if output is None:
        output = input_.with_suffix("")

    if output.exists() and not output.is_dir():
        raise OutputNotDirectoryError()
    output.mkdir(parents=True, exist_ok=True)

    stats = get_video_stats(input_)
    timer = Timer()

    handlers: list[FrameHandler] = []
    handlers.append(
        CropPanHandler(
            motion_handler=FlowMotion(
                history=history,
                threshold=threshold,
                kernel_size=kernel_size,
            ),
            expansion=crop_expansion,
            inertia=crop_inertia,
        )
    )
    handlers.append(FilterSSIM(similarity_minimum=similarity_minimum))
    handlers.append(
        Rescaler(
            stats=stats,
            x=x,
            y=y,
            fps=fps,
        )
    )
    handlers.append(FrameImageWriter(output))

    with FrameSourceFFMPEG(input_) as video_input:
        for frame in video_input:
            with timer:
                for handler in handlers:
                    frame = handler.handle(frame)

    typer.secho(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


if __name__ == "__main__":
    app()
