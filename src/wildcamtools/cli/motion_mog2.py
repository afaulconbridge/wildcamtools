from pathlib import Path
from typing import Annotated

import cv2
import typer

from wildcamtools.lib.frames import MotionFlowHighlighter
from wildcamtools.lib.motion import AvgMotion, FlowMotion, MogMotion, MotionHandler
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.vidio import VideoReader, VideoWriter

app = typer.Typer()


def _shared(
    input_: Path,
    output: Path,
    fps: float,
    history: int,
    handler: MotionHandler,
) -> None:
    timer = Timer()

    with (
        VideoWriter(output, fps=fps) as video_writer,
        VideoReader(input_) as video_input,
    ):
        frame_out = None
        for frame in video_input:
            with timer:
                frame_out = handler.handle(frame)
                if frame_out.motion_proportion > 0.001:
                    typer.secho(f"{frame.frame_no:4d} {frame_out.motion_proportion:0.3f}")
                if frame.frame_no >= history:
                    image_out = frame.raw.copy()
                    for bbox in handler.get_contour_bboxes():
                        cv2.rectangle(image_out, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 3)
                    video_writer.write(image_out)

    typer.secho(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


@app.command(name="flow")
def motion_flow(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT")],
    history: int = 5,
    threshold: float = 0.1,
    kernel_size: float = 0.02,
) -> None:
    stats = get_video_stats(input_)

    if stats.frame_count - history < 0:
        typer.secho("Must have input longer than history")
        raise typer.Exit(code=1)

    if threshold <= 0:
        typer.secho("Threshold must be greater than 0")
        raise typer.Exit(code=1)

    if kernel_size < 0:
        typer.secho("Kernel size cannot be negative")
        raise typer.Exit(code=1)

    motion = FlowMotion(history=history, threshold=threshold, kernel_size=kernel_size)
    _shared(input_, output, stats.fps, history, motion)


@app.command()
def flow_highlighter(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT")],
    alpha: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    magnitude: Annotated[float, typer.Option(min=0.0)] = 10.0,
) -> None:
    stats = get_video_stats(input_)
    handler = MotionFlowHighlighter(alpha=alpha, max_magnitude=magnitude)
    timer = Timer()

    with VideoWriter(output, fps=stats.fps) as video_writer, VideoReader(input_) as video_input:
        for frame in video_input:
            with timer:
                frame = handler.handle(frame)
            if frame.filter_keep:
                video_writer.write(frame.raw)

    typer.secho(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


@app.command(name="mog2")
def motion_mog2(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT")],
    history: int = 25,
    threshold: int = 16,
    kernel_size: float = 0.005,
) -> None:
    stats = get_video_stats(input_)

    if stats.frame_count - history < 0:
        typer.secho("Must have input longer than history")
        raise typer.Exit(code=1)

    if kernel_size < 0:
        typer.secho("Kernel size cannot be negative")
        raise typer.Exit(code=1)

    motion = MogMotion(history=history, threshold=threshold, detect_shadows=False, kernel_size=kernel_size)
    _shared(input_, output, stats.fps, history, motion)


@app.command(name="avg")
def motion_avg(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT")],
    history: int = 25,
    threshold: int = 16,
    kernel_size: float = 0.005,
) -> None:
    stats = get_video_stats(input_)

    if stats.frame_count - history < 0:
        typer.secho("Must have input longer than history")
        raise typer.Exit(code=1)

    if kernel_size < 0:
        typer.secho("Kernel size cannot be negative")
        raise typer.Exit(code=1)

    motion = AvgMotion(history=history, threshold=threshold, kernel_size=kernel_size)
    _shared(input_, output, stats.fps, history, motion)


if __name__ == "__main__":
    app()
