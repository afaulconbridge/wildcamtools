from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib.motion import AvgMotion, FlowMotion, MogMotion, MotionHandler
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.vidio import FrameSourceFFMPEG, FrameWriterFFMPEG

app = typer.Typer()


MOTION_REPORT_THRESHOLD = 0.001


def _shared(
    input_: Path,
    output: Path,
    fps: float,
    history: int,
    handler: MotionHandler,
) -> None:
    timer = Timer()

    with (
        FrameWriterFFMPEG(output, fps=fps) as video_writer,
        FrameSourceFFMPEG(input_) as video_input,
    ):
        for frame in video_input:
            with timer:
                frame_out = handler.handle(frame)
            if frame_out.motion_proportion > MOTION_REPORT_THRESHOLD:
                typer.secho(f"{frame.frame_no:4d} {frame_out.motion_proportion:0.3f}")
            if frame.frame_no >= history:
                video_writer.write(frame_out.raw)

    typer.secho(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


def _validate_common(stats_frame_count: int, history: int, threshold: float, kernel_size: float) -> None:
    if history >= stats_frame_count:
        typer.secho("Must have input longer than history")
        raise typer.Exit(code=1)

    if threshold <= 0:
        typer.secho("Threshold must be greater than 0")
        raise typer.Exit(code=1)

    if kernel_size < 0:
        typer.secho("Kernel size cannot be negative")
        raise typer.Exit(code=1)


@app.command()
def flow(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT")],
    history: int = 25,
    threshold: float = 5.0,
    kernel_size: float = 0.01,
) -> None:
    stats = get_video_stats(input_)
    _validate_common(stats.frame_count, history, threshold, kernel_size)

    motion = FlowMotion(history=history, threshold=threshold, kernel_size=kernel_size)
    _shared(input_, output, stats.fps, history, motion)


@app.command()
def mog2(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT")],
    history: int = 25,
    threshold: int = 16,
    kernel_size: float = 0.01,
) -> None:
    stats = get_video_stats(input_)
    _validate_common(stats.frame_count, history, threshold, kernel_size)

    motion = MogMotion(history=history, threshold=threshold, detect_shadows=False, kernel_size=kernel_size)
    _shared(input_, output, stats.fps, history, motion)


@app.command(name="avg")
def motion_avg(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT")],
    history: int = 25,
    threshold: int = 16,
    kernel_size: float = 0.01,
) -> None:
    stats = get_video_stats(input_)
    _validate_common(stats.frame_count, history, threshold, kernel_size)

    motion = AvgMotion(history=history, threshold=threshold, kernel_size=kernel_size)
    _shared(input_, output, stats.fps, history, motion)


if __name__ == "__main__":
    app()
