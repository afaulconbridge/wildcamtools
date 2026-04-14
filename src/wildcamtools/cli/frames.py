from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib import Frame, FrameHandler
from wildcamtools.lib.errors.cli import OutputNotDirectoryError
from wildcamtools.lib.frames import FrameImageWriter, Rescaler
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.vidio import FrameSourceFFMPEG

app = typer.Typer()


@app.command()
def frames(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path | None, typer.Option(metavar="OUTPUT")] = None,
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
    handlers.append(FrameImageWriter(output))

    timer = Timer()

    with FrameSourceFFMPEG(input_) as video_input:
        frame: Frame | None
        for frame in video_input:
            with timer:
                for handler in handlers:
                    if frame:
                        frame = handler.handle(frame)

    typer.secho(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


if __name__ == "__main__":
    app()
