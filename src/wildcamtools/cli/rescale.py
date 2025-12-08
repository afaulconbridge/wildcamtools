from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib.rescale import Rescaler
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.vidio import FrameSourceFFMPEG, FrameWriterFFMPEG

app = typer.Typer()


@app.command()
def rescale(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT")],
    x: int | None = None,
    y: int | None = None,
    fps: float | None = None,
) -> None:
    stats = get_video_stats(input_)

    rescaler = Rescaler(stats=stats, x=x, y=y, fps=fps)
    timer = Timer()

    with FrameWriterFFMPEG(output, fps=rescaler.fps) as video_writer:
        for frame in FrameSourceFFMPEG(input_):
            with timer:
                frame_rescaled = rescaler.handle(frame)
            if frame_rescaled is not None:
                video_writer.write(frame_rescaled)

    typer.secho(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


if __name__ == "__main__":
    app()
