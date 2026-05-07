from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib import Frame
from wildcamtools.lib.frames import Rescaler
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.vidio import VideoReader, VideoWriter

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

    handler = Rescaler(stats=stats, x=x, y=y, fps=fps)
    timer = Timer()

    with VideoWriter(output, fps=handler.fps) as video_writer, VideoReader(input_) as video_input:
        frame: Frame
        for frame in video_input:
            with timer:
                frame = handler.handle(frame)
            if frame.filter_keep:
                video_writer.write(frame.output)

    typer.secho(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


if __name__ == "__main__":
    app()
