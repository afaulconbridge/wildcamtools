from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib import Frame
from wildcamtools.lib.frames import Rescaler
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

    handler = Rescaler(stats=stats, x=x, y=y, fps=fps)
    timer = Timer()

    with FrameWriterFFMPEG(output, fps=handler.fps) as video_writer, FrameSourceFFMPEG(input_) as video_input:
        frame: Frame | None
        for frame in video_input:
            with timer:
                frame = handler.handle(frame)
            if frame is not None:
                video_writer.write(frame.raw)

    typer.secho(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


if __name__ == "__main__":
    app()
