from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.vidio import FrameSourceFFMPEG, FrameWriterFFMPEG

app = typer.Typer()


@app.command()
def motion_mog(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT")],
    history: int = 500,
    threshold: int = 16,
    kernel_size: int = 3,
) -> None:
    stats = get_video_stats(input_)

    if stats.frame_count - history < 0:
        typer.secho("Must have input longer than history")
        raise typer.Exit(code=1)

    mog_motion = MogMotion(history=history, threshold=threshold, detect_shadows=False, kernel_size=kernel_size)
    timer = Timer()

    with FrameWriterFFMPEG(output, fps=stats.fps) as video_writer:
        frame_out = None
        for frame in FrameSourceFFMPEG(input_):
            with timer:
                frame_out = mog_motion.handle(frame)
                prop = mog_motion.get_motion_proportion(frame_out.raw)
            if prop > 0.001:
                typer.secho(f"{frame.frame_no:4d} {prop:0.3f}")
            if frame.frame_no >= history:
                video_writer.write(frame_out.raw)

    typer.secho(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


if __name__ == "__main__":
    app()
