import math
from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib.rescale import Rescaler
from wildcamtools.lib.stats import VideoFileStats, get_video_stats
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.vidio import generate_frames_cv2, save_video

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
    output_stats = VideoFileStats(
        fps=rescaler.fps,
        frame_count=math.floor((rescaler.source_frametime / rescaler.target_frametime) * stats.frame_count),
        x=rescaler.x,
        y=rescaler.y,
        colourspace=stats.colourspace,
    )

    with save_video(output, stats=output_stats) as video_writer:
        for _, frame in generate_frames_cv2(input_):
            with timer:
                frame_rescaled = rescaler.handle(frame)
            if frame_rescaled is not None:
                video_writer.write(frame_rescaled)

    print(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


if __name__ == "__main__":
    app()
