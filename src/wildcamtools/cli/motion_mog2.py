from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.stats import Colourspace, VideoFileStats, get_video_stats
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.vidio import generate_frames_cv2, save_video

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
        print("Must have input longer than history")
        raise typer.Exit(code=1)

    output_stats = VideoFileStats(
        fps=stats.fps,
        frame_count=stats.frame_count - history,
        x=stats.x,
        y=stats.y,
        colourspace=Colourspace.boolean,
    )

    mog_motion = MogMotion(history=history, threshold=threshold, detect_shadows=False, kernel_size=kernel_size)
    timer = Timer()

    with save_video(output, stats=output_stats) as video_writer:
        frame_out = None
        for i, frame in generate_frames_cv2(input_):
            with timer:
                frame_out = mog_motion.handle(frame)
            if i >= history:
                video_writer.write(frame_out)

    print(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")


if __name__ == "__main__":
    app()
