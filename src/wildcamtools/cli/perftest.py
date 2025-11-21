import contextlib
import math

import typer

from wildcamtools.lib import FrameHandler
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.rescale import Rescaler
from wildcamtools.lib.stats import Colourspace, VideoFileStats, get_video_stats
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.vidio import generate_frames_cv2, save_video

app = typer.Typer()


def convert(input_: str, output: str, output_stats: VideoFileStats, timer: Timer, handler: FrameHandler) -> None:
    with save_video(output, stats=output_stats) as video_writer:
        for _, frame in generate_frames_cv2(input_):
            with timer:
                frame_rescaled = handler.handle(frame)
            if frame_rescaled is not None:
                video_writer.write(frame_rescaled)


@app.command()
def perftest(output: str = "-") -> None:
    with contextlib.ExitStack() as stack:
        if output != "-":
            stack.enter_context(output_file := open(output, "w"))  # noqa: SIM115
            stack.enter_context(contextlib.redirect_stdout(output_file))

        input_ = "tests/data/04-51-08.mp4"
        input_stats = get_video_stats(input_)
        # want to know at what resolution we can process faster than the source frame rate
        downscales = [(input_stats.x, input_stats.y, input_stats.fps)]
        # add halving sizes until the smallest is less than 300 pixels high
        while downscales[-1][1] > 300:
            downscales.append((
                downscales[-1][0] // 2,
                downscales[-1][1] // 2,
                downscales[-1][2],
            ))
        # add halving frame rates untill less than 10 fps
        # do this at all resolution scales
        downscaled_resolutions = tuple(downscales)
        while downscales[-1][2] > 10:
            new_fps = downscales[-1][2] // 2
            downscales.extend((d[0], d[1], new_fps) for d in downscaled_resolutions)

        # make files at each resolution
        downscaled_files: dict[tuple[int, int, float], str] = {}
        for x, y, fps in downscales:
            rescaler = Rescaler(stats=input_stats, x=x, y=y, fps=fps)
            output_stats = VideoFileStats(
                fps=fps,
                frame_count=math.floor(
                    (rescaler.source_frametime / rescaler.target_frametime) * input_stats.frame_count
                ),
                x=x,
                y=y,
                colourspace=input_stats.colourspace,
            )
            output = f"{x}x{y}f{fps:.2f}.mp4"
            print(output)
            timer = Timer()
            convert(input_, output, output_stats, timer, rescaler)
            print(f"Processed {timer.intervals:d} frames in {timer.elapsed:.2f} sec; {timer.per_second:.2f}FPS")
            downscaled_files[(x, y, fps)] = output

        # now do motion detection on each file that was downscaled
        for (x, y, fps), input_ in downscaled_files.items():
            input_stats = get_video_stats(input_)
            output_stats = VideoFileStats(
                fps=input_stats.fps,
                frame_count=input_stats.frame_count,
                x=input_stats.x,
                y=input_stats.y,
                colourspace=Colourspace.boolean,
            )
            motion_detector = MogMotion()  # TODO vary parameters?
            timer = Timer()
            convert(input_, "output.mp4", output_stats, timer, motion_detector)
            speed = input_stats.duration_in_sconds / timer.elapsed
            print(f"Processed {x}x{y}@{fps:.2f} in {timer.elapsed:.2f} sec; {speed:.2f}x")
