import sys
from collections.abc import Generator
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Annotated

import typer
from pydantic import BaseModel

from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.states import Watcher, WatcherStateEnum, WatcherTransitionMetrics
from wildcamtools.lib.stats import VideoStats, get_video_stats
from wildcamtools.lib.vidio import FrameSourceFFMPEG

app = typer.Typer()


class MotionWindow(BaseModel):
    start_frame: int
    start_time: datetime
    end_frame: int | None
    end_time: datetime | None


class FileResult(BaseModel):
    metadata: VideoStats
    motion: list[MotionWindow]


def _find_motion_times(source: str, stats: VideoStats, watcher: Watcher) -> Generator[MotionWindow]:
    start_frame: int | None = None
    start_time: datetime | None = None
    with FrameSourceFFMPEG(source, stats.x, stats.y) as video_input:
        for frame in video_input:
            frame = watcher.handle(frame)
            if start_frame is None and watcher.state == WatcherStateEnum.RED:
                start_frame = frame.frame_no
                start_time = datetime.now(UTC)
                yield MotionWindow(
                    start_frame=start_frame,
                    start_time=start_time,
                    end_frame=None,
                    end_time=None,
                )
            elif start_frame is not None and watcher.state == WatcherStateEnum.GREEN:
                end_frame = frame.frame_no
                end_time = datetime.now(UTC)
                yield MotionWindow(
                    start_frame=start_frame,
                    start_time=start_time,
                    end_frame=end_frame,
                    end_time=end_time,
                )
                start_frame = None
                start_time = None


@app.command()
def states(
    input_: Annotated[str, typer.Argument(metavar="INPUT")],
    history: int = 10,
    output: str | None = None,
):
    transition_metrics = WatcherTransitionMetrics(
        preparing_duration=history,
        green_to_amber_motion_min=0.01,
        amber_to_green_proportion_max=0.0075,
        amber_to_red_duration=5,
        red_to_red_amber_proportion_max=0.0075,
        red_amber_to_red_proportion_min=0.01,
        red_amber_to_green_duration=5,
    )
    watcher = Watcher(motion=MogMotion(history=history), transition_metrics=transition_metrics)

    stats = get_video_stats(input_)

    # output single JSON doc per file
    # do this before processing the video so errors surface at the start
    with nullcontext(sys.stdout) if output is None else open(output, "w") as output_target:
        for motion in _find_motion_times(input_, stats, watcher):
            output_target.write(motion.model_dump_json())
            output_target.write("\n")
            output_target.flush()
