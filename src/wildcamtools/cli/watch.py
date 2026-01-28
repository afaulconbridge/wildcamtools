import bisect
import contextlib
import logging
import os
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from time import sleep
from typing import Annotated

import typer
from typer_config import use_yaml_config

from wildcamtools.lib.concat import concat_ffmpeg
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.states import MotionWindow, Watcher, WatcherStateEnum, WatcherTransitionMetrics
from wildcamtools.lib.stats import VideoStats, get_video_stats
from wildcamtools.lib.vidio import FrameSourceFFMPEG

app = typer.Typer()
logger = logging.getLogger(__name__)


def cleanup_old_segments(path: Path, max_file_count: int) -> None:
    # list files in directory
    files = os.listdir(path)
    if len(files) > max_file_count:
        # sort in order
        files = sorted(files, reverse=True)
        # identify the oldest ones
        files_to_remove = files[max_file_count:]
        for file_to_remove in files_to_remove:
            logger.debug(f"removing {path / file_to_remove}")
            os.unlink(path / file_to_remove)


def find_segments_for_timespan(start_time: datetime, end_time: datetime, segments_dir: Path) -> tuple[Path, ...] | None:
    logger.info(f"Finding segments from {start_time} to {end_time}")
    # given a directory

    # build a list of files in segment directory
    segments_files = tuple(sorted(segments_dir.iterdir()))

    # turn the time into a path to a non-existant file
    ftime_string = "seg_%Y_%m_%d__%H_%M_%S.mp4"
    start_time_path = segments_dir / start_time.strftime(ftime_string)
    end_time_path = segments_dir / end_time.strftime(ftime_string)
    logger.info(f"Finding segments from {start_time_path} to {end_time_path}")

    # work out where in the list the start and end filenames would be inserted
    start_position = bisect.bisect_left(segments_files, start_time_path)
    end_position = bisect.bisect_right(segments_files, end_time_path)
    logger.info(f"Segment positions from {start_position} to {end_position}")

    if end_position == len(segments_files):
        # if we would cover the last file, beware
        # it might not be complete and parseable yet.
        # signal to caller to try again soon
        logger.info("Potentially incomplete file detected")
        return None
    else:
        # files that should be merged
        # TODO also offsets to trim
        return segments_files[max(start_position - 1, 0) : end_position]


def motion_states(
    rtsp_stream: str,
    queue: Queue,
    history: int = 30,
    threshold: int = 16,
    kernel_size: int = 3,
    scale: float = 0.25,
    green_to_amber_motion_min: float = 0.01,
    amber_to_green_proportion_max: float = 0.0075,
    amber_to_red_duration: int = 5,
    red_to_red_amber_proportion_max: float = 0.0075,
    red_amber_to_red_proportion_min: float = 0.01,
    red_amber_to_green_duration: int = 5,
) -> None:
    def _find_motion_times(source: str, stats: VideoStats, watcher: Watcher) -> Generator[MotionWindow]:
        start_frame: int | None = None
        start_time: datetime | None = None
        with FrameSourceFFMPEG(source, width=stats.x, height=stats.y, scale=scale) as video_input:
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

    transition_metrics = WatcherTransitionMetrics(
        preparing_duration=history,
        green_to_amber_motion_min=green_to_amber_motion_min,
        amber_to_green_proportion_max=amber_to_green_proportion_max,
        amber_to_red_duration=amber_to_red_duration,
        red_to_red_amber_proportion_max=red_to_red_amber_proportion_max,
        red_amber_to_red_proportion_min=red_amber_to_red_proportion_min,
        red_amber_to_green_duration=red_amber_to_green_duration,
    )
    watcher = Watcher(
        motion=MogMotion(
            history=history,
            threshold=threshold,
            detect_shadows=False,
            kernel_size=kernel_size,
        ),
        transition_metrics=transition_metrics,
    )
    stats = get_video_stats(rtsp_stream)
    for motion in _find_motion_times(rtsp_stream, stats, watcher):
        queue.put(motion)


class WatcherManagerStateEnum(StrEnum):
    WAITING = "WAITING"
    RECORDING = "RECORDING"


class WatcherManager:
    rtsp_stream: str
    segments_dir: Path
    output_dir: Path
    keep_count: int
    offset_start: float
    offset_end: float
    history: int
    threshold: int
    kernel_size: int
    scale: float
    green_to_amber_motion_min: float
    amber_to_green_proportion_max: float
    amber_to_red_duration: int
    red_to_red_amber_proportion_max: float
    red_amber_to_red_proportion_min: float
    red_amber_to_green_duration: int

    msg_queue: Queue
    stat: WatcherManagerStateEnum
    motion_process: Process | None = None

    def __init__(
        self,
        rtsp_stream: str,
        segments_dir: Path,
        output_dir: Path,
        keep_count: int,
        offset_start: float,
        offset_end: float,
        history: int,
        threshold: int,
        kernel_size: int,
        scale: float,
        green_to_amber_motion_min: float,
        amber_to_green_proportion_max: float,
        amber_to_red_duration: int,
        red_to_red_amber_proportion_max: float,
        red_amber_to_red_proportion_min: float,
        red_amber_to_green_duration: int,
    ) -> None:
        self.rtsp_stream = rtsp_stream
        self.segments_dir = segments_dir
        self.output_dir = output_dir
        self.keep_count = keep_count
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.history = history
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.scale = scale
        self.green_to_amber_motion_min = green_to_amber_motion_min
        self.amber_to_green_proportion_max = amber_to_green_proportion_max
        self.amber_to_red_duration = amber_to_red_duration
        self.red_to_red_amber_proportion_max = red_to_red_amber_proportion_max
        self.red_amber_to_red_proportion_min = red_amber_to_red_proportion_min
        self.red_amber_to_green_duration = red_amber_to_green_duration

        self.msg_queue = Queue()
        self.state = WatcherManagerStateEnum.WAITING

    def create_motion_process(self) -> None:
        self.motion_process = Process(
            target=motion_states,
            kwargs={
                "rtsp_stream": self.rtsp_stream,
                "queue": self.msg_queue,
                "history": self.history,
                "threshold": self.threshold,
                "kernel_size": self.kernel_size,
                "scale": self.scale,
                "green_to_amber_motion_min": self.green_to_amber_motion_min,
                "amber_to_green_proportion_max": self.amber_to_green_proportion_max,
                "amber_to_red_duration": self.amber_to_red_duration,
                "red_to_red_amber_proportion_max": self.red_to_red_amber_proportion_max,
                "red_amber_to_red_proportion_min": self.red_amber_to_red_proportion_min,
                "red_amber_to_green_duration": self.red_amber_to_green_duration,
            },
            daemon=True,
        )
        self.motion_process.start()

    def get_message(self) -> MotionWindow | None:
        if not self.motion_process:
            self.create_motion_process()
            return None

        if not self.motion_process.is_alive():
            logger.warning("Motion Process is dead, recreating")
            self.create_motion_process()
            return None

        msg = None
        with contextlib.suppress(Empty):
            msg = self.msg_queue.get_nowait()
        return msg

    def run(self) -> None:
        self.create_motion_process()

        # TODO start segment_process

        while True:
            # TODO check segment_process is alive
            # check queue for a message
            msg = None
            msg = self.get_message()
            if msg:
                if msg.end_time is None:
                    self.state = WatcherManagerStateEnum.RECORDING
                    logger.info("Starting recording")
                else:
                    start_time = msg.start_time - timedelta(seconds=self.offset_start)
                    end_time = msg.end_time + timedelta(seconds=self.offset_end)
                    while (to_merge := find_segments_for_timespan(start_time, end_time, self.segments_dir)) is None:
                        sleep(1)
                    output_file = self.output_dir / start_time.strftime("out_%Y_%m_%d__%H_%M_%S.mp4")
                    logger.info(f"Joining {len(to_merge)} segments into {output_file}")
                    concat_ffmpeg(to_merge, output_file)
                    self.state = WatcherManagerStateEnum.WAITING
            elif self.state == WatcherManagerStateEnum.WAITING:
                # waiting to record, cleanup old files
                cleanup_old_segments(
                    path=self.segments_dir,
                    max_file_count=self.keep_count,
                )
            else:
                # nothing to do, sleep
                sleep(1)


@app.command()
@use_yaml_config()
def watch(
    rtsp_stream: Annotated[str, typer.Argument(metavar="RTSP_URL", envvar="WCT_RTSP")],
    segments: Annotated[Path, typer.Argument(metavar="PATH", envvar="WCT_SEGMENTS")],
    output: Annotated[Path, typer.Argument(metavar="PATH", envvar="WCT_OUTPUT")],
    keep_count: Annotated[int, typer.Option(metavar="INT", envvar="WCT_KEEP")] = 4,
    offset_start: Annotated[float, typer.Option(metavar="FLOAT", envvar="WCT_OFFSET_START")] = 10.0,
    offset_end: Annotated[float, typer.Option(metavar="FLOAT", envvar="WCT_OFFSET_END")] = 10.0,
    history: Annotated[int, typer.Option(metavar="INT", envvar="WTC_HISTORY")] = 30,
    threshold: Annotated[int, typer.Option(metavar="INT", envvar="WTC_THRESHOLD")] = 16,
    kernel_size: Annotated[int, typer.Option(metavar="INT", envvar="WTC_KERNEL_SIZE")] = 3,
    scale: Annotated[float, typer.Option(metavar="FLOAT", envvar="WTC_SCALE")] = 0.25,
    green_to_amber_motion_min: Annotated[float, typer.Option(metavar="FLOAT", envvar="WTC_GREEN_2_AMBER_MIN")] = 0.01,
    amber_to_green_proportion_max: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WTC_AMBER_2_GREEN_MAX")
    ] = 0.0075,
    amber_to_red_duration: Annotated[int, typer.Option(metavar="INT", envvar="WTC_AMBER_2_RED_DURATION")] = 5,
    red_to_red_amber_proportion_max: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WTC_RED_2_RED_AMBER_MAX")
    ] = 0.0075,
    red_amber_to_red_proportion_min: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WTC_RED_AMBER_2_RED_MIN")
    ] = 0.01,
    red_amber_to_green_duration: Annotated[
        int, typer.Option(metavar="INT", envvar="WTC_RED_AMBER_2_GREEN_DURATION")
    ] = 5,
) -> None:

    # TODO validate rtsp_stream is a rtsp url

    segments = segments.resolve()
    if not segments.is_dir() or not segments.exists():
        raise typer.BadParameter("segments must be an existing directory")  # noqa: TRY003

    output = output.resolve()
    if not output.is_dir() or not output.exists():
        raise typer.BadParameter("output must be an existing directory")  # noqa: TRY003

    watcher = WatcherManager(
        rtsp_stream=rtsp_stream,
        segments_dir=segments,
        output_dir=output,
        keep_count=keep_count,
        offset_start=offset_start,
        offset_end=offset_end,
        history=history,
        threshold=threshold,
        kernel_size=kernel_size,
        scale=scale,
        green_to_amber_motion_min=green_to_amber_motion_min,
        amber_to_green_proportion_max=amber_to_green_proportion_max,
        amber_to_red_duration=amber_to_red_duration,
        red_to_red_amber_proportion_max=red_to_red_amber_proportion_max,
        red_amber_to_red_proportion_min=red_amber_to_red_proportion_min,
        red_amber_to_green_duration=red_amber_to_green_duration,
    )
    watcher.run()
