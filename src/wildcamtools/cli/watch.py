import bisect
import contextlib
import logging
import os
from datetime import datetime, timedelta
from enum import StrEnum
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from subprocess import Popen
from time import sleep
from typing import Annotated

import typer
from typer_config import use_yaml_config

from wildcamtools.lib.concat import concat_videos
from wildcamtools.lib.errors import (
    CannotCombineOpenWindowError,
    MotionMaskNotExistsError,
    MotionMaskNotFileError,
    MotionMaskNotReadableError,
)
from wildcamtools.lib.segment import create_segment_process
from wildcamtools.lib.states import (
    MotionWindow,
    WatcherTransitionMetrics,
    create_motion_process,
)

app = typer.Typer()
logger = logging.getLogger(__name__)


def find_segments_for_timespan(start_time: datetime, end_time: datetime, segments_dir: Path) -> tuple[Path, ...] | None:
    logger.info("Finding segments from %s to %s", start_time, end_time)
    # given a directory

    # build a list of files in segment directory
    segments_files = tuple(sorted(segments_dir.iterdir()))

    # turn the time into a path to a non-existant file
    ftime_string = "seg_%Y_%m_%d__%H_%M_%S.mp4"
    start_time_path = segments_dir / start_time.strftime(ftime_string)
    end_time_path = segments_dir / end_time.strftime(ftime_string)
    logger.info("Finding segments from %s to %s", start_time_path, end_time_path)

    # work out where in the list the start and end filenames would be inserted
    start_position = bisect.bisect_left(segments_files, start_time_path)
    end_position = bisect.bisect_right(segments_files, end_time_path)
    logger.info("Segment positions from %s to %s", start_position, end_position)

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
    fps: float
    hwaccel: str
    segment_duration: int
    transition_metrics: WatcherTransitionMetrics
    motion_mask: Path | None
    msg_queue: Queue
    stat: WatcherManagerStateEnum
    motion_process: Process | None = None
    segment_process: Popen | None = None

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
        fps: float,
        hwaccel: str,
        segment_duration: int,
        transition_metrics: WatcherTransitionMetrics,
        motion_mask: Path | None = None,
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
        self.fps = fps
        self.hwaccel = hwaccel
        self.segment_duration = segment_duration
        self.transition_metrics = transition_metrics
        self.motion_mask = motion_mask

        self.msg_queue = Queue()
        self.state = WatcherManagerStateEnum.WAITING

    def check_and_start_processes(self) -> None:
        # check and create motion process
        if self.motion_process and not self.motion_process.is_alive():
            logger.warning("Motion Process is dead")
            self.motion_process = None
        if not self.motion_process:
            logger.info("Creating Motion process...")
            self.motion_process = create_motion_process(
                rtsp_stream=self.rtsp_stream,
                msg_queue=self.msg_queue,
                threshold=self.threshold,
                kernel_size=self.kernel_size,
                scale=self.scale,
                fps=self.fps,
                hwaccel=self.hwaccel,
                transition_metrics=self.transition_metrics,
                motion_mask=self.motion_mask,
            )

        # check and create segment process
        if self.segment_process and self.segment_process.poll() is not None:
            logger.warning("Segmentation process is dead")
            self.segment_process = None
        if not self.segment_process:
            logger.info("Creating Segmentation process...")
            self.segment_process = create_segment_process(
                input_=self.rtsp_stream,
                output=self.segments_dir,
                duration=self.segment_duration,
            )

    def get_message(self) -> MotionWindow | None:
        self.check_and_start_processes()
        msg = None
        with contextlib.suppress(Empty):
            msg = self.msg_queue.get_nowait()
        return msg

    def combine_segments(self, motion_window: MotionWindow) -> None:
        start_time = motion_window.start_time - timedelta(seconds=self.offset_start)
        if motion_window.end_time is None:
            raise CannotCombineOpenWindowError()
        end_time = motion_window.end_time + timedelta(seconds=self.offset_end)

        while (to_merge := find_segments_for_timespan(start_time, end_time, self.segments_dir)) is None:
            # inner loop waiting for enough segments to be made
            self.check_and_start_processes()
            sleep(self.segment_duration)
        output_file = self.output_dir / start_time.strftime("out_%Y_%m_%d__%H_%M_%S.mp4")
        logger.info("Joining %s segments into %s", len(to_merge), output_file)
        concat_videos(to_merge, output_file)

        # also output a JSON summary
        with open(self.output_dir / start_time.strftime("out_%Y_%m_%d__%H_%M_%S.json"), "w") as json_out:
            json_out.write(motion_window.model_dump_json())

    def cleanup_old_segments(self, *, keep_override: int = -1) -> None:
        keep = keep_override if keep_override >= 0 else self.keep_count
        # list files in directory
        files = os.listdir(self.segments_dir)
        if len(files) > keep:
            # sort in order
            files = sorted(files, reverse=True)
            # identify the oldest ones
            files_to_remove = files[keep:]
            for file_to_remove in files_to_remove:
                logger.debug("removing %s", self.segments_dir / file_to_remove)
                os.unlink(self.segments_dir / file_to_remove)

    def run(self) -> None:
        # initially remove any old segments
        self.cleanup_old_segments(keep_override=0)

        try:
            while True:
                # check subprocesses while checking queue for a message
                msg = None
                msg = self.get_message()
                if msg:
                    if msg.end_time is None:
                        self.state = WatcherManagerStateEnum.RECORDING
                        logger.info("Starting recording")
                    else:
                        self.combine_segments(msg)
                        self.state = WatcherManagerStateEnum.WAITING
                elif self.state == WatcherManagerStateEnum.WAITING:
                    # waiting to record, cleanup old files
                    self.cleanup_old_segments()
                    # sleep till there's something else to do
                    sleep(1)
                else:
                    # nothing to do, sleep
                    sleep(1)
        finally:
            # cleanup any lingering processes
            if self.motion_process and self.motion_process.is_alive():
                self.motion_process.kill()
                self.motion_process = None
            if self.segment_process and self.segment_process.poll() is None:
                self.segment_process.kill()
                self.segment_process = None


@app.command()
@use_yaml_config()
def watch(
    rtsp_stream: Annotated[str, typer.Argument(metavar="RTSP_URL", envvar="WCT_RTSP")],
    segments: Annotated[Path, typer.Argument(metavar="PATH", envvar="WCT_SEGMENTS")],  # existing directory
    output: Annotated[Path, typer.Argument(metavar="PATH", envvar="WCT_OUTPUT")],  # existing directory
    keep_count: Annotated[
        int, typer.Option(metavar="INT", envvar="WCT_KEEP")
    ] = 4,  # no. segments # TODO calculate from offset_start and segment_duration
    offset_start: Annotated[float, typer.Option(metavar="FLOAT", envvar="WCT_OFFSET_START")] = 10.0,  # seconds
    offset_end: Annotated[float, typer.Option(metavar="FLOAT", envvar="WCT_OFFSET_END")] = 10.0,  # seconds
    history: Annotated[int, typer.Option(metavar="INT", envvar="WTC_HISTORY")] = 30,  # frames
    threshold: Annotated[int, typer.Option(metavar="INT", envvar="WTC_THRESHOLD")] = 16,  # < 128?
    kernel_size: Annotated[int, typer.Option(metavar="INT", envvar="WTC_KERNEL_SIZE")] = 3,  # pixels
    scale: Annotated[float, typer.Option(metavar="FLOAT", envvar="WTC_SCALE")] = 0.25,  # <1.0
    fps: Annotated[float, typer.Option(metavar="FLOAT", envvar="WTC_FPS")] = 5.0,  # >=1.0
    hwaccel: Annotated[
        str, typer.Option(metavar="STR", envvar="WTC_AWACCEL")
    ] = "",  # see https://trac.ffmpeg.org/wiki/HWAccelIntro
    segment_duration: Annotated[int, typer.Option(metavar="INT", envvar="WTC_SEG_DURATION")] = 15,  # seconds
    green_to_amber_motion_min: Annotated[float, typer.Option(metavar="FLOAT", envvar="WTC_GREEN_2_AMBER_MIN")] = 0.01,
    amber_to_green_proportion_max: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WTC_AMBER_2_GREEN_MAX")
    ] = 0.0075,
    amber_to_red_duration: Annotated[int, typer.Option(metavar="INT", envvar="WTC_AMBER_2_RED_DURATION")] = 5,  # frames
    red_to_red_amber_proportion_max: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WTC_RED_2_RED_AMBER_MAX")
    ] = 0.0075,
    red_amber_to_red_proportion_min: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WTC_RED_AMBER_2_RED_MIN")
    ] = 0.01,
    red_amber_to_green_duration: Annotated[
        int, typer.Option(metavar="INT", envvar="WTC_RED_AMBER_2_GREEN_DURATION")
    ] = 5,  # frames
    motion_mask: Annotated[Path | None, typer.Option(metavar="PATH", envvar="WTC_MOTION_MASK")] = None,
) -> None:
    if motion_mask:
        motion_mask = motion_mask.resolve()
        if not motion_mask.exists():
            raise MotionMaskNotExistsError(str(motion_mask))
        if not motion_mask.is_file():
            raise MotionMaskNotFileError(str(motion_mask))
        if not os.access(motion_mask, os.R_OK):
            raise MotionMaskNotReadableError(str(motion_mask))

    # TODO validate rtsp_stream is a rtsp url

    segments = segments.resolve()
    if not segments.is_dir() or not segments.exists():
        raise typer.BadParameter("segments must be an existing directory")

    output = output.resolve()
    if not output.is_dir() or not output.exists():
        raise typer.BadParameter("output must be an existing directory")
    transition_metrics = WatcherTransitionMetrics(
        preparing_duration=history,
        green_to_amber_motion_min=green_to_amber_motion_min,
        amber_to_green_proportion_max=amber_to_green_proportion_max,
        amber_to_red_duration=amber_to_red_duration,
        red_to_red_amber_proportion_max=red_to_red_amber_proportion_max,
        red_amber_to_red_proportion_min=red_amber_to_red_proportion_min,
        red_amber_to_green_duration=red_amber_to_green_duration,
    )
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
        fps=fps,
        hwaccel=hwaccel,
        segment_duration=segment_duration,
        transition_metrics=transition_metrics,
        motion_mask=motion_mask,
    )
    watcher.run()


if __name__ == "__main__":
    app()
