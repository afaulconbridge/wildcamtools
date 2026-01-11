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

from wildcamtools.lib.concat import concat_ffmpeg
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.states import MotionWindow, Watcher, WatcherStateEnum, WatcherTransitionMetrics
from wildcamtools.lib.stats import VideoStats, get_video_stats
from wildcamtools.lib.vidio import FrameSourceFFMPEG

app = typer.Typer()
logger = logging.getLogger(__name__)


class StorageWatcherStateEnum(StrEnum):
    WAITING = "WAITING"
    RECORDING = "RECORDING"


def enqueue_motion_windows(msgs: Path, queue: Queue) -> None:
    with open(msgs) as msg_input:
        for line in msg_input.readline():
            # NOTE assumes no new line characters inside the json object
            motion_window = MotionWindow.model_validate_json(line)
            queue.put(motion_window)


def get_message(queue: Queue, process: Process) -> MotionWindow | None:
    if not process.is_alive():
        raise RuntimeError("Process is dead")
    msg = None
    with contextlib.suppress(Empty):
        msg = queue.get_nowait()
    return msg


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
    logger.info(" ".join(str(s) for s in segments_files))

    # turn the time into a path to a non-existant file
    ftime_string = "seg_%Y_%m_%d__%H_%M_%S.mp4"
    start_time_path = segments_dir / start_time.strftime(ftime_string)
    end_time_path = segments_dir / end_time.strftime(ftime_string)
    logger.info(f"Finding segments from {start_time_path} to {end_time_path}")

    # work out where in the list the start and end filenames would be inserted
    start_position = bisect.bisect_left(segments_files, start_time_path)
    end_position = bisect.bisect_right(segments_files, end_time_path)
    logger.info(f"{start_position} - {end_position}")

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


def motion_states(rtsp_stream: str, queue: Queue) -> None:
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

    history = 10
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
    stats = get_video_stats(rtsp_stream)
    for motion in _find_motion_times(rtsp_stream, stats, watcher):
        queue.put(motion)


@app.command()
def watch(
    rtsp_stream: Annotated[str, typer.Argument(metavar="RTSP")],
    segments: Annotated[Path, typer.Argument(metavar="SEGMENTS")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT")],
    keep_count: int = 4,
) -> None:

    segments = segments.resolve()
    if not segments.is_dir() or not segments.exists():
        raise ValueError("segments must be an existing directory")

    output = output.resolve()
    if not output.is_dir() or not output.exists():
        raise ValueError("output must be an existing directory")

    msg_queue = Queue()
    motion_process = Process(target=motion_states, args=(rtsp_stream, msg_queue), daemon=True)
    motion_process.start()

    start_offset = timedelta(seconds=10)
    end_offset = timedelta(seconds=10)

    state = StorageWatcherStateEnum.WAITING
    while True:
        # check queue for a message
        msg = None
        msg = get_message(msg_queue, motion_process)
        if msg:
            if msg.end_time is None:
                state = StorageWatcherStateEnum.RECORDING
                logger.info("Starting recording")
            else:
                start_time = msg.start_time - start_offset
                end_time = msg.end_time + end_offset
                while (to_merge := find_segments_for_timespan(start_time, end_time, segments)) is None:
                    sleep(1)
                output_file = output / start_time.strftime("out_%Y_%m_%d__%H_%M_%S.mp4")
                logger.info(f"Joining {len(to_merge)} segments into {output_file}")
                concat_ffmpeg(to_merge, output_file)
                state = StorageWatcherStateEnum.WAITING
        elif state == StorageWatcherStateEnum.WAITING:
            # waiting to record, cleanup old files
            cleanup_old_segments(segments, keep_count)
        else:
            # nothing to do, sleep
            sleep(1)
