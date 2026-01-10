import bisect
import contextlib
import logging
import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from time import sleep
from typing import Annotated

import typer

from wildcamtools.lib.concat import concat_ffmpeg
from wildcamtools.lib.states import MotionWindow

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


def get_message(queue: Queue, thread: Thread) -> MotionWindow | None:
    if not thread.is_alive():
        raise RuntimeError("Thread is dead")
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


def find_segments_for_timespan(start_time: datetime, end_time: datetime, segments_dir: Path) -> tuple[Path, ...]:
    # given a directory

    # build a list of files in segment directory
    segments_files = tuple(sorted(segments_dir.iterdir()))

    # turn the time into a path to a non-existant file
    ftime_string = "seg_%Y_%m_%d__%H_%M_%S.mp4"
    start_time_path = segments_dir / start_time.strftime(ftime_string)
    end_time_path = segments_dir / end_time.strftime(ftime_string)

    # work out where in the list the start and end filenames would be inserted
    start_position = bisect.bisect_left(segments_files, start_time_path)
    end_position = bisect.bisect_right(segments_files, end_time_path)

    # files that should be merged
    # TODO also offsets to trim
    return segments_files[max(start_position - 1, 0) : end_position]


@app.command()
def watch(
    msgs: Annotated[Path, typer.Argument(metavar="MSGS")],
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

    # need a background thread that reads from the FIFO named pipe constantly
    # push messages from there into a queue for main thread handling
    msg_queue = Queue()
    msg_thread = Thread(target=enqueue_motion_windows, args=(msgs, msg_queue), daemon=True)
    msg_thread.start()

    state = StorageWatcherStateEnum.WAITING
    while True:
        # check queue for a message
        msg = None
        msg = get_message(msg_queue, msg_thread)
        if msg:
            if msg.end_time is None:
                state = StorageWatcherStateEnum.RECORDING
                logger.info("Starting recording")
            else:
                to_merge = find_segments_for_timespan(msg.start_time, msg.end_time, segments)
                output_file = output / msg.start_time.strftime("out_%Y_%m_%d__%H_%M_%S.mp4")
                concat_ffmpeg(to_merge, output_file)
                logger.info(f"Joined {len(to_merge)} segments into {output_file}")
                state = StorageWatcherStateEnum.WAITING
        elif state == StorageWatcherStateEnum.WAITING:
            # waiting to record, cleanup old files
            cleanup_old_segments(segments, keep_count)
        else:
            # nothing to do, sleep
            sleep(0.1)
