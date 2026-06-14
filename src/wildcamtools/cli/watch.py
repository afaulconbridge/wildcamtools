from __future__ import annotations

import contextlib
import logging
import os
from datetime import datetime
from enum import StrEnum
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from time import sleep
from typing import Annotated

import typer
from pydantic import BaseModel
from typer_config import use_yaml_config

from wildcamtools.lib.concat import concat_videos
from wildcamtools.lib.errors import (
    CannotCombineOpenWindowError,
    MotionMaskNotExistsError,
    MotionMaskNotFileError,
    MotionMaskNotReadableError,
)
from wildcamtools.lib.segment import _PyAVSegmentProcess, create_segment_process
from wildcamtools.lib.segment_metadata import SegmentMetadata
from wildcamtools.lib.states import (
    MotionProcessWrapper,
    MotionWindow,
    WatcherTransitionMetrics,
    create_motion_process,
)
from wildcamtools.lib.utils import is_stream_url

app = typer.Typer()
logger = logging.getLogger(__name__)


def find_segments_for_framerange(
    start_frame: int, end_frame: int, segments_dir: Path, is_segmenting: bool = True
) -> tuple[Path, ...] | None:
    """Find segment files that overlap with the given frame range.

    Args:
        start_frame: First frame number to include
        end_frame: Last frame number to include
        segments_dir: Directory containing segment files and metadata
        is_segmenting: If True, wait for all segments to be complete.
                      If False, accept the last segment even if incomplete.

    Returns:
        Tuple of segment file paths to merge, or None if not all segments are ready
    """
    logger.info("Finding segments for frames %d to %d", start_frame, end_frame)

    # Find all metadata files
    metadata_files = sorted(segments_dir.glob("*.meta.json"))

    if not metadata_files:
        logger.warning("No segment metadata files found in %s", segments_dir)
        return None

    # Load metadata and find overlapping segments
    matching_segments: list[Path] = []
    for metadata_path in metadata_files:
        metadata = SegmentMetadata.load(metadata_path)
        if metadata is None:
            logger.debug("Skipping invalid metadata file: %s", metadata_path)
            continue

        # Check if segment overlaps with requested frame range
        if metadata.end_frame < start_frame or metadata.start_frame > end_frame:
            continue

        # Get the corresponding segment file
        segment_path = SegmentMetadata.get_segment_path(metadata_path)
        if not segment_path.exists():
            logger.warning("Segment file missing: %s", segment_path)
            continue

        matching_segments.append(segment_path)

    if not matching_segments:
        logger.info("No segments found for frame range %d-%d", start_frame, end_frame)
        return ()

    # Check if the last segment might be incomplete (only if still segmenting)
    if is_segmenting:
        last_segment = matching_segments[-1]
        last_metadata_path = SegmentMetadata.get_metadata_path(last_segment)
        if last_metadata_path in metadata_files:
            last_metadata = SegmentMetadata.load(last_metadata_path)
            if last_metadata and last_metadata.end_frame < end_frame:
                # The last segment doesn't cover the full range, wait for more
                logger.info("Potentially incomplete segment range, waiting for more segments")
                return None

    logger.info("Found %d segments for frame range %d-%d", len(matching_segments), start_frame, end_frame)
    return tuple(sorted(matching_segments))


class WatcherManagerStateEnum(StrEnum):
    WAITING = "WAITING"
    RECORDING = "RECORDING"


class OutputClipMetadata(BaseModel):
    """Metadata for an output clip file.

    Attributes:
        start_frame: First frame number in the clip
        end_frame: Last frame number in the clip
        start_time: Start timestamp (for streams)
        end_time: End timestamp (for streams)
        motion_window: Original motion window that triggered the clip
    """

    start_frame: int
    end_frame: int
    start_time: datetime | None = None
    end_time: datetime | None = None
    motion_window: MotionWindow


class WatcherManager:
    rtsp_stream: str
    segments_dir: Path
    output_dir: Path
    keep_count: int
    offset_start: float
    offset_end: float
    history: int
    threshold: int
    kernel_size: float
    scale: float
    fps: float
    hwaccel: str
    segment_duration: int
    transition_metrics: WatcherTransitionMetrics
    motion_mask: Path | None
    msg_queue: Queue
    stat: WatcherManagerStateEnum
    motion_process: MotionProcessWrapper | None
    segment_process: _PyAVSegmentProcess | None
    _segment_process_completed: bool
    _motion_process_completed: bool
    _segment_restart_count: int
    _MAX_RESTART_ATTEMPTS: int = 3
    _is_stream: bool

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
        kernel_size: float,
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
        self._is_stream = is_stream_url(rtsp_stream)

        self.msg_queue = Queue()
        self.state = WatcherManagerStateEnum.WAITING
        self.motion_process = None
        self.segment_process = None
        self._segment_process_completed = False
        self._motion_process_completed = False
        self._segment_restart_count = 0

    def _handle_motion_process_restart(self) -> None:
        """Handle restarting the motion process.

        Called when motion process terminates and should_restart=True.
        Resets _motion_process_completed to False to allow future restarts.
        """
        self.motion_process = create_motion_process(
            rtsp_stream=self.rtsp_stream,
            msg_queue=self.msg_queue,
            history=self.history,
            threshold=self.threshold,
            kernel_size=self.kernel_size,
            scale=self.scale,
            fps=self.fps,
            hwaccel=self.hwaccel,
            transition_metrics=self.transition_metrics,
            motion_mask=self.motion_mask,
            restart_on_exit=None,  # Auto-detect
        )
        self._motion_process_completed = False

    def _handle_segment_process_restart(self) -> None:
        """Handle restarting the segment process with restart attempt limiting.

        Called when segment process terminates and should_restart=True.
        Resets _segment_process_completed to False to allow future restarts.
        Returns early if max restart attempts exceeded.
        """
        if self._segment_restart_count >= self._MAX_RESTART_ATTEMPTS:
            logger.error("Segmentation process exceeded max restart attempts (%d)", self._MAX_RESTART_ATTEMPTS)
            self.segment_process = None
            self._segment_process_completed = True
            return

        logger.warning(
            "Segmentation process terminated, restarting... (attempt %d/%d)",
            self._segment_restart_count + 1,
            self._MAX_RESTART_ATTEMPTS,
        )
        self.segment_process = create_segment_process(
            input_=self.rtsp_stream,
            output=self.segments_dir,
            duration=self.segment_duration,
        )
        self._segment_restart_count += 1
        self._segment_process_completed = False

    def check_and_start_processes(self) -> None:
        # check and create motion process
        if self.motion_process and not self.motion_process.is_alive():
            should_restart = getattr(self.motion_process, "restart_on_exit", True)

            if should_restart:
                logger.warning("Motion Process terminated, restarting...")
                self._handle_motion_process_restart()
            else:
                logger.info("Motion Process completed successfully")
                self._motion_process_completed = True
                self.motion_process = None
        elif self.motion_process is None and not self._motion_process_completed:
            self.motion_process = create_motion_process(
                rtsp_stream=self.rtsp_stream,
                msg_queue=self.msg_queue,
                history=self.history,
                threshold=self.threshold,
                kernel_size=self.kernel_size,
                scale=self.scale,
                fps=self.fps,
                hwaccel=self.hwaccel,
                transition_metrics=self.transition_metrics,
                motion_mask=self.motion_mask,
                restart_on_exit=None,  # Auto-detect
            )

        # check and create segment process
        if self.segment_process and self.segment_process.poll() is not None:
            should_restart = self.segment_process.restart_on_exit

            if self.segment_process.returncode != 0:
                logger.error(
                    "Segmentation process terminated with error (returncode %d)", self.segment_process.returncode
                )
                should_restart = False

            if should_restart:
                self._handle_segment_process_restart()
            else:
                if self.segment_process.returncode != 0:
                    logger.info("Segmentation process terminated (returncode %d)", self.segment_process.returncode)
                else:
                    logger.info("Segmentation process completed successfully")
                self._segment_restart_count = 0
                self._segment_process_completed = True

        # Create segment process if not running and hasn't completed yet
        if self.segment_process is None and not self._segment_process_completed:
            logger.info("Creating Segmentation process...")
            self.segment_process = create_segment_process(
                input_=self.rtsp_stream,
                output=self.segments_dir,
                duration=self.segment_duration,
                # Auto-detect: RTSP URL will default to restart_on_exit=True
            )

    def get_message(self) -> MotionWindow | None:
        self.check_and_start_processes()
        msg = None
        with contextlib.suppress(Empty):
            msg = self.msg_queue.get_nowait()
        return msg

    def _seconds_to_frames(self, seconds: float, fps: float) -> int:
        """Convert seconds to frame count."""
        return int(seconds * fps)

    def combine_segments(self, motion_window: MotionWindow) -> None:
        if motion_window.end_frame is None:
            raise CannotCombineOpenWindowError()

        # Get FPS from the first available segment metadata or use default
        fps = self._get_fps_from_segments()
        if fps is None:
            logger.warning("Could not determine FPS, using default 30.0")
            fps = 30.0

        # Convert offsets from seconds to frames for segment selection
        offset_start_frames = self._seconds_to_frames(self.offset_start, fps)
        offset_end_frames = self._seconds_to_frames(self.offset_end, fps)

        # Use offset-adjusted frames for selecting segments to merge
        segment_start_frame = max(0, motion_window.start_frame - offset_start_frames)
        segment_end_frame = motion_window.end_frame + offset_end_frames

        max_wait_time = self.segment_duration * 10  # Wait for at most 10 segment durations
        wait_time = 0

        while (
            to_merge := find_segments_for_framerange(
                segment_start_frame,
                segment_end_frame,
                self.segments_dir,
                is_segmenting=not self._segment_process_completed,
            )
        ) is None:
            # inner loop waiting for enough segments to be made
            self.check_and_start_processes()
            sleep(self.segment_duration)
            wait_time += self.segment_duration

            if wait_time >= max_wait_time:
                logger.error("Timeout waiting for segments (waited %ds)", wait_time)
                return  # Exit early to prevent infinite loop

        if not to_merge:
            logger.warning("No segments found for frame range %d-%d", segment_start_frame, segment_end_frame)
            return

        # Generate output filename based on input type using actual motion window frames
        if self._is_stream:
            # Timestamp-based naming for streams
            timestamp = motion_window.start_time
            output_base = f"out_{timestamp:%Y_%m_%d__%H_%M_%S}"
            output_metadata = OutputClipMetadata(
                start_frame=motion_window.start_frame,
                end_frame=motion_window.end_frame,
                start_time=motion_window.start_time,
                end_time=motion_window.end_time,
                motion_window=motion_window,
            )
        else:
            # Frame-based naming for files (use actual motion window frames, not offset-adjusted)
            output_base = f"out_frame{motion_window.start_frame:06d}_{motion_window.end_frame:06d}"
            output_metadata = OutputClipMetadata(
                start_frame=motion_window.start_frame,
                end_frame=motion_window.end_frame,
                motion_window=motion_window,
            )

        output_file = self.output_dir / f"{output_base}.mp4"
        logger.info("Joining %s segments into %s", len(to_merge), output_file)
        concat_videos(to_merge, output_file)

        # also output a JSON summary
        with open(self.output_dir / f"{output_base}.json", "w") as json_out:
            json_out.write(output_metadata.model_dump_json(indent=2))

    def _get_fps_from_segments(self) -> float | None:
        """Get FPS from the first available segment metadata file."""
        metadata_files = sorted(self.segments_dir.glob("*.meta.json"))
        for metadata_path in metadata_files:
            metadata = SegmentMetadata.load(metadata_path)
            if metadata is not None:
                return metadata.fps
        return None

    def _all_processes_completed(self) -> bool:
        """Check if all processes have completed (for file inputs)."""
        return self._motion_process_completed and self._segment_process_completed

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
                # check if all processes have completed (file input)
                if self._all_processes_completed():
                    logger.info("All processes completed, exiting...")
                    break

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
    history: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WCT_HISTORY")
    ] = 10.0,  # seconds; controls both the MOG2 background model frame count (history * fps) and the state machine's preparing_duration
    threshold: Annotated[int, typer.Option(metavar="INT", envvar="WCT_THRESHOLD")] = 16,  # < 128?
    kernel_size: Annotated[float, typer.Option(metavar="FLOAT", envvar="WCT_KERNEL_SIZE")] = 0.005,  # proportion
    scale: Annotated[float, typer.Option(metavar="FLOAT", envvar="WCT_SCALE")] = 0.25,  # <1.0
    fps: Annotated[float, typer.Option(metavar="FLOAT", envvar="WCT_FPS")] = 5.0,  # >=1.0
    hwaccel: Annotated[
        str, typer.Option(metavar="STR", envvar="WCT_HWACCEL")
    ] = "",  # see https://trac.ffmpeg.org/wiki/HWAccelIntro
    segment_duration: Annotated[int, typer.Option(metavar="INT", envvar="WCT_SEG_DURATION")] = 15,  # seconds
    green_to_amber_motion_min: Annotated[float, typer.Option(metavar="FLOAT", envvar="WCT_GREEN_2_AMBER_MIN")] = 0.01,
    amber_to_green_proportion_max: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WCT_AMBER_2_GREEN_MAX")
    ] = 0.0075,
    amber_to_red_duration: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WCT_AMBER_2_RED_DURATION")
    ] = 5.0,  # seconds
    red_to_red_amber_proportion_max: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WCT_RED_2_RED_AMBER_MAX")
    ] = 0.0075,
    red_amber_to_red_proportion_min: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WCT_RED_AMBER_2_RED_MIN")
    ] = 0.01,
    red_amber_to_green_duration: Annotated[
        float, typer.Option(metavar="FLOAT", envvar="WCT_RED_AMBER_2_GREEN_DURATION")
    ] = 5.0,  # seconds
    motion_mask: Annotated[Path | None, typer.Option(metavar="PATH", envvar="WCT_MOTION_MASK")] = None,
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
    if kernel_size < 0 or kernel_size > 1:
        raise typer.BadParameter("kernel_size must be a float proportion between 0 and 1")

    # The MOG2 background subtractor's ``history`` parameter is in number of
    # frames. Convert the state machine's seconds-based warm-up into frames
    # at the target FPS so the background model has roughly the same amount
    # of training data.
    mog_history = max(1, round(history * fps))

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
        history=mog_history,
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
