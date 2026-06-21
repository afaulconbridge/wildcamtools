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

from wildcamtools.lib.concat import SegmentInfo, concat_videos
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
    create_motion_process,
)
from wildcamtools.lib.utils import is_stream_url
from wildcamtools.lib.watch_config import WatchConfig

app = typer.Typer()
logger = logging.getLogger(__name__)


def find_segments_for_framerange(
    start_frame: int,
    end_frame: int,
    segments_dir: Path,
    is_segmenting: bool = True,
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


class ClipMetadata(BaseModel):
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


class OutputClipMetadata(BaseModel):
    """Output metadata containing clip details and configuration.

    Attributes:
        clip: Clip-specific metadata
        config: Configuration used for motion detection (for traceability)

    """

    clip: ClipMetadata
    config: WatchConfig


class WatcherManager:
    config: WatchConfig
    segments_dir: Path
    output_dir: Path
    hwaccel: str
    debug_video_path: Path | None
    msg_queue: Queue
    state: WatcherManagerStateEnum
    motion_process: MotionProcessWrapper | None
    segment_process: _PyAVSegmentProcess | None
    _segment_process_completed: bool
    _motion_process_completed: bool
    _segment_restart_count: int
    _MAX_RESTART_ATTEMPTS: int = 3
    _is_stream: bool

    def __init__(
        self,
        config: WatchConfig,
        segments_dir: Path,
        output_dir: Path,
        hwaccel: str = "",
        debug_video_path: Path | None = None,
    ) -> None:
        self.config = config
        self.segments_dir = segments_dir
        self.output_dir = output_dir
        self.hwaccel = hwaccel
        self.debug_video_path = debug_video_path
        self._is_stream = is_stream_url(config.rtsp_stream)

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
            rtsp_stream=self.config.rtsp_stream,
            msg_queue=self.msg_queue,
            config=self.config,
            restart_on_exit=None,  # Auto-detect
            hwaccel=self.hwaccel,
            debug_video_path=self.debug_video_path,
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
            input_=self.config.rtsp_stream,
            output=self.segments_dir,
            duration=self.config.motion_detection.segment_duration,
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
                rtsp_stream=self.config.rtsp_stream,
                msg_queue=self.msg_queue,
                config=self.config,
                restart_on_exit=None,  # Auto-detect
                hwaccel=self.hwaccel,
                debug_video_path=self.debug_video_path,
            )

        # check and create segment process
        if self.segment_process and self.segment_process.poll() is not None:
            should_restart = self.segment_process.restart_on_exit

            if self.segment_process.returncode != 0:
                logger.error(
                    "Segmentation process terminated with error (returncode %d)",
                    self.segment_process.returncode,
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
                input_=self.config.rtsp_stream,
                output=self.segments_dir,
                duration=self.config.motion_detection.segment_duration,
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

    def _build_segment_infos(self, to_merge: tuple[Path, ...]) -> list[SegmentInfo]:
        """Build segment metadata for accurate trim calculation."""
        segment_infos: list[SegmentInfo] = []
        for seg_path in to_merge:
            meta_path = SegmentMetadata.get_metadata_path(seg_path)
            meta = SegmentMetadata.load(meta_path)
            if meta:
                segment_infos.append(
                    SegmentInfo(
                        path=seg_path,
                        start_frame=meta.start_frame,
                        end_frame=meta.end_frame,
                        fps=meta.fps,
                        duration=meta.duration,
                        actual_frames=meta.actual_frames,
                    )
                )
        return segment_infos

    def combine_segments(self, motion_window: MotionWindow) -> None:
        if motion_window.end_frame is None:
            raise CannotCombineOpenWindowError()

        # Use source FPS from motion window for offset calculations
        # This ensures offsets are calculated in source video frame indices
        fps = motion_window.source_fps

        # Convert offsets from seconds to frames for segment selection
        offset_start_frames = self._seconds_to_frames(self.config.offset_start, fps)
        offset_end_frames = self._seconds_to_frames(self.config.offset_end, fps)

        # Use offset-adjusted frames for selecting segments to merge
        segment_start_frame = max(0, motion_window.start_frame - offset_start_frames)
        segment_end_frame = motion_window.end_frame + offset_end_frames

        max_wait_time = self.config.motion_detection.segment_duration * 10
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
            sleep(self.config.motion_detection.segment_duration)
            wait_time += self.config.motion_detection.segment_duration

            if wait_time >= max_wait_time:
                logger.error("Timeout waiting for segments (waited %ds)", wait_time)
                return  # Exit early to prevent infinite loop

        if not to_merge:
            logger.warning("No segments found for frame range %d-%d", segment_start_frame, segment_end_frame)
            self.state = WatcherManagerStateEnum.WAITING
            return

        # Generate output filename based on input type using actual motion window frames
        if self._is_stream:
            # Timestamp-based naming for streams
            timestamp = motion_window.start_time
            output_base = f"out_{timestamp:%Y_%m_%d__%H_%M_%S}"
            clip_metadata = ClipMetadata(
                start_frame=motion_window.start_frame,
                end_frame=motion_window.end_frame,
                start_time=motion_window.start_time,
                end_time=motion_window.end_time,
                motion_window=motion_window,
            )
        else:
            # Frame-based naming for files (use actual motion window frames, not offset-adjusted)
            output_base = f"out_frame{motion_window.start_frame:06d}_{motion_window.end_frame:06d}"
            clip_metadata = ClipMetadata(
                start_frame=motion_window.start_frame,
                end_frame=motion_window.end_frame,
                motion_window=motion_window,
            )

        output_metadata = OutputClipMetadata(
            clip=clip_metadata,
            config=self.config,
        )

        output_file = self.output_dir / f"{output_base}.mp4"
        logger.info(
            "Joining %s segments into %s (trimming to frames %d-%d)",
            len(to_merge),
            output_file,
            segment_start_frame,
            segment_end_frame,
        )

        # Build segment metadata for accurate trim calculation
        segment_infos = self._build_segment_infos(to_merge)

        concat_videos(
            to_merge,
            output_file,
            trim_start_frame=segment_start_frame,
            trim_end_frame=segment_end_frame,
            source_fps=fps,
            segment_metadata=segment_infos,
        )

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
        keep = keep_override if keep_override >= 0 else self.config.keep_count
        try:
            files = os.listdir(self.segments_dir)
        except OSError as e:
            logger.warning("Failed to list segments directory: %s", e)
            return

        if len(files) > keep:
            files = sorted(files, reverse=True)
            files_to_remove = files[keep:]
            for file_to_remove in files_to_remove:
                try:
                    logger.debug("removing %s", self.segments_dir / file_to_remove)
                    os.unlink(self.segments_dir / file_to_remove)
                except OSError as e:
                    logger.warning("Failed to remove segment file %s: %s", self.segments_dir / file_to_remove, e)

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
                try:
                    self.motion_process.kill()
                except Exception as e:
                    logger.warning("Failed to kill motion process: %s", e)
                self.motion_process = None
            if self.segment_process and self.segment_process.poll() is None:
                try:
                    self.segment_process.kill()
                except Exception as e:
                    logger.warning("Failed to kill segment process: %s", e)
                self.segment_process = None


@app.command("generate-config")
def generate_config_cmd(
    output: Annotated[Path | None, typer.Argument(metavar="OUTPUT")] = None,
) -> None:
    """Generate an example configuration file.

    Creates a JSON configuration file with default values that can be customized.
    The rtsp_stream field will use an environment variable reference.

    Args:
        output: Output file path (defaults to watch_config.json in current directory)

    """
    if output is None:
        output = Path("watch_config.json")

    config = WatchConfig(rtsp_stream="${WCT_RTSP}")
    config.to_json(output)
    typer.secho(f"Generated example configuration: {output}")
    typer.secho("Edit this file and set WCT_RTSP environment variable before running.")


@app.command()
def watch(  # noqa: C901
    config: Annotated[Path, typer.Argument(metavar="CONFIG", help="Path to JSON configuration file")],
    segments: Annotated[Path, typer.Argument(metavar="SEGMENTS", help="Directory for segment files")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT", help="Directory for output clips")],
    hwaccel: Annotated[
        str,
        typer.Option(metavar="STR", envvar="WCT_HWACCEL", help="Hardware acceleration method"),
    ] = "",
    debug_video: Annotated[
        Path | None,
        typer.Option(
            metavar="PATH",
            envvar="WCT_DEBUG_VIDEO",
            help="Path to write debug video output (1:1 frames with overlays)",
        ),
    ] = None,
) -> None:
    """Watch a video stream and generate motion-detected clips.

    Loads configuration from a JSON file and uses command-line arguments
    for deployment-specific paths (segments directory, output directory)
    and hardware acceleration settings.

    Args:
        config: Path to JSON configuration file
        segments: Directory for segment files (must exist)
        output: Directory for output clips (must exist)
        hwaccel: Hardware acceleration method (see https://trac.ffmpeg.org/wiki/HWAccelIntro)
        debug_video: Optional path to write debug video output

    """
    if not config.exists():
        typer.secho(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(code=1)
    if not config.is_file():
        typer.secho(f"Error: Config path is not a file: {config}", err=True)
        raise typer.Exit(code=1)

    segments = segments.resolve()
    if not segments.is_dir() or not segments.exists():
        typer.secho(f"Error: Segments directory does not exist: {segments}", err=True)
        raise typer.Exit(code=1)

    output = output.resolve()
    if not output.is_dir() or not output.exists():
        typer.secho(f"Error: Output directory does not exist: {output}", err=True)
        raise typer.Exit(code=1)

    if debug_video:
        debug_video = debug_video.resolve()
        if not debug_video.parent.exists():
            typer.secho(f"Error: Debug video parent directory does not exist: {debug_video.parent}", err=True)
            raise typer.Exit(code=1)

    logger.info("Loading config from %s", config)
    watch_config = WatchConfig.from_json(config)

    if watch_config.motion_mask:
        motion_mask = watch_config.motion_mask.resolve()
        if not motion_mask.exists():
            raise MotionMaskNotExistsError(str(motion_mask))
        if not motion_mask.is_file():
            raise MotionMaskNotFileError(str(motion_mask))
        if not os.access(motion_mask, os.R_OK):
            raise MotionMaskNotReadableError(str(motion_mask))

    logger.info("Starting watcher for %s", watch_config.rtsp_stream)
    watcher = WatcherManager(
        config=watch_config,
        segments_dir=segments,
        output_dir=output,
        hwaccel=hwaccel,
        debug_video_path=debug_video,
    )
    watcher.run()


if __name__ == "__main__":
    app()
