import logging
from collections.abc import Generator
from datetime import UTC, datetime
from enum import StrEnum
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Self

import cv2
import numpy as np
from pydantic import BaseModel

from wildcamtools.lib import Frame, FrameHandler
from wildcamtools.lib.frames import Rescaler
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.stats import VideoStats, get_video_stats
from wildcamtools.lib.utils import is_stream_url
from wildcamtools.lib.vidio import VideoReader
from wildcamtools.lib.watch_config import WatchConfig

logger = logging.getLogger(__name__)


class MotionProcessWrapper:
    """Typed wrapper for multiprocessing Process with restart_on_exit attribute."""

    restart_on_exit: bool

    def __init__(self, process: Process, restart_on_exit: bool) -> None:
        self._process = process
        self.restart_on_exit = restart_on_exit

    @property
    def pid(self) -> int | None:
        return self._process.pid

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def start(self) -> None:
        self._process.start()

    def terminate(self) -> None:
        self._process.terminate()

    def join(self, timeout: float | None = None) -> None:
        self._process.join(timeout=timeout)

    def kill(self) -> None:
        self._process.terminate()


class WatcherStateEnum(StrEnum):
    PREPARING = "PREPARING"
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"
    RED_AMBER = "RED_AMBER"
    DISABLED = "DISABLED"


class WatcherTransitionMetrics(BaseModel):
    # All duration fields are in seconds (wall-clock time of the frame).
    preparing_duration: float = 10.0
    green_to_amber_motion_min: float = 0.1
    amber_to_green_proportion_max: float = 0.075
    amber_to_red_duration: float = 1.0
    red_to_red_amber_proportion_max: float = 0.075
    red_amber_to_red_proportion_min: float = 0.1
    red_amber_to_green_duration: float = 1.0


class StateTransitionWindowMetrics(BaseModel):
    minimum: float
    maximum: float
    mean: float
    count: int

    def update(self, value: float) -> Self:
        self.count += 1
        self.mean = self.mean + ((value - self.mean) / self.count)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        return self


class MotionWindow(BaseModel):
    """
    Represents a motion detection window with frame and time boundaries.

    Note: ``start_frame``/``end_frame`` are in **source video indices** (native
    FPS, e.g., 30 fps), not post-filtering indices. When using Rescaler with
    ``fps < native_fps``, ``frame_no`` values will be spaced (e.g., 0, 6, 12,
    ... for 30fps source at 5fps). ``start_time``/``end_time`` are wall-clock
    timestamps captured from the frame's PTS.

    The state machine itself interprets duration-based thresholds
    (``preparing_duration``, ``amber_to_red_duration``,
    ``red_amber_to_green_duration``) in **seconds**, using the frame's
    timestamp — not frame counts. So the behavior is independent of the
    source/rescaled FPS configuration.

    Attributes:
        start_frame: First frame number in the motion window
        start_time: Start timestamp of the motion window
        end_frame: Last frame number in the motion window (None if still active)
        end_time: End timestamp of the motion window (None if still active)
        transition_metrics: State machine transition thresholds used
        transition_window_metrics: Per-state motion statistics during the window
        config: Configuration used for motion detection (for traceability)
    """

    start_frame: int
    start_time: datetime
    end_frame: int | None
    end_time: datetime | None
    transition_metrics: WatcherTransitionMetrics
    transition_window_metrics: dict[WatcherStateEnum, StateTransitionWindowMetrics]
    config: WatchConfig


class Watcher(FrameHandler):
    """
    ```mermaid
    stateDiagram-v2
        [*] --> preparing %% initializing background history
        preparing --> green %% fully initialized and ready
        green --> amber %% some motion detected, check if it is of sufficient duration
        amber --> green %% motion not of sufficient duration, go back to ready
        amber --> red %% motion of sufficient duration, trigger record start (with lookback)
        red --> red_amber %% motion stopped, prepare to stop recording
        red_amber --> red %% motion detected again, continue recording
        red_amber --> green %% no further motion detected, trigger record end
        green --> disabled
        disabled --> green
        [*]
    ```
    """

    motion: MogMotion
    state: WatcherStateEnum
    transition_metrics: WatcherTransitionMetrics
    transition_window_metrics: dict[WatcherStateEnum, StateTransitionWindowMetrics]
    # Timestamps (in seconds, from Frame.timestamp) of when each motion state
    # was entered. Used for time-based duration calculations in the state
    # machine. The state machine interprets ``amber_to_red_duration`` and
    # ``red_amber_to_green_duration`` as seconds, not frame counts, so its
    # behavior is independent of the source/rescaled FPS.
    preparing_start: float | None
    amber_start: float | None
    red_start: float | None
    red_amber_start: float | None

    def __init__(self, motion: MogMotion, transition_metrics: WatcherTransitionMetrics) -> None:
        self.motion = motion
        self.state = WatcherStateEnum.PREPARING
        self.transition_metrics = transition_metrics
        self.transition_window_metrics = {}
        self.preparing_start = None
        self.amber_start = None
        self.red_start = None
        self.red_amber_start = None

    def handle(self, frame: Frame) -> Frame:
        output = self.motion.handle(frame)

        next_state = self._get_next_state(output)
        self._update_state_transition_window_metrics(frame=output, next_state=next_state)

        self.state = next_state
        return output

    def _update_state_transition_window_metrics(self, frame: Frame, next_state: WatcherStateEnum) -> None:
        if next_state not in self.transition_window_metrics:
            self.transition_window_metrics[next_state] = StateTransitionWindowMetrics(
                minimum=frame.motion_proportion,
                maximum=frame.motion_proportion,
                mean=frame.motion_proportion,
                count=1,
            )
        else:
            self.transition_window_metrics[next_state].update(frame.motion_proportion)

    def _get_next_state(self, frame: Frame) -> WatcherStateEnum:
        logger.debug("Frame no: %s", frame.frame_no)
        logger.debug("Motion proportion: %s", frame.motion_proportion)
        match self.state:
            case WatcherStateEnum.PREPARING:
                return self._handle_preparing_state(frame)
            case WatcherStateEnum.GREEN:
                return self._handle_green_state(frame)
            case WatcherStateEnum.AMBER:
                return self._handle_amber_state(frame)
            case WatcherStateEnum.RED:
                return self._handle_red_state(frame)
            case WatcherStateEnum.RED_AMBER:
                return self._handle_red_amber_state(frame)
            case WatcherStateEnum.DISABLED:
                pass
        return self.state

    def _handle_preparing_state(self, frame: Frame) -> WatcherStateEnum:
        """Handle transition from PREPARING state."""
        if self.preparing_start is None and frame.timestamp is not None:
            self.preparing_start = frame.timestamp
        if (
            self.preparing_start is not None
            and frame.timestamp is not None
            and frame.timestamp - self.preparing_start >= self.transition_metrics.preparing_duration
        ):
            return WatcherStateEnum.GREEN
        return WatcherStateEnum.PREPARING

    def _handle_green_state(self, frame: Frame) -> WatcherStateEnum:
        """Handle transition from GREEN state."""
        if frame.motion_proportion > self.transition_metrics.green_to_amber_motion_min:
            if self.transition_metrics.amber_to_red_duration == 0:
                return WatcherStateEnum.RED
            self.amber_start = frame.timestamp
            return WatcherStateEnum.AMBER
        return WatcherStateEnum.GREEN

    def _handle_amber_state(self, frame: Frame) -> WatcherStateEnum:
        """Handle transition from AMBER state."""
        if frame.motion_proportion < self.transition_metrics.amber_to_green_proportion_max:
            return WatcherStateEnum.GREEN
        if (
            self.amber_start is not None
            and frame.timestamp is not None
            and frame.timestamp - self.amber_start >= self.transition_metrics.amber_to_red_duration
        ):
            self.red_start = frame.timestamp
            return WatcherStateEnum.RED
        return WatcherStateEnum.AMBER

    def _handle_red_state(self, frame: Frame) -> WatcherStateEnum:
        """Handle transition from RED state."""
        if frame.motion_proportion < self.transition_metrics.red_to_red_amber_proportion_max:
            if self.transition_metrics.red_amber_to_green_duration == 0:
                return WatcherStateEnum.GREEN
            self.red_amber_start = frame.timestamp
            return WatcherStateEnum.RED_AMBER
        return WatcherStateEnum.RED

    def _handle_red_amber_state(self, frame: Frame) -> WatcherStateEnum:
        """Handle transition from RED_AMBER state."""
        if frame.motion_proportion > self.transition_metrics.red_amber_to_red_proportion_min:
            self.red_start = frame.timestamp
            return WatcherStateEnum.RED
        if (
            self.red_amber_start is not None
            and frame.timestamp is not None
            and frame.timestamp - self.red_amber_start >= self.transition_metrics.red_amber_to_green_duration
        ):
            return WatcherStateEnum.GREEN
        return WatcherStateEnum.RED_AMBER


def _load_and_resize_mask(mask_path: Path | None, width: int, height: int, scale: float) -> np.ndarray | None:
    if mask_path is None:
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.error(
            "Failed to load motion mask from %s. The file may be corrupted or in an unsupported image format.",
            mask_path,
        )
        return None

    # Target size is the scaled dimensions of the frame
    target_width = int(width * scale)
    target_height = int(height * scale)

    # Aspect ratio check
    mask_h, mask_w = mask.shape[:2]
    stream_aspect = width / height
    mask_aspect = mask_w / mask_h

    if abs(stream_aspect - mask_aspect) > 0.05:
        logger.warning(
            "Motion mask aspect ratio (%.2f) differs significantly from stream aspect ratio (%.2f). The mask will be stretched to fit.",
            mask_aspect,
            stream_aspect,
        )

    # Resize using nearest neighbor to keep binary mask properties
    mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return mask


def _should_yield_motion_window(watcher: Watcher) -> bool:
    """Check if a motion window should be yielded (must have reached RED state)."""
    red_metrics = watcher.transition_window_metrics.get(WatcherStateEnum.RED)
    return red_metrics is not None and red_metrics.count > 0


def enqueue_motion_windows(  # noqa: C901
    rtsp_stream: str,
    queue: Queue,
    config: WatchConfig,
    hwaccel: str = "",
) -> None:
    """
    Extract motion windows from video stream.

    ``start_frame``/``end_frame`` on the returned ``MotionWindow`` objects use
    source video indices (native FPS), not post-filtering indices. When using
    Rescaler with ``fps < native_fps``, ``frame_no`` values will be spaced
    (e.g., 0, 6, 12, ... for 30fps source at 5fps).

    The Watcher state machine interprets duration-based thresholds
    (``preparing_duration``, ``amber_to_red_duration``,
    ``red_amber_to_green_duration``) in seconds (using ``Frame.timestamp``),
    so the behavior is independent of source/rescaled FPS.

    Args:
        rtsp_stream: RTSP URL or file path to process
        queue: Queue to put MotionWindow instances into
        config: WatchConfig with motion detection parameters
        hwaccel: Hardware acceleration method (deployment-specific)
    """

    def _find_motion_times(source: str, stats: VideoStats, watcher: Watcher) -> Generator[MotionWindow]:
        start_frame: int | None = None
        start_time: datetime | None = None
        prev_state: WatcherStateEnum | None = None
        last_frame_no: int | None = None
        logger.debug(
            "Reading %s at %sx%s@%s (%s)",
            source,
            stats.x,
            stats.y,
            config.motion_detection.fps,
            config.motion_detection.scale,
        )
        rescaler = Rescaler(
            stats,
            int(stats.x * config.motion_detection.scale),
            int(stats.y * config.motion_detection.scale),
            config.motion_detection.fps,
        )
        with VideoReader(source, hwaccel=hwaccel if hwaccel else None) as video_input:
            for frame in video_input:
                frame = rescaler.handle(frame)
                if not frame.filter_keep:
                    continue
                # If a new motion window is about to start (GREEN -> AMBER/RED),
                # reset the watcher's per-window metrics before processing the
                # entry frame. This scopes the metrics to just the new window
                # and gives exact min/max/mean/count (no lifetime accumulation
                # or approximation).
                if start_frame is None and start_time is None and prev_state == WatcherStateEnum.GREEN:
                    watcher.transition_window_metrics = {}
                frame = watcher.handle(frame)
                last_frame_no = frame.frame_no
                # Start tracking when transitioning from GREEN to any motion state (AMBER or RED)
                # This handles both normal transitions (GREEN->AMBER) and zero-duration transitions (GREEN->RED)
                if start_frame is None and start_time is None:
                    if watcher.state in (
                        WatcherStateEnum.AMBER,
                        WatcherStateEnum.RED,
                    ):
                        start_frame = frame.frame_no
                        start_time = datetime.now(UTC)
                # End tracking when returning to GREEN from any motion state
                # Only yield windows that reached RED state (discard GREEN->AMBER->GREEN)
                elif (
                    start_frame is not None
                    and start_time is not None
                    and watcher.state == WatcherStateEnum.GREEN
                    and prev_state
                    in (
                        WatcherStateEnum.AMBER,
                        WatcherStateEnum.RED,
                        WatcherStateEnum.RED_AMBER,
                    )
                ):
                    if not _should_yield_motion_window(watcher):
                        logger.debug(
                            "Discarding motion window %d-%d: never reached RED state",
                            start_frame,
                            last_frame_no,
                        )
                        start_frame = None
                        start_time = None
                        continue
                    end_frame = frame.frame_no
                    end_time = datetime.now(UTC)
                    yield MotionWindow(
                        start_frame=start_frame,
                        start_time=start_time,
                        end_frame=end_frame,
                        end_time=end_time,
                        transition_metrics=watcher.transition_metrics,
                        transition_window_metrics=dict(watcher.transition_window_metrics),
                        config=config,
                    )
                    start_frame = None
                    start_time = None

                prev_state = watcher.state

            # Yield any pending window if video ended during motion
            # Only yield if the window reached RED state
            if start_frame is not None and start_time is not None and last_frame_no is not None:
                if _should_yield_motion_window(watcher):
                    yield MotionWindow(
                        start_frame=start_frame,
                        start_time=start_time,
                        end_frame=last_frame_no,
                        end_time=datetime.now(UTC),
                        transition_metrics=watcher.transition_metrics,
                        transition_window_metrics=dict(watcher.transition_window_metrics),
                        config=config,
                    )
                else:
                    logger.debug(
                        "Discarding pending motion window %d-%d: never reached RED state",
                        start_frame,
                        last_frame_no,
                    )

    stats = get_video_stats(rtsp_stream)
    processed_mask = _load_and_resize_mask(
        mask_path=config.motion_mask,
        width=stats.x,
        height=stats.y,
        scale=config.motion_detection.scale,
    )

    watcher = Watcher(
        motion=MogMotion(
            history=config.get_mog_history(),
            threshold=config.motion_detection.threshold,
            detect_shadows=False,
            kernel_size=config.motion_detection.kernel_size,
            motion_mask=processed_mask,
        ),
        transition_metrics=config.transition_metrics.to_transition_metrics(),
    )
    for motion in _find_motion_times(rtsp_stream, stats, watcher):
        queue.put(motion)


def create_motion_process(
    rtsp_stream: str,
    msg_queue: Queue,
    config: WatchConfig,
    restart_on_exit: bool | None = None,
    hwaccel: str = "",
) -> MotionProcessWrapper:
    """Spawn a motion-detection process.

    The ``history`` parameter is the number of frames the MOG2 background
    subtractor uses to build its initial background model. This is separate
    from ``transition_metrics.preparing_duration`` (which is the state
    machine's warm-up time in seconds). Callers are expected to convert
    seconds to frames at the target FPS.

    Args:
        rtsp_stream: RTSP URL or file path to process
        msg_queue: Queue to put MotionWindow instances into
        config: WatchConfig with motion detection parameters
        restart_on_exit: Whether to restart the process on exit (auto-detected if None)
        hwaccel: Hardware acceleration method (deployment-specific)

    Returns:
        MotionProcessWrapper instance
    """
    motion_process = Process(
        target=enqueue_motion_windows,
        kwargs={
            "rtsp_stream": rtsp_stream,
            "queue": msg_queue,
            "config": config,
            "hwaccel": hwaccel,
        },
        daemon=True,
        name="wildcamtools-motion",
    )
    restart_on_exit_value = restart_on_exit if restart_on_exit is not None else is_stream_url(rtsp_stream)
    wrapper = MotionProcessWrapper(motion_process, restart_on_exit_value)
    wrapper.start()

    return wrapper
