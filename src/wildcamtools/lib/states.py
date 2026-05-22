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
    preparing_duration: int = 10
    green_to_amber_motion_min: float = 0.1
    amber_to_green_proportion_max: float = 0.075
    amber_to_red_duration: int = 1
    red_to_red_amber_proportion_max: float = 0.075
    red_amber_to_red_proportion_min: float = 0.1
    red_amber_to_green_duration: int = 1


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

    Note: Frame numbers use source video indices (native FPS, e.g., 30 fps),
    not post-filtering indices. When using Rescaler with fps < native_fps,
    frame_no values will be spaced (e.g., 0, 6, 12, ... for 30fps source at 5fps).
    """

    start_frame: int
    start_time: datetime
    end_frame: int | None
    end_time: datetime | None
    transition_metrics: WatcherTransitionMetrics
    transition_window_metrics: dict[WatcherStateEnum, StateTransitionWindowMetrics]


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
    amber_start: int | None
    red_start: int | None
    red_amber_start: int | None

    def __init__(self, motion: MogMotion, transition_metrics: WatcherTransitionMetrics) -> None:
        self.motion = motion
        self.state = WatcherStateEnum.PREPARING
        self.transition_metrics = transition_metrics
        self.transition_window_metrics = {}
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
        if next_state != self.state or next_state not in self.transition_window_metrics:
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
        if frame.frame_no >= self.transition_metrics.preparing_duration:
            return WatcherStateEnum.GREEN
        return WatcherStateEnum.PREPARING

    def _handle_green_state(self, frame: Frame) -> WatcherStateEnum:
        """Handle transition from GREEN state."""
        if frame.motion_proportion > self.transition_metrics.green_to_amber_motion_min:
            if self.transition_metrics.amber_to_red_duration == 0:
                return WatcherStateEnum.RED
            else:
                self.amber_start = frame.frame_no
                return WatcherStateEnum.AMBER
        return WatcherStateEnum.GREEN

    def _handle_amber_state(self, frame: Frame) -> WatcherStateEnum:
        """Handle transition from AMBER state."""
        if frame.motion_proportion < self.transition_metrics.amber_to_green_proportion_max:
            return WatcherStateEnum.GREEN
        if (
            self.amber_start is not None
            and frame.frame_no >= self.amber_start + self.transition_metrics.amber_to_red_duration
        ):
            self.red_start = frame.frame_no
            return WatcherStateEnum.RED
        return WatcherStateEnum.AMBER

    def _handle_red_state(self, frame: Frame) -> WatcherStateEnum:
        """Handle transition from RED state."""
        if frame.motion_proportion < self.transition_metrics.red_to_red_amber_proportion_max:
            if self.transition_metrics.red_amber_to_green_duration == 0:
                return WatcherStateEnum.GREEN
            else:
                self.red_amber_start = frame.frame_no
                return WatcherStateEnum.RED_AMBER
        return WatcherStateEnum.RED

    def _handle_red_amber_state(self, frame: Frame) -> WatcherStateEnum:
        """Handle transition from RED_AMBER state."""
        if frame.motion_proportion > self.transition_metrics.red_amber_to_red_proportion_min:
            self.red_start = frame.frame_no
            return WatcherStateEnum.RED
        if (
            self.red_amber_start is not None
            and frame.frame_no >= self.red_amber_start + self.transition_metrics.red_amber_to_green_duration
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


def enqueue_motion_windows(
    rtsp_stream: str,
    queue: Queue,
    history: int,
    threshold: int,
    kernel_size: float,
    scale: float,
    fps: float,
    hwaccel: str,
    transition_metrics: WatcherTransitionMetrics,
    motion_mask: Path | None = None,
) -> None:
    """
    Extract motion windows from video stream.

    Note: Frame numbers in returned MotionWindow objects use source video indices
    (native FPS), not post-filtering indices. When using Rescaler with fps < native_fps,
    frame_no values will be spaced (e.g., 0, 6, 12, ... for 30fps source at 5fps).
    """

    def _find_motion_times(source: str, stats: VideoStats, watcher: Watcher) -> Generator[MotionWindow]:
        start_frame: int | None = None
        start_time: datetime | None = None
        prev_state: WatcherStateEnum | None = None
        last_frame_no: int | None = None
        logger.debug("Reading %s at %sx%s@%s (%s)", source, stats.x, stats.y, fps, scale)
        rescaler = Rescaler(stats, int(stats.x * scale), int(stats.y * scale), fps)
        with VideoReader(source, hwaccel=hwaccel if hwaccel else None) as video_input:
            for frame in video_input:
                frame = rescaler.handle(frame)
                if not frame.filter_keep:
                    continue
                frame = watcher.handle(frame)
                last_frame_no = frame.frame_no
                # Start tracking when transitioning from GREEN to any motion state (AMBER or RED)
                # This handles both normal transitions (GREEN->AMBER) and zero-duration transitions (GREEN->RED)
                if start_frame is None and start_time is None:
                    if prev_state == WatcherStateEnum.GREEN and watcher.state in (
                        WatcherStateEnum.AMBER,
                        WatcherStateEnum.RED,
                    ):
                        start_frame = frame.frame_no
                        start_time = datetime.now(UTC)
                # End tracking when returning to GREEN from any motion state
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
                    end_frame = frame.frame_no
                    end_time = datetime.now(UTC)
                    yield MotionWindow(
                        start_frame=start_frame,
                        start_time=start_time,
                        end_frame=end_frame,
                        end_time=end_time,
                        transition_metrics=watcher.transition_metrics,
                        transition_window_metrics=watcher.transition_window_metrics,
                    )
                    start_frame = None
                    start_time = None

                prev_state = watcher.state

            # Yield any pending window if video ended during motion
            if start_frame is not None and start_time is not None and last_frame_no is not None:
                yield MotionWindow(
                    start_frame=start_frame,
                    start_time=start_time,
                    end_frame=last_frame_no,
                    end_time=datetime.now(UTC),
                    transition_metrics=watcher.transition_metrics,
                    transition_window_metrics=watcher.transition_window_metrics,
                )

    stats = get_video_stats(rtsp_stream)
    processed_mask = _load_and_resize_mask(
        mask_path=motion_mask,
        width=stats.x,
        height=stats.y,
        scale=scale,
    )

    watcher = Watcher(
        motion=MogMotion(
            history=history,
            threshold=threshold,
            detect_shadows=False,
            kernel_size=kernel_size,
            motion_mask=processed_mask,
        ),
        transition_metrics=transition_metrics,
    )
    for motion in _find_motion_times(rtsp_stream, stats, watcher):
        queue.put(motion)


def create_motion_process(
    rtsp_stream: str,
    msg_queue: Queue,
    threshold: float,
    kernel_size: float,
    scale: float,
    fps: float,
    hwaccel: str,
    transition_metrics: WatcherTransitionMetrics,
    motion_mask: Path | None = None,
    restart_on_exit: bool | None = None,
) -> MotionProcessWrapper:
    motion_process = Process(
        target=enqueue_motion_windows,
        kwargs={
            "rtsp_stream": rtsp_stream,
            "queue": msg_queue,
            "history": transition_metrics.preparing_duration,
            "threshold": threshold,
            "kernel_size": kernel_size,
            "scale": scale,
            "fps": fps,
            "hwaccel": hwaccel,
            "transition_metrics": transition_metrics,
            "motion_mask": motion_mask,
        },
        daemon=True,
        name="wildcamtools-motion",
    )
    restart_on_exit_value = restart_on_exit if restart_on_exit is not None else is_stream_url(rtsp_stream)
    wrapper = MotionProcessWrapper(motion_process, restart_on_exit_value)
    wrapper.start()

    return wrapper
