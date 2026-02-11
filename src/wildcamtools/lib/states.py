import logging
from collections.abc import Generator
from datetime import UTC, datetime
from enum import StrEnum
from multiprocessing import Process, Queue
from typing import Self

from pydantic import BaseModel

from wildcamtools.lib import Frame, FrameHandler
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.stats import VideoStats, get_video_stats
from wildcamtools.lib.vidio import FrameSourceFFMPEG

logger = logging.getLogger(__name__)


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

    def __init__(self, motion: MogMotion, transition_metrics: WatcherTransitionMetrics) -> None:
        self.motion = motion
        self.state = WatcherStateEnum.PREPARING
        self.transition_metrics = transition_metrics
        self.transition_window_metrics = {}

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
        logger.debug(f"Frame no: {frame.frame_no}")
        logger.debug(f"Motion proportion: {frame.motion_proportion}")
        match self.state:
            case WatcherStateEnum.PREPARING:
                if frame.frame_no >= self.transition_metrics.preparing_duration:
                    return WatcherStateEnum.GREEN

            case WatcherStateEnum.GREEN:
                if frame.motion_proportion > self.transition_metrics.green_to_amber_motion_min:
                    if self.transition_metrics.amber_to_red_duration == 0:
                        return WatcherStateEnum.RED
                    else:
                        self.amber_start = frame.frame_no
                        return WatcherStateEnum.AMBER

            case WatcherStateEnum.AMBER:
                if frame.motion_proportion < self.transition_metrics.amber_to_green_proportion_max:
                    return WatcherStateEnum.GREEN
                elif frame.frame_no >= self.amber_start + self.transition_metrics.amber_to_red_duration:
                    self.red_start = frame.frame_no
                    return WatcherStateEnum.RED

            case WatcherStateEnum.RED:
                if frame.motion_proportion < self.transition_metrics.red_to_red_amber_proportion_max:
                    if self.transition_metrics.red_amber_to_green_duration == 0:
                        return WatcherStateEnum.GREEN
                    else:
                        self.red_amber_start = frame.frame_no
                        return WatcherStateEnum.RED_AMBER

            case WatcherStateEnum.RED_AMBER:
                if frame.motion_proportion > self.transition_metrics.red_amber_to_red_proportion_min:
                    self.red_start = frame.frame_no
                    return WatcherStateEnum.RED
                elif frame.frame_no >= self.red_amber_start + self.transition_metrics.red_amber_to_green_duration:
                    return WatcherStateEnum.GREEN

            case WatcherStateEnum.DISABLED:
                pass
        return self.state


def enqueue_motion_windows(
    rtsp_stream: str,
    queue: Queue,
    history: int,
    threshold: int,
    kernel_size: int,
    scale: float,
    fps: float,
    hwaccel: str,
    transition_metrics: WatcherTransitionMetrics,
) -> None:
    def _find_motion_times(source: str, stats: VideoStats, watcher: Watcher) -> Generator[MotionWindow]:
        start_frame: int | None = None
        start_time: datetime | None = None
        with FrameSourceFFMPEG(
            source,
            width=stats.x,
            height=stats.y,
            scale=scale,
            fps=fps,
            hwaccel=hwaccel if hwaccel else None,
        ) as video_input:
            for frame in video_input:
                frame = watcher.handle(frame)
                # TODO include both amber and red-amber states in window
                if start_frame is None and start_time is None and watcher.state == WatcherStateEnum.RED:
                    start_frame = frame.frame_no
                    start_time = datetime.now(UTC)
                    yield MotionWindow(
                        start_frame=start_frame,
                        start_time=start_time,
                        end_frame=None,
                        end_time=None,
                        transition_metrics=watcher.transition_metrics,
                        transition_window_metrics=watcher.transition_window_metrics,
                    )
                elif start_frame is not None and start_time is not None and watcher.state == WatcherStateEnum.GREEN:
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


def create_motion_process(
    rtsp_stream: str,
    msg_queue: Queue,
    threshold: float,
    kernel_size: int,
    scale: float,
    fps: float,
    hwaccel: str,
    transition_metrics: WatcherTransitionMetrics,
) -> Process:
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
        },
        daemon=True,
        name="wildcamtools-motion",
    )
    motion_process.start()

    return motion_process
