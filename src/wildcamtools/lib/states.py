import logging
from enum import StrEnum

from pydantic import BaseModel

from wildcamtools.lib import Frame, FrameHandler
from wildcamtools.lib.motion import MogMotion

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

    def __init__(self, motion: MogMotion, transition_metrics: WatcherTransitionMetrics) -> None:
        self.motion = motion
        self.state = WatcherStateEnum.PREPARING
        self.transition_metrics = transition_metrics

    def handle(self, frame: Frame) -> Frame:
        output = self.motion.handle(frame)

        self.state = self._get_next_state(output)

        return output

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
