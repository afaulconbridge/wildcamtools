from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.states import Watcher, WatcherStateEnum, WatcherTransitionMetrics
from wildcamtools.lib.vidio import VideoReader


def test_states(rtsp_server: str):
    watcher = Watcher(
        motion=MogMotion(history=10),
        transition_metrics=WatcherTransitionMetrics(
            preparing_duration=10,
            green_to_amber_motion_min=0.01,
            amber_to_green_proportion_max=0.0075,
            amber_to_red_duration=1,
            red_to_red_amber_proportion_max=0.0075,
            red_amber_to_red_proportion_min=0.01,
            red_amber_to_green_duration=1,
        ),
    )
    visited_states = {watcher.state}
    with VideoReader(rtsp_server, 3840, 2160) as video_reader:
        for frame in video_reader:
            if frame.frame_no > 150:
                break
            watcher.handle(frame)
            visited_states.add(watcher.state)

    assert WatcherStateEnum.PREPARING in visited_states
    assert WatcherStateEnum.GREEN in visited_states
    assert WatcherStateEnum.AMBER in visited_states
    assert WatcherStateEnum.RED in visited_states
    assert WatcherStateEnum.RED_AMBER in visited_states
