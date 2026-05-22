from multiprocessing import Queue
from pathlib import Path

import pytest

from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.states import Watcher, WatcherStateEnum, WatcherTransitionMetrics
from wildcamtools.lib.vidio import VideoReader


@pytest.mark.skip(reason="Test video lacks sufficient motion to trigger all state transitions")
def test_states(video_path: str) -> None:
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
    with VideoReader(video_path, 3840, 2160) as video_reader:
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


def test_create_motion_process_restart_on_exit_auto_detect_file(video_path: Path) -> None:
    """Test that file paths auto-detect as restart_on_exit=False."""
    from wildcamtools.lib.states import WatcherTransitionMetrics, create_motion_process

    msg_queue: Queue = Queue()
    metrics = WatcherTransitionMetrics()

    process = create_motion_process(
        rtsp_stream=str(video_path),
        msg_queue=msg_queue,
        threshold=16,
        kernel_size=0.005,
        scale=0.25,
        fps=5.0,
        hwaccel="",
        transition_metrics=metrics,
        restart_on_exit=None,  # Auto-detect
    )

    assert getattr(process, "restart_on_exit", None) is False
    process.terminate()
    process.join(timeout=1.0)


def test_create_motion_process_restart_on_exit_auto_detect_rtsp() -> None:
    """Test that RTSP URLs auto-detect as restart_on_exit=True."""
    from wildcamtools.lib.states import WatcherTransitionMetrics, create_motion_process

    msg_queue: Queue = Queue()
    metrics = WatcherTransitionMetrics()

    process = create_motion_process(
        rtsp_stream="rtsp://localhost:8554/stream",
        msg_queue=msg_queue,
        threshold=16,
        kernel_size=0.005,
        scale=0.25,
        fps=5.0,
        hwaccel="",
        transition_metrics=metrics,
        restart_on_exit=None,  # Auto-detect
    )

    assert getattr(process, "restart_on_exit", None) is True
    process.terminate()
    process.join(timeout=1.0)


def test_create_motion_process_restart_on_exit_explicit_override() -> None:
    """Test explicit restart_on_exit override."""
    from wildcamtools.lib.states import WatcherTransitionMetrics, create_motion_process

    msg_queue: Queue = Queue()
    metrics = WatcherTransitionMetrics()

    # Override file to restart
    process = create_motion_process(
        rtsp_stream="samples/synth/synth_0.0100_1.000.mp4",
        msg_queue=msg_queue,
        threshold=16,
        kernel_size=0.005,
        scale=0.25,
        fps=5.0,
        hwaccel="",
        transition_metrics=metrics,
        restart_on_exit=True,  # Explicit override
    )

    assert getattr(process, "restart_on_exit", None) is True
    process.terminate()
    process.join(timeout=1.0)
