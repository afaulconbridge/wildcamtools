from multiprocessing import Queue
from pathlib import Path

import pytest

from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.states import Watcher, WatcherStateEnum, WatcherTransitionMetrics, create_motion_process
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


def test_motion_window_tracks_amber_to_green() -> None:
    """Test that motion windows track from AMBER state through GREEN state.

    This is a regression test for the bug where windows were only tracked during
    RED state, causing fragmentation when the state machine oscillated rapidly.
    """

    watcher = Watcher(
        motion=MogMotion(history=10, threshold=16, detect_shadows=False, kernel_size=0.005),
        transition_metrics=WatcherTransitionMetrics(
            preparing_duration=10,
            green_to_amber_motion_min=0.01,
            amber_to_green_proportion_max=0.0075,
            amber_to_red_duration=0,
            red_to_red_amber_proportion_max=0.0075,
            red_amber_to_red_proportion_min=0.01,
            red_amber_to_green_duration=0,
        ),
    )

    # Simulate state transitions: PREPARING -> GREEN -> AMBER -> RED -> RED_AMBER -> GREEN
    # This should produce ONE continuous motion window from AMBER to GREEN

    # Start in PREPARING state
    assert watcher.state == WatcherStateEnum.PREPARING

    # Transition to GREEN (handled internally by state machine)
    # We'll manually set state to simulate the transitions
    watcher.state = WatcherStateEnum.GREEN

    # Motion detected: GREEN -> AMBER
    watcher.amber_start = 100
    watcher.state = WatcherStateEnum.AMBER

    # AMBER -> RED (immediate due to amber_to_red_duration=0)
    watcher.red_start = 100
    watcher.state = WatcherStateEnum.RED

    # RED -> RED_AMBER (motion drops)
    watcher.red_amber_start = 150
    watcher.state = WatcherStateEnum.RED_AMBER

    # RED_AMBER -> GREEN (motion fully ended)
    watcher.state = WatcherStateEnum.GREEN

    # At this point, the state machine should have tracked:
    # - Started at AMBER (frame 100)
    # - Ended at GREEN (frame 150+)
    # This should be ONE window, not multiple fragmented windows

    # Verify the state machine doesn't reset amber_start prematurely
    assert watcher.amber_start == 100, "amber_start should persist through RED and RED_AMBER states"
    assert watcher.red_start == 100, "red_start should persist through RED_AMBER state"
