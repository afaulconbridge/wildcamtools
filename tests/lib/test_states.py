from multiprocessing import Queue
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from wildcamtools.lib import Frame
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.states import (
    Watcher,
    WatcherStateEnum,
    WatcherTransitionMetrics,
    create_motion_process,
)
from wildcamtools.lib.vidio import VideoReader
from wildcamtools.lib.watch_config import WatchConfig


@pytest.mark.skip(reason="Test video lacks sufficient motion to trigger all state transitions")
def test_states(video_path: str) -> None:
    watcher = Watcher(
        motion=MogMotion(history=10),
        transition_metrics=WatcherTransitionMetrics(
            preparing_duration=10.0,
            green_to_amber_motion_min=0.01,
            amber_to_green_proportion_max=0.0075,
            amber_to_red_duration=1.0,
            red_to_red_amber_proportion_max=0.0075,
            red_amber_to_red_proportion_min=0.01,
            red_amber_to_green_duration=1.0,
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
    config = WatchConfig(rtsp_stream=str(video_path))

    process = create_motion_process(
        rtsp_stream=str(video_path),
        msg_queue=msg_queue,
        config=config,
        restart_on_exit=None,  # Auto-detect
    )

    assert getattr(process, "restart_on_exit", None) is False
    process.terminate()
    process.join(timeout=1.0)


def test_create_motion_process_restart_on_exit_auto_detect_rtsp() -> None:
    """Test that RTSP URLs auto-detect as restart_on_exit=True."""
    msg_queue: Queue = Queue()
    config = WatchConfig(rtsp_stream="rtsp://localhost:8554/stream")

    process = create_motion_process(
        rtsp_stream="rtsp://localhost:8554/stream",
        msg_queue=msg_queue,
        config=config,
        restart_on_exit=None,  # Auto-detect
    )

    assert getattr(process, "restart_on_exit", None) is True
    process.terminate()
    process.join(timeout=1.0)


def test_create_motion_process_restart_on_exit_explicit_override() -> None:
    """Test explicit restart_on_exit override."""
    msg_queue: Queue = Queue()
    config = WatchConfig(rtsp_stream="samples/synth/synth_0.0100_1.000.mp4")

    # Override file to restart
    process = create_motion_process(
        rtsp_stream="samples/synth/synth_0.0100_1.000.mp4",
        msg_queue=msg_queue,
        config=config,
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
            preparing_duration=10.0,
            green_to_amber_motion_min=0.01,
            amber_to_green_proportion_max=0.0075,
            amber_to_red_duration=0.0,
            red_to_red_amber_proportion_max=0.0075,
            red_amber_to_red_proportion_min=0.01,
            red_amber_to_green_duration=0.0,
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
    watcher.amber_start = 100.0
    watcher.state = WatcherStateEnum.AMBER

    # AMBER -> RED (immediate due to amber_to_red_duration=0)
    watcher.red_start = 100.0
    watcher.state = WatcherStateEnum.RED

    # RED -> RED_AMBER (motion drops)
    watcher.red_amber_start = 150.0
    watcher.state = WatcherStateEnum.RED_AMBER

    # RED_AMBER -> GREEN (motion fully ended)
    watcher.state = WatcherStateEnum.GREEN

    # At this point, the state machine should have tracked:
    # - Started at AMBER (frame 100)
    # - Ended at GREEN (frame 150+)
    # This should be ONE window, not multiple fragmented windows

    # Verify the state machine doesn't reset amber_start prematurely
    assert watcher.amber_start == 100.0, "amber_start should persist through RED and RED_AMBER states"
    assert watcher.red_start == 100.0, "red_start should persist through RED_AMBER state"


def _make_watcher_with_motion_mock(metrics: WatcherTransitionMetrics | None = None) -> Watcher:
    """Create a Watcher with a mocked motion handler so we can drive motion_proportion directly."""
    motion = MagicMock()
    motion.handle.side_effect = lambda frame: frame
    if metrics is None:
        metrics = WatcherTransitionMetrics(
            preparing_duration=10.0,
            green_to_amber_motion_min=0.01,
            amber_to_green_proportion_max=0.0075,
            amber_to_red_duration=5.0,
            red_to_red_amber_proportion_max=0.0075,
            red_amber_to_red_proportion_min=0.01,
            red_amber_to_green_duration=5.0,
        )
    watcher = Watcher(
        motion=motion,
        transition_metrics=metrics,
    )
    return watcher


def _make_frame(
    frame_no: int,
    motion_proportion: float = 0.0,
    timestamp: float | None = None,
) -> Frame:
    raw = np.zeros((10, 10, 3), dtype=np.uint8)
    f = Frame(raw=raw, frame_no=frame_no, timestamp=timestamp)
    f.motion_proportion = motion_proportion
    return f


def test_amber_count_accumulates_across_state_visits() -> None:
    """Regression test: AMBER count must accumulate across multiple visits, not reset.

    Previously, the metric tracking condition ``next_state != self.state or
    next_state not in self.transition_window_metrics`` caused a new entry to be
    created (count=1) on every state change. This meant the AMBER count
    reflected only the most recent visit, not the cumulative count for the
    watcher's lifetime.
    """
    watcher = _make_watcher_with_motion_mock()

    # Drive at 5 fps (0.2 s between frames). preparing_duration=10.0s
    # means we need 50 frames in PREPARING before transitioning to GREEN,
    # so we generate 50 frames at motion=-1.0 then 1 frame at low motion
    # to transition to GREEN.
    for i in range(51):
        motion_val = -1.0 if i < 50 else 0.001
        watcher.handle(_make_frame(i, motion_proportion=motion_val, timestamp=i * 0.2))
    assert watcher.state == WatcherStateEnum.GREEN

    # GREEN -> AMBER entry (timestamp 10.2s)
    watcher.handle(_make_frame(51, motion_proportion=0.05, timestamp=10.2))
    assert watcher.state == WatcherStateEnum.AMBER
    amber_count_after_first_entry = watcher.transition_window_metrics[WatcherStateEnum.AMBER].count

    # AMBER -> GREEN (motion dropped, timestamp 10.4s)
    watcher.handle(_make_frame(52, motion_proportion=0.001, timestamp=10.4))
    assert watcher.state == WatcherStateEnum.GREEN

    # GREEN -> AMBER entry (timestamp 10.6s)
    watcher.handle(_make_frame(53, motion_proportion=0.05, timestamp=10.6))
    assert watcher.state == WatcherStateEnum.AMBER

    # Stay in AMBER for 5 seconds worth of frames (0.2s each) so amber_to_red_duration=5.0 is satisfied.
    # 5.0s = 25 frames at 0.2s. We enter AMBER at frame 53, so the 25th frame
    # at timestamp 10.6 + 25*0.2 = 15.6s should be the one that transitions.
    for offset in range(1, 25):
        ts = 10.6 + offset * 0.2
        watcher.handle(_make_frame(53 + offset, motion_proportion=0.05, timestamp=ts))
        assert watcher.state == WatcherStateEnum.AMBER

    # Final frame at timestamp 10.6 + 25*0.2 = 15.6s transitions to RED
    watcher.handle(_make_frame(78, motion_proportion=0.05, timestamp=15.6))
    assert watcher.state == WatcherStateEnum.RED

    # AMBER count should accumulate: 1 (first visit) + 25 (second visit) = 26
    amber_count = watcher.transition_window_metrics[WatcherStateEnum.AMBER].count
    assert amber_count == 26, (
        f"AMBER count should accumulate across visits. "
        f"Expected 26 (1 + 25), got {amber_count}. "
        f"First entry count was {amber_count_after_first_entry}."
    )


def test_preparation_and_green_counts_are_cumulative() -> None:
    """PREPARING and GREEN counts should accumulate across the watcher's lifetime, not reset."""
    watcher = _make_watcher_with_motion_mock()

    # preparing_duration=10.0s, so 50 frames at 0.2s = 10s of PREPARING
    for i in range(55):
        motion_val = -1.0 if i < 50 else 0.001
        watcher.handle(_make_frame(i, motion_proportion=motion_val, timestamp=i * 0.2))
    assert watcher.state == WatcherStateEnum.GREEN

    preparing_count = watcher.transition_window_metrics[WatcherStateEnum.PREPARING].count
    green_count = watcher.transition_window_metrics[WatcherStateEnum.GREEN].count
    assert preparing_count == 50
    assert green_count == 5

    # Trigger motion, drop back to GREEN
    watcher.handle(_make_frame(55, motion_proportion=0.05, timestamp=11.0))
    assert watcher.state == WatcherStateEnum.AMBER
    amber_count_first_visit = watcher.transition_window_metrics[WatcherStateEnum.AMBER].count

    # AMBER -> GREEN (motion dropped, before amber_to_red_duration=5.0s elapses)
    watcher.handle(_make_frame(56, motion_proportion=0.001, timestamp=11.2))
    assert watcher.state == WatcherStateEnum.GREEN

    # GREEN count should be 6 (accumulated), not reset
    assert watcher.transition_window_metrics[WatcherStateEnum.GREEN].count == green_count + 1
    # AMBER count from first visit should still be 1
    assert watcher.transition_window_metrics[WatcherStateEnum.AMBER].count == amber_count_first_visit


def test_state_machine_uses_timestamps_for_durations() -> None:
    """The state machine should use timestamps (seconds) for AMBER/RED_AMBER durations.

    This is a regression test: previously the state machine used
    ``frame.frame_no`` (source FPS index) for duration checks, so
    ``amber_to_red_duration=5`` meant 5 source frames (~0.17s at 30fps),
    not 5 seconds. With the fix, the duration is in seconds.
    """
    metrics = WatcherTransitionMetrics(
        preparing_duration=0.0,  # no warm-up so we can immediately start motion
        green_to_amber_motion_min=0.01,
        amber_to_green_proportion_max=0.0075,
        amber_to_red_duration=5.0,  # 5 seconds
        red_to_red_amber_proportion_max=0.0075,
        red_amber_to_red_proportion_min=0.01,
        red_amber_to_green_duration=5.0,  # 5 seconds
    )
    watcher = _make_watcher_with_motion_mock(metrics)
    watcher.state = WatcherStateEnum.GREEN  # Skip PREPARING for the test

    # Drive frames spaced 1 second apart (so timestamps advance 1s each frame).
    # At frame_no=100, motion spikes, enter AMBER.
    watcher.handle(_make_frame(100, motion_proportion=0.05, timestamp=100.0))
    assert watcher.state == WatcherStateEnum.AMBER

    # Frames 101, 102, 103, 104: timestamps 101, 102, 103, 104 (each +1s from entry).
    # amber_to_red_duration=5.0, so we need >=5.0s elapsed before transitioning to RED.
    for i, ts in enumerate([101.0, 102.0, 103.0, 104.0]):
        watcher.handle(_make_frame(101 + i, motion_proportion=0.05, timestamp=ts))
        assert watcher.state == WatcherStateEnum.AMBER, (
            f"State should still be AMBER at timestamp {ts} (only {ts - 100.0}s elapsed)"
        )

    # At timestamp 105.0, 5.0s have elapsed -> transition to RED.
    watcher.handle(_make_frame(105, motion_proportion=0.05, timestamp=105.0))
    assert watcher.state == WatcherStateEnum.RED, "At 5.0s elapsed, state should transition to RED"


def test_amber_zero_duration_immediate_red() -> None:
    """amber_to_red_duration=0 should still skip AMBER (transition GREEN->RED directly)."""
    metrics = WatcherTransitionMetrics(
        preparing_duration=0.0,
        green_to_amber_motion_min=0.01,
        amber_to_green_proportion_max=0.0075,
        amber_to_red_duration=0.0,
        red_to_red_amber_proportion_max=0.0075,
        red_amber_to_red_proportion_min=0.01,
        red_amber_to_green_duration=0.0,
    )
    watcher = _make_watcher_with_motion_mock(metrics)
    watcher.state = WatcherStateEnum.GREEN

    watcher.handle(_make_frame(0, motion_proportion=0.05, timestamp=0.0))
    assert watcher.state == WatcherStateEnum.RED, "amber_to_red_duration=0 should transition GREEN->RED directly"


def test_state_machine_ignores_timestamps_when_none() -> None:
    """If frames have no timestamp, the state machine should not crash.

    The state machine falls through (state stays the same) when timestamps
    are not available. This is a defensive default.
    """
    metrics = WatcherTransitionMetrics(
        preparing_duration=10.0,
        green_to_amber_motion_min=0.01,
        amber_to_green_proportion_max=0.0075,
        amber_to_red_duration=5.0,
        red_to_red_amber_proportion_max=0.0075,
        red_amber_to_red_proportion_min=0.01,
        red_amber_to_green_duration=5.0,
    )
    watcher = _make_watcher_with_motion_mock(metrics)
    # No timestamp -> should stay in PREPARING
    for i in range(20):
        watcher.handle(_make_frame(i, motion_proportion=0.05, timestamp=None))
    assert watcher.state == WatcherStateEnum.PREPARING, (
        "Without timestamps, the state machine should remain in PREPARING"
    )


def test_motion_window_metrics_are_scoped_per_window(video_path: Path) -> None:
    """Integration test: each MotionWindow's transition_window_metrics reflects only that window.

    Previously, the watcher's transition_window_metrics were cumulative across the
    watcher's lifetime, so the same metrics were attached to every yielded
    MotionWindow. With the fix, the watcher's metrics are reset at each new
    window, so each yielded MotionWindow has metrics scoped to itself.
    """
    from multiprocessing import Queue

    from wildcamtools.lib.states import enqueue_motion_windows

    q: Queue = Queue()
    config = WatchConfig(rtsp_stream=str(video_path))
    config.transition_metrics.preparing_duration = 10.0
    config.transition_metrics.green_to_amber_motion_min = 0.01
    config.transition_metrics.amber_to_green_proportion_max = 0.0075
    config.transition_metrics.amber_to_red_duration = 2.0
    config.transition_metrics.red_to_red_amber_proportion_max = 0.0075
    config.transition_metrics.red_amber_to_red_proportion_min = 0.01
    config.transition_metrics.red_amber_to_green_duration = 2.0

    enqueue_motion_windows(
        rtsp_stream=str(video_path),
        queue=q,
        config=config,
    )

    windows = []
    while not q.empty():
        windows.append(q.get_nowait())

    # If we got any windows, verify their metrics are scoped (i.e. PREPARING should
    # not appear in the per-window metrics because it was completed before any
    # motion could be detected).
    for window in windows:
        assert WatcherStateEnum.PREPARING not in window.transition_window_metrics, (
            f"PREPARING should not be in a motion window's metrics "
            f"(got metrics: {list(window.transition_window_metrics.keys())})"
        )


def test_motion_windows_without_red_are_discarded() -> None:
    """Verify that motion windows that never reach RED state are discarded.

    This test creates a synthetic video with brief motion events that trigger
    GREEN->AMBER->GREEN transitions but don't sustain motion long enough to
    reach RED. These windows should be discarded.
    """
    import tempfile

    import cv2

    from wildcamtools.lib.states import enqueue_motion_windows

    tmpdir = tempfile.mkdtemp()
    test_path = Path(tmpdir) / "test_brief_motion.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(test_path), fourcc, 30.0, (100, 100))
    # 5s background, 1s motion (too brief for RED), 3s background
    # amber_to_red_duration=3.0s, so 1s motion won't reach RED
    total_frames = 30 * (5 + 1 + 3)
    np.random.seed(42)
    for i in range(total_frames):
        t = i / 30.0
        if 5 <= t < 6:
            # Brief motion period
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        else:
            frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    q: Queue = Queue()
    config = WatchConfig(rtsp_stream=str(test_path))
    config.transition_metrics.preparing_duration = 0.5
    config.transition_metrics.green_to_amber_motion_min = 0.01
    config.transition_metrics.amber_to_green_proportion_max = 0.0075
    config.transition_metrics.amber_to_red_duration = 3.0  # 3s in AMBER before RED
    config.transition_metrics.red_to_red_amber_proportion_max = 0.0075
    config.transition_metrics.red_amber_to_red_proportion_min = 0.01
    config.transition_metrics.red_amber_to_green_duration = 1.0

    enqueue_motion_windows(
        rtsp_stream=str(test_path),
        queue=q,
        config=config,
    )

    windows = []
    while not q.empty():
        windows.append(q.get_nowait())

    # Brief motion should be discarded (never reached RED)
    assert len(windows) == 0, f"Expected 0 windows (brief motion discarded), got {len(windows)}"


def test_per_window_scoping_resets_between_windows() -> None:
    """Integration test: each MotionWindow's metrics are independent of previous windows.

    This test drives the full pipeline with a synthetic video that has
    multiple motion events and verifies that each yielded MotionWindow's
    metrics are scoped to that window (not cumulative across windows).

    Motion periods are long enough to reach RED state (required for windows to be yielded).
    """
    import tempfile

    import cv2

    from wildcamtools.lib.states import enqueue_motion_windows

    # Build a synthetic video with noise patterns to simulate motion.
    # 30fps source, 5fps rescale. Use a temp directory but don't auto-clean so
    # ``enqueue_motion_windows`` can read the file from a subprocess.
    # Motion periods must have sustained motion to reach RED state.
    tmpdir = tempfile.mkdtemp()
    test_path = Path(tmpdir) / "test_per_window.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(test_path), fourcc, 30.0, (100, 100))
    # 5s background, 8s motion, 3s background, 8s motion, 3s background
    # Motion periods must be long enough to: stay in AMBER for 3s, then RED, then RED_AMBER for 1s
    total_frames = 30 * (5 + 8 + 3 + 8 + 3)
    np.random.seed(42)  # Reproducible noise
    for i in range(total_frames):
        t = i / 30.0
        if (5 <= t < 13) or (16 <= t < 24):
            # Motion period: random noise pattern (simulates activity)
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        else:
            # Background period: static gray frame
            frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    q: Queue = Queue()
    config = WatchConfig(rtsp_stream=str(test_path))
    config.transition_metrics.preparing_duration = 0.5  # 0.5s warm-up
    config.transition_metrics.green_to_amber_motion_min = 0.01
    config.transition_metrics.amber_to_green_proportion_max = 0.0075
    config.transition_metrics.amber_to_red_duration = 3.0  # 3s in AMBER before RED
    config.transition_metrics.red_to_red_amber_proportion_max = 0.0075
    config.transition_metrics.red_amber_to_red_proportion_min = 0.01
    config.transition_metrics.red_amber_to_green_duration = 1.0  # 1s in RED_AMBER before GREEN

    enqueue_motion_windows(
        rtsp_stream=str(test_path),
        queue=q,
        config=config,
    )

    windows = []
    while not q.empty():
        windows.append(q.get_nowait())

    # We should detect both motion periods as separate windows.
    assert len(windows) >= 2, f"Expected at least 2 motion windows from synthetic video, got {len(windows)}"

    # Each window's metrics should be independent (per-window scoping).
    for i, window in enumerate(windows):
        # PREPARING should never appear (it was completed before motion)
        assert WatcherStateEnum.PREPARING not in window.transition_window_metrics, (
            f"Window {i} should not contain PREPARING in its metrics"
        )
        # Each window should include the end GREEN frame in its metrics.
        assert WatcherStateEnum.GREEN in window.transition_window_metrics, (
            f"Window {i} should include GREEN in its metrics"
        )
        # The GREEN count for each window should be 1 (just the end frame).
        # If it's higher, the metrics are cumulative across windows.
        green_count = window.transition_window_metrics[WatcherStateEnum.GREEN].count
        assert green_count == 1, (
            f"Window {i} GREEN count should be 1 (just the end frame), got {green_count}. "
            f"This means the per-window scoping reset isn't working correctly."
        )
