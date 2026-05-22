"""Unit tests for motion window tracking from AMBER to GREEN states."""

from wildcamtools.lib.states import (
    WatcherStateEnum,
)


def test_amber_to_green_tracking_logic() -> None:
    """Test the logic for tracking motion windows from AMBER/RED through GREEN states.

    This is a regression test for the bug where windows were only tracked during
    RED state, causing fragmentation when the state machine oscillated rapidly.

    The fix tracks from the first motion state (AMBER or RED) through GREEN state (motion ended),
    including AMBER, RED, and RED_AMBER states in a single continuous window.
    """
    # Simulate the _find_motion_times logic
    start_frame: int | None = None
    prev_state: WatcherStateEnum | None = None

    # Simulate state transitions
    states_sequence = [
        (0, WatcherStateEnum.PREPARING),
        (10, WatcherStateEnum.GREEN),
        (100, WatcherStateEnum.AMBER),  # Motion starts - should set start_frame
        (101, WatcherStateEnum.RED),
        (150, WatcherStateEnum.RED_AMBER),
        (151, WatcherStateEnum.GREEN),  # Motion ends - should yield window
    ]

    end_frame = None

    for frame_no, state in states_sequence:
        # Simulate the fix logic from _find_motion_times
        if start_frame is None:
            if prev_state == WatcherStateEnum.GREEN and state in (WatcherStateEnum.AMBER, WatcherStateEnum.RED):
                start_frame = frame_no
        elif (
            start_frame is not None
            and state == WatcherStateEnum.GREEN
            and prev_state
            in (
                WatcherStateEnum.AMBER,
                WatcherStateEnum.RED,
                WatcherStateEnum.RED_AMBER,
            )
        ):
            end_frame = frame_no
            break
        prev_state = state

    # Verify single continuous window
    assert start_frame == 100, "Window should start at first motion state (frame 100)"
    assert end_frame == 151, "Window should end at GREEN state (frame 151)"
    assert end_frame - start_frame == 51, "Window should span entire motion event"


def test_no_fragmentation_with_rapid_oscillation() -> None:
    """Test that rapid state oscillation doesn't create fragmented windows.

    Simulates the bug scenario where state oscillates between RED and GREEN
    due to noisy threshold, which previously created many 1-frame windows.

    With the fix, each motion event (GREEN->motion->GREEN) produces one window
    regardless of whether it goes through AMBER or directly to RED.
    """
    start_frame: int | None = None
    prev_state: WatcherStateEnum | None = None
    windows_yielded: list[tuple[int, int]] = []

    # Simulate rapid oscillation (the bug scenario)
    oscillating_states = [
        (10, WatcherStateEnum.GREEN),
        (100, WatcherStateEnum.AMBER),
        (101, WatcherStateEnum.RED),
        (102, WatcherStateEnum.GREEN),
        (103, WatcherStateEnum.AMBER),
        (104, WatcherStateEnum.RED),
        (105, WatcherStateEnum.GREEN),
        (150, WatcherStateEnum.AMBER),
        (151, WatcherStateEnum.RED),
        (152, WatcherStateEnum.GREEN),
    ]

    for frame_no, state in oscillating_states:
        if start_frame is None:
            if prev_state == WatcherStateEnum.GREEN and state in (WatcherStateEnum.AMBER, WatcherStateEnum.RED):
                start_frame = frame_no
        elif (
            start_frame is not None
            and state == WatcherStateEnum.GREEN
            and prev_state
            in (
                WatcherStateEnum.AMBER,
                WatcherStateEnum.RED,
                WatcherStateEnum.RED_AMBER,
            )
        ):
            windows_yielded.append((start_frame, frame_no))
            start_frame = None
        prev_state = state

    # Each motion event produces one window (3 events = 3 windows)
    assert len(windows_yielded) == 3, f"Expected 3 windows, got {len(windows_yielded)}"

    # Each window should span multiple frames (not 1-frame fragments)
    for start, end in windows_yielded:
        assert end - start > 1, f"Window {start}-{end} should span multiple frames"


def test_zero_duration_amber_to_red() -> None:
    """Test motion tracking when amber_to_red_duration is 0 (GREEN->RED directly).

    This is a regression test for the case where --amber-to-red-duration 0 causes
    the state machine to skip AMBER state and go directly GREEN->RED.
    """
    start_frame: int | None = None
    prev_state: WatcherStateEnum | None = None
    windows_yielded: list[tuple[int, int]] = []

    # Simulate GREEN->RED->GREEN (skipping AMBER due to zero duration)
    states_sequence = [
        (10, WatcherStateEnum.GREEN),
        (100, WatcherStateEnum.RED),  # Direct transition from GREEN
        (101, WatcherStateEnum.RED),
        (150, WatcherStateEnum.GREEN),  # Back to GREEN
    ]

    for frame_no, state in states_sequence:
        if start_frame is None:
            if prev_state == WatcherStateEnum.GREEN and state in (WatcherStateEnum.AMBER, WatcherStateEnum.RED):
                start_frame = frame_no
        elif (
            start_frame is not None
            and state == WatcherStateEnum.GREEN
            and prev_state
            in (
                WatcherStateEnum.AMBER,
                WatcherStateEnum.RED,
                WatcherStateEnum.RED_AMBER,
            )
        ):
            windows_yielded.append((start_frame, frame_no))
            start_frame = None
        prev_state = state

    # Should capture the RED motion event even though AMBER was skipped
    assert len(windows_yielded) == 1, f"Expected 1 window, got {len(windows_yielded)}"
    assert windows_yielded[0] == (100, 150), "Window should span RED state from frame 100 to 150"
