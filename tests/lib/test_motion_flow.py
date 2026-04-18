from collections.abc import Generator

import numpy as np
import pytest

from wildcamtools.lib import Frame
from wildcamtools.lib.motion import FlowMotion, MogMotion


# ---------------------------------------------------------------------------
# _compute_kernel tests (via MogMotion as a concrete subclass)
# ---------------------------------------------------------------------------


def _make_frame(height: int, width: int) -> np.ndarray:
    """Return a random RGB frame of the given dimensions."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_compute_kernel_small_frame_returns_minimum() -> None:
    """Small frames produce k_size below 3, so the minimum of 3 is enforced."""
    # max_dim=100, k_size=int(100*0.01)=1, odd but <3 → 3
    motion = MogMotion(history=1, kernel_size=0.01)
    frame = _make_frame(100, 100)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (3, 3)
    assert kernel.dtype == np.uint8
    assert np.all(kernel == 1)


def test_compute_kernel_even_result_rounds_down_to_odd() -> None:
    """When the raw proportion gives an even k_size, it is decremented to odd."""
    # max_dim=1000, k_size=int(1000*0.01)=10, even → 10-1=9 → max(3,9)=9
    motion = MogMotion(history=1, kernel_size=0.01)
    frame = _make_frame(800, 1000)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (9, 9)


def test_compute_kernel_odd_result_unchanged() -> None:
    """When the raw proportion gives an odd k_size >= 3, it is used as-is."""
    # max_dim=1000, k_size=int(1000*0.051)=51, odd → 51
    motion = MogMotion(history=1, kernel_size=0.051)
    frame = _make_frame(800, 1000)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (51, 51)


def test_compute_kernel_uses_max_dimension() -> None:
    """Kernel size is based on the longest side (height or width)."""
    # Tall frame: max_dim=height=500
    # k_size=int(500*0.1)=50, even → 49
    motion = MogMotion(history=1, kernel_size=0.1)
    tall_frame = _make_frame(500, 100)
    kernel = motion._compute_kernel(tall_frame)
    assert kernel.shape == (49, 49)

    # Wide frame: max_dim=width=500
    wide_frame = _make_frame(100, 500)
    kernel_wide = motion._compute_kernel(wide_frame)
    assert kernel_wide.shape == (49, 49)


def test_compute_kernel_zero_proportion_returns_minimum() -> None:
    """A very small kernel_size proportion that yields 0 still returns minimum 3."""
    # max_dim=10, k_size=int(10*0.001)=0, even → -1, max(3,-1)=3
    motion = MogMotion(history=1, kernel_size=0.001)
    frame = _make_frame(10, 10)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (3, 3)


def test_compute_kernel_large_proportion() -> None:
    """A large kernel_size proportion produces a large odd kernel."""
    # max_dim=200, k_size=int(200*0.3)=60, even → 59
    motion = MogMotion(history=1, kernel_size=0.3)
    frame = _make_frame(200, 150)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (59, 59)


def test_compute_kernel_returns_ones_array() -> None:
    """The kernel is always a structuring element filled with 1s."""
    motion = MogMotion(history=1, kernel_size=0.05)
    frame = _make_frame(300, 300)
    kernel = motion._compute_kernel(frame)
    assert np.all(kernel == 1)
    assert kernel.dtype == np.uint8


# ---------------------------------------------------------------------------
# FlowMotion tests
# ---------------------------------------------------------------------------


def test_flow_motion_init_defaults() -> None:
    """FlowMotion initialises with expected default parameters."""
    motion = FlowMotion()
    assert motion.history == 500
    assert motion.threshold == pytest.approx(1.0)
    assert motion.kernel_size == pytest.approx(0.01)
    assert motion.prev_gray is None
    assert motion.motion_mask is None


def test_flow_motion_init_custom() -> None:
    """FlowMotion accepts and stores custom parameters."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    motion = FlowMotion(history=25, threshold=5.0, kernel_size=0.02, motion_mask=mask)
    assert motion.history == 25
    assert motion.threshold == pytest.approx(5.0)
    assert motion.kernel_size == pytest.approx(0.02)
    assert motion.prev_gray is None


def test_flow_motion_first_frame_returns_zeros() -> None:
    """On the first call, update_background stores prev_gray and returns a zero mask."""
    motion = FlowMotion(history=1, threshold=1.0)
    frame = _make_frame(100, 100)
    result = motion.update_background(frame)

    assert result.shape == (100, 100)
    assert result.dtype == np.uint8
    assert np.all(result == 0)
    assert motion.prev_gray is not None


def test_flow_motion_first_frame_sets_prev_gray() -> None:
    """After the first frame, prev_gray is populated with a grayscale version."""
    motion = FlowMotion(history=1, threshold=1.0)
    frame = _make_frame(80, 120)
    motion.update_background(frame)

    assert motion.prev_gray is not None
    assert motion.prev_gray.shape == (80, 120)
    assert motion.prev_gray.dtype == np.uint8


def test_flow_motion_identical_frames_produce_no_motion() -> None:
    """Two identical frames should produce near-zero optical flow and no motion mask."""
    motion = FlowMotion(history=1, threshold=0.5)
    frame = _make_frame(100, 100)

    motion.update_background(frame)  # first frame - sets prev_gray
    result = motion.update_background(frame)  # second identical frame

    assert result.shape == (100, 100)
    assert result.dtype == np.uint8
    # Identical frames → zero flow → no pixels above threshold
    assert np.all(result == 0)


def test_flow_motion_different_frames_produce_motion() -> None:
    """Two very different frames should produce a non-empty motion mask."""
    motion = FlowMotion(history=1, threshold=0.1)

    rng = np.random.default_rng(42)
    frame1 = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    # Create a clearly different second frame by shifting the content
    frame2 = np.roll(frame1, shift=20, axis=1)

    motion.update_background(frame1)
    result = motion.update_background(frame2)

    assert result.shape == (100, 100)
    assert np.any(result > 0), "Expected some motion pixels between different frames"


def test_flow_motion_updates_prev_gray() -> None:
    """prev_gray is updated to the current frame's grayscale after each call."""
    motion = FlowMotion(history=1, threshold=1.0)

    frame1 = _make_frame(50, 50)
    frame2 = _make_frame(50, 50)

    motion.update_background(frame1)
    gray1 = motion.prev_gray.copy()  # type: ignore[union-attr]

    motion.update_background(frame2)
    gray2 = motion.prev_gray

    # prev_gray should have changed after the second frame
    assert not np.array_equal(gray1, gray2)


def test_flow_motion_high_threshold_suppresses_motion() -> None:
    """A very high threshold should suppress all motion even for different frames."""
    motion = FlowMotion(history=1, threshold=1e6)

    rng = np.random.default_rng(7)
    frame1 = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    frame2 = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)

    motion.update_background(frame1)
    result = motion.update_background(frame2)

    assert np.all(result == 0)


def test_flow_motion_mask_values_are_0_or_255() -> None:
    """Motion mask output values must be exactly 0 or 255."""
    motion = FlowMotion(history=1, threshold=0.1)

    rng = np.random.default_rng(99)
    frame1 = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    frame2 = np.roll(frame1, 15, axis=0)

    motion.update_background(frame1)
    result = motion.update_background(frame2)

    unique_values = np.unique(result)
    assert all(v in (0, 255) for v in unique_values)


def test_flow_motion_handle_returns_valid_frame(video_frame_generator: Generator[Frame]) -> None:
    """FlowMotion.handle returns valid Frame objects with motion_proportion in [0,1]."""
    motion = FlowMotion(history=5, threshold=1.0, kernel_size=0.01)

    for frame in video_frame_generator():
        result = motion.handle(frame)
        assert isinstance(result, Frame)
        assert result.frame_no == frame.frame_no
        if result.frame_no > motion.history:
            assert 0.0 <= result.motion_proportion <= 1.0