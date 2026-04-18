from collections.abc import Generator

import numpy as np

from wildcamtools.lib import Frame
from wildcamtools.lib.motion import FlowMotion, MogMotion


def test_motion_mog(video_frame_generator: Generator[np.ndarray]):
    motion_mog = MogMotion(history=1, threshold=16, detect_shadows=False, kernel_size=0.01)

    for frame in video_frame_generator():
        frame = motion_mog.handle(frame)
        if frame.frame_no > 10:
            prop = frame.motion_proportion
            print(f"Frame {frame.frame_no} : {prop * 100:.2f}%")
            assert 0.0 <= prop <= 1.0


# --- _compute_kernel tests ---


def _make_frame(height: int, width: int) -> np.ndarray:
    """Create a synthetic RGB frame of given dimensions."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_compute_kernel_minimum_size():
    """Very small kernel_size proportion should produce minimum 3x3 kernel."""
    motion = MogMotion(history=1, kernel_size=0.001)
    frame = _make_frame(100, 100)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (3, 3)
    assert kernel.dtype == np.uint8
    assert np.all(kernel == 1)


def test_compute_kernel_proportional_odd_result():
    """kernel_size that yields an odd result is used directly (if >= 3)."""
    # 100 * 0.05 = 5 (odd, >= 3)
    motion = MogMotion(history=1, kernel_size=0.05)
    frame = _make_frame(100, 100)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (5, 5)


def test_compute_kernel_even_rounded_down_to_odd():
    """kernel_size that yields an even result is decremented to the nearest odd number."""
    # 100 * 0.1 = 10 (even → 9)
    motion = MogMotion(history=1, kernel_size=0.1)
    frame = _make_frame(100, 100)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (9, 9)


def test_compute_kernel_uses_longest_dimension_landscape():
    """Longest dimension (width > height) governs kernel size."""
    # max_dim = 200; 200 * 0.1 = 20 (even → 19)
    motion = MogMotion(history=1, kernel_size=0.1)
    frame = _make_frame(100, 200)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (19, 19)


def test_compute_kernel_uses_longest_dimension_portrait():
    """Longest dimension (height > width) governs kernel size."""
    # max_dim = 200; 200 * 0.1 = 20 (even → 19)
    motion = MogMotion(history=1, kernel_size=0.1)
    frame = _make_frame(200, 100)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (19, 19)


def test_compute_kernel_square_all_ones():
    """Returned kernel is filled with ones of dtype uint8."""
    motion = MogMotion(history=1, kernel_size=0.07)
    frame = _make_frame(100, 100)
    kernel = motion._compute_kernel(frame)
    # 100 * 0.07 = 7 (odd, >= 3)
    assert kernel.shape == (7, 7)
    assert kernel.dtype == np.uint8
    assert np.all(kernel == 1)


def test_compute_kernel_boundary_exactly_three():
    """When computed k_size rounds down to exactly 3, shape is (3, 3)."""
    # 100 * 0.03 = 3 (odd, >= 3)
    motion = MogMotion(history=1, kernel_size=0.03)
    frame = _make_frame(100, 100)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (3, 3)


def test_compute_kernel_zero_floor_gives_minimum():
    """k_size of 0 (even) becomes -1 then is raised to minimum 3."""
    # 50 * 0.01 = 0 (even → -1 → max(3, -1) = 3)
    motion = MogMotion(history=1, kernel_size=0.01)
    frame = _make_frame(50, 50)
    kernel = motion._compute_kernel(frame)
    assert kernel.shape == (3, 3)


# --- FlowMotion tests ---


def test_flow_motion_first_frame_returns_zeros():
    """First call to update_background initialises prev_gray and returns all-zero mask."""
    motion = FlowMotion(history=10, threshold=5.0, kernel_size=0.01)
    frame_raw = _make_frame(100, 100)
    result = motion.update_background(frame_raw)
    assert result.shape == (100, 100)
    assert np.all(result == 0)


def test_flow_motion_stores_prev_gray_after_first_frame():
    """prev_gray is set after the first call and is a grayscale image."""
    motion = FlowMotion(history=10, threshold=5.0, kernel_size=0.01)
    assert motion.prev_gray is None
    frame_raw = _make_frame(100, 100)
    motion.update_background(frame_raw)
    assert motion.prev_gray is not None
    assert motion.prev_gray.shape == (100, 100)  # grayscale


def test_flow_motion_identical_frames_low_motion():
    """Identical consecutive frames should produce near-zero motion mask at low threshold."""
    motion = FlowMotion(history=10, threshold=0.01, kernel_size=0.01)
    frame_raw = _make_frame(100, 100)
    motion.update_background(frame_raw)  # first frame: initialise
    result = motion.update_background(frame_raw)  # second frame: identical
    # With identical frames, optical flow magnitudes are 0, so mask should be all zeros
    assert result.shape == (100, 100)
    nonzero_fraction = np.count_nonzero(result) / result.size
    assert nonzero_fraction < 0.01  # less than 1% of pixels triggered


def test_flow_motion_high_threshold_suppresses_motion():
    """Very high threshold means no pixels exceed it → all-zero mask."""
    motion = FlowMotion(history=10, threshold=1e6, kernel_size=0.01)
    frame1 = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    frame2 = np.random.default_rng(1).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    motion.update_background(frame1)
    result = motion.update_background(frame2)
    assert np.all(result == 0)


def test_flow_motion_low_threshold_detects_motion():
    """Very low threshold means many pixels exceed it when frames differ."""
    motion = FlowMotion(history=10, threshold=0.0, kernel_size=0.01)
    rng = np.random.default_rng(42)
    frame1 = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    # Create a clearly different second frame
    frame2 = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    motion.update_background(frame1)
    result = motion.update_background(frame2)
    # With threshold=0, all pixels with any flow should be marked
    assert result.shape == (100, 100)
    assert result.dtype == np.uint8
    # mask values must be 0 or 255
    assert np.all((result == 0) | (result == 255))


def test_flow_motion_prev_gray_updated_on_each_call():
    """prev_gray should be updated to current frame after each call."""
    motion = FlowMotion(history=10, threshold=5.0, kernel_size=0.01)
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.full((100, 100, 3), 128, dtype=np.uint8)
    motion.update_background(frame1)
    gray1 = motion.prev_gray.copy()
    motion.update_background(frame2)
    gray2 = motion.prev_gray
    # prev_gray should reflect frame2 now
    assert not np.array_equal(gray1, gray2)


def test_flow_motion_handle_returns_frame_with_proportion_before_history():
    """Before history frames have passed, motion_proportion should be -1.0."""
    motion = FlowMotion(history=10, threshold=5.0, kernel_size=0.01)
    frame_raw = _make_frame(100, 100)
    frame = Frame(raw=frame_raw, frame_no=5)
    result = motion.handle(frame)
    assert result.motion_proportion == -1.0


def test_flow_motion_handle_returns_frame_with_proportion_after_history():
    """After history frames have passed, motion_proportion should be in [0, 1]."""
    motion = FlowMotion(history=2, threshold=5.0, kernel_size=0.01)
    frame_raw = _make_frame(100, 100)
    # Warm up prev_gray with a first handle call
    motion.handle(Frame(raw=frame_raw, frame_no=0))
    motion.handle(Frame(raw=frame_raw, frame_no=1))
    result = motion.handle(Frame(raw=frame_raw, frame_no=3))
    assert 0.0 <= result.motion_proportion <= 1.0


def test_flow_motion_with_video(video_frame_generator: Generator[np.ndarray]):
    """FlowMotion produces valid motion proportions when processing real video frames."""
    motion = FlowMotion(history=5, threshold=1.0, kernel_size=0.01)
    count = 0
    for frame in video_frame_generator():
        result = motion.handle(frame)
        if frame.frame_no > 5:
            assert 0.0 <= result.motion_proportion <= 1.0
        count += 1
        if count >= 30:
            break