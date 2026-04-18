from collections.abc import Generator

import numpy as np

from wildcamtools.lib import Frame
from wildcamtools.lib.motion import AvgMotion, FlowMotion, MogMotion


def test_motion_mog(video_frame_generator: Generator[np.ndarray]):
    motion_mog = MogMotion(history=1, threshold=16, detect_shadows=False, kernel_size=3)

    for frame in video_frame_generator():
        frame = motion_mog.handle(frame)
        if frame.frame_no > 10:
            prop = frame.motion_proportion
            print(f"Frame {frame.frame_no} : {prop * 100:.2f}%")
            assert 0.0 <= prop <= 1.0


# --- _compute_kernel tests ---


def _make_rgb_frame(height: int, width: int) -> np.ndarray:
    """Create a solid grey RGB frame of given dimensions."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def test_compute_kernel_minimum_is_3():
    """Very small frames should still produce a minimum kernel of 3x3."""
    handler = MogMotion(history=1, kernel_size=0.01)
    # 10x10 frame: max_dim=10, k_size=int(10*0.01)=0, clamped to max(3,...)=3
    small_frame = _make_rgb_frame(10, 10)
    kernel = handler._compute_kernel(small_frame)
    assert kernel.shape == (3, 3)
    assert np.all(kernel == 1)


def test_compute_kernel_is_odd():
    """Kernel size must always be odd."""
    handler = MogMotion(history=1, kernel_size=0.01)
    for dim in [100, 200, 300, 400, 500, 640, 1280, 1920, 3840]:
        frame = _make_rgb_frame(dim, dim)
        kernel = handler._compute_kernel(frame)
        k = kernel.shape[0]
        assert k % 2 == 1, f"Kernel size {k} for dim {dim} is not odd"


def test_compute_kernel_proportional_to_max_dimension():
    """Larger frames produce larger kernels proportional to max dimension."""
    handler = MogMotion(history=1, kernel_size=0.1)
    small_frame = _make_rgb_frame(100, 100)
    large_frame = _make_rgb_frame(1000, 1000)
    small_k = handler._compute_kernel(small_frame).shape[0]
    large_k = handler._compute_kernel(large_frame).shape[0]
    assert large_k > small_k


def test_compute_kernel_uses_max_of_height_width():
    """Kernel should be based on the larger of height/width, not just width."""
    handler = MogMotion(history=1, kernel_size=0.1)
    # 200 tall, 100 wide — max dim is 200
    tall_frame = _make_rgb_frame(200, 100)
    # 100 tall, 200 wide — max dim is 200
    wide_frame = _make_rgb_frame(100, 200)
    k_tall = handler._compute_kernel(tall_frame).shape[0]
    k_wide = handler._compute_kernel(wide_frame).shape[0]
    assert k_tall == k_wide


def test_compute_kernel_even_adjusted_to_odd():
    """When raw k_size is even, it must be decremented to an odd number."""
    # kernel_size=0.02, frame dim=200: k_size=int(200*0.02)=4 (even), adjusted to 3
    handler = MogMotion(history=1, kernel_size=0.02)
    frame = _make_rgb_frame(200, 200)
    kernel = handler._compute_kernel(frame)
    assert kernel.shape[0] % 2 == 1


# --- FlowMotion tests ---


def test_flow_motion_instantiation():
    """FlowMotion should initialise with sensible defaults."""
    fm = FlowMotion()
    assert fm.history == 500
    assert fm.threshold == 1.0
    assert fm.kernel_size == 0.01
    assert fm.prev_gray is None


def test_flow_motion_custom_params():
    """FlowMotion should accept custom parameters."""
    fm = FlowMotion(history=25, threshold=5.0, kernel_size=0.02)
    assert fm.history == 25
    assert fm.threshold == 5.0
    assert fm.kernel_size == 0.02


def test_flow_motion_first_frame_returns_zeros():
    """First call to update_background should return an all-zero mask."""
    fm = FlowMotion(history=1, threshold=1.0)
    frame = _make_rgb_frame(100, 100)
    result = fm.update_background(frame)
    assert result.shape == (100, 100)
    assert np.all(result == 0)


def test_flow_motion_stores_prev_gray_after_first_frame():
    """After the first frame, prev_gray must be set."""
    fm = FlowMotion(history=1, threshold=1.0)
    frame = _make_rgb_frame(80, 80)
    fm.update_background(frame)
    assert fm.prev_gray is not None
    assert fm.prev_gray.shape == (80, 80)


def test_flow_motion_second_frame_returns_mask():
    """Second call should return a binary (0/255) mask of the same spatial size."""
    fm = FlowMotion(history=1, threshold=1.0)
    frame1 = _make_rgb_frame(60, 60)
    frame2 = _make_rgb_frame(60, 60)
    fm.update_background(frame1)
    result = fm.update_background(frame2)
    assert result.shape == (60, 60)
    # All values must be either 0 or 255
    unique_vals = np.unique(result)
    assert set(unique_vals.tolist()).issubset({0, 255})


def test_flow_motion_high_threshold_produces_no_motion():
    """With a very high threshold, identical frames should produce zero motion."""
    fm = FlowMotion(history=1, threshold=1e9)
    frame = _make_rgb_frame(64, 64)
    fm.update_background(frame)
    result = fm.update_background(frame)
    assert np.all(result == 0)


def test_flow_motion_low_threshold_detects_motion():
    """With a very low threshold, a frame shifted by 1 pixel should trigger motion."""
    fm = FlowMotion(history=1, threshold=0.001)
    # Two different frames: solid colour vs noisy
    rng = np.random.default_rng(42)
    frame1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    frame2 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    fm.update_background(frame1)
    result = fm.update_background(frame2)
    assert np.any(result == 255)


def test_flow_motion_updates_prev_gray_each_frame():
    """prev_gray should be updated to the latest frame after each call."""
    fm = FlowMotion(history=1, threshold=1.0)
    frame1 = _make_rgb_frame(50, 50)
    frame2 = np.full((50, 50, 3), 200, dtype=np.uint8)
    fm.update_background(frame1)
    prev_after_frame1 = fm.prev_gray.copy()
    fm.update_background(frame2)
    prev_after_frame2 = fm.prev_gray
    assert not np.array_equal(prev_after_frame1, prev_after_frame2)


def test_flow_motion_handle_before_history_returns_minus_one():
    """handle() should return motion_proportion=-1 for frames within the history window."""
    fm = FlowMotion(history=10, threshold=1.0)
    raw = _make_rgb_frame(64, 64)
    # frame_no=5 is within history=10
    frame_in = Frame(raw=raw, frame_no=5)
    frame_out = fm.handle(frame_in)
    assert frame_out.motion_proportion == -1.0


def test_flow_motion_handle_after_history_returns_proportion():
    """handle() should return a valid proportion for frames past the history window."""
    fm = FlowMotion(history=1, threshold=1.0)
    raw = _make_rgb_frame(64, 64)
    # Prime the prev_gray with frame 0
    fm.handle(Frame(raw=raw, frame_no=0))
    frame_out = fm.handle(Frame(raw=raw, frame_no=2))
    assert 0.0 <= frame_out.motion_proportion <= 1.0


def test_flow_motion_with_video(video_frame_generator: Generator[np.ndarray]):
    """Integration: FlowMotion proportions are in [0, 1] for all post-history frames."""
    fm = FlowMotion(history=1, threshold=5.0)
    for frame in video_frame_generator():
        result = fm.handle(frame)
        if result.frame_no > fm.history:
            assert 0.0 <= result.motion_proportion <= 1.0


# --- Float kernel_size compatibility tests ---


def test_mog_motion_float_kernel_size(video_frame_generator: Generator[np.ndarray]):
    """MogMotion should work correctly with a float kernel_size."""
    motion_mog = MogMotion(history=1, threshold=16, detect_shadows=False, kernel_size=0.01)
    for frame in video_frame_generator():
        result = motion_mog.handle(frame)
        if result.frame_no > motion_mog.history:
            assert 0.0 <= result.motion_proportion <= 1.0


def test_avg_motion_float_kernel_size(video_frame_generator: Generator[np.ndarray]):
    """AvgMotion should work correctly with a float kernel_size."""
    motion_avg = AvgMotion(history=1, threshold=16, kernel_size=0.01)
    for frame in video_frame_generator():
        result = motion_avg.handle(frame)
        if result.frame_no > motion_avg.history:
            assert 0.0 <= result.motion_proportion <= 1.0


def test_flow_motion_zero_kernel_skips_morphology():
    """kernel_size=0 should skip morphological operations (no crash, still returns mask)."""
    fm = FlowMotion(history=1, threshold=1.0, kernel_size=0.0)
    frame1 = _make_rgb_frame(64, 64)
    frame2 = _make_rgb_frame(64, 64)
    fm.update_background(frame1)
    # handle() should not raise even when morphology is skipped
    result = fm.handle(Frame(raw=frame2, frame_no=2))
    assert result.motion_proportion >= 0.0 or result.motion_proportion == -1.0