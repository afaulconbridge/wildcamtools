from collections.abc import Callable, Generator

import numpy as np
import pytest

from wildcamtools.lib import Frame
from wildcamtools.lib.motion import AvgMotion, FlowMotion, MogMotion


def test_motion_mog(video_frame_generator: Callable[[], Generator[Frame]]):
    motion_mog = MogMotion(history=1, threshold=16, detect_shadows=False, kernel_size=0.005)

    for frame in video_frame_generator():
        frame = motion_mog.handle(frame)
        if frame.frame_no > 10:
            prop = frame.motion_proportion
            print(f"Frame {frame.frame_no} : {prop * 100:.2f}%")
            assert 0.0 <= prop <= 1.0


def test_motion_mog_with_resolution(video_frame_generator: Callable[[], Generator[Frame]]):
    motion_mog = MogMotion(
        history=1,
        threshold=16,
        detect_shadows=False,
        kernel_size=0.005,
        resolution=(320, 240),
    )

    for frame in video_frame_generator():
        frame = motion_mog.handle(frame)
        if frame.frame_no > 10:
            prop = frame.motion_proportion
            print(f"Frame {frame.frame_no} : {prop * 100:.2f}%")
            assert 0.0 <= prop <= 1.0


def test_motion_flow_with_resolution(video_frame_generator: Callable[[], Generator[Frame]]):
    motion_flow = FlowMotion(history=1, threshold=1.0, kernel_size=0.005, resolution=(320, 240))

    for frame in video_frame_generator():
        frame = motion_flow.handle(frame)
        if frame.frame_no > 10:
            prop = frame.motion_proportion
            print(f"Frame {frame.frame_no} : {prop * 100:.2f}%")
            assert 0.0 <= prop <= 1.0


def test_motion_avg_with_resolution(video_frame_generator: Callable[[], Generator[Frame]]):
    motion_avg = AvgMotion(history=1, threshold=16, kernel_size=0.005, resolution=(320, 240))

    for frame in video_frame_generator():
        frame = motion_avg.handle(frame)
        if frame.frame_no > 10:
            prop = frame.motion_proportion
            print(f"Frame {frame.frame_no} : {prop * 100:.2f}%")
            assert 0.0 <= prop <= 1.0


def test_motion_resolution_initialization():
    mog = MogMotion(history=1, resolution=(640, 480))
    assert mog.resolution == (640, 480)

    flow = FlowMotion(history=1, resolution=(320, 240))
    assert flow.resolution == (320, 240)

    avg = AvgMotion(history=1, resolution=(160, 120))
    assert avg.resolution == (160, 120)


def test_motion_default_resolution_is_none():
    mog = MogMotion(history=1)
    assert mog.resolution is None

    flow = FlowMotion(history=1)
    assert flow.resolution is None

    avg = AvgMotion(history=1)
    assert avg.resolution is None


def test_motion_resolution_processes_frames():
    raw = np.zeros((100, 200, 3), dtype=np.uint8)
    raw[:, :] = [100, 100, 100]
    frame = Frame(raw=raw, frame_no=0)

    motion = MogMotion(history=1, resolution=(50, 50))
    result = motion.handle(frame)

    assert result.motion_proportion == -1.0
    assert result.frame_no == 0

    frame2 = Frame(raw=raw, frame_no=5)
    result2 = motion.handle(frame2)
    assert result2.motion_proportion >= 0.0


def test_motion_exclusion_mask_resized_with_resolution():
    motion_mask = np.zeros((100, 200), dtype=np.uint8)
    motion_mask[50:75, 100:150] = 255

    motion = MogMotion(history=1, resolution=(50, 50), motion_mask=motion_mask)

    assert motion.exclusion_mask is not None
    assert motion.exclusion_mask.shape == (50, 50)


def test_motion_exclusion_mask_without_resolution():
    motion_mask = np.zeros((100, 200), dtype=np.uint8)
    motion_mask[50:75, 100:150] = 255

    motion = MogMotion(history=1, motion_mask=motion_mask)

    assert motion.exclusion_mask is motion_mask
    assert motion.exclusion_mask.shape == (100, 200)


def test_motion_exclusion_mask_none_with_resolution():
    motion = MogMotion(history=1, resolution=(50, 50))

    assert motion.exclusion_mask is None


def test_motion_bounding_boxes_scaled_to_original_resolution(video_frame_generator):
    frames = list(video_frame_generator())
    if len(frames) < 10:
        pytest.skip("Need at least 10 frames")

    frame = frames[0]
    original_h, original_w = frame.raw.shape[:2]

    motion = MogMotion(history=1, resolution=(50, 50), threshold=10)

    for frame in frames[:10]:
        motion.handle(frame)

    bboxes = motion.get_contour_bboxes()

    if bboxes:
        for bbox in bboxes:
            assert bbox.x1 <= original_w
            assert bbox.y1 <= original_h
            assert bbox.x2 <= original_w
            assert bbox.y2 <= original_h


def test_motion_exclusion_mask_filtering_with_resolution():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:, :] = 255

    motion = MogMotion(history=1, motion_mask=mask, resolution=(50, 50))

    assert motion.exclusion_mask.shape == (50, 50)

    motion_mask = np.zeros((50, 50), dtype=np.uint8)
    motion_mask[10:20, 10:20] = 255
    motion_mask[30:40, 30:40] = 255
    motion.motion_mask = motion_mask

    bboxes = motion.get_contour_bboxes()

    assert len(bboxes) == 1
    assert bboxes[0].y2 < 30
